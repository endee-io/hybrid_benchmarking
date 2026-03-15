import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tqdm
from datasets import load_dataset
from endee import Endee

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEV_PATH = "https://dev.endee.io/api/v1"


class EndeeIndexing:

    def __init__(self, token: str, base_url: str = DEV_PATH):
        self.vx = Endee(token=token)
        self.vx.set_base_url(base_url)
        self.base_url = base_url
        logger.info("Connected to Endee at %s", base_url)

    def create_index(
        self,
        name: str,
        dimension: int,
        space_type: str = "cosine",
        sparse_dim: int = 30522,
        sparse_scoring_model: str = "default",
        precision: str = "float32",
    ):
        result = self.vx.create_index(
            name=name,
            dimension=dimension,
            space_type=space_type,
            sparse_dim=sparse_dim,
            sparse_scoring_model=sparse_scoring_model,
            precision=precision,
        )
        logger.info("Index '%s' created", name)
        return result

    def get_index(self, name: str):
        return self.vx.get_index(name)

    def delete_index(self, name: str):
        result = self.vx.delete_index(name)
        logger.info("Index '%s' deleted", name)
        return result

    @staticmethod
    def _safe_load(raw: Any) -> Optional[Dict]:
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception as e:
            logger.warning("Failed to parse JSON: %s", e)
            return None

    @staticmethod
    def _streaming_batches(stream_iter, batch_size: int):
        """
        A Python generator that builds and yields batches at runtime without
        precomputing all batches upfront.

        Iterates through stream_iter record by record, validates each one, and
        appends valid records to a batch. When the batch reaches batch_size it
        is yielded to the caller — pausing here — while the caller processes
        (builds points + upserts) that batch. On the next caller iteration the
        generator resumes exactly where it paused, resets the batch to empty,
        and continues reading from stream_iter.

        At any point only one batch worth of records is held in memory, so the
        full dataset is never loaded at once. The final partial batch (if any
        records remain after the last full batch) is yielded at the end.
        """
        batch: List[Dict] = []
        for rec_raw in stream_iter:
            rec = EndeeIndexing._safe_load(rec_raw)
            if rec is None:
                continue
            meta = rec.get("meta", {})
            pid_raw = meta.get("id") or rec.get("id")
            dv = rec.get("dense_vector")
            sv = rec.get("sparse_vector")
            text = meta.get("text")
            if not isinstance(text, str):
                logger.warning("Skipping record missing text: %s", pid_raw)
                continue
            if not (isinstance(dv, list) and len(dv) == 384):
                logger.warning("Skipping %s: wrong dense dimension", pid_raw)
                continue
            if not (
                isinstance(sv, dict)
                and isinstance(sv.get("indices"), list)
                and isinstance(sv.get("values"), list)
                and len(sv["indices"]) == len(sv["values"])
            ):
                logger.warning("Skipping %s: invalid sparse indices/values", pid_raw)
                continue
            if len(sv["indices"]) == 0:
                logger.warning("Record %s has empty sparse vector, inserting dense only", pid_raw)
            batch.append(rec)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def index_from_jsonl(
        self,
        index_name: str,
        jsonl_path: str,
        batch_size: int = 1000,
        max_retries: int = 15,
        cache_dir: str = None,
    ):
        index = self.get_index(index_name)
        dataset = load_dataset(
            "json", data_files=jsonl_path, split="train", streaming=False, cache_dir=cache_dir
        )
        batches = self._streaming_batches(dataset, batch_size)
        upsert_times: List[float] = []
        total_inserted = 0
        start_time = time.perf_counter()

        for _, batch in enumerate(tqdm.tqdm(batches, unit="batch"), start=1):
            points = []
            for rec in batch:
                pid_raw = rec["meta"].get("id") or rec.get("id")
                points.append({
                    "id": str(pid_raw),
                    "vector": rec["dense_vector"],
                    "sparse_indices": rec["sparse_vector"]["indices"],
                    "sparse_values": rec["sparse_vector"]["values"],
                    "meta": {
                        "text": rec["meta"]["text"],
                        "id": pid_raw,
                    },
                })
            t0 = time.perf_counter()
            for attempt in range(max_retries):
                try:
                    index.upsert(points)
                    break
                except Exception as e:
                    print(f"⚠️ Server busy (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1.5)
                    else:
                        raise
            upsert_times.append((time.perf_counter() - t0) * 1000)
            total_inserted += len(points)

        total_time_sec = time.perf_counter() - start_time
        logger.info("Indexing complete in %.2f seconds", total_time_sec)

        self._save_index_performance(index_name, upsert_times, total_inserted, total_time_sec, batch_size)

    def _save_index_performance(
        self,
        index_name: str,
        upsert_times_ms: List[float],
        total_inserted: int,
        total_time_sec: float,
        batch_size: int,
    ):
        out_dir = Path(index_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "indexing_performance.json"

        t = np.array(upsert_times_ms)
        performance = {
            "index_name":        index_name,
            "total_vectors":     total_inserted,
            "total_batches":     len(upsert_times_ms),
            "batch_size":        batch_size,
            "total_time_sec":    round(total_time_sec, 3),
            "upsert_latency_ms": {
                "min":  round(float(t.min()),                        3),
                "p50":  round(float(np.percentile(t, 50)),           3),
                "p95":  round(float(np.percentile(t, 95)),           3),
                "p99":  round(float(np.percentile(t, 99)),           3),
                "max":  round(float(t.max()),                        3),
                "mean": round(float(t.mean()),                       3),
            },
        }

        with open(out_file, "w") as f:
            json.dump(performance, f, indent=2)
        logger.info("Performance saved to %s", out_file)


def main():
    parser = argparse.ArgumentParser(description="Create an Endee index and ingest vectors from a JSONL file.")
    parser.add_argument("--index_name", help="Name of the index to create and populate")
    parser.add_argument("--jsonl_path", help="Path to the JSONL embeddings file")
    parser.add_argument("--token", default="12345678", help="Endee API token (default: 12345678)")
    parser.add_argument("--base-url", default=DEV_PATH, help=f"Endee base URL (default: {DEV_PATH})")
    parser.add_argument("--batch-size", type=int, default=1000, help="Upsert batch size (default: 1000)")
    parser.add_argument("--create-index", type=lambda x: x.lower() != "false", default=True,
                        help="Create the index before ingesting (default: true)")
    parser.add_argument("--dimension", type=int, default=384, help="Dense vector dimension (default: 384)")
    parser.add_argument("--space-type", default="cosine", help="Distance metric: cosine, dot, euclidean (default: cosine)")
    parser.add_argument("--sparse-dim", type=int, default=30522, help="Sparse vector dimension (default: 30522)")
    parser.add_argument(
        "--sparse-scoring-model",
        default="default",
        help="Sparse scoring model to use, e.g. default or endee_bm25_server_idf (default: default)",
    )
    parser.add_argument("--precision", default="float32", help="Vector precision: float32, float16 (default: float32)")
    parser.add_argument("--cache-dir", default=None, help="Local cache directory for HuggingFace datasets (default: HF default cache)")
    args = parser.parse_args()

    ei = EndeeIndexing(token=args.token, base_url=args.base_url)
    if args.create_index:
        ei.create_index(
            name=args.index_name,
            dimension=args.dimension,
            space_type=args.space_type,
            sparse_dim=args.sparse_dim,
            sparse_scoring_model=args.sparse_scoring_model,
            precision=args.precision,
        )
    ei.index_from_jsonl(
        index_name=args.index_name,
        jsonl_path=args.jsonl_path,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
