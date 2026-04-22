import json
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import tqdm

from src.interface import HybridDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_dense(base: Path, dataset_name: str):
    """Load dense corpus embeddings via memmap."""
    dense_path     = base / f"{dataset_name}_dense_corpus.npy"
    dense_ids_path = base / f"{dataset_name}_dense_corpus_ids.npy"
    logger.info("Loading dense embeddings from %s", dense_path)
    dense     = np.load(str(dense_path), mmap_mode="r")
    dense_ids = np.load(str(dense_ids_path), allow_pickle=True)
    n_docs, dim = dense.shape
    logger.info("Dense: %d vectors, dim=%d", n_docs, dim)
    return dense, dense_ids, n_docs, dim


def load_sparse(base: Path, dataset_name: str, sparse_mode: str):
    """Load sparse corpus flat arrays via memmap."""
    sp_base = base / f"{dataset_name}_sparse_corpus_{sparse_mode}"
    logger.info("Loading sparse embeddings from %s_*.npy", sp_base)
    sp_values  = np.load(str(sp_base) + "_values.npy",     mmap_mode="r")
    sp_indices = np.load(str(sp_base) + "_col_indices.npy", mmap_mode="r")
    sp_indptr  = np.load(str(sp_base) + "_indptr.npy")
    sp_ids     = np.load(str(sp_base) + "_ids.npy",         allow_pickle=True)
    logger.info("Sparse: %d vectors, %d nnz total", len(sp_ids), sp_indptr[-1])
    return sp_values, sp_indices, sp_indptr, sp_ids


def save_index_performance(
    output_base: str,
    index_name: str,
    upsert_times_ms: List[float],
    total_inserted: int,
    total_time_sec: float,
    batch_size: int,
):
    out_dir = Path(output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{index_name}.json"

    t = np.array(upsert_times_ms)
    performance = {
        "index_name":        index_name,
        "total_vectors":     total_inserted,
        "total_batches":     len(upsert_times_ms),
        "batch_size":        batch_size,
        "total_time_sec":    round(total_time_sec, 3),
        "upsert_latency_ms": {
            "min":  round(float(t.min()),                  3),
            "p50":  round(float(np.percentile(t, 50)),     3),
            "p95":  round(float(np.percentile(t, 95)),     3),
            "p99":  round(float(np.percentile(t, 99)),     3),
            "max":  round(float(t.max()),                  3),
            "mean": round(float(t.mean()),                 3),
        },
    }

    with open(out_file, "w") as f:
        json.dump(performance, f, indent=2)
    logger.info("Indexing performance saved to %s", out_file)


def index_from_npy(
    db: HybridDB,
    index_name: str,
    data_dir: str,
    dataset_name: str,
    output_base: str,
    sparse_mode: str = "bm25",
    batch_size: int = 1000,
    texts: dict = None,
):
    """
    Load .npy embeddings and index them using the provided HybridDB instance.

    Expects db.init() to have already been called before this function.

    Files expected under data_dir/dataset_name/:
      <dataset_name>_dense_corpus.npy
      <dataset_name>_dense_corpus_ids.npy
      <dataset_name>_sparse_corpus_<sparse_mode>_values.npy
      <dataset_name>_sparse_corpus_<sparse_mode>_col_indices.npy
      <dataset_name>_sparse_corpus_<sparse_mode>_indptr.npy
      <dataset_name>_sparse_corpus_<sparse_mode>_ids.npy
    """
    base = Path(data_dir) / dataset_name

    dense, dense_ids, n_docs, _      = load_dense(base, dataset_name)
    sp_values, sp_indices, sp_indptr, sp_ids = load_sparse(base, dataset_name, sparse_mode)
    n_sparse = len(sp_ids)

    upsert_times: List[float] = []
    total_inserted = 0
    dense_ptr = 0
    start_time = time.perf_counter()

    try:
        for batch_start in tqdm.tqdm(range(0, n_sparse, batch_size), unit="batch"):
            batch_end = min(batch_start + batch_size, n_sparse)

            points = []
            for j in range(batch_start, batch_end):
                sp_doc_id = str(sp_ids[j])

                while dense_ptr < n_docs and str(dense_ids[dense_ptr]) != sp_doc_id:
                    dense_ptr += 1

                if dense_ptr >= n_docs:
                    logger.warning("Dense doc not found for sparse doc %s, skipping", sp_doc_id)
                    continue

                dv = dense[dense_ptr].tolist()
                dense_ptr += 1

                s, e = int(sp_indptr[j]), int(sp_indptr[j + 1])
                point = {
                    "id":             sp_doc_id,
                    "vector":         dv,
                    "sparse_indices": sp_indices[s:e].tolist(),
                    "sparse_values":  sp_values[s:e].tolist(),
                    "meta":           {"id": sp_doc_id},
                }
                if texts is not None:
                    point["text"] = texts.get(sp_doc_id, "")
                points.append(point)

            t0 = time.perf_counter()
            db.index_batch(points)
            upsert_times.append((time.perf_counter() - t0) * 1000)
            total_inserted += len(points)

        total_time_sec = time.perf_counter() - start_time
        logger.info("Indexing complete: %d vectors in %.2f seconds", total_inserted, total_time_sec)
        save_index_performance(output_base, index_name, upsert_times, total_inserted, total_time_sec, batch_size)

    except Exception as e:
        logger.error("Indexing failed: %s", e)
        raise
