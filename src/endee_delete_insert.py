"""
Endee delete/insert runner with scenario controls + persistence of last deletes + insert/delete verification.

Change in this version:
- Duplicate deletes that fail with "not found" are treated as EXPECTED if the ID has already been deleted in this run.
  They are logged at INFO as "Expected duplicate delete" instead of WARNING.
- Duplicate deletes for the same doc_id are issued consecutively (not randomly interleaved).
- The deleted-ids JSON file stores each doc_id only once (deduplicated).
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import List, Set, Optional, Dict, Any

import numpy as np
from datasets import load_dataset as hf_load_dataset

from endee import Endee
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEV_PATH = "https://dev.endee.io/api/v1"

DATASET_CONFIG = {
    "beir_scifact": {"hf_name": "BeIR/scifact-qrels", "split": "train"},
    "beir_quora":   {"hf_name": "BeIR/quora-qrels",   "split": "test"},
}


def load_dataset(dataset_name: str, cache_dir: Optional[str] = None):
    """Load a HuggingFace BeIR qrels dataset by name."""
    cfg = DATASET_CONFIG[dataset_name]
    logger.info("Loading %s split=%s", cfg["hf_name"], cfg["split"])
    return hf_load_dataset(cfg["hf_name"], split=cfg["split"], cache_dir=cache_dir)


def extract_gt_corpus_ids(dataset, dataset_name: str) -> List[str]:
    """Extract and return unique ground-truth corpus IDs from a HuggingFace dataset (no file saving)."""
    unique_ids = sorted(set(str(entry["corpus-id"]) for entry in dataset))
    logger.info("Extracted %d ground-truth corpus IDs for dataset '%s'", len(unique_ids), dataset_name)
    return unique_ids


def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    """Atomically write JSON to path (write temp then replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(p)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def _is_not_found_exc(e: Exception) -> bool:
    """
    Best-effort detection for "not found" errors from Endee client.
    We avoid importing SDK-specific exception types and instead match common text.
    """
    s = str(e).lower()
    return ("resource not found" in s) or ("does not exist" in s) or ("not found" in s)


def _load_npy_arrays(data_dir: str, dataset_name: str, sparse_mode: str):
    """Load dense + sparse corpus npy arrays. Returns mmap'd arrays + ids."""
    base = Path(data_dir) / dataset_name

    dense_path = base / f"{dataset_name}_dense_corpus.npy"
    dense_ids_path = base / f"{dataset_name}_dense_corpus_ids.npy"
    logger.info("Loading dense from %s", dense_path)
    dense = np.load(str(dense_path), mmap_mode="r")
    dense_ids = np.load(str(dense_ids_path), allow_pickle=True)

    sp_base = str(base / f"{dataset_name}_sparse_corpus_{sparse_mode}")
    logger.info("Loading sparse from %s_*.npy", sp_base)
    sp_values  = np.load(sp_base + "_values.npy",      mmap_mode="r")
    sp_indices = np.load(sp_base + "_col_indices.npy",  mmap_mode="r")
    sp_indptr  = np.load(sp_base + "_indptr.npy")
    sp_ids     = np.load(sp_base + "_ids.npy",          allow_pickle=True)

    logger.info("Dense: %d vectors  Sparse: %d vectors", len(dense_ids), len(sp_ids))
    return dense, dense_ids, sp_values, sp_indices, sp_indptr, sp_ids


class EndeeDeleteInsert:
    def __init__(self, token: str, base_url: str, index_name: str):
        vx = Endee(token=token)
        vx.set_base_url(base_url)
        self.index = vx.get_index(index_name)
        self.index_name = index_name
        total_vectors = self.index.count
        logger.info("Index '%s' has %d vectors", index_name, total_vectors)

    @staticmethod
    def load_gt_corpus_ids_from_hf(dataset_name: str, cache_dir: Optional[str] = None) -> List[str]:
        """Load HuggingFace dataset and return unique ground-truth corpus IDs (no file I/O)."""
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        return extract_gt_corpus_ids(dataset, dataset_name)

    def load_all_record_ids(self, data_dir: str, dataset_name: str, sparse_mode: str) -> List[str]:
        """Load all corpus record IDs from the sparse npy ids file."""
        base = Path(data_dir) / dataset_name
        sp_ids_path = base / f"{dataset_name}_sparse_corpus_{sparse_mode}_ids.npy"
        sp_ids = np.load(str(sp_ids_path), allow_pickle=True)
        all_ids = [str(sid) for sid in sp_ids]
        logger.info("Total corpus records: %d", len(all_ids))
        return all_ids

    @staticmethod
    def check_data_files(
        data_dir: str, dataset_name: str, sparse_mode: str, need_ground_truth: bool = False
    ) -> None:
        """Verify all required .npy data files exist. Raises FileNotFoundError listing missing files."""
        base = Path(data_dir) / dataset_name
        required = [
            base / f"{dataset_name}_dense_corpus.npy",
            base / f"{dataset_name}_dense_corpus_ids.npy",
            base / f"{dataset_name}_sparse_corpus_{sparse_mode}_values.npy",
            base / f"{dataset_name}_sparse_corpus_{sparse_mode}_col_indices.npy",
            base / f"{dataset_name}_sparse_corpus_{sparse_mode}_indptr.npy",
            base / f"{dataset_name}_sparse_corpus_{sparse_mode}_ids.npy",
        ]
        if need_ground_truth:
            required.append(base / f"{dataset_name}_ground_truth_ids.npy")
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required data files:\n" + "\n".join(f"  {p}" for p in missing)
            )
        logger.info("All required data files present in %s", base)

    @staticmethod
    def load_ground_truth_ids(data_dir: str, dataset_name: str) -> Set[str]:
        """Load ground-truth corpus IDs from the .npy file produced by embedding_creation_v2.py."""
        path = Path(data_dir) / dataset_name / f"{dataset_name}_ground_truth_ids.npy"
        ids_arr = np.load(str(path), allow_pickle=True)
        ids = set(str(x) for x in ids_arr)
        logger.info("Loaded %d ground-truth IDs from %s", len(ids), path)
        return ids

    def select_ground_truth_ids(self, all_record_ids: List[str], ground_truth_ids: Set[str]) -> List[str]:
        """Select IDs that are in both all_record_ids and ground_truth_ids."""
        common_ids = sorted(set(str(i) for i in all_record_ids) & ground_truth_ids)
        logger.info("Selected %d ground-truth IDs in candidate pool", len(common_ids))
        return common_ids

    def select_non_ground_truth_ids(self, all_record_ids: List[str], ground_truth_ids: Set[str]) -> List[str]:
        """Select IDs that are in all_record_ids but not in ground_truth_ids."""
        non_gt_ids = sorted(set(str(i) for i in all_record_ids) - ground_truth_ids)
        logger.info("Selected %d non-ground-truth IDs in candidate pool", len(non_gt_ids))
        return non_gt_ids


    def build_id_list(
        self,
        candidate_pool: List[str],
        percentage: int,
        pick: str,
        order: str,
        repeat: int,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Build an ID list for operations:
          1) pick a subset of IDs from candidate_pool (begin/end/random) based on percentage
          2) order them (seq/reverse/random)
          3) repeat each ID 'repeat' times — duplicates are always consecutive
             (e.g. [A, A, B, B, C, C] not [A, B, C, A, B, C])
        """
        if seed is not None:
            random.seed(seed)

        if not candidate_pool:
            return []

        percentage = max(0, min(int(percentage), 100))
        n = int(len(candidate_pool) * percentage / 100)
        n = max(0, min(n, len(candidate_pool)))

        # 1) pick subset
        if pick == "begin":
            base = candidate_pool[:n]
        elif pick == "end":
            base = candidate_pool[len(candidate_pool) - n :]
        else:  # random
            base = random.sample(candidate_pool, n)

        # 2) order operations
        if order == "seq":
            ordered = list(base)
        elif order == "reverse":
            ordered = list(reversed(base))
        else:  # random order
            ordered = list(base)
            random.shuffle(ordered)

        # 3) duplicate operations — duplicates are always consecutive
        repeat = max(1, int(repeat))
        if repeat == 1:
            return ordered

        expanded: List[str] = []
        for vid in ordered:
            expanded.extend([vid] * repeat)

        return expanded


    def delete_vectors(self, ids_to_delete: List[str]) -> Dict[str, Any]:
        """
        Delete vectors one by one.

        Improvement:
        - If the same ID appears multiple times in ids_to_delete and we get a "not found" error
          after we've already successfully deleted that ID earlier in this run, we treat it as expected.
        """
        delete_success = 0
        delete_fail = 0
        expected_dup_not_found = 0
        start = time.perf_counter()

        deleted_once: Set[str] = set()

        for vid in tqdm(ids_to_delete, desc="Deleting vectors"):
            vid = str(vid)
            try:
                self.index.delete_vector(vid)
                delete_success += 1
                deleted_once.add(vid)
            except Exception as e:
                # Expected case: duplicate delete after a successful delete in this run
                if vid in deleted_once and _is_not_found_exc(e):
                    expected_dup_not_found += 1
                    logger.info("Expected duplicate delete (already deleted in this run): ID %s", vid)
                    continue

                delete_fail += 1
                logger.warning("Failed to delete ID %s: %s", vid, e)

        elapsed = time.perf_counter() - start
        logger.info(
            "Delete complete — success: %d  failed: %d  expected_dup_not_found: %d  time: %.2fs",
            delete_success,
            delete_fail,
            expected_dup_not_found,
            elapsed,
        )
        return {
            "deleted": delete_success,
            "failed": delete_fail,
            "expected_dup_not_found": expected_dup_not_found,
            "elapsed_sec": round(elapsed, 2),
        }

    def verify_deletion(self, ids_to_delete: List[str]) -> Dict[str, Any]:
        """Verify that all deleted IDs are gone from the index."""
        confirmed_deleted = 0
        still_exists = 0

        for vid in tqdm(ids_to_delete, desc="Verifying deletions"):
            try:
                self.index.get_vector(str(vid))
                still_exists += 1
            except Exception:
                confirmed_deleted += 1

        accuracy = confirmed_deleted / len(ids_to_delete) * 100 if ids_to_delete else 0.0
        logger.info(
            "Delete Verification — confirmed deleted: %d  still exists: %d  accuracy: %.2f%%",
            confirmed_deleted, still_exists, accuracy,
        )
        return {
            "confirmed_deleted": confirmed_deleted,
            "still_exists": still_exists,
            "delete_accuracy_pct": round(accuracy, 2),
        }

    def reinsert_vectors(
        self,
        ids_to_insert: List[str],
        data_dir: str,
        dataset_name: str,
        sparse_mode: str,
        batch_size: int = 1000,
        max_retries: int = 5,
        sparse_output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert/upsert vectors whose IDs appear in ids_to_insert from npy files.

        IMPORTANT:
        - Order of ids_to_insert is preserved (no sorting).
        - Duplicates in ids_to_insert will cause repeated upserts of the same ID.
        """
        dense, dense_ids, sp_values, sp_indices, sp_indptr, sp_ids = _load_npy_arrays(
            data_dir, dataset_name, sparse_mode
        )

        # Build fast lookup: id -> row index
        dense_id_to_idx: Dict[str, int] = {str(did): i for i, did in enumerate(dense_ids)}
        sp_id_to_idx: Dict[str, int]    = {str(sid): i for i, sid in enumerate(sp_ids)}

        batch: List[Dict[str, Any]] = []
        upsert_times: List[float] = []
        insert_count = 0
        insert_fail = 0
        not_found = 0
        sparse_log: List[Dict[str, Any]] = []

        def flush_batch(batch_local: List[Dict[str, Any]]) -> None:
            nonlocal insert_count, insert_fail
            t0 = time.perf_counter()
            for attempt in range(max_retries):
                try:
                    self.index.upsert(batch_local)
                    insert_count += len(batch_local)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1.5)
                    else:
                        insert_fail += len(batch_local)
                        logger.error("Batch upsert failed after %d retries: %s", max_retries, e)
            upsert_times.append(time.perf_counter() - t0)
            batch_local.clear()

        start = time.perf_counter()

        for vid in tqdm(ids_to_insert, desc="Inserting vectors"):
            vid = str(vid)
            d_idx = dense_id_to_idx.get(vid)
            s_idx = sp_id_to_idx.get(vid)

            if d_idx is None or s_idx is None:
                not_found += 1
                logger.warning("ID %s not found in npy data (dense=%s, sparse=%s), skipping",
                               vid, d_idx is not None, s_idx is not None)
                continue

            dv = dense[d_idx].tolist()
            s, e = int(sp_indptr[s_idx]), int(sp_indptr[s_idx + 1])

            raw_vals = sp_values[s:e]
            raw_idxs = sp_indices[s:e]
            mask = raw_vals != 0

            filtered_indices = raw_idxs[mask].tolist()
            filtered_values  = raw_vals[mask].tolist()

            if sparse_output_file is not None:
                sparse_log.append({
                    "id": vid,
                    "sparse_indices": filtered_indices,
                    "sparse_values":  filtered_values,
                })

            batch.append({
                "id": vid,
                "vector": dv,
                "sparse_indices": filtered_indices,
                "sparse_values":  filtered_values,
                "meta": {"id": vid},
            })
            if len(batch) >= batch_size:
                flush_batch(batch)

        if batch:
            flush_batch(batch)

        if sparse_output_file is not None:
            _atomic_write_json(sparse_output_file, {"vectors": sparse_log})
            logger.info("Saved sparse data for %d vectors to %s", len(sparse_log), sparse_output_file)

        elapsed = time.perf_counter() - start
        total_upsert = sum(upsert_times)

        summary: Dict[str, Any] = {
            "inserted": insert_count,
            "failed": insert_fail,
            "not_found_in_npy": not_found,
            "total_elapsed_sec": round(elapsed, 2),
            "upsert_time_sec": round(total_upsert, 2),
            "scan_time_sec": round(elapsed - total_upsert, 2),
            "num_batches": len(upsert_times),
        }
        if upsert_times:
            summary["upsert_per_batch"] = {
                "min_sec": round(min(upsert_times), 2),
                "max_sec": round(max(upsert_times), 2),
                "avg_sec": round(total_upsert / len(upsert_times), 2),
            }

        logger.info(
            "Insert complete — inserted: %d  failed: %d  not_found: %d  time: %.2fs",
            insert_count, insert_fail, not_found, elapsed,
        )
        return summary

    def verify_insertion(self, ids_to_insert: List[str]) -> Dict[str, Any]:
        """Verify that all inserted IDs exist in the index."""
        confirmed_present = 0
        missing = 0

        for vid in tqdm(ids_to_insert, desc="Verifying insertions"):
            try:
                self.index.get_vector(str(vid))
                confirmed_present += 1
            except Exception:
                missing += 1

        accuracy = confirmed_present / len(ids_to_insert) * 100 if ids_to_insert else 0.0
        logger.info(
            "Insert Verification — confirmed present: %d  missing: %d  accuracy: %.2f%%",
            confirmed_present, missing, accuracy,
        )
        return {
            "confirmed_present": confirmed_present,
            "missing": missing,
            "insert_accuracy_pct": round(accuracy, 2),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Delete and/or insert Endee vectors from .npy embedding files with configurable scenarios."
    )
    parser.add_argument("--index_name",    required=True, help="Name of the Endee index")
    parser.add_argument("--data-dir",      required=True, help="Root data directory (contains <dataset_name>/ subfolder)")
    parser.add_argument("--dataset-name",  required=True, help="Dataset name (e.g. beir_scifact, beir_quora)")
    parser.add_argument("--sparse-mode",   default="splade", help="Sparse embedding type: splade or bm25 (default: splade)")

    parser.add_argument("--token",    default="12345678", help="Endee API token (default: 12345678)")
    parser.add_argument("--base-url", default=DEV_PATH,   help=f"Endee base URL (default: {DEV_PATH})")

    parser.add_argument(
        "--delete-percentage",
        type=int, default=10,
        help="Percent of candidate pool to pick IDs from (default: 10)",
    )
    parser.add_argument(
        "--delete-pick",
        choices=["random", "begin", "end"], default="random",
        help="Which IDs to pick from candidate pool before ordering (default: random)",
    )
    parser.add_argument(
        "--delete-order",
        choices=["random", "seq", "reverse"], default="random",
        help="Execution order for deletes (default: random)",
    )
    parser.add_argument(
        "--delete-repeat",
        type=int, default=1,
        help="Repeat deletes per selected ID (delete same doc multiple times). Default: 1",
    )

    parser.add_argument(
        "--gt-filter",
        choices=["ground-truth", "non-ground-truth"], default=None,
        help="Filter candidate pool to ground-truth or non-ground-truth IDs before picking; "
             "loads GT corpus IDs from HuggingFace dataset (requires dataset in DATASET_CONFIG)",
    )

    parser.add_argument(
        "--insert-only",
        action="store_true", default=False,
        help="Only insert (no deletes). Uses the same pick/order/percentage logic to choose IDs.",
    )
    parser.add_argument(
        "--insert-order",
        choices=["same-as-delete", "random"], default="same-as-delete",
        help="Insertion order: same as delete list, or random shuffle (default: same-as-delete)",
    )
    parser.add_argument(
        "--insert-repeat",
        type=int, default=1,
        help="Repeat inserts per ID (insert same doc multiple times). Default: 1",
    )

    parser.add_argument("--batch-size", type=int, default=1000, help="Insert batch size (default: 1000)")
    parser.add_argument(
        "--sparse-output-file",
        default=None,
        help="If set, save inserted sparse indices/values to this JSON file (default: disabled)",
    )
    parser.add_argument(
        "--skip-insert",
        action="store_true", default=False,
        help="Skip insertion step entirely (default: false)",
    )
    parser.add_argument(
        "--verify-delete",
        type=lambda x: x.lower() == "true", default=False,
        help="Verify deletions after delete step (default: false)",
    )
    parser.add_argument(
        "--verify-insert",
        type=lambda x: x.lower() == "true", default=False,
        help="Verify insertions after insert step (default: false)",
    )

    parser.add_argument(
        "--seed",
        type=int, default=None,
        help="Optional RNG seed for reproducible random pick/order and insert-order randomization",
    )

    parser.add_argument(
        "--deleted-ids-file",
        default="last_deleted_ids.json",
        help="Path to store IDs deleted in the last run (default: last_deleted_ids.json).",
    )
    parser.add_argument(
        "--use-last-deletes",
        action="store_true", default=False,
        help="Ignore selection logic and load IDs from --deleted-ids-file for insertion (useful for 'insert later').",
    )

    args = parser.parse_args()

    EndeeDeleteInsert.check_data_files(args.data_dir, args.dataset_name, args.sparse_mode)

    edi = EndeeDeleteInsert(token=args.token, base_url=args.base_url, index_name=args.index_name)

    all_record_ids = edi.load_all_record_ids(args.data_dir, args.dataset_name, args.sparse_mode)

    if args.gt_filter:
        gt_ids = set(EndeeDeleteInsert.load_gt_corpus_ids_from_hf(args.dataset_name))
        if args.gt_filter == "ground-truth":
            candidate_pool = edi.select_ground_truth_ids(all_record_ids, gt_ids)
        else:
            candidate_pool = edi.select_non_ground_truth_ids(all_record_ids, gt_ids)
    else:
        candidate_pool = all_record_ids

    if args.use_last_deletes:
        saved = _load_json(args.deleted_ids_file)
        if not saved or "ids" not in saved:
            raise RuntimeError(f"--use-last-deletes set but no valid file found at {args.deleted_ids_file}")
        delete_ids = [str(x) for x in saved["ids"]]
        logger.info("Loaded %d IDs from %s for insertion", len(delete_ids), args.deleted_ids_file)
    else:
        delete_ids = edi.build_id_list(
            candidate_pool=candidate_pool,
            percentage=args.delete_percentage,
            pick=args.delete_pick,
            order=args.delete_order,
            repeat=args.delete_repeat,
            seed=args.seed,
        )

    did_delete = False
    if args.insert_only or args.use_last_deletes:
        if args.insert_only:
            logger.info("--insert-only set: skipping deletes")
        if args.use_last_deletes:
            logger.info("--use-last-deletes set: skipping deletes")
    else:
        edi.delete_vectors(delete_ids)
        did_delete = True
        if args.verify_delete:
            edi.verify_deletion(delete_ids)

        # Deduplicate IDs before saving — each doc_id is stored only once
        seen: Set[str] = set()
        unique_delete_ids: List[str] = []
        for vid in delete_ids:
            if vid not in seen:
                seen.add(vid)
                unique_delete_ids.append(vid)

        payload = {
            "index_name":   args.index_name,
            "base_url":     args.base_url,
            "data_dir":     args.data_dir,
            "dataset_name": args.dataset_name,
            "sparse_mode":  args.sparse_mode,
            "created_at_unix": time.time(),
            "ids": unique_delete_ids,
        }
        _atomic_write_json(args.deleted_ids_file, payload)
        logger.info("Saved %d unique deleted IDs to %s", len(unique_delete_ids), args.deleted_ids_file)

    if args.skip_insert:
        logger.info("--skip-insert set: skipping insertion step")
        return

    insert_ids = list(delete_ids)

    if args.insert_order == "random":
        if args.seed is not None:
            random.seed(args.seed + 1)
        random.shuffle(insert_ids)

    args.insert_repeat = max(1, int(args.insert_repeat))
    if args.insert_repeat > 1:
        expanded: List[str] = []
        for vid in insert_ids:
            expanded.extend([vid] * args.insert_repeat)
        insert_ids = expanded

    edi.reinsert_vectors(
        ids_to_insert=insert_ids,
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        sparse_mode=args.sparse_mode,
        batch_size=args.batch_size,
        sparse_output_file=args.sparse_output_file,
    )

    if args.verify_insert:
        edi.verify_insertion(insert_ids)

    if did_delete:
        logger.info("Run complete: delete+insert finished")
    else:
        logger.info("Run complete: insert finished")


if __name__ == "__main__":
    main()
