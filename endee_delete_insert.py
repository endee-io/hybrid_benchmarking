import argparse
import json
import logging
import random
import time
from typing import List, Set

import numpy as np
from scipy.sparse import csr_matrix

from endee import Endee
from tqdm import tqdm

DUMMY_DENSE_DIM = 10
DUMMY_DENSE_VECTOR = [0.1] * DUMMY_DENSE_DIM


def _read_csr_matrix(path: str) -> csr_matrix:
    """Read a .csr file in spmat format (NeurIPS sparse benchmark format)."""
    with open(path, "rb") as f:
        sizes = np.fromfile(f, dtype="int64", count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype="int64", count=nrow + 1)
        indices = np.fromfile(f, dtype="int32", count=nnz)
        data = np.fromfile(f, dtype="float32", count=nnz)
    return csr_matrix((data, indices, indptr), shape=(int(nrow), int(ncol)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEV_PATH = "https://dev.endee.io/api/v1"


class EndeeDeleteInsert:

    def __init__(self, token: str, base_url: str, index_name: str):
        vx = Endee(token=token)
        vx.set_base_url(base_url)
        self.index = vx.get_index(index_name)
        self.index_name = index_name
        total_vectors = self.index.count
        logger.info("Index '%s' has %d vectors", index_name, total_vectors)

    def load_all_record_ids(self, jsonl_path: str) -> List[str]:
        """Read every record['id'] from the JSONL file into a list."""
        all_ids = []
        with open(jsonl_path, "r") as f:
            for line in tqdm(f, desc="Reading IDs"):
                record = json.loads(line)
                all_ids.append(record["id"])
                del record
        logger.info("Total records in JSONL: %d", len(all_ids))
        return all_ids

    def select_random_ids(self, all_record_ids: List[str], delete_percentage: int) -> List[str]:
        """Randomly sample delete_percentage% of all_record_ids."""
        num_to_delete = int(len(all_record_ids) * delete_percentage / 100)
        ids_to_delete = sorted(random.sample(all_record_ids, num_to_delete))
        logger.info("Selected %d random IDs (%d%%) to delete", len(ids_to_delete), delete_percentage)
        return ids_to_delete

    def select_last_n_percent_ids(self, all_record_ids: List[str], delete_percentage: int) -> List[str]:
        """Select the last delete_percentage% of all_record_ids (tail of the list)."""
        start_idx = len(all_record_ids) * (100 - delete_percentage) // 100
        ids_to_delete = all_record_ids[start_idx:]
        logger.info(
            "Selected last %d%% = %d IDs (from index %d to %d)",
            delete_percentage, len(ids_to_delete), start_idx, len(all_record_ids) - 1,
        )
        return ids_to_delete
    def select_first_n_percent_ids(self, all_record_ids: List[str], delete_percentage: int) -> List[str]:
        """Select the first delete_percentage% of all_record_ids (head of the list)."""
        end_idx = len(all_record_ids) * delete_percentage // 100
        ids_to_delete = all_record_ids[:end_idx]
        logger.info(
            "Selected first %d%% = %d IDs (from index 0 to %d)",
            delete_percentage, len(ids_to_delete), end_idx - 1, 
        )
        return ids_to_delete
    def select_ground_truth_ids(self, all_record_ids: List[str], ground_truth_ids: Set[str]) -> List[str]:
        """Select IDs that are in both all_record_ids and ground_truth_ids."""
        common_ids = sorted(set(str(i) for i in all_record_ids) & ground_truth_ids)
        logger.info("Selected %d ground-truth IDs to delete", len(common_ids))
        return common_ids

    def select_non_ground_truth_ids(self, all_record_ids: List[str], ground_truth_ids: Set[str]) -> List[str]:
        """Select IDs that are in all_record_ids but not in ground_truth_ids."""
        non_gt_ids = sorted(set(str(i) for i in all_record_ids) - ground_truth_ids)
        logger.info("Selected %d non-ground-truth IDs to delete", len(non_gt_ids))
        return non_gt_ids

    @staticmethod
    def load_ground_truth_file(path: str) -> Set[str]:
        """Load a ground-truth ID file (one ID per line) produced by endee_validation.py."""
        ids = set()
        with open(path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    ids.add(stripped)
        logger.info("Loaded %d ground-truth IDs from %s", len(ids), path)
        return ids

    def delete_vectors(self, ids_to_delete: List[str]) -> dict:
        """Delete vectors one by one. Returns a summary dict."""
        delete_success = 0
        delete_fail = 0
        start = time.perf_counter()

        for vid in tqdm(ids_to_delete, desc="Deleting vectors"):
            try:
                self.index.delete_vector(str(vid))
                delete_success += 1
            except Exception as e:
                delete_fail += 1
                logger.warning("Failed to delete ID %s: %s", vid, e)

        elapsed = time.perf_counter() - start
        logger.info(
            "Delete complete — success: %d  failed: %d  time: %.2fs",
            delete_success, delete_fail, elapsed,
        )
        return {"deleted": delete_success, "failed": delete_fail, "elapsed_sec": round(elapsed, 2)}
    

    def verify_deletion(self, ids_to_delete: List[str]) -> dict:
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
            "Verification — confirmed deleted: %d  still exists: %d  accuracy: %.2f%%",
            confirmed_deleted, still_exists, accuracy,
        )
        return {
            "confirmed_deleted": confirmed_deleted,
            "still_exists": still_exists,
            "delete_accuracy_pct": round(accuracy, 2),
        }

    def load_ids_from_csr(self, csr_path: str) -> List[str]:
        """Return all record IDs for a sparse-only index.

        IDs are the string row-indices (0, 1, 2, …) of the CSR matrix.
        Only the header is read so this is fast even for large files.
        """
        with open(csr_path, "rb") as f:
            nrow = int(np.fromfile(f, dtype="int64", count=1)[0])
        all_ids = [str(i) for i in range(nrow)]
        logger.info("Sparse CSR '%s' has %d rows → %d IDs", csr_path, nrow, len(all_ids))
        return all_ids

    def reinsert_vectors(
        self,
        ids_to_delete: List[str],
        jsonl_path: str = None,
        csr_path: str = None,
        batch_size: int = 1000,
        max_retries: int = 5,
    ) -> dict:
        """Reinsert vectors whose IDs appear in ids_to_delete.

        Pass jsonl_path for hybrid/dense indexes or csr_path for sparse-only indexes.
        """
        batch = []
        upsert_times = []
        reinsert_count = 0
        reinsert_fail = 0

        def flush_batch(batch):
            nonlocal reinsert_count, reinsert_fail
            t0 = time.perf_counter()
            for attempt in range(max_retries):
                try:
                    self.index.upsert(batch)
                    reinsert_count += len(batch)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1.5)
                    else:
                        reinsert_fail += len(batch)
                        logger.error("Batch upsert failed after %d retries: %s", max_retries, e)
            upsert_times.append(time.perf_counter() - t0)
            batch.clear()

        start = time.perf_counter()

        if csr_path:
            # Sparse-only mode: IDs are row indices; load matrix and index directly
            logger.info("Loading CSR matrix from '%s' …", csr_path)
            data = _read_csr_matrix(csr_path)
            for row_idx in tqdm(sorted(int(i) for i in ids_to_delete), desc="Reinserting sparse vectors"):
                row = data[row_idx]
                batch.append({
                    "id": str(row_idx),
                    "vector": DUMMY_DENSE_VECTOR,
                    "sparse_indices": row.indices.tolist(),
                    "sparse_values": row.data.tolist(),
                    "meta": {"point_id": row_idx},
                })
                if len(batch) >= batch_size:
                    flush_batch(batch)
            not_found = 0
            scan_time = 0.0
        else:
            # JSONL mode: scan file and match by ID
            target_ids: Set[str] = set(str(i) for i in ids_to_delete)
            found_ids: Set[str] = set()
            with open(jsonl_path, "r") as f:
                for line in tqdm(f, desc="Scanning JSONL"):
                    record = json.loads(line)
                    if str(record["id"]) not in target_ids:
                        del record
                        continue
                    found_ids.add(str(record["id"]))
                    sv = record.get("sparse_vector", {})
                    batch.append({
                        "id": record["id"],
                        "vector": record["dense_vector"],
                        "sparse_indices": sv.get("indices", []),
                        "sparse_values": sv.get("values", []),
                        "meta": {
                            "text": record["meta"]["text"],
                            "id": record["id"],
                        },
                    })
                    del record, sv
                    if len(batch) >= batch_size:
                        flush_batch(batch)
                    if len(found_ids) == len(target_ids):
                        break
            not_found = len(target_ids - found_ids)
            scan_time = None  # computed below

        if batch:
            flush_batch(batch)

        elapsed = time.perf_counter() - start
        total_upsert = sum(upsert_times)

        summary = {
            "reinserted": reinsert_count,
            "failed": reinsert_fail,
            "total_elapsed_sec": round(elapsed, 2),
            "upsert_time_sec": round(total_upsert, 2),
            "num_batches": len(upsert_times),
        }
        if not csr_path:
            summary["not_found_in_jsonl"] = not_found
            summary["scan_time_sec"] = round(elapsed - total_upsert, 2)
        if upsert_times:
            summary["upsert_per_batch"] = {
                "min_sec": round(min(upsert_times), 2),
                "max_sec": round(max(upsert_times), 2),
                "avg_sec": round(total_upsert / len(upsert_times), 2),
            }

        logger.info(
            "Reinsert complete — reinserted: %d  failed: %d  time: %.2fs",
            reinsert_count, reinsert_fail, elapsed,
        )
        return summary

def main():
    parser = argparse.ArgumentParser(description="Delete and reinsert Endee vectors from a JSONL or CSR file.")
    parser.add_argument("--index_name",    help="Name of the Endee index")
    parser.add_argument("--jsonl_path",    help="Path to the JSONL embeddings file (hybrid/dense indexes)")
    parser.add_argument("--sparse-only",   action="store_true", default=False,
                        help="Use sparse-only mode: load IDs and vectors from a .csr file")
    parser.add_argument("--csr-path",      default=None,
                        help="Path to the .csr data file (required when --sparse-only is set)")
    parser.add_argument("--token",       default="12345678",  help="Endee API token (default: 12345678)")
    parser.add_argument("--base-url",    default=DEV_PATH,    help=f"Endee base URL (default: {DEV_PATH})")
    parser.add_argument("--delete-percentage", type=int, default=10,
                        help="Percentage of vectors to delete from the candidate pool (default: 10)")
    parser.add_argument("--mode", choices=["random", "last-n-percent", "first-n-percent"], default="random",
                        help="ID selection mode applied to the candidate pool (default: random)")
    parser.add_argument("--ground-truth-file", default=None,
                        help="Path to unique_doc_ids.txt produced by endee_validation.py")
    parser.add_argument("--gt-filter", choices=["ground-truth", "non-ground-truth"], default=None,
                        help="Filter candidate pool to ground-truth or non-ground-truth IDs before applying --mode; requires --ground-truth-file")
    parser.add_argument("--batch-size",  type=int, default=1000, help="Reinsert batch size (default: 1000)")
    parser.add_argument("--skip-reinsert", type=lambda x: x.lower() != "false", default=False,
                        help="Skip reinsertion after deletion (default: false)")
    parser.add_argument("--verify-delete", type=lambda x: x.lower() == "true", default=False,
                        help="Verify deletions after delete step (default: false)")
    args = parser.parse_args()

    if args.sparse_only and not args.csr_path:
        parser.error("--sparse-only requires --csr-path")
    if not args.sparse_only and not args.jsonl_path:
        parser.error("--jsonl_path is required unless --sparse-only is set")
    if args.gt_filter and not args.ground_truth_file:
        parser.error("--gt-filter requires --ground-truth-file")

    edi = EndeeDeleteInsert(
        token=args.token,
        base_url=args.base_url,
        index_name=args.index_name,
    )

    # Load IDs from the appropriate source
    if args.sparse_only:
        all_record_ids = edi.load_ids_from_csr(args.csr_path)
    else:
        all_record_ids = edi.load_all_record_ids(args.jsonl_path)

    # Optionally narrow the candidate pool using ground-truth filter
    if args.gt_filter:
        gt_ids = EndeeDeleteInsert.load_ground_truth_file(args.ground_truth_file)
        if args.gt_filter == "ground-truth":
            candidate_pool = edi.select_ground_truth_ids(all_record_ids, gt_ids)
        else:
            candidate_pool = edi.select_non_ground_truth_ids(all_record_ids, gt_ids)
    else:
        candidate_pool = all_record_ids

    # Apply selection mode to the candidate pool
    if args.mode == "random":
        ids_to_delete = edi.select_random_ids(candidate_pool, args.delete_percentage)
    elif args.mode == "last-n-percent":
        ids_to_delete = edi.select_last_n_percent_ids(candidate_pool, args.delete_percentage)
    else:  # first-n-percent
        ids_to_delete = edi.select_first_n_percent_ids(candidate_pool, args.delete_percentage)

    edi.delete_vectors(ids_to_delete)

    if args.verify_delete:
        edi.verify_deletion(ids_to_delete)

    if not args.skip_reinsert:
        edi.reinsert_vectors(
            ids_to_delete=ids_to_delete,
            jsonl_path=args.jsonl_path if not args.sparse_only else None,
            csr_path=args.csr_path if args.sparse_only else None,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
