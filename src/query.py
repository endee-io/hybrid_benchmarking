import os
import json
import time
import logging
import math
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, List, Tuple, Any

import numpy as np

from src.interface import HybridDB
from src.utils import create_db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_worker_db_cache: Dict[int, HybridDB] = {}


def get_or_init_db(db_name: str, db_config: dict, index_name: str) -> HybridDB:
    """Get or create a DB instance for the current worker process."""
    worker_pid = os.getpid()
    if worker_pid not in _worker_db_cache:
        db = create_db(db_name, db_config)
        db.init(index_name, create=False)
        _worker_db_cache[worker_pid] = db
        logger.info("Worker %d initialized %s with index '%s'", worker_pid, db_name, index_name)
    return _worker_db_cache[worker_pid]


def load_queries_from_npy(data_dir: str, dataset_name: str, sparse_mode: str) -> List[Dict]:
    """
    Load query embeddings from .npy files.

    Expects files under data_dir/dataset_name/:
      <dataset_name>_dense_queries.npy
      <dataset_name>_dense_queries_ids.npy
      <dataset_name>_sparse_queries_<mode>_values.npy
      <dataset_name>_sparse_queries_<mode>_col_indices.npy
      <dataset_name>_sparse_queries_<mode>_indptr.npy
      <dataset_name>_sparse_queries_<mode>_ids.npy
    """
    base           = Path(data_dir) / dataset_name
    dense_path     = base / f"{dataset_name}_dense_queries.npy"
    dense_ids_path = base / f"{dataset_name}_dense_queries_ids.npy"
    sp_base        = base / f"{dataset_name}_sparse_queries_{sparse_mode}"

    logger.info("Loading dense query embeddings from %s", dense_path)
    dense     = np.load(str(dense_path), mmap_mode="r")
    dense_ids = np.load(str(dense_ids_path), allow_pickle=True)

    logger.info("Loading sparse query embeddings from %s_*.npy", sp_base)
    sp_values  = np.load(str(sp_base) + "_values.npy",      mmap_mode="r")
    sp_indices = np.load(str(sp_base) + "_col_indices.npy",  mmap_mode="r")
    sp_indptr  = np.load(str(sp_base) + "_indptr.npy")
    sp_ids     = np.load(str(sp_base) + "_ids.npy",          allow_pickle=True)

    n_dense  = len(dense_ids)
    n_sparse = len(sp_ids)
    logger.info("Loaded %d dense / %d sparse queries", n_dense, n_sparse)

    queries   = []
    dense_ptr = 0
    skipped   = 0

    for j in range(n_sparse):
        sp_qid = str(sp_ids[j])

        while dense_ptr < n_dense and str(dense_ids[dense_ptr]) != sp_qid:
            dense_ptr += 1

        if dense_ptr >= n_dense:
            logger.warning("Dense query not found for sparse query %s, skipping", sp_qid)
            skipped += 1
            continue

        s, e = int(sp_indptr[j]), int(sp_indptr[j + 1])
        queries.append({
            "query_id":     sp_qid,
            "dense_vector": dense[dense_ptr].tolist(),
            "sparse_vector": {
                "indices": sp_indices[s:e].tolist(),
                "values":  sp_values[s:e].tolist(),
            },
        })
        dense_ptr += 1

    if skipped:
        logger.warning("Skipped %d sparse queries with no matching dense query", skipped)

    return queries


def process_query_batch(batch_data: Tuple) -> Dict[str, Any]:
    """Process a batch of queries synchronously inside a worker process."""
    batch_id, queries, top_k, index_name, db_name, db_config = batch_data
    worker_pid = os.getpid()
    db = get_or_init_db(db_name, db_config, index_name)

    results    = {}
    latencies  = []
    query_logs = []
    total      = len(queries)
    completed  = 0

    logger.info("Worker %d - Batch %d: Processing %d queries", worker_pid, batch_id, total)

    for query in queries:
        query_id   = query["query_id"]
        start_time = time.time()
        try:
            search_results = db.search(
                dense_vector=query["dense_vector"],
                sparse_indices=query["sparse_vector"]["indices"],
                sparse_values=query["sparse_vector"]["values"],
                top_k=top_k,
            )
            if search_results is None:
                logger.error("Worker %d - Query %s returned None", worker_pid, query_id)
                err = {"query_id": query_id, "latency_ms": 0, "worker_id": worker_pid, "batch_id": batch_id, "error": "None result"}
                latencies.append(err)
                query_logs.append({**err, "status": "error"})
                results[query_id] = {}
                continue

            elapsed_ms    = (time.time() - start_time) * 1000
            query_results = {p["id"]: p["score"] for p in search_results}
            results[query_id] = query_results
            latencies.append({
                "query_id":    query_id,
                "latency_ms":  elapsed_ms,
                "worker_id":   worker_pid,
                "batch_id":    batch_id,
                "num_results": len(query_results),
            })
            query_logs.append({
                "query_id":    query_id,
                "status":      "success",
                "latency_ms":  elapsed_ms,
                "num_results": len(query_results),
                "worker_id":   worker_pid,
                "batch_id":    batch_id,
            })
            logger.debug("Worker %d - Query %s: %.2fms, %d results", worker_pid, query_id, elapsed_ms, len(query_results))

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error("Worker %d - Query %s failed: %s", worker_pid, query_id, e)
            latencies.append({"query_id": query_id, "latency_ms": elapsed_ms, "worker_id": worker_pid, "batch_id": batch_id, "error": str(e)})
            query_logs.append({"query_id": query_id, "status": "error", "latency_ms": elapsed_ms, "error": str(e), "worker_id": worker_pid, "batch_id": batch_id})
            results[query_id] = {}

        completed += 1
        if completed % 100 == 0:
            logger.info("Worker %d - Batch %d: %d/%d completed", worker_pid, batch_id, completed, total)

    logger.info("Worker %d - Batch %d: completed %d queries", worker_pid, batch_id, total)
    return {
        "batch_id":    batch_id,
        "worker_id":   worker_pid,
        "results":     results,
        "latencies":   latencies,
        "query_logs":  query_logs,
        "num_queries": len(queries),
    }


def split_into_batches(
    queries: List[Dict],
    concurrency: int,
    top_k: int,
    index_name: str,
    db_name: str,
    db_config: dict,
) -> List[Tuple]:
    """Split queries into batches for parallel processing."""
    total_queries = len(queries)
    if total_queries == 0:
        return []

    batch_size = math.ceil(total_queries / concurrency)
    batches = [
        (batch_id, queries[i : i + batch_size], top_k, index_name, db_name, db_config)
        for batch_id, i in enumerate(range(0, total_queries, batch_size))
        if queries[i : i + batch_size]
    ]

    total_in_batches = sum(len(b[1]) for b in batches)
    if total_in_batches != total_queries:
        logger.warning("Batch split mismatch: %d total, %d in batches", total_queries, total_in_batches)

    logger.info("Split %d queries into %d batches (concurrency=%d, batch_size=%d)",
                total_queries, len(batches), concurrency, batch_size)
    return batches


def calculate_p99_latency(latencies: List[Dict]) -> float:
    values = sorted(l["latency_ms"] for l in latencies if "error" not in l)
    if not values:
        return 0.0
    return values[min(int(len(values) * 0.99), len(values) - 1)]


def save_results(
    all_results: Dict[str, Dict],
    all_latencies: List[Dict],
    all_logs: List[Dict],
    output_dir: Path,
    results: str,
    concurrency: int,
    total_time: float,
):
    """Save all query results, latencies, and summary to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_latencies = sorted(all_latencies, key=lambda x: x["latency_ms"])
    merged_results   = {l["query_id"]: all_results[l["query_id"]] for l in sorted_latencies if l["query_id"] in all_results}

    with open(output_dir / "merged_results.json", "w") as f:
        json.dump(merged_results, f, indent=2)
    logger.info("Saved merged results: %d queries", len(merged_results))

    with open(output_dir / "all_latencies.json", "w") as f:
        json.dump(sorted_latencies, f, indent=2)
    logger.info("Saved latencies: %d queries", len(sorted_latencies))

    p99_latency = calculate_p99_latency(all_latencies)
    with open(output_dir / "p99_latency.json", "w") as f:
        json.dump({
            "p99_latency_ms":     p99_latency,
            "total_queries":      len(all_latencies),
            "successful_queries": len([l for l in all_latencies if "error" not in l]),
            "failed_queries":     len([l for l in all_latencies if "error" in l]),
        }, f, indent=2)
    logger.info("P99 latency: %.2fms", p99_latency)

    with open(output_dir / "detailed_logs.json", "w") as f:
        json.dump(all_logs, f, indent=2)

    successful_latencies = [l["latency_ms"] for l in all_latencies if "error" not in l]
    summary = {
        "results":            results,
        "concurrency":        concurrency,
        "total_queries":      len(all_latencies),
        "successful_queries": len(successful_latencies),
        "failed_queries":     len(all_latencies) - len(successful_latencies),
        "p99_latency_ms":     p99_latency,
        "p95_latency_ms":     float(np.percentile(successful_latencies, 95)) if successful_latencies else 0,
        "p90_latency_ms":     float(np.percentile(successful_latencies, 90)) if successful_latencies else 0,
        "p50_latency_ms":     float(np.percentile(successful_latencies, 50)) if successful_latencies else 0,
        "mean_latency_ms":    float(np.mean(successful_latencies))            if successful_latencies else 0,
        "min_latency_ms":     min(successful_latencies)                       if successful_latencies else 0,
        "max_latency_ms":     max(successful_latencies)                       if successful_latencies else 0,
        "total_time_seconds": total_time,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary statistics")


def run_query(
    db_name: str,
    db_config: dict,
    index_name: str,
    data_dir: str,
    dataset_name: str,
    sparse_mode: str,
    output_dir: Path,
    results: str,
    concurrency: int,
    top_k: int,
):
    """
    Full query pipeline entry point called from main.py.

    db_config  : kwargs passed directly to the DB constructor (e.g. vector_token, base_url)
    output_dir : pre-built Path (e.g. results/endee/run1_concurrency16)
    """
    logger.info("Starting query pipeline")
    logger.info("  DB:                %s", db_name)
    logger.info("  Dataset:           %s/%s", data_dir, dataset_name)
    logger.info("  Sparse mode:       %s", sparse_mode)
    logger.info("  Index name:        %s", index_name)
    logger.info("  Process concurrency: %d", concurrency)
    logger.info("  Results:           %s", results)
    logger.info("  Output directory:  %s", output_dir)

    queries = load_queries_from_npy(data_dir, dataset_name, sparse_mode)
    batches = split_into_batches(queries, concurrency, top_k, index_name, db_name, db_config)

    logger.info("Starting parallel processing with %d workers", concurrency)
    start_time = time.time()
    with Pool(processes=concurrency) as pool:
        batch_results = pool.map(process_query_batch, batches)
    total_time = time.time() - start_time
    logger.info("Completed all queries in %.2f seconds", total_time)

    all_results, all_latencies, all_logs = {}, [], []
    for br in batch_results:
        all_results.update(br["results"])
        all_latencies.extend(br["latencies"])
        all_logs.extend(br["query_logs"])

    logger.info("Merged results: %d queries, %d latency entries", len(all_results), len(all_latencies))
    save_results(all_results, all_latencies, all_logs, output_dir, results, concurrency, total_time)
    logger.info("All results saved to %s", output_dir)
