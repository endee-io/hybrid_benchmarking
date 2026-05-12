import os
import json
import time
import random
import logging
import concurrent.futures
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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


def run_serial_correctness(
    queries: List[Dict],
    db_name: str,
    db_config: dict,
    index_name: str,
    top_k: int,
    qrel_query_ids: Optional[set] = None,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Run queries serially for correctness evaluation.

    Filters to qrel_query_ids to ensure every query has ground truth.
    Uses perf_counter per query for accurate individual latency measurement.
    """
    if qrel_query_ids is not None:
        queries = [q for q in queries if q["query_id"] in qrel_query_ids]
        logger.info("Filtered to %d queries with qrel ground truth", len(queries))

    queries_to_run = queries
    logger.info("Serial correctness run: %d queries", len(queries_to_run))

    db = create_db(db_name, db_config)
    db.init(index_name, create=False)

    results:   Dict[str, Dict] = {}
    latencies: List[Dict]      = []

    for i, query in enumerate(queries_to_run):
        query_id = query["query_id"]
        s = time.perf_counter()
        try:
            search_results = db.search(
                dense_vector=query["dense_vector"],
                sparse_indices=query["sparse_vector"]["indices"],
                sparse_values=query["sparse_vector"]["values"],
                top_k=top_k,
                text=query.get("text", ""),
            )
            elapsed_ms = (time.perf_counter() - s) * 1000
            results[query_id] = {p["id"]: p["score"] for p in search_results} if search_results else {}
            latencies.append({"query_id": query_id, "latency_ms": elapsed_ms})
        except Exception as e:
            elapsed_ms = (time.perf_counter() - s) * 1000
            logger.error("Serial query %s failed: %s", query_id, e)
            results[query_id] = {}
            latencies.append({"query_id": query_id, "latency_ms": elapsed_ms, "error": str(e)})

        if (i + 1) % 100 == 0:
            logger.info("Serial correctness: %d / %d completed", i + 1, len(queries_to_run))

    logger.info("Serial correctness complete: %d queries", len(queries_to_run))
    return results, latencies


def calculate_p99_latency(latencies: List[Dict]) -> float:
    values = [l["latency_ms"] for l in latencies if "error" not in l]
    if not values:
        return 0.0
    return float(np.percentile(values, 99))


def qps_worker(
    worker_id: int,
    duration: int,
    queries: List[Dict],
    q,
    cond,
    db_name: str,
    db_config: dict,
    index_name: str,
    top_k: int,
) -> Tuple[int, int, List[float]]:
    """Duration-based QPS worker (VectorDBBench-style). Cycles through queries for `duration` seconds."""
    q.put(1)
    with cond:
        cond.wait()

    db = get_or_init_db(db_name, db_config, index_name)
    num = len(queries)
    idx = random.randint(0, num - 1)

    start_time = time.perf_counter()
    success_count = 0
    failed_count = 0
    latencies_ms: List[float] = []

    while time.perf_counter() < start_time + duration:
        query = queries[idx]
        s = time.perf_counter()
        try:
            db.search(
                dense_vector=query["dense_vector"],
                sparse_indices=query["sparse_vector"]["indices"],
                sparse_values=query["sparse_vector"]["values"],
                top_k=top_k,
                text=query.get("text", ""),
            )
            success_count += 1
            latencies_ms.append((time.perf_counter() - s) * 1000)
        except Exception as e:
            failed_count += 1
            logger.warning("Worker-%d query failed: %s", worker_id, e)

        idx = idx + 1 if idx < num - 1 else 0

    total_dur = time.perf_counter() - start_time
    per_process_qps = round(success_count / total_dur, 4) if total_dur else 0
    logger.info(
        "Worker-%d search %ds: actual_dur=%.4fs, count=%d, qps in this process: %.4f",
        worker_id, duration, total_dur, success_count, per_process_qps,
    )
    return success_count, failed_count, latencies_ms


def run_qps_benchmark(
    queries: List[Dict],
    concurrency: int,
    duration: int,
    db_name: str,
    db_config: dict,
    index_name: str,
    top_k: int,
) -> Dict:
    """Run VectorDBBench-style duration-based QPS benchmark.

    All workers start simultaneously via Queue+Condition synchronization,
    fire queries for `duration` seconds, then report aggregate QPS.
    """
    logger.info("Starting QPS benchmark: concurrency=%d, duration=%ds", concurrency, duration)

    with mp.Manager() as manager:
        q    = manager.Queue()
        cond = manager.Condition()

        with concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("spawn"),
            max_workers=concurrency,
        ) as executor:
            future_iter = [
                executor.submit(
                    qps_worker, i + 1, duration, queries, q, cond,
                    db_name, db_config, index_name, top_k,
                )
                for i in range(concurrency)
            ]

            # Wait until all workers signal ready
            while q.qsize() < concurrency:
                time.sleep(1)

            # Release all workers simultaneously
            with cond:
                cond.notify_all()
            logger.info("All %d workers synchronized and released", concurrency)

            # Collect results — cost ≈ duration since all workers run for the same window
            start = time.perf_counter()
            results = [f.result() for f in future_iter]
            cost = time.perf_counter() - start

    total_success  = sum(r[0] for r in results)
    total_failed   = sum(r[1] for r in results)
    all_latencies  = [lat for r in results for lat in r[2]]

    qps = round(total_success / cost, 4) if cost else 0.0
    p99 = float(np.percentile(all_latencies, 99)) if all_latencies else 0.0
    p95 = float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
    avg = float(np.mean(all_latencies))            if all_latencies else 0.0

    logger.info(
        "End search in concurrency %d: dur=%.4fs, total_count=%d, qps=%.4f",
        concurrency, cost, total_success, qps,
    )
    return {
        "qps":            qps,
        "total_success":  total_success,
        "total_failed":   total_failed,
        "p99_latency_ms": p99,
        "p95_latency_ms": p95,
        "avg_latency_ms": avg,
        "cost_seconds":   cost,
    }


def save_results(
    all_results: Dict[str, Dict],
    all_latencies: List[Dict],
    output_dir: Path,
    results: str,
    concurrency: int,
    total_time: float,
    qps_override: Optional[float] = None,
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
        json.dump(sorted_latencies, f, indent=2)

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
        "qps":                qps_override if qps_override is not None else (round(len(successful_latencies) / total_time, 2) if total_time else 0),
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
    query_texts: dict = None,
    qps_duration: int = 30,
    qrel_query_ids: Optional[set] = None,
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
    if query_texts:
        for q in queries:
            q["text"] = query_texts.get(q["query_id"], "")

    logger.info("--- Starting QPS benchmark (duration=%ds) ---", qps_duration)
    qps_result = run_qps_benchmark(
        queries=queries,
        concurrency=concurrency,
        duration=qps_duration,
        db_name=db_name,
        db_config=db_config,
        index_name=index_name,
        top_k=top_k,
    )
    logger.info("--- QPS benchmark complete: qps=%.2f ---", qps_result["qps"])

    logger.info("--- Starting serial correctness run ---")
    start_time = time.perf_counter()
    all_results, all_latencies = run_serial_correctness(
        queries=queries,
        db_name=db_name,
        db_config=db_config,
        index_name=index_name,
        top_k=top_k,
        qrel_query_ids=qrel_query_ids,
    )
    total_time = time.perf_counter() - start_time
    logger.info("--- Serial correctness complete in %.2fs ---", total_time)

    save_results(all_results, all_latencies, output_dir, results, concurrency, total_time,
                 qps_override=qps_result["qps"])
    logger.info("All results saved to %s", output_dir)
