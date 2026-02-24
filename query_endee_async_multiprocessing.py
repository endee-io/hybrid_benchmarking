import json
import os
import time
import logging
import argparse
import math
import asyncio
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, List, Tuple, Any
import numpy as np
from endee import Endee
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEV_PATH = "https://dev.endee.io/api/v1"

# Module-level variables for worker initialization (per process)
_worker_index_cache = {}


def get_or_init_index(index_name: str, vector_token: str, base_url: str):
    """Get or initialize index for current worker process."""
    worker_id = os.getpid()
    if worker_id not in _worker_index_cache:
        vx = Endee(token=vector_token)
        vx.set_base_url(base_url)
        index = vx.get_index(index_name)
        _worker_index_cache[worker_id] = index
        print(f"Worker {worker_id} initialized with index {index_name}")

    return _worker_index_cache[worker_id]


async def process_single_query(
    query: Dict,
    index,
    top_k: int,
    batch_id: int,
    worker_pid: int
) -> Tuple[str, Dict[str, Any], Dict, Dict]:
    """
    Process a single query asynchronously.
    
    Returns:
        Tuple of (query_id, query_results, latency_dict, query_log)
    """
    query_id = query.get("query_id") or query["id"]
    start_time = time.time()
    
    try:
        dense_vector = query.get("dense_embedding") or query["dense_vector"]
        sparse_embedding = query.get("sparse_embedding") or query["sparse_vector"]
        
        # Execute query in thread pool (since Endee API is synchronous)
        loop = asyncio.get_event_loop()
        search_results = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=dense_vector,
                sparse_indices=sparse_embedding["indices"],
                sparse_values=sparse_embedding["values"],
                top_k=top_k
            )
        )
        
        elapsed_time = time.time() - start_time
        
        # Process results
        query_results = {
            str(point['meta']['id']): point['similarity']
            for point in search_results
        }
        
        latency_dict = {
            "query_id": query_id,
            "latency_ms": elapsed_time * 1000,
            "worker_id": worker_pid,
            "batch_id": batch_id,
            "num_results": len(query_results)
        }
        
        query_log = {
            "query_id": query_id,
            "status": "success",
            "latency_ms": elapsed_time * 1000,
            "num_results": len(query_results),
            "worker_id": worker_pid,
            "batch_id": batch_id
        }
        
        logger.debug(f"Worker {worker_pid} - Query {query_id}: {elapsed_time*1000:.2f}ms, {len(query_results)} results")
        
        return query_id, query_results, latency_dict, query_log
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Worker {worker_pid} - Query {query_id} failed: {str(e)}")
        
        query_log = {
            "query_id": query_id,
            "status": "error",
            "latency_ms": elapsed_time * 1000,
            "error": str(e),
            "worker_id": worker_pid,
            "batch_id": batch_id
        }
        
        latency_dict = {
            "query_id": query_id,
            "latency_ms": elapsed_time * 1000,
            "worker_id": worker_pid,
            "batch_id": batch_id,
            "error": str(e)
        }
        
        return query_id, {}, latency_dict, query_log


async def process_query_batch_async(
    queries: List[Dict],
    index,
    top_k: int,
    batch_id: int,
    worker_pid: int,
    async_concurrency: int
) -> Dict[str, Any]:
    """
    Process a batch of queries asynchronously with controlled concurrency.
    
    Args:
        queries: List of queries to process
        index: Endee index instance
        top_k: Number of top results to retrieve
        batch_id: Batch identifier
        worker_pid: Worker process ID
        async_concurrency: Number of concurrent async queries per worker
    
    Returns:
        Dictionary with results, latencies, and metadata
    """
    results = {}
    latencies = []
    query_logs = []
    
    total_queries = len(queries)
    
    logger.info(f"Worker {worker_pid} - Batch {batch_id}: Processing {total_queries} queries with async concurrency {async_concurrency}")
    
    # Process queries in chunks based on async_concurrency
    for chunk_start in range(0, total_queries, async_concurrency):
        chunk_end = min(chunk_start + async_concurrency, total_queries)
        chunk = queries[chunk_start:chunk_end]
        chunk_num = (chunk_start // async_concurrency) + 1
        total_chunks = math.ceil(total_queries / async_concurrency)
        
        logger.info(f"Worker {worker_pid} - Batch {batch_id}: Processing async chunk {chunk_num}/{total_chunks} ({len(chunk)} queries)")
        
        # Create tasks for concurrent execution
        tasks = [
            process_single_query(query, index, top_k, batch_id, worker_pid)
            for query in chunk
        ]
        
        # Wait for all tasks in this chunk to complete
        chunk_results = await asyncio.gather(*tasks)
        
        # Process results
        for query_id, query_results, latency_dict, query_log in chunk_results:
            results[query_id] = query_results
            latencies.append(latency_dict)
            query_logs.append(query_log)
        
        logger.info(f"Worker {worker_pid} - Batch {batch_id}: Completed async chunk {chunk_num}/{total_chunks}")
    
    logger.info(f"Worker {worker_pid} - Batch {batch_id}: Completed all {total_queries} queries")
    
    return {
        "batch_id": batch_id,
        "worker_id": worker_pid,
        "results": results,
        "latencies": latencies,
        "query_logs": query_logs,
        "num_queries": len(queries)
    }


def process_query_batch(batch_data: Tuple[int, List[Dict], int, str, str, int]) -> Dict[str, Any]:
    """
    Process a batch of queries (wrapper to run async function in process).
    
    Args:
        batch_data: Tuple of (batch_id, queries, top_k, index_name, vector_token, async_concurrency)
    
    Returns:
        Dictionary with results, latencies, and metadata
    """
    batch_id, queries, top_k, index_name, vector_token, base_url, async_concurrency = batch_data
    worker_pid = os.getpid()

    # Get or initialize index for this worker
    index = get_or_init_index(index_name, vector_token, base_url)
    
    # Run async function in this process's event loop
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the async batch processing
    result = loop.run_until_complete(
        process_query_batch_async(
            queries, index, top_k, batch_id, worker_pid, async_concurrency
        )
    )
    
    return result


def load_queries(query_file: str) -> List[Dict]:
    """Load queries from a JSON or JSONL file."""
    logger.info(f"Loading queries from {query_file}")
    with open(query_file, 'r') as f:
        content = f.read().strip()
    if content.startswith('['):
        queries = json.loads(content)
    else:
        queries = [json.loads(line) for line in content.splitlines() if line.strip()]
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def split_into_batches(
    queries: List[Dict],
    concurrency: int,
    top_k: int,
    index_name: str,
    vector_token: str,
    base_url: str,
    async_concurrency: int
) -> List[Tuple[int, List[Dict], int, str, str, str, int]]:
    """Split queries into batches for parallel processing."""
    total_queries = len(queries)

    if total_queries == 0:
        return []

    # Calculate batch size using ceiling division to ensure all queries are covered
    batch_size = math.ceil(total_queries / concurrency)

    batches = []
    batch_id = 0

    for i in range(0, total_queries, batch_size):
        batch = queries[i:i + batch_size]
        if batch:  # Only add non-empty batches
            batches.append((batch_id, batch, top_k, index_name, vector_token, base_url, async_concurrency))
            batch_id += 1

    # Verify all queries are covered
    total_in_batches = sum(len(batch) for _, batch, _, _, _, _, _ in batches)
    if total_in_batches != total_queries:
        logger.warning(f"Batch splitting mismatch: {total_queries} queries total, {total_in_batches} in batches")
    
    logger.info(f"Split {total_queries} queries into {len(batches)} batches (concurrency: {concurrency}, batch_size: {batch_size}, async_concurrency: {async_concurrency})")
    return batches


def calculate_p99_latency(latencies: List[Dict]) -> float:
    """Calculate p99 latency from latency list."""
    if not latencies:
        return 0.0
    
    latency_values = [l["latency_ms"] for l in latencies if "error" not in l]
    if not latency_values:
        return 0.0
    
    latency_values.sort()
    p99_index = int(len(latency_values) * 0.99)
    p99_index = min(p99_index, len(latency_values) - 1)
    return latency_values[p99_index]


def save_results(
    all_results: Dict[str, Dict],
    all_latencies: List[Dict],
    all_logs: List[Dict],
    output_dir: Path,
    test_cycle: int,
    concurrency: int,
    total_time: float
):
    """Save all results, latencies, and logs to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort latencies in ascending order
    sorted_latencies = sorted(all_latencies, key=lambda x: x["latency_ms"])
    
    # Merge results in ascending order of latency
    merged_results = {}
    for latency_entry in sorted_latencies:
        query_id = latency_entry["query_id"]
        if query_id in all_results:
            merged_results[query_id] = all_results[query_id]
    
    # Save merged results
    merged_results_file = output_dir / "merged_results.json"
    with open(merged_results_file, 'w') as f:
        json.dump(merged_results, f, indent=2)
    logger.info(f"Saved merged results: {len(merged_results)} queries")
    
    # Save all latencies sorted
    latencies_file = output_dir / "all_latencies.json"
    with open(latencies_file, 'w') as f:
        json.dump(sorted_latencies, f, indent=2)
    logger.info(f"Saved latencies: {len(sorted_latencies)} queries")
    
    # Calculate and save p99 latency
    p99_latency = calculate_p99_latency(all_latencies)
    p99_file = output_dir / "p99_latency.json"
    with open(p99_file, 'w') as f:
        json.dump({
            "p99_latency_ms": p99_latency,
            "total_queries": len(all_latencies),
            "successful_queries": len([l for l in all_latencies if "error" not in l]),
            "failed_queries": len([l for l in all_latencies if "error" in l])
        }, f, indent=2)
    logger.info(f"P99 latency: {p99_latency:.2f}ms")
    
    # Save detailed logs
    logs_file = output_dir / "detailed_logs.json"
    with open(logs_file, 'w') as f:
        json.dump(all_logs, f, indent=2)
    logger.info(f"Saved detailed logs: {len(all_logs)} entries")
    
    # Save summary statistics
    successful_latencies = [l["latency_ms"] for l in all_latencies if "error" not in l]
    summary = {
        "test_cycle": test_cycle,
        "concurrency": concurrency,
        "total_queries": len(all_latencies),
        "successful_queries": len(successful_latencies),
        "failed_queries": len(all_latencies) - len(successful_latencies),
        "p99_latency_ms": p99_latency,
        "p95_latency_ms": np.percentile(successful_latencies, 95) if successful_latencies else 0,
        "p90_latency_ms": np.percentile(successful_latencies, 90) if successful_latencies else 0,
        "p50_latency_ms": np.percentile(successful_latencies, 50) if successful_latencies else 0,
        "mean_latency_ms": np.mean(successful_latencies) if successful_latencies else 0,
        "min_latency_ms": min(successful_latencies) if successful_latencies else 0,
        "max_latency_ms": max(successful_latencies) if successful_latencies else 0,
        "total_time_seconds": total_time
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary statistics")


def main():
    parser = argparse.ArgumentParser(description="Async multiprocessed Endee query execution")
    parser.add_argument("--query-file", type=str, required=True,
                       help="Path to query embeddings JSON file")
    parser.add_argument("--index-name", type=str, required=True,
                       help="Name of the Endee index")
    parser.add_argument("--vector-token", type=str, required=False,default="12345678",
                       help="Endee vector token")
    parser.add_argument("--concurrency", type=int, required=True,
                       help="Number of concurrent processes")
    parser.add_argument("--async-concurrency", type=int, default=10,
                       help="Number of concurrent async queries per worker process (default: 10)")
    parser.add_argument("--test-cycle", type=int, required=True,
                       help="Test cycle number")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of top results to retrieve (default: 10)")
    parser.add_argument("--base-url", type=str, default=DEV_PATH,
                       help=f"Endee base URL (default: {DEV_PATH})")
    parser.add_argument("--output-base", type=str, default="test",
                       help="Base output directory (default: test)")

    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_base) / f"testcycle{args.test_cycle}" / f"concurrency{args.concurrency}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting async query execution")
    logger.info(f"  Query file: {args.query_file}")
    logger.info(f"  Index name: {args.index_name}")
    logger.info(f"  Process concurrency: {args.concurrency}")
    logger.info(f"  Async concurrency per worker: {args.async_concurrency}")
    logger.info(f"  Test cycle: {args.test_cycle}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Load queries
    queries = load_queries(args.query_file)
    
    # Split into batches (includes async_concurrency for worker initialization)
    batches = split_into_batches(
        queries, args.concurrency, args.top_k, args.index_name, args.vector_token, args.base_url, args.async_concurrency
    )
    
    # Process queries in parallel
    logger.info(f"Starting parallel processing with {args.concurrency} workers")
    logger.info(f"Each worker will process queries with {args.async_concurrency} async concurrency")
    start_time = time.time()
    with Pool(processes=args.concurrency) as pool:
        batch_results = pool.map(process_query_batch, batches)
    
    end_time_pool = time.time()
    logger.info(f"Completed parallel processing in {end_time_pool - start_time:.2f} seconds")
    total_time = time.time() - start_time
    logger.info(f"Completed all queries in {total_time:.2f} seconds")
    
    # Merge all results
    all_results = {}
    all_latencies = []
    all_logs = []
    
    for batch_result in batch_results:
        all_results.update(batch_result["results"])
        all_latencies.extend(batch_result["latencies"])
        all_logs.extend(batch_result["query_logs"])
    
    logger.info(f"Merged results: {len(all_results)} queries, {len(all_latencies)} latency entries")
    
    # Save results
    save_results(
        all_results,
        all_latencies,
        all_logs,
        output_dir,
        args.test_cycle,
        args.concurrency,
        total_time
    )
    
    logger.info(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()

