import asyncio
import math
import os
from multiprocessing import Pool

from endee import Endee
import click
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from src.download import download_gz_file, download_file
from src.sparse_matrix import read_sparse_matrix, knn_result_read
from src.stats import compute_dataset_stats, compare_floats_percentage

# Dummy dense vector (10 dimensions) used for all upserts and queries
# since endee requires a dense vector but we only care about sparse search
DUMMY_DENSE_DIM = 10
DUMMY_DENSE_VECTOR = [0.1] * DUMMY_DENSE_DIM

# Per-process index cache (avoids re-initializing on every task)
_worker_index_cache = {}


def _get_or_init_worker_index(index_name, endee_token, endee_base_url):
    worker_pid = os.getpid()
    if worker_pid not in _worker_index_cache:
        vx = Endee(token=endee_token)
        vx.set_base_url(endee_base_url)
        _worker_index_cache[worker_pid] = vx.get_index(index_name)
        print(f"Worker {worker_pid} initialized with index {index_name}")
    return _worker_index_cache[worker_pid]


async def _query_single(query_dict, worker_index, search_limit):
    loop = asyncio.get_event_loop()
    start = time.time()
    try:
        results = await loop.run_in_executor(
            None,
            lambda: worker_index.query(
                vector=DUMMY_DENSE_VECTOR,
                sparse_indices=query_dict["indices"],
                sparse_values=query_dict["values"],
                top_k=search_limit
            )
        )
        duration_ms = (time.time() - start) * 1000
        return query_dict["query_idx"], results, duration_ms
    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        print(f"Worker {os.getpid()} - Query {query_dict['query_idx']} failed: {e}")
        return query_dict["query_idx"], [], duration_ms


async def _process_batch_async(queries, worker_index, search_limit, async_concurrency, batch_id):
    results = []
    worker_pid = os.getpid()
    pbar = tqdm(
        total=len(queries),
        desc=f"Worker {worker_pid} batch {batch_id}",
        position=batch_id,
        leave=True
    )
    for chunk_start in range(0, len(queries), async_concurrency):
        chunk = queries[chunk_start:chunk_start + async_concurrency]
        tasks = [_query_single(q, worker_index, search_limit) for q in chunk]
        chunk_results = await asyncio.gather(*tasks)
        results.extend(chunk_results)
        pbar.update(len(chunk))
    pbar.close()
    return results


def _process_batch(batch_args):
    batch_id, queries, index_name, endee_token, endee_base_url, search_limit, async_concurrency = batch_args
    worker_index = _get_or_init_worker_index(index_name, endee_token, endee_base_url)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _process_batch_async(queries, worker_index, search_limit, async_concurrency, batch_id)
        )
    finally:
        loop.close()


def csr_to_sparse_lists(point_csr):
    """Convert a CSR matrix row to sparse indices and values lists."""
    indices = point_csr.indices.tolist()
    values = point_csr.data.tolist()
    return indices, values


def longest_posting_list_for_vector(sparse_indices, stats):
    longest = 0
    for index in sparse_indices:
        if index in stats.sizes:
            longest = max(longest, stats.sizes[index])
    return longest


def sum_posting_list_for_vector(sparse_indices, stats):
    total = 0
    for index in sparse_indices:
        if index in stats.sizes:
            total += stats.sizes[index]
    return total


# query data files
QUERY_DATA_FILE_NAME = "queries.dev.csr"
# small dataset
SMALL_DATA_FILE_NAME = "base_small.csr"
SMALL_GT_FILE_NAME = "base_small.dev.gt"
# 1M dataset
M1_DATA_FILE_NAME = "base_1M.csr"
M1_GT_FILE_NAME = "base_1M.dev.gt"
# full dataset
FULL_DATA_FILE_NAME = "base_full.csr"
FULL_GT_FILE_NAME = "base_full.dev.gt"


def file_names_for_dataset(dataset) -> (str, str):
    if dataset == "small":
        return SMALL_DATA_FILE_NAME, SMALL_GT_FILE_NAME
    elif dataset == "1M":
        return M1_DATA_FILE_NAME, M1_GT_FILE_NAME
    elif dataset == "full":
        return FULL_DATA_FILE_NAME, FULL_GT_FILE_NAME
    else:
        print(f"Unknown dataset {dataset}")
        exit(1)


@click.command()
@click.option('--endee-token', required=False, default="12345678", help="Endee API token")
@click.option('--endee-base-url', default="http://51.89.231.115:8080/api/v1", help="Endee base URL")
@click.option('--skip-creation', default=True, help="Whether to skip index creation")
@click.option('--dataset', default="small", help="Dataset to use: small, 1M, full")
@click.option('--slow-ms', default=500, help="Slow query threshold in milliseconds")
@click.option('--search-limit', default=10, help="Search limit")
@click.option('--data-path', default="./data", help="Path to the data files")
@click.option('--results-path', default="./results", help="Path to the results files")
@click.option('--analyze-data', default=False, help="Whether to analyze data")
@click.option('--check-ground-truth', default=False, help="Whether to check results against ground truth")
@click.option('--graph-y-range', default=None, help="Y axis range for the graph to help compare plots")
@click.option('--upsert-batch-size', default=1000, help="Number of vectors per batch upsert")
@click.option('--max-retries', default=5, help="Number of retries for upsert on failure")
@click.option('--concurrency', default=1, help="Number of worker processes for parallel querying")
@click.option('--async-concurrency', default=10, help="Number of concurrent async queries per worker")
@click.option('--index-name', default="neurIPS_sparse_bench", help="Endee index name")
def sparse_vector_benchmark(
        endee_token,
        endee_base_url,
        skip_creation,
        dataset,
        slow_ms,
        search_limit,
        data_path,
        results_path,
        analyze_data,
        check_ground_truth,
        graph_y_range,
        upsert_batch_size,
        max_retries,
        concurrency,
        async_concurrency,
        index_name,
        ):
    """Sparse vector benchmark tool for Endee."""

    # Make sure working folders exist
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    index_name = index_name.replace(" ", "_")  # avoid spaces in index name 

    # pick dataset
    data_file_name, gt_file_name = file_names_for_dataset(dataset)

    # check if files exist
    query_data_file_path = f"{data_path}/{QUERY_DATA_FILE_NAME}"
    if not os.path.isfile(query_data_file_path):
        print(f"Query data file {query_data_file_path} doesn't exist")
        download_gz_file(data_path, QUERY_DATA_FILE_NAME)

    data_file_path = f"{data_path}/{data_file_name}"
    if not os.path.isfile(data_file_path):
        print(f"Data file {data_file_path} doesn't exist")
        download_gz_file(data_path, data_file_name)

    gt_file_path = f"{data_path}/{gt_file_name}"
    if check_ground_truth and not os.path.isfile(gt_file_path):
        print(f"Ground truth file {gt_file_path} doesn't exist")
        download_file(data_path, gt_file_name)

    # ground truth data
    gt_indices = []
    gt_scores = []
    if check_ground_truth:
        gt_indices, gt_scores = knn_result_read(gt_file_path)
        assert len(gt_indices) == len(gt_scores)
        gt_len = len(gt_indices)
        top_len = len(gt_indices[0])
        assert top_len == len(gt_scores[0])
        print(f"Ground truth contains {gt_len} entries for top {top_len} vectors")

    data = {}
    vec_count = 0

    if analyze_data or not skip_creation:
        print(f"Reading {data_file_path} ...")
        data = read_sparse_matrix(data_file_path)
        vec_count = data.shape[0]

    # data analyze behind flag as it can be expensive
    if analyze_data:
        stats = compute_dataset_stats(data)
        dim_count = len(stats.posting_len_per_dimension)
        print(f"Dataset contains {vec_count} sparse vectors with {dim_count} unique dimensions")
        # bar chart of posting list lengths
        plt.scatter(*zip(*stats.count_per_posting_len.items()), )
        plt.grid(True)
        plt.xlabel('Posting list length')
        plt.ylabel('Number of dimensions')
        plt.title(f"Posting list length distribution ({dim_count} dimensions, {vec_count} vectors)")
        plt.savefig(f"./results/neurIPS_bench_{dataset}_posting_len.png")
        plt.close()
        print("Posting list lengths (top 20)")
        # sort by count
        sorted_count_per_index = sorted(stats.posting_len_per_dimension.items(), key=lambda item: item[1], reverse=True)
        for idx, count in sorted_count_per_index[:20]:
            # percentage of vectors with this index
            percentage = round(count / vec_count * 100, 2)
            print(f"{idx}: {count} ({percentage}%)")

    # Endee client
    print("Connecting to Endee...")
    vx = Endee(token=endee_token)
    vx.set_base_url(endee_base_url)
    print("Connected to Endee")
    indexing_start = time.time_ns()
    if not skip_creation:
        # Compute sparse_dim from the CSR matrix column count
        sparse_dim = data.shape[1]

        # Delete existing index if it exists
        try:
            vx.delete_index(index_name)
            print(f"Deleted existing index '{index_name}'")
        except Exception:
            pass

        print(f"Creating index '{index_name}' with sparse_dim={sparse_dim}")
        vx.create_index(
            name=index_name,
            dimension=DUMMY_DENSE_DIM,
            space_type="cosine",
            sparse_dim=sparse_dim,
            precision="float32"
        )
        print("Index created")
        index = vx.get_index(index_name)

        print(f"Uploading {vec_count} sparse vectors into '{index_name}'")
        batch = []
        for i in tqdm(range(vec_count), desc="Uploading"):
            point = data[i]
            indices, values = csr_to_sparse_lists(point)
            batch.append({
                "id": str(i),
                "vector": DUMMY_DENSE_VECTOR,
                "sparse_indices": indices,
                "sparse_values": values,
                "meta": {"point_id": i}
            })
            if len(batch) >= upsert_batch_size:
                for attempt in range(max_retries):
                    try:
                        index.upsert(batch)
                        break
                    except Exception as e:
                        print(f"Upsert failed (attempt {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(1.5)
                        else:
                            raise
                batch = []
        # flush remaining batch
        if batch:
            for attempt in range(max_retries):
                try:
                    index.upsert(batch)
                    break
                except Exception as e:
                    print(f"Upsert failed (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1.5)
                    else:
                        raise
        print("Upload done")
    else:
        print("Skipping index creation")
        index = vx.get_index(index_name)

    indexing_end = time.time_ns()
    indexing_duration_sec = round((indexing_end - indexing_start) / 1_000_000_000, 2)
    print(f"Upload & indexing took {indexing_duration_sec} seconds")

    # Index stats
    print("Index is ready for querying:")
    print(f"- {index.count} vectors")

    # Read queries from CSR and convert to plain dicts (picklable for multiprocessing)
    query_data = read_sparse_matrix(query_data_file_path)
    query_count = query_data.shape[0]
    queries_list = []
    for i in range(query_count):
        point = query_data[i]
        indices, values = csr_to_sparse_lists(point)
        queries_list.append({"query_idx": i, "indices": indices, "values": values})

    # Split queries into batches — one batch per worker
    batch_sz = math.ceil(query_count / concurrency)
    batches = []
    for b_id, start in enumerate(range(0, query_count, batch_sz)):
        batches.append((
            b_id,
            queries_list[start:start + batch_sz],
            index_name,
            endee_token,
            endee_base_url,
            search_limit,
            async_concurrency
        ))

    # data for plotting
    latency = []
    dimensions = []
    recall_scores = []
    print("---------------------------------------")
    print(f"Querying {query_count} sparse vectors "
          f"({concurrency} workers x {async_concurrency} async concurrency)")

    query_start = time.time()
    try:
        with Pool(processes=concurrency) as pool:
            batch_results_list = list(tqdm(
                pool.imap(_process_batch, batches),
                total=len(batches),
                desc="Query batches"
            ))
    except KeyboardInterrupt:
        print("Bye - generating partial report")
        batch_results_list = []
    total_query_time_sec = time.time() - query_start

    # Flatten and restore original query order
    all_query_results = []
    for batch_result in batch_results_list:
        all_query_results.extend(batch_result)
    all_query_results.sort(key=lambda x: x[0])

    # Compute latency, dimensions, and ground truth recall
    for query_idx, results, duration_ms in all_query_results:
        latency.append(duration_ms)
        dim = len(queries_list[query_idx]["indices"])
        dimensions.append(dim)
        if duration_ms > slow_ms:
            print(f"Slow query with dim {dim} took {duration_ms:.2f} millis")
        if check_ground_truth and results:
            expected_scores = gt_scores[query_idx]
            expected_ids = gt_indices[query_idx]
            result_ids = set()
            for j in range(len(results)):
                result = results[j]
                result_score = result["similarity"]
                expected_score = float(expected_scores[j])
                if not compare_floats_percentage(result_score, expected_score, 1):
                    print(f"GT score mismatch vector:{query_idx} result:{j}/{search_limit}: {result_score} != {expected_score}")
                result_id = int(result["id"])
                result_ids.add(result_id)
                expected_id = expected_ids[j]
                if result_id != expected_id:
                    print(f"GT id mismatch vector:{query_idx} result:{j}/{search_limit}: {result_id} != {expected_id}")
            expected_id_set = set(expected_ids[:search_limit])
            hits = len(result_ids & expected_id_set)
            recall = hits / len(expected_id_set) if expected_id_set else 0.0
            recall_scores.append(recall)

    print("---------------------------------------")
    print("Search latency distribution:")
    quantiles = np.quantile(latency, [0, 0.5, 0.95, 0.99, 0.999, 1])
    print(f"min:   {round(quantiles[0], 2)} millis")
    print(f"50p:   {round(quantiles[1], 2)} millis")
    print(f"95p:   {round(quantiles[2], 2)} millis")
    print(f"99p:   {round(quantiles[3], 2)} millis")
    print(f"999p:  {round(quantiles[4], 2)} millis")
    print(f"max:   {round(quantiles[5], 2)} millis")
    print(f"total: {round(total_query_time_sec, 2)} seconds ({query_count} queries)")
    print("")

    if check_ground_truth and recall_scores:
        mean_recall = np.mean(recall_scores)
        print(f"Recall@{search_limit}: {round(mean_recall * 100, 2)}% (mean over {len(recall_scores)} queries)")
        recall_quantiles = np.quantile(recall_scores, [0, 0.5, 0.95, 0.99, 1])
        print(f"  min:  {round(recall_quantiles[0] * 100, 2)}%")
        print(f"  50p:  {round(recall_quantiles[1] * 100, 2)}%")
        print(f"  95p:  {round(recall_quantiles[2] * 100, 2)}%")
        print(f"  99p:  {round(recall_quantiles[3] * 100, 2)}%")
        print(f"  max:  {round(recall_quantiles[4] * 100, 2)}%")

    # query dimensions distribution
    if analyze_data:
        print("Query dimensions distribution:")
        quantiles = np.quantile(dimensions, [0, 0.5, 0.95, 0.99, 0.999, 1])
        print(f"min: {quantiles[0]}")
        print(f"50p: {quantiles[1]}")
        print(f"95p: {quantiles[2]}")
        print(f"99p: {quantiles[3]}")
        print(f"999p: {quantiles[4]}")
        print(f"max: {quantiles[5]}")

    # Create a 2D histogram of the query dimensions and latencies
    timestamp = int(time.time())
    title = f"Sparse NeurIPS {dataset} (Endee)"
    plt.hist2d(dimensions, latency, bins=100, cmap="rainbow")
    plt.grid(True)
    # force y-axis limits to be able to compare plots
    if graph_y_range is not None:
        axis = plt.gca()
        split = graph_y_range.split(" ")
        bottom = int(split[0])
        top = int(split[1])
        axis.set_ylim(bottom=bottom, top=top)
    cbar = plt.colorbar()
    cbar.set_label('Frequency')
    plt.xlabel('Query dimension count')
    plt.ylabel('Latency (ms)')
    plt.title(title)
    plot_file_name = f"./results/sparse_bench_{dataset}_{timestamp}.png"
    print(f"Saving plot to {plot_file_name}")
    plt.savefig(plot_file_name)
    plt.close()
