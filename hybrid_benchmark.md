# Hybrid Benchmark — Script Reference

---

## Environments

> **Python version:** Use **Python 3.12** for both environments to avoid dependency conflicts (e.g. `numpy` version incompatibilities seen on older Pythons).

### Install Python 3.12 (macOS)

```bash
brew install python@3.12
```

Verify:
```bash
python3.12 --version   # Python 3.12.x
```

---

### Create environments

Two separate Python 3.12 environments must be maintained:

| Environment | Purpose | Key packages |
|---|---|---|
| **index-env** | Indexing, querying, delete/reinsert | `endee`, `datasets`, `sentence-transformers`, `tqdm`, `numpy` |
| **validation-env** | Metrics evaluation, qrels, embedding creation | `ranx`, `datasets` |

> **Why separate?** The `endee` SDK and `ranx` have conflicting dependency trees. Keep them isolated to avoid breakage.

```bash
# Create both environments with Python 3.12
python3.12 -m venv ~/index-env
python3.12 -m venv ~/validation-env
```

Install dependencies into each environment:
```bash
# index-env dependencies
source ~/index-env/bin/activate
pip install -r index-env.txt
deactivate

# validation-env dependencies
source ~/validation-env/bin/activate
pip install -r validation-env.txt
deactivate
```

Activate before running each script:
```bash
source ~/index-env/bin/activate       # for indexing / querying scripts
source ~/validation-env/bin/activate  # for validation / embedding scripts
```

---

## Scripts

### 1. `embedding_creation.py`
> **Environment: validation-env**

Creates dense + sparse embeddings from a BeIR-style HuggingFace dataset and writes both corpus and query embeddings as JSONL in a single run.

**Command**
```bash
python embedding_creation.py \
  --dataset-id <HF_DATASET_ID> \
  [--output-path <PREFIX>] \
  [--batch-size 1000] \
  [--device cpu] \
  [--dense-model sentence-transformers/all-MiniLM-L6-v2] \
  [--sparse-model prithivida/Splade_PP_en_v1] \
  [--cache-dir <PATH>]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--dataset-id` | **Yes** | — | HuggingFace dataset ID — use `BeIR/scifact` for the 5k dataset, `BeIR/quora` for the 500k dataset |
| `--output-path` | No | `embeddings` | Output filename prefix; produces `<prefix>_corpus.jsonl` and `<prefix>_queries.jsonl` |
| `--batch-size` | No | `1000` | Encoding batch size |
| `--device` | No | `cpu` | Torch device (`cpu`, `cuda`) |
| `--dense-model` | No | `sentence-transformers/all-MiniLM-L6-v2` | Dense model ID (HuggingFace) |
| `--sparse-model` | No | `prithivida/Splade_PP_en_v1` | Sparse model ID (HuggingFace) |
| `--cache-dir` | No | HF default | Local directory to cache downloaded datasets and models |

**Output files**
- `<output-path>_corpus.jsonl` — corpus embeddings
- `<output-path>_queries.jsonl` — query embeddings

**Dataset loading behaviour**

The script uses the standard BeIR split convention tested across common datasets:
```python
load_dataset(dataset_id, "corpus",  split="corpus")   # corpus
load_dataset(dataset_id, "queries", split="queries")  # queries
```
> **Note:** This general structure works for most BeIR datasets. If a dataset uses different config or split names, edit `load_hf_dataset()` in `embedding_creation.py` directly to match the correct `load_dataset()` call.

**Embedding models**

Dense and sparse models are loaded from HuggingFace. The defaults are:
- Dense: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Sparse: `prithivida/Splade_PP_en_v1`

> **Note:** To use a different model, pass any compatible HuggingFace model ID via `--dense-model` or `--sparse-model`. Make sure the dense model output dimension matches the `dimension` used when creating the Endee index (default: `384`). And sparse model output match the `sparse_dim` used when creating the Endee index(default:`30522`)

**Dataset IDs by size**

| Size | `--dataset-id` | `--output-path`  |
|------|---------------|-------------------------------|
| 5k   | `BeIR/scifact` | `scifact` |
| 500k | `BeIR/quora`   | `quora` |

**Examples**
```bash
# 5k dataset (scifact)
python embedding_creation.py \
  --dataset-id BeIR/scifact \
  --output-path scifact

# 500k dataset (quora)
python embedding_creation.py \
  --dataset-id BeIR/quora \
  --output-path quora
```
This produces `<output-path>_corpus.jsonl` and `<output-path>_queries.jsonl`.

---

### 2. `endee_indexing.py`
> **Environment: index-env**

Creates an Endee index and ingests vectors from a JSONL embeddings file.

**Command**
```bash
python endee_indexing.py \
  --index_name <INDEX_NAME> \
  --jsonl_path <PATH.jsonl> \
  [--token 12345678] \
  [--base-url https://dev.endee.io/api/v1] \
  [--batch-size 1000] \
  [--create-index true] \
  [--dimension 384] \
  [--space-type cosine] \
  [--sparse-dim 30522] \
  [--precision float32] \
  [--cache-dir <PATH>]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--index_name` | **Yes** | — | Name of the index to create and populate |
| `--jsonl_path` | **Yes** | — | Path to the corpus JSONL embeddings file created during embedding_creation.py|
| `--token` | No | `12345678` | Endee API token |
| `--base-url` | No | `https://dev.endee.io/api/v1` | Endee base URL |
| `--batch-size` | No | `1000` | Upsert batch size |
| `--create-index` | No | `true` | Create the index before ingesting; set `false` to skip if index already exists |
| `--dimension` | No | `384` | Dense vector dimension; must match the embedding model output |
| `--space-type` | No | `cosine` | Distance metric: `cosine`, `l2`, `ip` |
| `--sparse-dim` | No | `30522` | Sparse vector dimension; must match the sparse model vocabulary size |
| `--precision` | No | `float32` | Vector precision: `float32`, `float16`,`int16d`,`int8d`,`binary` |
| `--cache-dir` | No | HF default | Local directory to cache downloaded datasets |

**Output**
Saves `<index_name>/indexing_performance.json` with upsert latency stats (min, p50, p95, p99, max, mean).

**Examples**
```bash
# Create index and ingest
python endee_indexing.py \
  --index_name quora_index \
  --jsonl_path quora_embeddings.jsonl \
  --token mytoken --base-url http://51.89.231.115:8080/api/v1

# Ingest into existing index (skip creation)
python endee_indexing.py \
  --index_name quora_index \
  --jsonl_path quora_embeddings.jsonl \
  --create-index false
```

---

### 3. `endee_delete_insert.py`
> **Environment: index-env**

Deletes a subset of vectors from an Endee index and reinserts them from the JSONL file. Supports ground-truth filtering before applying the selection mode.

**Command**
```bash
python endee_delete_insert.py \
  --index_name <INDEX_NAME> \
  --jsonl_path <PATH.jsonl> \
  [--token 12345678] \
  [--base-url https://dev.endee.io/api/v1] \
  [--delete-percentage 10] \
  [--mode random] \
  [--ground-truth-file <PATH>] \
  [--gt-filter <ground-truth|non-ground-truth>] \
  [--batch-size 1000] \
  [--skip-reinsert false] \
  [--verify-delete]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--index_name` | **Yes** | — | Name of the Endee index |
| `--jsonl_path` | **Yes** | — | Path to the corpus JSONL embeddings file  created during embeddin_creation.py|
| `--token` | No | `12345678` | Endee API token |
| `--base-url` | No | `https://dev.endee.io/api/v1` | Endee base URL |
| `--delete-percentage` | No | `10` | Percentage of vectors to delete from the candidate pool |
| `--mode` | No | `random` | Selection mode: `random`, `last-n-percent`, `first-n-percent` |
| `--ground-truth-file` | No* | — | Path to `unique_doc_ids.txt` from `endee_validation.py`; required when `--gt-filter` is set |
| `--gt-filter` | No | — | Filter pool before mode: `ground-truth` (only GT ids) or `non-ground-truth` (only non-GT ids) |
| `--batch-size` | No | `1000` | Reinsert batch size |
| `--skip-reinsert` | No | `false` | Set `true` to delete only without reinserting |
| `--verify-delete` | No | `false` | After deletion, verify each deleted ID is gone from the index |

**Selection flow**
1. Load all record IDs from JSONL
2. *(Optional)* Narrow pool with `--gt-filter` using `--ground-truth-file`
3. Apply `--mode` + `--delete-percentage` to the pool

**Examples**
```bash
# Delete random 10% of all vectors and reinsert
python endee_delete_insert.py \
  --index_name quora_index \
  --jsonl_path quora_embeddings.jsonl

# Delete last 30% of non-ground-truth vectors
python endee_delete_insert.py \
  --index_name quora_index \
  --jsonl_path quora_embeddings.jsonl \
  --gt-filter non-ground-truth \
  --ground-truth-file quora/unique_doc_ids.txt \
  --mode last-n-percent --delete-percentage 30

# Delete only, skip reinsert
python endee_delete_insert.py \
  --index_name quora_index \
  --jsonl_path quora_embeddings.jsonl \
  --skip-reinsert true
```

---

### 4. `query_endee_async_multiprocessing.py`
> **Environment: index-env**

Runs hybrid queries against an Endee index using multiprocessing + async concurrency. Saves per-batch results, latencies, and a merged results file.

**Command**
```bash
python query_endee_async_multiprocessing.py \
  --query-file <PATH> \
  --index-name <INDEX_NAME> \
  --vector-token <TOKEN> \
  --concurrency <N> \
  --test-cycle <N> \
  [--async-concurrency 10] \
  [--top-k 10] \
  [--base-url https://dev.endee.io/api/v1] \
  [--output-base test]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--query-file` | **Yes** | — | Path to query embeddings JSON file created during embedding_creation.py |
| `--index-name` | **Yes** | — | Name of the Endee index |
| `--vector-token` | No | `12345678` | Endee API token |
| `--concurrency` | **Yes** | — | Number of parallel worker processes |
| `--test-cycle` | **Yes** | — | Test cycle number (used in output path) |
| `--async-concurrency` | No | `10` | Concurrent async queries per worker |
| `--top-k` | No | `10` | Number of top results to retrieve |
| `--base-url` | No | `https://dev.endee.io/api/v1` | Endee base URL |
| `--output-base` | No | `test` | Base output directory |

**Output directory:** `<output-base>/testcycle<N>/concurrency<N>/`

| File | Description |
|---|---|
| `merged_results.json` | `{ query_id: { doc_id: score, ... }, ... }` — ranked results for every query, sorted by latency ascending; **input for `endee_validation.py`** |
| `all_latencies.json` | Per-query latency entries `{ query_id, latency_ms, worker_id, batch_id, num_results }`, sorted ascending |
| `p99_latency.json` | `{ p99_latency_ms, total_queries, successful_queries, failed_queries }` |
| `summary.json` | Aggregate stats: p50/p90/p95/p99/mean/min/max latency, total time, success/fail counts |
| `detailed_logs.json` | Per-query log entries with status, latency, worker, batch; includes error field on failures |

**Example**
```bash
python query_endee_async_multiprocessing.py \
  --query-file quora_query_embeddings.json \
  --index-name quora_index \
  --vector-token mytoken \
  --concurrency 16 \
  --async-concurrency 5 \
  --test-cycle 1 \
  --top-k 10
```

---

### 5. `endee_validation.py`
> **Environment: validation-env**

Evaluates search results (from `query_endee_async_multiprocessing.py`) against BeIR ground-truth qrels. Computes NDCG@10, MAP@10, Recall@10.

**Command**
```bash
python endee_validation.py \
  --dataset <5k|500k> \
  --run-file <PATH/merged_results.json> \
  --testcycle <OUTPUT_DIR> \
  [--need-unique-doc-ids false] \
  [--validation true] \
  [--cache-dir <PATH>]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--dataset` | **Yes** | — | `5k` → BeIR/scifact-qrels (train) · `500k` → BeIR/quora-qrels (test) |
| `--run-file` | **Yes** | — | Path to `merged_results.json` from the query script |
| `--testcycle` | **Yes** | — | Output directory for `correctness.json` |
| `--need-unique-doc-ids` | No | `false` | Save unique ground-truth doc IDs to file |
| `--validation` | No | `true` | Run validation (load qrels, compute NDCG/MAP/Recall); set `false` to only save unique doc IDs without computing metrics |
| `--cache-dir` | No | HF default | Local directory to cache downloaded datasets |

**Unique doc ID output paths**
- `5k` → `scifact/unique_doc_ids.txt`
- `500k` → `quora/unique_doc_ids.txt`

These files are used as `--ground-truth-file` input in `endee_delete_insert.py`.

**Examples**
```bash
# Evaluate quora 500k results and save ground-truth IDs
python endee_validation.py \
  --dataset 500k \
  --run-file test/testcycle1/concurrency4/merged_results.json \
  --testcycle test/testcycle1/concurrency4 \
  --need-unique-doc-ids true

# Evaluate scifact 5k results
python endee_validation.py \
  --dataset 5k \
  --run-file test/testcycle1/concurrency4/merged_results.json \
  --testcycle test/testcycle1/concurrency4
```

---

## Sparse-Only Benchmark (`sparse-vectors-benchmark/`)

> **Environment: index-env**

Benchmarks pure sparse vector search against Endee using the [NeurIPS 2023](https://big-ann-benchmarks.com/neurips23.html) MSMARCO sparse dataset (SPLADE-encoded). Measures query latency and optionally checks recall against ground truth.

> **Server-side requirement:** For a true sparse-only benchmark, the **dense vector search code must be commented out on the Endee server** so queries are evaluated using only sparse vectors.

> **Dummy dense vector:** Since the Endee API requires a dense vector in every upsert and query call, the benchmark upserts a fixed dummy dense vector `[0.1] * 10` (10-dim) alongside each sparse vector. Queries also send this dummy vector. This is purely for API compatibility — all scoring is done via the sparse path.

**Command**
```bash
cd sparse-vectors-benchmark
python main.py \
  --endee-token <TOKEN> \
  [--endee-base-url http://51.89.231.115:8080/api/v1] \
  [--skip-creation true] \
  [--dataset small] \
  [--slow-ms 500] \
  [--search-limit 10] \
  [--data-path ./data] \
  [--results-path ./results] \
  [--analyze-data false] \
  [--check-ground-truth false] \
  [--graph-y-range "0 500"] \
  [--upsert-batch-size 1000] \
  [--max-retries 5] \
  [--concurrency 16] \
  [--async-concurrency 5]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--endee-token` | No | — | Endee API token |
| `--endee-base-url` | No | `https://dev.endee.io/api/v1` | Endee base URL |
| `--skip-creation` | No | `true` | Skip index creation and data upload; set `false` to create index and upload data |
| `--dataset` | No | `small` | Dataset size: `small` (100k), `1M` (1M), `full` (8.8M) |
| `--slow-ms` | No | `500` | Print a warning for any query exceeding this latency (ms) |
| `--search-limit` | No | `10` | Top-k results to retrieve per query |
| `--data-path` | No | `./data` | Directory where dataset files are downloaded and cached |
| `--results-path` | No | `./results` | Directory where result plots are saved |
| `--analyze-data` | No | `false` | Print dataset stats and posting list distribution |
| `--check-ground-truth` | No | `false` | Compare results against ground-truth file and compute recall |
| `--graph-y-range` | No | auto | Fix the y-axis range of the output plot, e.g. `"0 500"` (helps compare runs) |
| `--upsert-batch-size` | No | `1000` | Vectors per upsert batch during index creation |
| `--max-retries` | No | `5` | Retry count on failed upserts |
| `--concurrency` | No | `1` | Number of parallel worker processes for querying |
| `--async-concurrency` | No | `10` | Concurrent async queries per worker |

**Datasets** (auto-downloaded from Google Storage if not present)

| `--dataset` | Vectors | Download size |
|---|---|---|
| `small` | 100,000 | 64 MB |
| `1M` | 1,000,000 | 636 MB |
| `full` | 8,841,823 | 5.5 GB |

**Output**
- Console: latency percentiles (min, p50, p95, p99, p99.9, max), total time, recall (if `--check-ground-truth true`)
- Plot saved to `<results-path>/sparse_bench_<dataset>_<timestamp>.png` — 2D histogram of query dimension count vs latency

**Examples**
```bash
# First run: create index and upload small dataset
python main.py \
  --endee-token mytoken \
  --skip-creation false \
  --dataset small

# Subsequent runs: skip creation, query only
python main.py \
  --endee-token mytoken \
  --dataset small \
  --concurrency 4 \
  --async-concurrency 10

# 1M dataset with ground truth check and fixed y-axis for comparison
python main.py \
  --endee-token mytoken \
  --dataset 1M \
  --check-ground-truth true \
  --graph-y-range "0 500" \
  --concurrency 4
```

---

## Typical End-to-End Workflow

```
[validation-env]
1. embedding_creation.py   → <prefix>corpus.jsonl + <prefix>queries.jsonl  (single run)

[index-env]
2. endee_indexing.py       → index ingested
3. query_endee_async_multiprocessing.py → merged_results.json

[validation-env]
4. endee_validation.py     → correctness.json + unique_doc_ids.txt

[index-env]
5. endee_delete_insert.py  → delete/reinsert with GT filter

[index-env]
6. query_endee_async_multiprocessing.py → merged_results.json (post delete/reinsert)

[validation-env]
7. endee_validation.py     → correctness.json (compare with step 4)
```
