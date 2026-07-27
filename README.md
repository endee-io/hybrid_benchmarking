# Hybrid Vector Benchmark 

End-to-end hybrid vector benchmark framework. A single command runs **indexing → querying → validation** for any supported DB.

---

## Quick setup

On a fresh Linux machine, `setup.sh` handles everything in one command — Python 3.13 (via pyenv if needed), cloning the repo, and creating both virtual environments:

```bash
bash setup.sh
```

Then activate `index-env` before running benchmarks:

```bash
source index-env/bin/activate
```

> `validation-env` does **not** need to be manually activated — it is invoked automatically by the benchmark pipeline via `--validation-venv validation-env`.

Skip to [Generating embeddings](#generating-embeddings) if setup is done.

---

## Prerequisites

### Python version

```bash
# macOS
brew install python@3.13

# Debian
sudo apt install python3.13 python3.13-venv
```

Verify:
```bash
python3.13 --version
```

---

### Create environments

Two separate environments are required:

| Environment | Purpose | Key packages |
|---|---|---|
| **index-env** | Indexing, querying (runs `main.py`) | `endee`, `qdrant-client`, `numpy`, `tqdm` |
| **validation-env** | Metrics evaluation | `ranx`, `datasets` |

> **Why separate?** `endee` and `ranx` have conflicting dependency trees.

```bash
python3.13 -m venv index-env
python3.13 -m venv validation-env
```

```bash
# index-env
source index-env/bin/activate
pip install -r index-env.txt
deactivate

# validation-env
source validation-env/bin/activate
pip install -r validation-env.txt
deactivate
```

Activate before running:
```bash
source index-env/bin/activate    # for main.py
```

---

### Data directory

The benchmark reads pre-built `.npy` embedding files. Place them under `data/<dataset-name>/`:

```
data/
└── beir_quora/
    ├── beir_quora_dense_corpus.npy
    ├── beir_quora_dense_corpus_ids.npy
    ├── beir_quora_sparse_corpus_splade_values.npy
    ├── beir_quora_sparse_corpus_splade_col_indices.npy
    ├── beir_quora_sparse_corpus_splade_indptr.npy
    ├── beir_quora_sparse_corpus_splade_ids.npy
    ├── beir_quora_dense_queries.npy
    ├── beir_quora_dense_queries_ids.npy
    ├── beir_quora_sparse_queries_splade_values.npy
    ├── beir_quora_sparse_queries_splade_col_indices.npy
    ├── beir_quora_sparse_queries_splade_indptr.npy
    └── beir_quora_sparse_queries_splade_ids.npy
```

Same structure applies for `beir_scifact` and `beir_msmarco`. Use `src/embedding_creation.py` (index-env) to generate these files.

---

## Generating embeddings

All `.npy` files are produced by `src/embedding_creation.py`. Run it inside `index-env` before any benchmark.

### Flags

| Flag | Required | Default | Description |
|---|---|---|---|
| `--dataset-id` | **Yes** | — | HuggingFace dataset ID, e.g. `BeIR/quora`, `BeIR/msmarco` |
| `--dense` | No | `false` | Generate dense embeddings |
| `--sparse-mode` | No | — | One or more sparse modes to generate (see table below); accepts multiple values |
| `--batch-size` | No | `1000` | Encoding batch size |
| `--device` | No | `cpu` | Torch device: `cpu`, `cuda`, or `mps` |
| `--dense-model` | No | `sentence-transformers/all-MiniLM-L6-v2` | Dense encoder model (output dim: 384) |
| `--splade-model` | No | `prithivida/Splade_PP_en_v1` | SPLADE model for `--sparse-mode splade` |
| `--milvus-splade-model` | No | `naver/splade-cocondenser-selfdistil` | SPLADE model for `--sparse-mode milvus_splade` |
| `--workers` | No | `5` | Parallel CPU workers for dense and SPLADE encoding |
| `--cache-dir` | No | — | Local directory to cache HuggingFace datasets and models (e.g. `model_cache`); avoids re-downloading on subsequent runs. If omitted, HuggingFace defaults to `~/.cache/huggingface/` |

### Sparse modes

| `--sparse-mode` | Description |
|---|---|
| `pymilvus_bm25` | Client-side BM25 via PyMilvus `BM25EmbeddingFunction`; fits IDF on the corpus first, saves a model JSON, then encodes queries using that model; works with any DB |
| `endee_bm25` | Endee BM25 via `endee_model.SparseModel`; uses the Endee tokenizer with server-side IDF. **Endee only.** |
| `splade` | SPLADE via `sentence-transformers` `SparseEncoder`; works with any DB |
| `milvus_splade` | SPLADE via PyMilvus `SpladeEmbeddingFunction`. **Milvus only.** |
| `bm25` | rank_bm25 BM25L; IDF built from full corpus each run; works with any DB |

Multiple modes can be passed in one command — corpus and queries are processed for each mode in the same run.

> **Restart support:** If the process is interrupted, it resumes from the last checkpoint automatically. No need to restart from scratch.

### Examples

**Dense only:**
```bash
source index-env/bin/activate

python3 src/embedding_creation.py \
  --dataset-id BeIR/quora \
  --dense \
  --cache-dir model_cache
```

**Dense + pymilvus BM25 sparse:**
```bash
python3 src/embedding_creation.py \
  --dataset-id BeIR/quora \
  --dense \
  --sparse-mode pymilvus_bm25 \
  --cache-dir model_cache
```

**Dense + endee BM25 sparse:**
```bash
python3 src/embedding_creation.py \
  --dataset-id BeIR/quora \
  --dense \
  --sparse-mode endee_bm25 \
  --cache-dir model_cache
```

**Dense + SPLADE sparse:**
```bash
python3 src/embedding_creation.py \
  --dataset-id BeIR/quora \
  --dense \
  --sparse-mode splade \
  --cache-dir model_cache
```

**Multiple sparse modes in one run:**
```bash
python3 src/embedding_creation.py \
  --dataset-id BeIR/msmarco \
  --dense \
  --sparse-mode pymilvus_bm25 endee_bm25 splade \
  --cache-dir model_cache
```

### Output files

All files are written to `data/<dataset-name>/`. The dataset name is derived from the `--dataset-id` by lowercasing and replacing `/` with `_` (e.g. `BeIR/quora` → `beir_quora`).

```
data/beir_quora/
├── beir_quora_dense_corpus.npy                        — float64 (N_corpus, 384)
├── beir_quora_dense_corpus_ids.npy                    — str     (N_corpus,)
├── beir_quora_dense_queries.npy                       — float64 (N_queries, 384)
├── beir_quora_dense_queries_ids.npy                   — str     (N_queries,)
├── beir_quora_sparse_corpus_<mode>_values.npy         — float64 (total_nnz,)
├── beir_quora_sparse_corpus_<mode>_col_indices.npy    — int32   (total_nnz,)
├── beir_quora_sparse_corpus_<mode>_indptr.npy         — int64   (N_corpus+1,)
├── beir_quora_sparse_corpus_<mode>_ids.npy            — str     (N_corpus,)
├── beir_quora_sparse_queries_<mode>_values.npy        — float64 (total_nnz,)
├── beir_quora_sparse_queries_<mode>_col_indices.npy   — int32   (total_nnz,)
├── beir_quora_sparse_queries_<mode>_indptr.npy        — int64   (N_queries+1,)
├── beir_quora_sparse_queries_<mode>_ids.npy           — str     (N_queries,)
└── beir_quora_pymilvus_bm25_model.json                — fitted BM25 model (pymilvus_bm25 only)
```

Replace `<mode>` with the sparse mode name: `pymilvus_bm25`, `endee_bm25`, `splade`, `milvus_splade`, or `bm25`.

---

## Output structure

All results are written under `results/<db>/`:

```
results/
└── endee/
    ├── <index_name>.json                  ← indexing performance: total upsert time (`total_time_sec`), total vectors, batch size, and per-batch upsert latency stats (min/p50/p95/p99/max/mean)
    └── <results>_concurrency<N>/
        ├── summary.json                   ← QPS, latency percentiles (p50/p90/p95/p99)
        ├── correctness.json               ← NDCG, Recall, MAP at top-k
        ├── p99_latency.json               ← standalone p99 latency (used by automation scripts)
        ├── merged_results.json            ← raw search results (input to validation)
        ├── all_latencies.json             ← per-query latencies sorted ascending
        └── detailed_logs.json             ← per-query status, latency, worker, batch
```

| File | Key fields | Description |
|---|---|---|
| `<index_name>.json` | `total_time_sec`, `total_vectors`, `batch_size`, `upsert_latency_ms.{min,p50,p95,p99,max,mean}` | Total time to upsert/load the dataset plus per-batch upsert latency breakdown |
| `merged_results.json` | `{ query_id: { doc_id: score } }` | Raw search results — input to the validation step |
| `all_latencies.json` | `query_id`, `latency_ms` | Per-query latency entries, sorted ascending |
| `summary.json` | **`qps`**, `p50/p90/p95/p99/mean/min/max_latency_ms`, `total_time_seconds`, `successful_queries`, `failed_queries` | Aggregate query performance. **QPS is here.** |
| `p99_latency.json` | **`p99_latency_ms`**, `total_queries`, `successful_queries`, `failed_queries` | Standalone p99 latency file read by the automation benchmark scripts |
| `correctness.json` | **`ndcg@{k}`**, **`recall@{k}`**, **`map@{k}`** | Relevance metrics at the top-k used for the run. **NDCG, Recall, and MAP are here.** |
| `detailed_logs.json` | `query_id`, `status`, `latency_ms`, `worker_id`, `batch_id` | Per-query log with status, latency, worker, and batch info |

---

## Common flags

These flags apply to all DBs.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--db` | **Yes** | — | DB to benchmark: `endee` or `qdrant`  |
| `--index-name` | **Yes** | — | Index / collection name |
| `--dataset-name` | **Yes** | — | Dataset name matching the folder under `--data-dir` (e.g. `beir_scifact`, `beir_quora`) |
| `--results` | **Yes** | — | Results folder label used as output folder name (e.g. `run1`) |
| `--concurrency` | **Yes** | — | Number of parallel worker processes for querying |
| `--sparse-mode` | No | `splade` | Sparse embedding type: `pymilvus_bm25` (PyMilvus BM25EmbeddingFunction), `endee_bm25` (Endee-only), `bm25` (rank_bm25, any DB), or `splade` |
| `--data-dir` | No | `data` | Root directory containing dataset `.npy` files |
| `--top-k` | No | `10` | Number of top results to retrieve per query |
| `--batch-size` | No | `1000` | Upsert batch size during indexing |
| `--dimension` | No | `384` | Dense vector dimension; must match the embedding model output |
| `--space-type` | No | `cosine` | Distance metric: `cosine`, `dot`, `l2` |
| `--create-index` | No | `true` | Create the index before indexing; set `false` to skip if already exists |
| `--skip-indexing` | No | `false` | Skip the indexing step; use when the collection is already populated and you only want to query |
| `--skip-query` | No | `false` | Skip the query step; no search requests are sent, so QPS, latency, and `merged_results.json` are not produced — validation is also skipped since it depends on query output |
| `--skip-validation` | No | `false` | Skip the validation step; query runs normally so QPS and concurrent latency (p50/p95/p99 under parallel load, not serial per-query latency) are measured, but `correctness.json` is not produced — Recall, NDCG, and MAP are not computed |
| `--validation-venv` | No | `validation-env` | Path to the validation virtual environment folder (e.g. `validation-env`) |

---

## Endee

### Endee-specific flags

| Flag | Required | Default | Description |
|---|---|---|---|
| `--vector-token` | No | `12345678` | Endee API token in `dbname:secret` format |
| `--base-url` | No | `https://dev.endee.io/api/v2` | Endee server base URL — must end with `/api/v2` |
| `--sparse-scoring-model` | No | `default` | Sparse scoring model used by the collection: `default` (pymilvus BM25) or `endee_bm25`. Only required for `endee_bm25` mode; all other modes use the default |
| `--precision` | No | `float32` | Dense vector storage precision: `float32`, `float16`, `int16`, `int8`, `int8e`, `binary` |
| `--query-mode` | No | `hybrid` | Query mode: `hybrid` (dense + sparse with RRF fusion) or `sparse` (sparse field only, no fusion) |

> **Important:** The base URL must always include `/api/v2`. Example: `http://148.113.37.113:8080/api/v2`.

---

### Sparse modes

The `--sparse-mode` flag controls which `.npy` embedding files are loaded and which sparse representations are sent to the server. The `--sparse-scoring-model` flag tells the **server** which scoring model the collection was created with — these two must be consistent.

| `--sparse-mode` | `--sparse-scoring-model` | Notes |
|---|---|---|
| `pymilvus_bm25` | *(omit — uses default)* | Client-side BM25 via PyMilvus `BM25EmbeddingFunction` |
| `endee_bm25` | `endee_bm25` | Endee's own BM25; collection must be created with `sparse_model: endee_bm25` |
| `splade` | *(omit — uses default)* | SPLADE dense-sparse encoder; collection uses default sparse model |

---

### Required data files

All `.npy` files must be present under `data/<dataset-name>/` before running. The filenames follow a fixed pattern based on dataset name and sparse mode.

**Example — `beir_quora` with `pymilvus_bm25`:**
```
data/beir_quora/
├── beir_quora_dense_corpus.npy
├── beir_quora_dense_corpus_ids.npy
├── beir_quora_dense_queries.npy
├── beir_quora_dense_queries_ids.npy
├── beir_quora_sparse_corpus_pymilvus_bm25_col_indices.npy
├── beir_quora_sparse_corpus_pymilvus_bm25_ids.npy
├── beir_quora_sparse_corpus_pymilvus_bm25_indptr.npy
├── beir_quora_sparse_corpus_pymilvus_bm25_values.npy
├── beir_quora_sparse_queries_pymilvus_bm25_col_indices.npy
├── beir_quora_sparse_queries_pymilvus_bm25_ids.npy
├── beir_quora_sparse_queries_pymilvus_bm25_indptr.npy
└── beir_quora_sparse_queries_pymilvus_bm25_values.npy
```

For other sparse modes replace `pymilvus_bm25` with `endee_bm25` or `splade` in the filenames. The dense files (`dense_corpus*`, `dense_queries*`) are shared across all sparse modes.

---

### Query modes

**Hybrid mode** (default): sends both dense and sparse vectors per query. Results from the two fields are fused using RRF (Reciprocal Rank Fusion, k=60).

**Sparse-only mode**: sends only the sparse vector. No fusion — single field result. Precision has no effect on sparse queries (sparse indices and values are stored independently of the dense precision setting).

The index always stores both dense and sparse vectors regardless of `--query-mode`. The query mode only affects what is sent at search time.

---

### Full run — index + query + validate

**Hybrid, pymilvus BM25, float32:**
```bash
python3 -m src.main \
  --db endee \
  --index-name quora_bench_pymilvus_float32 \
  --dataset-name beir_quora \
  --results run1_quora_hybrid_pmbm25_float32 \
  --concurrency 16 \
  --top-k 10 \
  --vector-token mydb:yoursecrettoken \
  --base-url http://your-server:8080/api/v2 \
  --sparse-mode pymilvus_bm25 \
  --precision float32 \
  --query-mode hybrid \
  --validation-venv validation-env
```

**Hybrid, endee BM25, int16:**
```bash
python3 -m src.main \
  --db endee \
  --index-name quora_bench_endee_bm25_int16 \
  --dataset-name beir_quora \
  --results run1_quora_hybrid_endeebm25_int16 \
  --concurrency 16 \
  --top-k 10 \
  --vector-token mydb:yoursecrettoken \
  --base-url http://your-server:8080/api/v2 \
  --sparse-mode endee_bm25 \
  --sparse-scoring-model endee_bm25 \
  --precision int16 \
  --query-mode hybrid \
  --validation-venv validation-env
```

**Hybrid, SPLADE, int16:**
```bash
python3 -m src.main \
  --db endee \
  --index-name quora_bench_splade_int16 \
  --dataset-name beir_quora \
  --results run1_quora_hybrid_splade_int16 \
  --concurrency 16 \
  --top-k 10 \
  --vector-token mydb:yoursecrettoken \
  --base-url http://your-server:8080/api/v2 \
  --sparse-mode splade \
  --precision int16 \
  --query-mode hybrid \
  --validation-venv validation-env
```

---

### Skip indexing — query + validate only

Use `--skip-indexing` when the collection is already indexed. This is the most common mode for repeated benchmark runs.

```bash
python3 -m src.main \
  --db endee \
  --index-name quora_bench_pymilvus_float32 \
  --dataset-name beir_quora \
  --results run2_quora_hybrid_pmbm25_float32 \
  --concurrency 16 \
  --top-k 10 \
  --vector-token mydb:yoursecrettoken \
  --base-url http://your-server:8080/api/v2 \
  --sparse-mode pymilvus_bm25 \
  --precision float32 \
  --query-mode hybrid \
  --skip-indexing \
  --validation-venv validation-env
```

---

### Sparse-only query

Set `--query-mode sparse` to query using the sparse field only. Only sparse `.npy` files are needed for the query step. The `--precision` flag is still required (it was used during indexing) but has no effect on the sparse query itself.

**pymilvus BM25, sparse-only:**
```bash
python3 -m src.main \
  --db endee \
  --index-name quora_bench_pymilvus_float32 \
  --dataset-name beir_quora \
  --results run1_quora_sparse_pmbm25 \
  --concurrency 16 \
  --top-k 10 \
  --vector-token mydb:yoursecrettoken \
  --base-url http://your-server:8080/api/v2 \
  --sparse-mode pymilvus_bm25 \
  --precision float32 \
  --query-mode sparse \
  --skip-indexing \
  --validation-venv validation-env
```

**endee BM25, sparse-only:**
```bash
python3 -m src.main \
  --db endee \
  --index-name quora_bench_endee_bm25_int16 \
  --dataset-name beir_quora \
  --results run1_quora_sparse_endeebm25 \
  --concurrency 16 \
  --top-k 10 \
  --vector-token mydb:yoursecrettoken \
  --base-url http://your-server:8080/api/v2 \
  --sparse-mode endee_bm25 \
  --sparse-scoring-model endee_bm25 \
  --precision int16 \
  --query-mode sparse \
  --skip-indexing \
  --validation-venv validation-env
```

**SPLADE, sparse-only:**
```bash
python3 -m src.main \
  --db endee \
  --index-name quora_bench_splade_int16 \
  --dataset-name beir_quora \
  --results run1_quora_sparse_splade \
  --concurrency 16 \
  --top-k 10 \
  --vector-token mydb:yoursecrettoken \
  --base-url http://your-server:8080/api/v2 \
  --sparse-mode splade \
  --precision int16 \
  --query-mode sparse \
  --skip-indexing \
  --validation-venv validation-env
```

---

### Indexing only (no query, no validation)

Use `--create-index false` if the collection already exists and you just want to upsert more data without recreating it.

```bash
# Create collection and index data
python3 -m src.main \
  --db endee \
  --index-name quora_bench_pymilvus_float32 \
  --dataset-name beir_quora \
  --results run1 \
  --concurrency 16 \
  --vector-token mydb:yoursecrettoken \
  --base-url http://your-server:8080/api/v2 \
  --sparse-mode pymilvus_bm25 \
  --precision float32 \
  --skip-query \
  --skip-validation

# Upsert into existing collection (skip creation)
python3 -m src.main \
  --db endee \
  --index-name quora_bench_pymilvus_float32 \
  --dataset-name beir_quora \
  --results run1 \
  --concurrency 16 \
  --vector-token mydb:yoursecrettoken \
  --base-url http://your-server:8080/api/v2 \
  --sparse-mode pymilvus_bm25 \
  --precision float32 \
  --create-index false \
  --skip-query \
  --skip-validation
```

---

### Automation scripts

Pre-built scripts run a full sweep across top-k values `[10, 30, 60, 100, 500, 1000]`, execute 3 runs per top-k, and export results to an Excel file. All scripts assume the collection is already indexed (`--skip-indexing` is set internally).

| Script | Dataset | Sparse mode | Precision |
|---|---|---|---|
| `bench_endee_sparse_topk.py` | beir_quora | pymilvus_bm25 | float32 |
| `bench_endee_bm25_sparse_topk.py` | beir_quora | endee_bm25 | float32 |
| `bench_endee_splade_sparse_topk.py` | beir_quora | splade | int16 |
| `bench_endee_msmarco_sparse_topk.py` | beir_msmarco | pymilvus_bm25 | float32 |
| `bench_endee_msmarco_endee_bm25_sparse_topk.py` | beir_msmarco | endee_bm25 | int16 |
| `bench_endee_msmarco_splade_sparse_topk.py` | beir_msmarco | splade | int16 |

Edit the `CONFIGURATION` block at the top of each script to set `BASE_URL`, `VECTOR_TOKEN`, and `INDEX_NAME` before running:

```bash
# Activate index-env first
source index-env/bin/activate

python3 bench_endee_sparse_topk.py
```

The output Excel file is saved in the project root with a timestamp, e.g. `bench_endee_sparse_topk_20260709_143012.xlsx`.

---

## Typical end-to-end workflow

```
[index-env]
1. python src/embedding_creation.py   →  data/<dataset>/*.npy

[index-env]
2. python -m src.main --db <endee|qdrant> ...   →  indexing + querying + validation in one command
   Output: results/<db>/<results>_concurrency<N>/
```


---

## Adding a new DB

1. **Create `src/dbs/<dbname>/__init__.py`** — empty file

2. **Create `src/dbs/<dbname>/db.py`** — implement `HybridDB` from `src/interface.py`:

   | Method | What to implement |
   |---|---|
   | `init(index_name, dimension, space_type, create)` | Connect to DB; create collection/index if `create=True` |
   | `index_batch(points)` | Upsert a list of `{id, vector, sparse_indices, sparse_values, meta}` dicts; include retry logic |
   | `search(dense_vector, sparse_indices, sparse_values, top_k)` | Run query; return `[{"id": str, "score": float}]` |
   | `list_indices()` | Return list of existing index/collection names |
   | `add_args(parser)` *(static)* | Register all DB-specific argparse flags |
   | `build_config(args)` *(static)* | Return a dict of kwargs to pass to `__init__` |

   > For a complete reference implementation of all methods and static helpers, see [`src/dbs/qdrant/db.py`](src/dbs/qdrant/db.py).

3. **Register in `src/utils.py`** — add one line to `DB_REGISTRY`:
   ```python
   from src.dbs.<dbname>.db import <NewDB>
   DB_REGISTRY = {
       "endee":  EndeeDB,
       "qdrant": QdrantDB,
       "<dbname>": <NewDB>,   # add this
   }
   ```

No changes needed in `src/main.py`, `src/indexing.py`, or `src/query.py`.

---

## Adding a new BeIR dataset

1. **Generate embeddings** — pass the HuggingFace dataset ID to `embedding_creation.py`:
   ```bash
   python src/embedding_creation.py \
     --dataset-id BeIR/<dataset> \
     --dense \
     --sparse-mode splade \
     --cache-dir model_cache
   ```
   The dataset folder name is derived automatically: `BeIR/<dataset>` → `beir_<dataset>` (e.g. `beir_fiqa`).

2. **Register in `src/endee_validation.py`** — add an entry to `DATASET_CONFIG`:
   ```python
   DATASET_CONFIG = {
       "beir_scifact": {"hf_name": "BeIR/scifact-qrels", "split": "train", "out_dir": "beir_scifact"},
       "beir_quora":   {"hf_name": "BeIR/quora-qrels",   "split": "test",  "out_dir": "beir_quora"},
       "beir_<dataset>": {"hf_name": "BeIR/<dataset>-qrels", "split": "test", "out_dir": "beir_<dataset>"},
   }
   ```
   > Check the HuggingFace dataset page for the correct qrels split name (`"train"` or `"test"`).

3. **Run** — use the derived dataset name as `--dataset-name`:
   ```bash
   python -m src.main \
     --db endee \
     --dataset-name beir_<dataset> \
     ...
   ```

