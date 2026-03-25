# Hybrid Vector Benchmark — `main.py` Reference

End-to-end hybrid vector benchmark framework. A single command runs **indexing → querying → validation** for any supported DB.

---

## Prerequisites

### Python version

Use **Python 3.12** for both environments.

```bash
# macOS
brew install python@3.12

# Debian / Ubuntu
sudo apt install python3.12 python3.12-venv
```

Verify:
```bash
python3.12 --version   # Python 3.12.x
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
python3.12 -m venv ~/index-env
python3.12 -m venv ~/validation-env
```

```bash
# index-env
source ~/index-env/bin/activate
pip install -r index-env.txt
deactivate

# validation-env
source ~/validation-env/bin/activate
pip install -r validation-env.txt
deactivate
```

Activate before running:
```bash
source ~/index-env/bin/activate    # for main.py
```

---

### Data directory

The benchmark reads pre-built `.npy` embedding files. Place them under `data/<dataset-name>/`:

```
data/
└── quora/
    ├── quora_dense_corpus.npy
    ├── quora_dense_corpus_ids.npy
    ├── quora_sparse_corpus_bm25_values.npy
    ├── quora_sparse_corpus_bm25_col_indices.npy
    ├── quora_sparse_corpus_bm25_indptr.npy
    ├── quora_sparse_corpus_bm25_ids.npy
    ├── quora_dense_queries.npy
    ├── quora_dense_queries_ids.npy
    ├── quora_sparse_queries_bm25_values.npy
    ├── quora_sparse_queries_bm25_col_indices.npy
    ├── quora_sparse_queries_bm25_indptr.npy
    └── quora_sparse_queries_bm25_ids.npy
```

Same structure applies for `scifact`. Use `embedding_creation_v2.py` (index-env) to generate these files.

---

## Output structure

All results are written under `dbs/<db>/test/<testcycle>/concurrency<N>/`:

```
dbs/
└── qdrant/
    └── test/
        └── qdrant_test1/
            ├── indexing_performance.json
            └── concurrency16/
                ├── merged_results.json
                ├── all_latencies.json
                ├── summary.json
                └── detailed_logs.json
```

| File | Description |
|---|---|
| `indexing_performance.json` | Upsert latency stats (min, p50, p95, p99, max, mean), total time, batch size |
| `merged_results.json` | `{ query_id: { doc_id: score } }` — input for validation |
| `all_latencies.json` | Per-query latency entries sorted ascending |
| `summary.json` | Aggregate stats: p50/p90/p95/p99/mean/min/max latency, total time, success/fail counts |
| `detailed_logs.json` | Per-query log with status, latency, worker, batch |

---

## Common flags

These flags apply to all DBs.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--db` | **Yes** | — | DB to benchmark: `endee` or `qdrant` |
| `--index-name` | **Yes** | — | Index / collection name |
| `--dataset-name` | **Yes** | — | Dataset: `scifact` (5k) or `quora` (500k) |
| `--testcycle` | **Yes** | — | Test cycle label used as output folder name (e.g. `testcycle1`) |
| `--concurrency` | **Yes** | — | Number of parallel worker processes for querying |
| `--sparse-mode` | No | `bm25` | Sparse embedding type: `bm25` or `splade` |
| `--data-dir` | No | `data` | Root directory containing dataset `.npy` files |
| `--async-concurrency` | No | `1` | Concurrent async queries per worker process |
| `--top-k` | No | `10` | Number of top results to retrieve per query |
| `--batch-size` | No | `1000` | Upsert batch size during indexing |
| `--dimension` | No | `384` | Dense vector dimension; must match the embedding model output |
| `--space-type` | No | `cosine` | Distance metric: `cosine`, `dot`, `l2` |
| `--create-index` | No | `true` | Create the index before indexing; set `false` to skip if already exists |
| `--skip-indexing` | No | `false` | Skip the indexing step |
| `--skip-query` | No | `false` | Skip the query step |
| `--skip-validation` | No | `false` | Skip the validation step |
| `--validation-env` | No | — | Path to validation venv (e.g. `~/validation-env`); uses current Python if omitted |

---

## Endee

### Endee-specific flags

| Flag | Required | Default | Description |
|---|---|---|---|
| `--vector-token` | No | `12345678` | Endee API token |
| `--base-url` | No | `https://dev.endee.io/api/v1` | Endee server base URL including port and path |
| `--sparse-scoring-model` | No | `default` | Sparse scoring model name on the Endee server |
| `--precision` | No | `float32` | Vector storage precision: `float32`, `float16`, `int8d`, `binary` |

### Full run (index + query + validation)

```bash
python main.py \
  --db endee \
  --index-name quora_bench \
  --dataset-name quora \
  --testcycle testcycle1 \
  --concurrency 16 \
  --async-concurrency 5 \
  --top-k 10 \
  --vector-token mytoken \
  --base-url http://51.89.231.115:8080/api/v1 \
  --validation-env ~/validation-env
```

### Skip indexing (query + validate only)

```bash
python main.py \
  --db endee \
  --index-name quora_bench \
  --dataset-name quora \
  --testcycle testcycle1 \
  --concurrency 16 \
  --vector-token mytoken \
  --base-url http://51.89.231.115:8080/api/v1 \
  --skip-indexing \
  --validation-env ~/validation-env
```

### Query only (no index creation, no validation)

```bash
python main.py \
  --db endee \
  --index-name quora_bench \
  --dataset-name quora \
  --testcycle testcycle1 \
  --concurrency 16 \
  --vector-token mytoken \
  --base-url http://51.89.231.115:8080/api/v1 \
  --skip-indexing \
  --skip-validation
```

### Ingest into existing index (skip creation, run indexing)

```bash
python main.py \
  --db endee \
  --index-name quora_bench \
  --dataset-name quora \
  --testcycle testcycle1 \
  --concurrency 16 \
  --vector-token mytoken \
  --base-url http://51.89.231.115:8080/api/v1 \
  --create-index false \
  --skip-query \
  --skip-validation
```

---

## Qdrant

### Starting the Qdrant server (Docker)

```bash
# Pull image
docker pull qdrant/qdrant

# Create storage directory
mkdir -p ~/qdrant_storage

# Run container
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v ~/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

> Replace `~/qdrant_storage` with an absolute path (e.g. `/home/debian/qdrant_storage`).

Verify Qdrant is running and accessible:
```bash
curl http://<host>:6333/collections
```

```

### Qdrant-specific flags

**Connection**

| Flag | Default | Description |
|---|---|---|
| `--host` | `localhost` | Qdrant server hostname or IP address |
| `--port` | `6333` | REST port |

**Query**

| Flag | Default | Description |
|---|---|---|
| `--query-mode` | `hybrid` | `hybrid` — dense + sparse with RRF fusion · `sparse` — sparse-only |
| `--modifier` | `none` | Sparse scoring modifier: `none` or `idf` (BM25 IDF weighting) |

**Advanced** *(rarely need to change)*

| Flag | Default | Description |
|---|---|---|
| `--sparse-vector-name` | `sparse` | Sparse vector field name in the collection |
| `--dense-vector-name` | `dense` | Dense vector field name in the collection |
| `--on-disk-index` | `true` | Store sparse index on disk instead of RAM |
| `--segment-number` | `8` | Number of collection segments |

### Full run — hybrid mode (index + query + validation)

```bash
python main.py \
  --db qdrant \
  --index-name quora_bench \
  --dataset-name quora \
  --testcycle testcycle1 \
  --concurrency 16 \
  --async-concurrency 5 \
  --top-k 10 \
  --host 139.99.218.208 \
  --port 6333 \
  --query-mode hybrid \
  --modifier idf \
  --validation-env ~/validation-env
```

### Full run — sparse-only mode

```bash
python main.py \
  --db qdrant \
  --index-name quora_bench \
  --dataset-name quora \
  --testcycle testcycle1 \
  --concurrency 16 \
  --async-concurrency 5 \
  --top-k 10 \
  --host 139.99.218.208 \
  --port 6333 \
  --query-mode sparse \
  --modifier idf \
  --validation-env ~/validation-env
```

### Skip indexing (query + validate only)

```bash
python main.py \
  --db qdrant \
  --index-name quora_bench \
  --dataset-name quora \
  --testcycle testcycle1 \
  --concurrency 16 \
  --host 139.99.218.208 \
  --skip-indexing \
  --validation-env ~/validation-env
```

### Query only (no validation)

```bash
python main.py \
  --db qdrant \
  --index-name quora_bench \
  --dataset-name quora \
  --testcycle testcycle1 \
  --concurrency 16 \
  --host 139.99.218.208 \
  --skip-indexing \
  --skip-validation
```

---

## Typical end-to-end workflow

```
[index-env]
1. embedding_creation.py  →  data/<dataset>/*.npy

[index-env]
2. python main.py --db <endee|qdrant> ...   →  indexing + querying + validation in one command
   Output: dbs/<db>/test/<testcycle>/concurrency<N>/
             ├── indexing_performance.json
             ├── merged_results.json
             ├── summary.json
             └── detailed_logs.json
```


