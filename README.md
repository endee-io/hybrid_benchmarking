# Hybrid Vector Benchmark 

End-to-end hybrid vector benchmark framework. A single command runs **indexing → querying → validation** for any supported DB.

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

Same structure applies for `beir_scifact`. Use `src/embedding_creation.py` (index-env) to generate these files.

---

## Output structure

All results are written under `results/<db>/`:

```
results/
└── qdrant/
    ├── <index_name>.json          ← indexing performance
    └── <results>_concurrency<N>/
        ├── merged_results.json
        ├── all_latencies.json
        ├── summary.json
        └── detailed_logs.json
```

| File | Description |
|---|---|
| `<index_name>.json` | Upsert latency stats (min, p50, p95, p99, max, mean), total time, batch size |
| `merged_results.json` | `{ query_id: { doc_id: score } }` — input for validation |
| `all_latencies.json` | Per-query latency entries sorted ascending |
| `summary.json` | Aggregate stats: p50/p90/p95/p99/mean/min/max latency, total time, success/fail counts |
| `detailed_logs.json` | Per-query log with status, latency, worker, batch |

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
| `--sparse-mode` | No | `splade` | Sparse embedding type: `endee_bm25` (Endee-only), `bm25` (rank_bm25, any DB), or `splade` |
| `--data-dir` | No | `data` | Root directory containing dataset `.npy` files |
| `--top-k` | No | `10` | Number of top results to retrieve per query |
| `--batch-size` | No | `1000` | Upsert batch size during indexing |
| `--dimension` | No | `384` | Dense vector dimension; must match the embedding model output |
| `--space-type` | No | `cosine` | Distance metric: `cosine`, `dot`, `l2` |
| `--create-index` | No | `true` | Create the index before indexing; set `false` to skip if already exists |
| `--skip-indexing` | No | `false` | Skip the indexing step |
| `--skip-query` | No | `false` | Skip the query step |
| `--skip-validation` | No | `false` | Skip the validation step |
| `--validation-venv` | No | `validation-env` | Path to the validation virtual environment folder (e.g. `validation-env`) |

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
python -m src.main \
  --db endee \
  --index-name quora_bench \
  --dataset-name beir_quora \
  --results run1 \
  --concurrency 16 \
  --top-k 10 \
  --vector-token mytoken \
  --base-url https://dev.endee.io/api/v1 \
  --validation-venv validation-env
```

### Skip indexing (query + validate only)

```bash
python -m src.main \
  --db endee \
  --index-name quora_bench \
  --dataset-name beir_quora \
  --results run1 \
  --concurrency 16 \
  --vector-token mytoken \
  --base-url https://dev.endee.io/api/v1 \
  --skip-indexing \
  --validation-venv validation-env
```

### Query only (no index creation, no validation)

```bash
python -m src.main \
  --db endee \
  --index-name quora_bench \
  --dataset-name beir_quora \
  --results run1 \
  --concurrency 16 \
  --vector-token mytoken \
  --base-url https://dev.endee.io/api/v1 \
  --skip-indexing \
  --skip-validation
```

### Ingest into existing index (skip creation, run indexing)

```bash
python -m src.main \
  --db endee \
  --index-name quora_bench \
  --dataset-name beir_quora \
  --results run1 \
  --concurrency 16 \
  --vector-token mytoken \
  --base-url https://dev.endee.io/api/v1 \
  --create-index false \
  --skip-query \
  --skip-validation
```

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

