import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset as hf_load_dataset

from src.dataset_config import DATASET_CONFIG
from src.indexing import index_from_npy
from src.query import run_query
from src.utils import create_db, DB_REGISTRY, add_db_args, build_db_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_validation_python(validation_env: str) -> Path:
    """Check the validation venv exists and return its Python interpreter path."""
    venv_path   = Path(validation_env)
    python_path = venv_path / "bin" / "python"

    if not venv_path.exists():
        logger.error("Validation venv not found: '%s'", validation_env)
        raise SystemExit(1)

    if not python_path.exists():
        logger.error("Python interpreter not found in venv: '%s'", python_path)
        raise SystemExit(1)

    logger.info("Validation venv found: %s", validation_env)
    return python_path


def check_data_files(data_dir: str, dataset_name: str, sparse_mode: str,
                     need_corpus: bool, need_queries: bool):
    """Verify all required .npy files exist before starting the pipeline."""
    base = Path(data_dir) / dataset_name
    missing = []

    if need_corpus:
        corpus_files = [
            base / f"{dataset_name}_dense_corpus.npy",
            base / f"{dataset_name}_dense_corpus_ids.npy",
            base / f"{dataset_name}_sparse_corpus_{sparse_mode}_values.npy",
            base / f"{dataset_name}_sparse_corpus_{sparse_mode}_col_indices.npy",
            base / f"{dataset_name}_sparse_corpus_{sparse_mode}_indptr.npy",
            base / f"{dataset_name}_sparse_corpus_{sparse_mode}_ids.npy",
        ]
        missing += [str(f) for f in corpus_files if not f.exists()]

    if need_queries:
        query_files = [
            base / f"{dataset_name}_dense_queries.npy",
            base / f"{dataset_name}_dense_queries_ids.npy",
            base / f"{dataset_name}_sparse_queries_{sparse_mode}_values.npy",
            base / f"{dataset_name}_sparse_queries_{sparse_mode}_col_indices.npy",
            base / f"{dataset_name}_sparse_queries_{sparse_mode}_indptr.npy",
            base / f"{dataset_name}_sparse_queries_{sparse_mode}_ids.npy",
        ]
        missing += [str(f) for f in query_files if not f.exists()]

    if missing:
        logger.error("Missing data files in '%s':", base)
        for f in missing:
            logger.error("  - %s", f)
        raise SystemExit(1)

    logger.info("Data check passed for dataset '%s' (sparse_mode=%s)", dataset_name, sparse_mode)




def main():
    parser = argparse.ArgumentParser(description="Hybrid Vector Benchmark")

    parser.add_argument("--db",            choices=list(DB_REGISTRY), required=True,
                        help="Vector DB to benchmark")
    parser.add_argument("--index-name",    required=True,
                        help="Index name")
    parser.add_argument("--dataset-name",  required=True,
                        help="Dataset name (e.g. beir_scifact, beir_quora)")
    parser.add_argument(
        "--sparse-mode", default="splade",
        choices=["endee_bm25", "bm25", "splade", "pymilvus_bm25", "milvus_splade"],
        help=(
            "Sparse embedding type:\n"
            "  endee_bm25    – Endee BM25 (--db endee only)\n"
            "  bm25          – rank_bm25 (any DB; also use with --milvus-sparse-mode builtin_bm25)\n"
            "  splade        – prithivida/Splade_PP_en_v1 (default)\n"
            "  pymilvus_bm25 – PyMilvus BM25EmbeddingFunction (--db milvus)\n"
            "  milvus_splade – PyMilvus SpladeEmbeddingFunction (--db milvus)"
        ),
    )
    parser.add_argument("--data-dir",      default="data",
                        help="Root data directory (default: data)")
    parser.add_argument("--results",        required=True,
                        help="Results folder label (e.g. run1)")
    parser.add_argument("--concurrency",   type=int, required=True,
                        help="Number of parallel worker processes")
    parser.add_argument("--qps-duration",  type=int, default=30,
                        help="Duration in seconds for the QPS benchmark (default: 30)")
    parser.add_argument("--top-k",         type=int, default=10,
                        help="Top-k results per query (default: 10)")
    parser.add_argument("--batch-size",    type=int, default=1000,
                        help="Upsert batch size (default: 1000)")
    parser.add_argument("--dimension",     type=int, default=384,
                        help="Dense vector dimension (default: 384)")
    parser.add_argument("--space-type",    default="cosine",
                        help="Distance metric (default: cosine)")
    parser.add_argument("--create-index",  type=lambda x: x.lower() != "false", default=True,
                        help="Create index before indexing (default: true)")
    
    parser.add_argument("--skip-indexing",   action="store_true", help="Skip indexing and deploy entirely")
    parser.add_argument("--deploy-only",     action="store_true", help="Deploy new schema but skip feeding data (useful for updating rerank_count etc.)")
    parser.add_argument("--skip-query",      action="store_true", help="Skip the query step")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the validation step")
    parser.add_argument("--validation-venv",  default="validation-env",
                        help="Path to the validation virtual environment folder (e.g. validation-env)")
    parser.add_argument("--cache-dir",       default=None,
                        help="HuggingFace cache directory passed to endee_validation.py (default: HF default)")
    parser.add_argument("--hf-dataset-id",   default=None,
                        help="HuggingFace dataset ID (e.g. BeIR/scifact). Required for Vespa native BM25 text modes.")

    # Pre-parse --db so we only register that DB's flags (avoids cross-DB arg conflicts)
    pre, _ = parser.parse_known_args()
    add_db_args(parser, pre.db)

    args = parser.parse_args()

    if args.sparse_mode == "endee_bm25" and args.db != "endee":
        parser.error(f"--sparse-mode endee_bm25 is only supported with --db endee (got --db {args.db})")

    if args.sparse_mode == "milvus_splade" and args.db != "milvus":
        parser.error(f"--sparse-mode milvus_splade is only supported with --db milvus (got --db {args.db})")

    # Output paths — db-specific: results/{db}/{results}_concurrency{N}
    db_output_base = Path("results") / args.db
    output_dir     = db_output_base / f"{args.results}_concurrency{args.concurrency}"

    db_config = build_db_config(args.db, args)

    check_data_files(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        sparse_mode=args.sparse_mode,
        need_corpus=not args.skip_indexing,
        need_queries=not args.skip_query,
    )

    logger.info("=== Hybrid Vector Benchmark ===")
    logger.info("  DB:                  %s", args.db)
    logger.info("  Dataset:             %s", args.dataset_name)
    logger.info("  Index:               %s", args.index_name)
    logger.info("  Results label:       %s", args.results)
    logger.info("  Concurrency:         %d", args.concurrency)
    logger.info("  Top-k:               %d", args.top_k)
    logger.info("  Output dir:          %s", output_dir)
    logger.info("  Skip indexing:       %s", args.skip_indexing)
    logger.info("  Deploy only:         %s", args.deploy_only)
    logger.info("  Skip query:          %s", args.skip_query)
    logger.info("  Skip validation:     %s", args.skip_validation)

    # ── Load raw texts from HuggingFace (optional, for Vespa native BM25) ────
    corpus_texts = None
    query_texts = None
    if args.hf_dataset_id:
        from datasets import load_dataset
        logger.info("Loading corpus texts from HuggingFace: %s", args.hf_dataset_id)
        corpus_ds = load_dataset(args.hf_dataset_id, "corpus", cache_dir=args.cache_dir)["corpus"]
        corpus_texts = {
            str(row["_id"]): (row.get("title", "") + " " + row.get("text", "")).strip()
            for row in corpus_ds
        }
        logger.info("Loaded %d corpus texts", len(corpus_texts))
        logger.info("Loading query texts from HuggingFace: %s", args.hf_dataset_id)
        query_ds = load_dataset(args.hf_dataset_id, "queries", cache_dir=args.cache_dir)["queries"]
        query_texts = {str(row["_id"]): row.get("text", "") for row in query_ds}
        logger.info("Loaded %d query texts", len(query_texts))

    # ── Indexing ──────────────────────────────────────────────────────────────
    if not args.skip_indexing:
        logger.info("--- Starting Indexing ---")
        db = create_db(args.db, db_config)
        db.init(
            index_name=args.index_name,
            dimension=args.dimension,
            space_type=args.space_type,
            create=args.create_index,
        )
        if not args.deploy_only:
            index_from_npy(
                db=db,
                index_name=args.index_name,
                data_dir=args.data_dir,
                dataset_name=args.dataset_name,
                output_base=str(db_output_base),
                sparse_mode=args.sparse_mode,
                batch_size=args.batch_size,
                texts=corpus_texts,
            )
        logger.info("--- Indexing Complete ---")
    else:
        logger.info("Skipping indexing")

    # ── Qrel query IDs (for correctness filtering) ────────────────────────────
    qrel_query_ids = None
    if args.dataset_name in DATASET_CONFIG:
        cfg = DATASET_CONFIG[args.dataset_name]
        logger.info("Loading qrel query IDs from %s (split=%s)", cfg["hf_name"], cfg["split"])
        qrel_ds = hf_load_dataset(cfg["hf_name"], split=cfg["split"], cache_dir=args.cache_dir)
        qrel_query_ids = {str(entry["query-id"]) for entry in qrel_ds}
        logger.info("Loaded %d qrel query IDs", len(qrel_query_ids))
    else:
        logger.warning("Dataset %s not in DATASET_CONFIG — skipping qrel filtering", args.dataset_name)

    # ── Query ─────────────────────────────────────────────────────────────────
    if not args.skip_query:
        logger.info("--- Starting Query ---")
        run_query(
            db_name=args.db,
            db_config=db_config,
            index_name=args.index_name,
            data_dir=args.data_dir,
            dataset_name=args.dataset_name,
            sparse_mode=args.sparse_mode,
            output_dir=output_dir,
            results=args.results,
           concurrency=args.concurrency, 
            top_k=args.top_k,
            query_texts=query_texts,
            qps_duration=args.qps_duration,
            qrel_query_ids=qrel_query_ids,
        )
        logger.info("--- Query Complete ---")
    else:
        logger.info("Skipping query")

    # ── Validation ────────────────────────────────────────────────────────────
    if not args.skip_validation:
        logger.info("--- Starting Validation ---")
        python = str(get_validation_python(args.validation_venv)) if args.validation_venv else sys.executable
        cmd = [
            python, "-m", "src.endee_validation",
            "--dataset-name", args.dataset_name,
            "--output-base",  str(db_output_base),
            "--results",      args.results,
            "--concurrency",  str(args.concurrency),
            "--top-k",        str(args.top_k),
        ]
        if args.cache_dir:
            cmd += ["--cache-dir", args.cache_dir]
        logger.info("Running validation: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        if result.returncode != 0:
            logger.error("Validation failed with return code %d", result.returncode)
            raise SystemExit(result.returncode)

        logger.info("--- Validation Complete ---")
    else:
        logger.info("Skipping validation")

    print_summary(
        output_dir=output_dir,
        db_output_base=db_output_base,
        index_name=args.index_name,
        top_k=args.top_k,
        skip_indexing=args.skip_indexing,
        skip_query=args.skip_query,
        skip_validation=args.skip_validation,
    )


def print_summary(
    output_dir: Path,
    db_output_base: Path,
    index_name: str,
    top_k: int,
    skip_indexing: bool,
    skip_query: bool,
    skip_validation: bool,
):
    lines = ["", "=" * 56, "  BENCHMARK SUMMARY", "=" * 56]

    if not skip_indexing:
        idx_path = db_output_base / f"{index_name}.json"
        if idx_path.exists():
            idx = json.loads(idx_path.read_text())
            lines.append(f"  Indexing time     : {idx.get('total_time_sec', 'N/A')} s")
        else:
            lines.append("  Indexing time     : (file not found)")

    if not skip_query:
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            lines.append(f"  Total Queries     : {s.get('total_queries')}")
            lines.append(f"  Successful Queries: {s.get('successful_queries')}")
            lines.append(f"  p99 latency       : {s.get('p99_latency_ms', 'N/A')} ms")
            lines.append(f"  QPS               : {s.get('qps', 'N/A')}")
        else:
            lines.append("  Query stats       : (file not found)")

    if not skip_validation:
        correctness_path = output_dir / "correctness.json"
        if correctness_path.exists():
            m = json.loads(correctness_path.read_text())
            lines.append(f"  ndcg@{top_k:<3}           : {round(m.get(f'ndcg@{top_k}', 0), 4)}")
            lines.append(f"  map@{top_k:<4}           : {round(m.get(f'map@{top_k}', 0), 4)}")
            lines.append(f"  recall@{top_k:<1}           : {round(m.get(f'recall@{top_k}', 0), 4)}")
        else:
            lines.append("  Validation stats  : (file not found)")

    lines.append("=" * 56)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
