import argparse
import logging
import subprocess
import sys
from pathlib import Path

from indexing import index_from_npy
from query import run_query
from utils import create_db, DB_REGISTRY, add_all_db_args, build_db_config

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
    parser.add_argument("--dataset-name",  choices=["scifact", "quora"], required=True,
                        help="Dataset name")
    parser.add_argument("--sparse-mode",   default="bm25", choices=["bm25", "splade"],
                        help="Sparse embedding type (default: bm25)")
    parser.add_argument("--data-dir",      default="data",
                        help="Root data directory (default: data)")
    parser.add_argument("--testcycle",     required=True,
                        help="Test cycle folder name (e.g. testcycle1)")
    parser.add_argument("--concurrency",   type=int, required=True,
                        help="Number of parallel worker processes")
    parser.add_argument("--async-concurrency", type=int, default=1,
                        help="Concurrent async queries per worker (default: 1)")
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
    
    parser.add_argument("--skip-indexing",   action="store_true", help="Skip the indexing step")
    parser.add_argument("--skip-query",      action="store_true", help="Skip the query step")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the validation step")
    parser.add_argument("--validation-env",  default=None,
                        help="Path to virtual environment to use for validation (e.g. validation-env)")

    add_all_db_args(parser)

    args = parser.parse_args()

    # Output paths — db-specific: dbs/{db}/test/{testcycle}/concurrency{N}
    db_output_base = Path("dbs") / args.db / "test"
    testcycle_path = db_output_base / args.testcycle
    output_dir     = testcycle_path / f"concurrency{args.concurrency}"

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
    logger.info("  Testcycle:           %s", args.testcycle)
    logger.info("  Concurrency:         %d", args.concurrency)
    logger.info("  Async concurrency:   %d", args.async_concurrency)
    logger.info("  Top-k:               %d", args.top_k)
    logger.info("  Output dir:          %s", output_dir)
    logger.info("  Skip indexing:       %s", args.skip_indexing)
    logger.info("  Skip query:          %s", args.skip_query)
    logger.info("  Skip validation:     %s", args.skip_validation)

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
        index_from_npy(
            db=db,
            index_name=args.index_name,
            data_dir=args.data_dir,
            dataset_name=args.dataset_name,
            testcycle=str(testcycle_path),
            sparse_mode=args.sparse_mode,
            batch_size=args.batch_size,
        )
        logger.info("--- Indexing Complete ---")
    else:
        logger.info("Skipping indexing")

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
            testcycle=args.testcycle,
            concurrency=args.concurrency,
            async_concurrency=args.async_concurrency,
            top_k=args.top_k,
        )
        logger.info("--- Query Complete ---")
    else:
        logger.info("Skipping query")

    # ── Validation ────────────────────────────────────────────────────────────
    if not args.skip_validation:
        logger.info("--- Starting Validation ---")
        python = str(get_validation_python(args.validation_env)) if args.validation_env else sys.executable
        cmd = [
            python, "endee_validation.py",
            "--dataset-name", args.dataset_name,
            "--output-base",  str(db_output_base),
            "--testcycle",    args.testcycle,
            "--concurrency",  str(args.concurrency),
            "--top-k",        str(args.top_k),
        ]
        logger.info("Running validation: %s", " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error("Validation failed with return code %d", result.returncode)
            raise SystemExit(result.returncode)

        logger.info("--- Validation Complete ---")
    else:
        logger.info("Skipping validation")


if __name__ == "__main__":
    main()
