import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset as hf_load_dataset
from ranx import Qrels, Run, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_CONFIG = {
    "5k":   {"hf_name": "BeIR/scifact-qrels", "split": "train", "out_dir": "scifact"},
    "500k": {"hf_name": "BeIR/quora-qrels",   "split": "test",  "out_dir": "quora"},
}

METRICS = ["ndcg@10", "map@10", "recall@10"]


def load_dataset(dataset_size: str, cache_dir: str = None):
    cfg = DATASET_CONFIG[dataset_size]
    logger.info("Loading %s split=%s", cfg["hf_name"], cfg["split"])
    return hf_load_dataset(cfg["hf_name"], split=cfg["split"], cache_dir=cache_dir)


def load_query_qrels(dataset, name: str = "qrels") -> Qrels:
    qrels_dict = defaultdict(dict)
    for entry in dataset:
        query_id = str(entry["query-id"])
        doc_id   = str(entry["corpus-id"])
        qrels_dict[query_id][doc_id] = entry["score"]
    logger.info("Built Qrels with %d queries", len(qrels_dict))
    return Qrels(qrels_dict, name=name)


def calculate_metrics(qrels: Qrels, run_file: str, testcycle_dir: str):
    run_file_path = Path(run_file)
    with open(run_file_path, "r") as f:
        run_dict = json.load(f)

    run = Run(run_dict, name=run_file_path.stem)
    results = evaluate(qrels, run, METRICS, make_comparable=True)

    out_dir = Path(testcycle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "correctness.json"

    serializable = {k: float(v) for k, v in results.items()}
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Metrics saved to %s", out_file)

    for metric, value in serializable.items():
        print(f"  {metric}: {round(value, 4)}")

    return serializable


def save_unique_doc_ids(dataset, dataset_size: str):
    cfg = DATASET_CONFIG[dataset_size]
    unique_ids = sorted(set(str(entry["corpus-id"]) for entry in dataset))

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "unique_doc_ids.txt"

    with open(out_file, "w") as f:
        for doc_id in unique_ids:
            f.write(f"{doc_id}\n")
    logger.info("Wrote %d unique doc IDs to %s", len(unique_ids), out_file)


def main():
    parser = argparse.ArgumentParser(description="Validate Endee search results against ground-truth qrels.")
    parser.add_argument("--dataset", choices=["5k", "500k"], required=True,
                        help="Dataset size: '5k' (scifact) or '500k' (quora)")
    parser.add_argument("--run-file", required=True,
                        help="Path to merged_results.json produced by the benchmark")
    parser.add_argument("--testcycle", required=True,
                        help="Output directory for correctness.json (e.g. testcycle152/concurrency16)")
    parser.add_argument("--need-unique-doc-ids", type=lambda x: x.lower() != "false", default=False,
                        help="Save unique ground-truth doc IDs to file (default: false)")
    parser.add_argument("--validation", type=lambda x: x.lower() != "false", default=True,
                        help="Run validation (load dataset, build qrels, calculate metrics) (default: true)")
    parser.add_argument("--cache-dir", default=None,
                        help="Local cache directory for HuggingFace datasets (default: HF default cache)")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)

    if args.need_unique_doc_ids:
        save_unique_doc_ids(dataset, args.dataset)

    if args.validation:
        qrels = load_query_qrels(dataset)
        calculate_metrics(qrels, args.run_file, args.testcycle)


if __name__ == "__main__":
    main()
