import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.endee_delete_insert import EndeeDeleteInsert
from endee import Endee

DEV_PATH = "https://dev.endee.io/api/v1"
DELETE_PERCENTAGES = [10, 20]
DELETE_PICKS       = ["random", "begin", "end"]
DELETE_ORDERS      = ["seq", "random"]
GT_FILTERS         = [None, "ground-truth", "non-ground-truth"]

INSERT_SCENARIOS = [
    (False, "same-as-delete"),
    (False, "random"),
    (True,  None),          
]

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _save(path: str, run_ts: str, results: list) -> None:
    """Incrementally write results so nothing is lost on failure."""
    with open(path, "w") as f:
        json.dump({"run_timestamp": run_ts, "results": results}, f, indent=2)


def _save_excel(json_path: str, results: list) -> str:
    """Flatten results into a pandas DataFrame and write to .xlsx alongside the JSON."""
    import pandas as pd

    rows = []
    for r in results:
        cfg = r.get("config", {}) or {}
        us  = r.get("upload_summary", {}) or {}
        dv  = r.get("delete_validation", {}) or {}
        iv  = r.get("insert_validation", {}) or {}
        ql  = r.get("query_latency", {}) or {}
        vm  = r.get("validation_metrics", {}) or {}

        rows.append({
            "timestamp":               r.get("timestamp"),
            "mode":                    cfg.get("mode"),
            "delete_percentage":       cfg.get("delete_percentage"),
            "delete_pick":             cfg.get("delete_pick"),
            "delete_order":            cfg.get("delete_order"),
            "gt_filter":               cfg.get("gt_filter"),
            "skip_insert":             cfg.get("skip_insert"),
            "insert_order":            cfg.get("insert_order"),
            "candidate_pool_size":     r.get("candidate_pool_size"),
            "num_deleted":             r.get("num_deleted"),
            "delete_confirmed":        dv.get("confirmed_deleted"),
            "delete_still_exists":     dv.get("still_exists"),
            "delete_validation_passed": r.get("delete_validation_passed"),
            "insert_confirmed":        iv.get("confirmed_present"),
            "insert_not_found":        iv.get("missing"),
            "insert_validation_passed": r.get("insert_validation_passed"),
            "recall@10":               vm.get("recall@10"),
            "ndcg@10":                 vm.get("ndcg@10"),
            "map@10":                  vm.get("map@10"),
        })

    df = pd.DataFrame(rows)
    xlsx_path = json_path.replace(".json", ".xlsx")
    if not xlsx_path.endswith(".xlsx"):
        xlsx_path += ".xlsx"
    df.to_excel(xlsx_path, index=False, sheet_name="Results")
    return xlsx_path


def _run_indexing(
    index_name: str,
    token: str,
    base_url: str,
    data_dir: str,
    dataset_name: str,
    sparse_mode: str,
    dimension: int,
    space_type: str,
    sparse_scoring_model: str,
    precision: str,
    batch_size: int,
    output_base: str,
    vx: Endee,
) -> Dict[str, Any]:
    """Delete old index then run endee_indexing.py via subprocess to create + upload."""
    try:
        vx.delete_index(index_name)
        print(f"  Deleted existing index '{index_name}'")
    except Exception:
        pass

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "endee_indexing.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"endee_indexing.py not found at: {script}")

    cmd = [
        sys.executable, script,
        "--index-name",           index_name,
        "--data-dir",             data_dir,
        "--dataset-name",         dataset_name,
        "--sparse-mode",          sparse_mode,
        "--vector-token",         token,
        "--base-url",             base_url,
        "--batch-size",           str(batch_size),
        "--create-index",         "true",
        "--dimension",            str(dimension),
        "--space-type",           space_type,
        "--sparse-scoring-model", sparse_scoring_model,
        "--precision",            precision,
        "--output-base",          output_base,
    ]
    print(f"  cmd: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"endee_indexing.py exited with code {proc.returncode}")

    return {"exit_code": proc.returncode}


def _reinsert_ids(
    edi: EndeeDeleteInsert,
    delete_ids: List[str],
    insert_order: str,
    data_dir: str,
    dataset_name: str,
    sparse_mode: str,
    batch_size: int,
    max_retries: int,
    seed: Optional[int],
    sparse_output_file: Optional[str] = None,
) -> dict:
    insert_ids = list(delete_ids)
    if insert_order == "random":
        if seed is not None:
            random.seed(seed + 1)
        random.shuffle(insert_ids)

    return edi.reinsert_vectors(
        ids_to_insert=insert_ids,
        data_dir=data_dir,
        dataset_name=dataset_name,
        sparse_mode=sparse_mode,
        batch_size=batch_size,
        max_retries=max_retries,
        sparse_output_file=sparse_output_file,
    )


def _run_query_benchmark(
    index_name: str,
    token: str,
    base_url: str,
    data_dir: str,
    dataset_name: str,
    sparse_mode: str,
    concurrency: int,
    top_k: int,
    combo_idx: int,
    output_base: str,
) -> Dict[str, Any]:
    """
    Run query_endee_async_multiprocessing.py via subprocess (index-env / current Python).
    Returns the output directory path and any parsed latency summary.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "query_endee_async_multiprocessing.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"query_endee_async_multiprocessing.py not found at: {script}")
    results    = f"results{combo_idx}"
    output_dir = Path(output_base) / f"{results}_concurrency{concurrency}"

    cmd = [
        sys.executable, script,
        "--data-dir",          data_dir,
        "--dataset-name",      dataset_name,
        "--sparse-mode",       sparse_mode,
        "--index-name",        index_name,
        "--vector-token",      token,
        "--base-url",          base_url,
        "--concurrency",       str(concurrency),
        "--results",           results,
        "--top-k",             str(top_k),
        "--output-base",       output_base,
    ]
    print(f"  cmd: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    merged_results_path = output_dir / "merged_results.json"
    summary_path        = output_dir / "summary.json"

    latency_summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            latency_summary = json.load(f)

    return {
        "merged_results_path": str(merged_results_path),
        "output_dir":          str(output_dir),
        "exit_code":           proc.returncode,
        "latency_summary":     latency_summary,
    }


def _run_validation(
    validation_python: str,
    validation_dataset: str,
    output_base: str,
    results: str,
    concurrency: int,
    cache_dir: Optional[str],
) -> Dict[str, Any]:
    """
    Run endee_validation.py using the validation-env Python executable.
    Returns parsed metrics dict from correctness.json.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "endee_validation.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"endee_validation.py not found at: {script}")

    cmd = [
        validation_python, script,
        "--dataset-name", validation_dataset,
        "--output-base",  output_base,
        "--results",      results,
        "--concurrency",  str(concurrency),
    ]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]

    print(f"  [validation-env] cmd: {' '.join(cmd)}", flush=True)

    captured = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        captured.append(line)
    proc.wait()

    # correctness.json is saved inside output_base/{results}_concurrency{n}/
    correctness_path = Path(output_base) / f"{results}_concurrency{concurrency}" / "correctness.json"
    if correctness_path.exists():
        with open(correctness_path) as f:
            metrics = json.load(f)
        return {"metrics": metrics, "exit_code": proc.returncode}

    # Fallback: parse stdout for metric lines like "  ndcg@10: 0.7234"
    metrics = {}
    for line in captured:
        m = re.search(r"(ndcg@\d+|map@\d+|recall@\d+):\s+([\d.]+)", line)
        if m:
            metrics[m.group(1)] = float(m.group(2))

    return {
        "metrics":   metrics,
        "exit_code": proc.returncode,
        "error":     "correctness.json not found; metrics parsed from stdout" if not metrics else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Automate delete/insert + query + validation for all combinations.\n"
            "Iterates over delete_percentage × delete_pick × delete_order × gt_filter × insert_scenario."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data source
    parser.add_argument("--data-dir",     required=False,default="data",
                        help="Root data directory (contains <dataset-name>/ subfolder)")
    parser.add_argument("--dataset-name", required=True,
                        help="Dataset name (e.g. beir_scifact, beir_quora)")
    parser.add_argument("--sparse-mode",  default="splade", choices=["endee_bm25", "bm25", "splade"],
                        help="Sparse embedding type: endee_bm25 (Endee-only), bm25 (any DB), splade (default: splade)")

    parser.add_argument("--index-name", required=True, help="Endee index name")
    parser.add_argument("--token",      default="12345678", help="Endee API token (default: 12345678)")
    parser.add_argument("--base-url",   default=DEV_PATH,
                        help=f"Endee base URL (default: {DEV_PATH})")

    # GT filter — auto-loaded from data/<dataset>/<dataset>_ground_truth_ids.npy if present

    # Index creation
    parser.add_argument("--dimension",            type=int, default=384,
                        help="Dense vector dimension (default: 384)")
    parser.add_argument("--space-type",           default="cosine",
                        help="Index space type: cosine | l2 | ip  (default: cosine)")
    parser.add_argument("--sparse-scoring-model", default="default",
                        help="Sparse scoring model passed to endee_indexing.py (default: default)")
    parser.add_argument("--precision",            default="float32",
                        help="Index precision: float32 | float16 | int8d | binary  (default: float32)")

    # Query benchmark
    parser.add_argument("--query-concurrency",        type=int, default=4,
                        help="Worker processes for query benchmark (default: 4)")
    parser.add_argument("--top-k",                    type=int, default=10,
                        help="Top-k results per query (default: 10)")
    parser.add_argument("--query-output-base",        default="delete_insert_query_results",
                        help="Base output dir for query results (default: delete_insert_query_results)")

    # Validation
    parser.add_argument("--validation-venv",      required=False,default="validation-env",
                        help="Path to the validation virtual environment folder (e.g. validation-env). "
                             "Script will use <validation-venv>/bin/python automatically.")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace cache dir passed to endee_validation.py (optional)")

    # Operational
    parser.add_argument("--keep-results", action="store_true", default=False,
                        help="Keep results output folders after each combination (default: delete to save disk space)")
    parser.add_argument("--batch-size",  type=int, default=1000, help="Upsert batch size (default: 1000)")
    parser.add_argument("--max-retries", type=int, default=5,    help="Upsert retry count (default: 5)")
    parser.add_argument(
        "--sparse-output-file",
        default=None,
        help="If set, save inserted sparse indices/values to this JSON file during reinsert (default: disabled)",
    )
    parser.add_argument("--seed",        type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--output-file", default="delete_insert_results.json",
                        help="Output JSON file (default: delete_insert_results.json)")
    parser.add_argument(
        "--gt-corpus-ids",
        action="store_true", default=False,
        help="Load ground-truth corpus IDs from HuggingFace dataset once at start instead of the gt npy file. "
             "Requires the dataset to be in DATASET_CONFIG in endee_delete_insert.py.",
    )

    args = parser.parse_args()

    validation_python = os.path.expanduser(
        os.path.join(args.validation_venv, "bin", "python")
    )


    if not os.path.isfile(validation_python):
        print(f"ERROR: validation Python not found at '{validation_python}'\n"
              f"  Make sure --validation-venv points to an existing virtualenv folder "
              f"(e.g. validation-env)", file=sys.stderr)
        sys.exit(1)
    print(f"Validation Python : {validation_python}")



    # ── Check required data files ────────────────────────────────────────────
    try:
        EndeeDeleteInsert.check_data_files(args.data_dir, args.dataset_name, args.sparse_mode)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Pre-load record IDs (once) ───────────────────────────────────────────
    sp_ids_path = (Path(args.data_dir) / args.dataset_name /
                   f"{args.dataset_name}_sparse_corpus_{args.sparse_mode}_ids.npy")
    print(f"Loading record IDs from '{sp_ids_path}' …")
    sp_ids_arr     = np.load(str(sp_ids_path), allow_pickle=True)
    all_record_ids = [str(sid) for sid in sp_ids_arr]
    print(f"Found {len(all_record_ids)} corpus records\n")

    # ── Load GT IDs once (only when --gt-corpus-ids flag is set) ─────────
    gt_ids: Optional[Set[str]] = None
    if args.gt_corpus_ids:
        print(f"Loading ground-truth corpus IDs from HuggingFace dataset '{args.dataset_name}' …")
        gt_ids = set(EndeeDeleteInsert.load_gt_corpus_ids_from_hf(args.dataset_name))
        print(f"Loaded {len(gt_ids)} ground-truth corpus IDs\n")
    else:
        print("--gt-corpus-ids not set → ground-truth/non-ground-truth combos will be skipped.\n")



    # ── Build combinations ───────────────────────────────────────────────────
    active_gt_filters = [f for f in GT_FILTERS if f is None or gt_ids is not None]

    combinations = [
        (pct, pick, order, gt_filter, skip_insert, insert_order)
        for pct         in DELETE_PERCENTAGES
        for pick        in DELETE_PICKS
        for order       in DELETE_ORDERS
        for gt_filter   in active_gt_filters
        for skip_insert, insert_order in INSERT_SCENARIOS
    ]



    vx = Endee(token=args.token)
    vx.set_base_url(args.base_url)

    run_ts  = _now_iso()
    results: List[Dict[str, Any]] = []

    print(f"{'='*72}")
    print(f"  Delete/Insert + Query + Validation Automation")
    print(f"  Combinations  : {len(combinations)}")
    print(f"  Index         : {args.index_name}")
    print(f"  Dataset       : {args.dataset_name}  sparse_mode={args.sparse_mode}")
    print(f"  GT filter     : {'enabled (' + str(len(gt_ids)) + ' HF GT corpus IDs)' if gt_ids else 'disabled'}")
    print(f"  Validation    : {args.dataset_name} via {validation_python}")
    print(f"  Output        : {os.path.abspath(args.output_file)}")
    print(f"{'='*72}\n")

    for combo_idx, (del_pct, del_pick, del_order, gt_filter, skip_insert, insert_order) in \
            enumerate(combinations, 1):

        insert_label = "skip" if skip_insert else f"insert({insert_order})"
        gt_label     = gt_filter if gt_filter else "no-filter"

        print(f"\n{'='*72}")
        print(f"  [{combo_idx}/{len(combinations)}]  "
              f"pct={del_pct}%  pick={del_pick}  order={del_order}  "
              f"gt={gt_label}  insert={insert_label}")
        print(f"{'='*72}")



        # ── 1. Create fresh index + upload ───────────────────────────────────
        print(f"\n[1/7] Create index + upload all {len(all_record_ids)} vectors …")
        try:
            upload_summary = _run_indexing(
                index_name=args.index_name,
                token=args.token,
                base_url=args.base_url,
                data_dir=args.data_dir,
                dataset_name=args.dataset_name,
                sparse_mode=args.sparse_mode,
                dimension=args.dimension,
                space_type=args.space_type,
                sparse_scoring_model=args.sparse_scoring_model,
                precision=args.precision,
                batch_size=args.batch_size,
                output_base=args.query_output_base,
                vx=vx,
            )
        except Exception as e:
            print(f"  ERROR during upload: {e}")
            results.append({
                "timestamp": _now_iso(),
                "config": _make_config(args, del_pct, del_pick, del_order, gt_label,
                                       skip_insert, insert_order),
                "error": f"Upload failed: {e}",
            })
            _save(args.output_file, run_ts, results)
            continue



        # ── 2. Fresh index handle + candidate pool ───────────────────────────
        edi = EndeeDeleteInsert(
            token=args.token,
            base_url=args.base_url,
            index_name=args.index_name,
        )

        if gt_filter and gt_ids:
            if gt_filter == "ground-truth":
                candidate_pool = edi.select_ground_truth_ids(all_record_ids, gt_ids)
            else:
                candidate_pool = edi.select_non_ground_truth_ids(all_record_ids, gt_ids)
        else:
            candidate_pool = all_record_ids



        # ── 3. Build deletion list ────────────────────────────────────────────
        ids_for_deletion = edi.build_id_list(
            candidate_pool=candidate_pool,
            percentage=del_pct,
            pick=del_pick,
            order=del_order,
            repeat=1,
            seed=args.seed,
        )



        # ── 4. Delete ────────────────────────────────────────────────────────
        print(f"\n[2/7] Delete  —  {len(ids_for_deletion)} vectors  "
              f"({del_pct}%  pick={del_pick}  order={del_order}  gt={gt_label})")
        delete_summary = edi.delete_vectors(ids_for_deletion)



        # ── 4b. Delete validation ─────────────────────────────────────────────
        print(f"\n[3/7] Delete validation …")
        delete_validation = edi.verify_deletion(ids_for_deletion)
        delete_validation_passed = delete_validation["still_exists"] == 0
        print(f"  delete_validation_passed={delete_validation_passed}  "
              f"(confirmed={delete_validation['confirmed_deleted']}  "
              f"still_exists={delete_validation['still_exists']})")



        # ── 5. Insert (unless skip_insert) ───────────────────────────────────
        insert_summary: Optional[Dict[str, Any]] = None
        insert_validation: Optional[Dict[str, Any]] = None
        insert_validation_passed: Optional[bool] = None
        if skip_insert:
            print(f"\n[4/7] Insert — SKIPPED")
            print(f"\n[5/7] Insert validation — SKIPPED")
        else:
            print(f"\n[4/7] Insert  —  {len(ids_for_deletion)} vectors  (order={insert_order})")
            try:
                insert_summary = _reinsert_ids(
                    edi=edi,
                    delete_ids=ids_for_deletion,
                    insert_order=insert_order,
                    data_dir=args.data_dir,
                    dataset_name=args.dataset_name,
                    sparse_mode=args.sparse_mode,
                    batch_size=args.batch_size,
                    max_retries=args.max_retries,
                    seed=args.seed,
                    sparse_output_file=args.sparse_output_file,
                )
            except Exception as e:
                print(f"  ERROR during insert: {e}")
                insert_summary = {"error": str(e)}

            # ── 5b. Insert validation ─────────────────────────────────────────
            print(f"\n[5/7] Insert validation …")
            insert_validation = edi.verify_insertion(ids_for_deletion)
            insert_validation_passed = insert_validation["missing"] == 0
            print(f"  insert_validation_passed={insert_validation_passed}  "
                  f"(confirmed={insert_validation['confirmed_present']}  "
                  f"missing={insert_validation['missing']})")



        # ── 6. Query benchmark ───────────────────────────────────────────────
        print(f"\n[6/7] Query benchmark …")
        query_result = _run_query_benchmark(
            index_name=args.index_name,
            token=args.token,
            base_url=args.base_url,
            data_dir=args.data_dir,
            dataset_name=args.dataset_name,
            sparse_mode=args.sparse_mode,
            concurrency=args.query_concurrency,
            top_k=args.top_k,
            combo_idx=combo_idx,
            output_base=args.query_output_base,
        )



        # ── 7. Validation (validation-env) ───────────────────────────────────
        validation_result: Dict[str, Any] = {}
        merged_path = query_result.get("merged_results_path", "")
        if os.path.isfile(merged_path):
            print(f"\n[7/7] Validation (validation-env) …")
            validation_result = _run_validation(
                validation_python=validation_python,
                validation_dataset=args.dataset_name,
                output_base=args.query_output_base,
                results=f"results{combo_idx}",
                concurrency=args.query_concurrency,
                cache_dir=args.cache_dir,
            )
        else:
            print(f"\n[7/7] Validation — SKIPPED (merged_results.json not found at {merged_path})")
            validation_result = {"error": f"merged_results.json missing: {merged_path}"}



        # ── Save ─────────────────────────────────────────────────────────────
        entry = {
            "timestamp":                _now_iso(),
            "config":                   _make_config(args, del_pct, del_pick, del_order, gt_label,
                                                     skip_insert, insert_order),
            "upload_summary":           upload_summary,
            "delete_summary":           delete_summary,
            "delete_validation":        delete_validation,
            "delete_validation_passed": delete_validation_passed,
            "insert_summary":           insert_summary,
            "insert_validation":        insert_validation,
            "insert_validation_passed": insert_validation_passed,
            "query_latency":            query_result.get("latency_summary", {}),
            "validation_metrics":       validation_result.get("metrics", {}),
            "num_deleted":              len(ids_for_deletion),
            "candidate_pool_size":      len(candidate_pool),
        }
        results.append(entry)
        _save(args.output_file, run_ts, results)
        metrics = validation_result.get("metrics", {})
        if metrics:
            print(f"  Metrics: " + "  ".join(f"{k}={round(v,4)}" for k, v in metrics.items()))

        if not args.keep_results:
            results_path = Path(args.query_output_base) / f"results{combo_idx}_concurrency{args.query_concurrency}"
            if results_path.exists():
                shutil.rmtree(results_path)
                print(f"  Deleted results folder: {results_path}")



    # ── Final summary table ──────────────────────────────────────────────────
    xlsx = _save_excel(args.output_file, results)
    print(f"\n{'='*72}")
    print(f"  Done — all {len(combinations)} combinations complete.")
    print(f"  Results JSON  : {os.path.abspath(args.output_file)}")
    print(f"  Results Excel : {os.path.abspath(xlsx)}")
    print(f"{'='*72}")
    print(f"\n  {'del%':>5}  {'pick':<8}  {'order':>6}  {'gt_filter':<20}  "
          f"{'insert':<22}  {'recall@10':>10}  {'ndcg@10':>8}")
    print(f"  {'-----':>5}  {'----':<8}  {'-----':>6}  {'--------':<20}  "
          f"{'------':<22}  {'--------':>10}  {'-------':>8}")
    for r in results:
        cfg = r["config"]
        m   = r.get("validation_metrics", {})
        ins_col     = "SKIPPED" if cfg["skip_insert"] else cfg.get("insert_order", "?")
        recall_val  = f"{m.get('recall@10', 'N/A')}" if m else "N/A"
        ndcg_val    = f"{m.get('ndcg@10',   'N/A')}" if m else "N/A"
        print(f"  {cfg['delete_percentage']:>4}%  {cfg['delete_pick']:<8}  "
              f"{cfg['delete_order']:>6}  {cfg['gt_filter']:<20}  "
              f"{ins_col:<22}  {recall_val:>10}  {ndcg_val:>8}")
    print()


def _make_config(args, del_pct, del_pick, del_order, gt_label, skip_insert, insert_order):
    return {
        "index_name":        args.index_name,
        "dataset_name":      args.dataset_name,
        "sparse_mode":       args.sparse_mode,
        "delete_percentage": del_pct,
        "delete_pick":       del_pick,
        "delete_order":      del_order,
        "gt_filter":         gt_label,
        "skip_insert":       skip_insert,
        "insert_order":      insert_order if not skip_insert else None,
        "space_type":        args.space_type,
        "precision":         args.precision,
    }


if __name__ == "__main__":
    main()
