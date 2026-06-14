"""Run SC / PPL / RPC confidence aggregation on a generated BIRD JSON.

This reuses the RPC evaluator *logic* (compute_perp.prep_evaluator,
compute_sc.sc_evaluator, compute_rpc.wpc_evaluator) unchanged, but injects
SQLite execution-match ``equal_func`` / ``check_equal`` closures instead of the
sympy math comparators. Because exec-match needs a per-problem database, the
RPC ``Evaluator.process`` / ``worker`` (which hardcode ``numberic_compare`` and
the math ``check_equal``) cannot be used directly; we replicate the small
solve loop here and install the right SQL closure for each problem index.

Output line appended to results_bird.txt:
    {method} BIRD {model} {K} {'Accuracy': ..., 'ECE': ...}

The math equality cache is disabled (cache_file=None) since exec-match has its
own per-call semantics.

Example:
    python run_bird.py --model Qwen/Qwen3-8B --K 16 --repeats 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

# Import the RPC package (read-only reuse). Add RPC dir to sys.path so its
# intra-package imports (metrics, eval, data_processing) resolve.
RPC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "RPC")
)
sys.path.insert(0, RPC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics  # noqa: E402  (from RPC/)
from compute_perp import prep_evaluator  # noqa: E402
from compute_sc import sc_evaluator  # noqa: E402
from compute_rpc import wpc_evaluator  # noqa: E402
from make_equal_funcs import SQLFuncFactory  # noqa: E402

BIRD_ROOT_DEFAULT = "/mnt/nfs/ssd2/bird_data/dev_20240627"

EVALUATORS = {
    "PPL": prep_evaluator,
    "SC": sc_evaluator,
    "RPC": wpc_evaluator,
}


def run_once(results: dict, factory: SQLFuncFactory, evaluator, K: int, seed: int):
    """Replicates Evaluator.process for one seed, but installs the per-problem
    SQL equal_func/check_equal closures. Returns (accuracy%, maximum_metrics)."""
    n = len(results["predict"])
    m = len(results["predict"][0])
    indices = list(range(m))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[:K]

    outputs = []
    for idx in range(n):
        preds = [results["predict"][idx][j] for j in indices]
        comps = [results["completion"][idx][j] for j in indices]
        perps = [results["mean_logprob"][idx][j] for j in indices]
        gold = results["answer"][idx]
        equal_func, check_equal = factory.funcs(idx)
        res = evaluator(preds, comps, perps, gold, equal_func, check_equal)
        outputs.append(res)

    maximum, _ = metrics.compute_maximum_metrics([x[1] for x in outputs])
    acc = np.mean([x[0] for x in outputs]) * 100.0
    return acc, maximum


def solve(results: dict, factory: SQLFuncFactory, evaluator, K: int, repeats: int):
    """Aggregate over ``repeats`` seeds, mirroring Evaluator.solve output."""
    accs, maxs = [], []
    for seed in range(repeats):
        acc, maximum = run_once(results, factory, evaluator, K, seed)
        accs.append(acc)
        maxs.append(maximum)
    accs = np.array(accs)
    maxs = np.array([m[0] for m in maxs])  # ECE is maximum[0]
    return {
        "Accuracy": f"{accs.mean():.2f} ± {accs.std():.2f}",
        "ECE": f"{maxs.mean() * 100.0:.2f} ± {maxs.std() * 100.0:.2f}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run SC/PPL/RPC on a generated BIRD JSON using SQLite "
        "execution-match equality (CPU-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                    help="Model tag used in the generated filename.")
    ap.add_argument("--json", type=str, default=None,
                    help="Path to bird_<model>.json (overrides --model/--in_dir).")
    ap.add_argument("--meta", type=str, default=None,
                    help="Path to bird_<model>.meta.json (defaults next to --json).")
    ap.add_argument("--in_dir", type=str, default=".",
                    help="Directory holding the generated files.")
    ap.add_argument("--K", type=int, default=16, help="Paths per problem to use.")
    ap.add_argument("--repeats", type=int, default=10, help="Random seeds to average.")
    ap.add_argument("--methods", type=str, default="SC,PPL,RPC",
                    help="Comma-separated subset of SC,PPL,RPC.")
    ap.add_argument("--bird_root", type=str, default=BIRD_ROOT_DEFAULT)
    ap.add_argument("--timeout", type=int, default=30,
                    help="Per-query SQLite execution timeout (seconds).")
    ap.add_argument("--out", type=str, default="results_bird.txt")
    args = ap.parse_args()

    model_tag = args.model.replace("/", "_")
    json_path = args.json or os.path.join(args.in_dir, f"bird_{model_tag}.json")
    meta_path = args.meta or json_path.replace(".json", ".meta.json")

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert len(meta) == len(results["predict"]), \
        "meta length must match number of problems"

    db_base = os.path.join(args.bird_root, "dev_databases")
    factory = SQLFuncFactory.from_meta(meta, db_base, timeout=args.timeout)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        if method not in EVALUATORS:
            print(f"Skipping unknown method {method}")
            continue
        res = solve(results, factory, EVALUATORS[method], args.K, args.repeats)
        line = f"{method} BIRD {args.model} {args.K} {res}"
        with open(args.out, "a") as f:
            f.write(line + "\n")
        print(line)


if __name__ == "__main__":
    main()
