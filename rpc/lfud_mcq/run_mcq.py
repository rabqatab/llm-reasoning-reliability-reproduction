"""Aggregate sampled MCQ answers with SC / PPL / RPC (reuses Paper A evaluators).

Equality is trivial integer match on option indices, so the RPC repo's
sc_evaluator / prep_evaluator (PPL) / wpc_evaluator (RPC) are reused unchanged;
only the equal/check funcs are swapped. CPU-only.

Usage: uv run python run_mcq.py --json mcq_Qwen_Qwen3-8B.json --K 16 --repeats 10
"""
from __future__ import annotations
import argparse, json, os, sys, random
import numpy as np

RPC_DIR = "/home/alphabridge/Study/reliableAI_final/rpc/RPC"
sys.path.insert(0, RPC_DIR)
import metrics                                  # noqa: E402
from compute_perp import prep_evaluator         # noqa: E402
from compute_sc import sc_evaluator             # noqa: E402
from compute_rpc import wpc_evaluator           # noqa: E402

EVALS = {"PPL": prep_evaluator, "SC": sc_evaluator, "RPC": wpc_evaluator}


def int_equal(ai, aj, ci, cj):
    return ai == aj and ai != -1          # unparseable (-1) never matches


def int_check(ans, gold):
    return ans == gold


def solve(data, evaluator, K, repeats=10):
    n = len(data["predict"])
    accs, eces = [], []
    for seed in range(repeats):
        random.seed(seed)
        outs = []
        for i in range(n):
            m = len(data["predict"][i])
            idx = list(range(m)); random.shuffle(idx); idx = idx[:K]
            preds = [data["predict"][i][j] for j in idx]
            comps = [data["completion"][i][j] for j in idx]
            perps = [data["mean_logprob"][i][j] for j in idx]
            outs.append(evaluator(preds, comps, perps, data["answer"][i], int_equal, int_check))
        maximum, _ = metrics.compute_maximum_metrics([o[1] for o in outs])
        accs.append(100.0 * np.mean([o[0] for o in outs]))
        eces.append(maximum[0] * 100.0)
    return np.array(accs), np.array(eces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--methods", default="SC,PPL,RPC")
    ap.add_argument("--out", default="results_mcq.txt")
    args = ap.parse_args()
    data = json.load(open(args.json))
    tag = os.path.basename(args.json).replace("mcq_", "").replace(".json", "")
    # parse-rate sanity
    flat = [p for q in data["predict"] for p in q]
    print(f"[mcq] {len(data['predict'])} questions, K={len(data['predict'][0])}, "
          f"parse-rate={100*np.mean([p!=-1 for p in flat]):.1f}%")
    with open(args.out, "a") as f:
        for mname in args.methods.split(","):
            accs, eces = solve(data, EVALS[mname], args.K, args.repeats)
            line = (f"{mname} LFUD-MCQ {tag} {args.K} "
                    f"{{'Accuracy': '{accs.mean():.2f} ± {accs.std():.2f}', "
                    f"'ECE': '{eces.mean():.2f} ± {eces.std():.2f}'}}")
            print(line); f.write(line + "\n")


if __name__ == "__main__":
    main()
