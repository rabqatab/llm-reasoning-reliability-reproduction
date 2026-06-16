"""Aggregate sampled KCC relevance answers with SC / PPL / RPC (reuses Paper A evaluators).

Equality is trivial integer match on the 0-3 graded-relevance label, so the RPC
repo's sc_evaluator / prep_evaluator (PPL) / wpc_evaluator (RPC) are reused
UNCHANGED; only the equal/check funcs are swapped — the run_mcq.py pattern. CPU-only.

KCC is a 4-class graded-relevance task (labels 0-3). On the class-balanced subset
we report:
  * Accuracy           — plain top-1 (the evaluator's own `correct`)
  * BalancedAccuracy   — mean of per-class accuracy over all 4 grades
  * ECE                — calibration of the selected vote (RPC repo metric)
Chance / majority-class baseline on the balanced subset is ~25%.

Usage (eval, CPU, RPC repo venv):
  cd /home/alphabridge/Study/reliableAI_final/rpc/RPC
  uv run python /home/alphabridge/Study/reliableAI_final/rpc/kcc_ext/run_kcc.py \
      --json /home/alphabridge/Study/reliableAI_final/rpc/kcc_ext/kcc_Qwen_Qwen3-8B.json \
      --K 8 --repeats 10
"""
from __future__ import annotations
import argparse, json, os, sys, random
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
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


def chosen_label(answers):
    """The selected prediction = the (max-prob) representative the evaluator scored.

    `answers` is the evaluator's returned list of [pred, prob, flag]; the entry
    with the largest prob is the one that contributes to `correct` (ties → first).
    Returns its predicted 0/1 label (or -1 if unparseable / empty)."""
    if not answers:
        return -1
    best = max(answers, key=lambda x: (x[1] if x[1] == x[1] else -1.0))  # x[1]==x[1] guards NaN
    return best[0]


def solve(data, evaluator, K, repeats=10):
    n = len(data["predict"])
    golds = [data["answer"][i] for i in range(n)]
    accs, baccs, eces = [], [], []
    for seed in range(repeats):
        random.seed(seed)
        outs = []
        chosen = []
        for i in range(n):
            m = len(data["predict"][i])
            idx = list(range(m)); random.shuffle(idx); idx = idx[:K]
            preds = [data["predict"][i][j] for j in idx]
            comps = [data["completion"][i][j] for j in idx]
            perps = [data["mean_logprob"][i][j] for j in idx]
            out = evaluator(preds, comps, perps, golds[i], int_equal, int_check)
            outs.append(out)
            chosen.append(chosen_label(out[1]))
        # plain accuracy (evaluator's own fractional correct)
        accs.append(100.0 * np.mean([o[0] for o in outs]))
        # balanced accuracy: mean of per-class top-1 accuracy over ALL gold classes
        # present (KCC is 4-class graded relevance 0-3; chance = 25%).
        per_class = []
        for cls in sorted(set(golds)):
            idxs = [i for i in range(n) if golds[i] == cls]
            if idxs:
                per_class.append(np.mean([1.0 if chosen[i] == cls else 0.0 for i in idxs]))
        baccs.append(100.0 * np.mean(per_class) if per_class else 0.0)
        # ECE from the RPC repo metric
        maximum, _ = metrics.compute_maximum_metrics([o[1] for o in outs])
        eces.append(maximum[0] * 100.0)
    return np.array(accs), np.array(baccs), np.array(eces)


def _selftest():
    """Tiny synthetic RPC-format dict — verifies the pipeline runs on CPU.

    4 pairs, K=4. Golds: [1,0,1,0]. Most paths agree with gold (with noise / a
    couple -1 unparseables). SC/PPL/RPC plain & balanced acc should be high."""
    data = {
        "predict": [
            [1, 1, 1, 0],     # gold 1
            [0, 0, 1, 0],     # gold 0
            [1, 1, -1, 1],    # gold 1 (one unparseable)
            [0, 0, 0, 1],     # gold 0
        ],
        "completion": [
            ["...Answer: 1"] * 4,
            ["...Answer: 0"] * 4,
            ["...Answer: 1"] * 4,
            ["...Answer: 0"] * 4,
        ],
        "mean_logprob": [
            [-0.3, -0.4, -0.5, -1.2],
            [-0.3, -0.35, -1.0, -0.4],
            [-0.3, -0.4, -2.0, -0.6],
            [-0.3, -0.4, -0.5, -1.5],
        ],
        "answer": [1, 0, 1, 0],
    }
    print("[selftest] synthetic 4-pair dict, K=4, golds=[1,0,1,0]")
    for mname, ev in EVALS.items():
        accs, baccs, eces = solve(data, ev, K=4, repeats=3)
        print(f"  {mname:4s} Acc={accs.mean():.2f}±{accs.std():.2f}  "
              f"BalAcc={baccs.mean():.2f}±{baccs.std():.2f}  "
              f"ECE={eces.mean():.2f}±{eces.std():.2f}")
    assert int_check(1, 1) and not int_check(1, 0)
    assert not int_equal(-1, -1, "", "")   # unparseable never matches
    assert int_equal(1, 1, "", "")
    print("[selftest] equality assertions passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="kcc_<model>.json (omit to run --selftest)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--methods", default="SC,PPL,RPC")
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "results_kcc.txt"))
    args = ap.parse_args()

    if args.selftest or not args.json:
        _selftest()
        return

    data = json.load(open(args.json))
    tag = os.path.basename(args.json).replace("kcc_", "").replace(".json", "")
    n = len(data["predict"])
    golds = data["answer"]
    from collections import Counter
    dist = Counter(golds)
    n_cls = len(dist)
    flat = [p for q in data["predict"] for p in q]
    parse_rate = 100 * np.mean([p != -1 for p in flat])
    maj = 100.0 * max(dist.values()) / n
    print(f"[kcc] {n} pairs, class dist {dict(sorted(dist.items()))} ({n_cls}-class graded relevance), "
          f"K={len(data['predict'][0])}, parse-rate={parse_rate:.1f}%")
    print(f"[kcc] majority-class baseline ~{maj:.1f}% plain acc (= {100.0/n_cls:.1f}% balanced acc / chance)")
    with open(args.out, "a") as f:
        for mname in args.methods.split(","):
            accs, baccs, eces = solve(data, EVALS[mname], args.K, args.repeats)
            line = (f"{mname} KCC-relevance {tag} {args.K} "
                    f"{{'Accuracy': '{accs.mean():.2f} ± {accs.std():.2f}', "
                    f"'BalancedAccuracy': '{baccs.mean():.2f} ± {baccs.std():.2f}', "
                    f"'ECE': '{eces.mean():.2f} ± {eces.std():.2f}'}}")
            print(line); f.write(line + "\n")


if __name__ == "__main__":
    main()
