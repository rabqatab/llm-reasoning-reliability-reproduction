"""Aggregate sampled JurisNet answers with SC / PPL / RPC (reuses Paper A evaluators).

Equality is canonical (law, article)-set match (normalize.answer_match), so the
RPC repo's sc_evaluator / prep_evaluator (PPL) / wpc_evaluator (RPC) are reused
UNCHANGED; only the equal/check funcs are swapped — exactly as run_mcq.py does
for integer MCQ equality. CPU-only.

Usage (eval, CPU, RPC repo venv):
  cd /home/alphabridge/Study/reliableAI_final/rpc/RPC
  uv run python /home/alphabridge/Study/reliableAI_final/rpc/jurisnet_ext/run_jurisnet.py \
      --json /home/alphabridge/Study/reliableAI_final/rpc/jurisnet_ext/jurisnet_Qwen_Qwen3-8B.json \
      --K 8 --repeats 10
"""
from __future__ import annotations
import argparse, json, os, sys, random
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RPC_DIR = "/home/alphabridge/Study/reliableAI_final/rpc/RPC"
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, RPC_DIR)

from normalize import answer_match                # noqa: E402
import metrics                                     # noqa: E402
from compute_perp import prep_evaluator            # noqa: E402
from compute_sc import sc_evaluator                # noqa: E402
from compute_rpc import wpc_evaluator              # noqa: E402

EVALS = {"PPL": prep_evaluator, "SC": sc_evaluator, "RPC": wpc_evaluator}


def juris_equal(ai, aj, ci, cj):
    # empty prediction (no parsed pair) never matches another empty one
    if not ai or not aj:
        return False
    return answer_match(ai, aj)


def juris_check(ans, gold):
    return answer_match(ans, gold)


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
            outs.append(
                evaluator(preds, comps, perps, data["answer"][i], juris_equal, juris_check)
            )
        maximum, _ = metrics.compute_maximum_metrics([o[1] for o in outs])
        accs.append(100.0 * np.mean([o[0] for o in outs]))
        eces.append(maximum[0] * 100.0)
    return np.array(accs), np.array(eces)


def _selftest():
    """Tiny synthetic RPC-format dict — verifies the pipeline runs on CPU.

    2 cases, K=4. Case 0: gold "민법/제103조" reached by 3/4 paths (one wrong).
    Case 1: gold "형법/제250조의2" reached by 2/4. SC/PPL/RPC should all be high.
    """
    data = {
        "predict": [
            ["민법/제103조", "민법/제103조", "민법/제103조", "상법/제5조"],
            ["형법/제250조의2", "형법/제250조의2", "형법/제251조", ""],
        ],
        "completion": [
            ["민법 제103조", "민법 제103조", "민법 제103조", "상법 제5조"],
            ["형법 제250조의2", "형법 제250조의2", "형법 제251조", "모름"],
        ],
        "mean_logprob": [
            [-0.5, -0.4, -0.6, -1.5],
            [-0.3, -0.35, -0.9, -2.0],
        ],
        "answer": ["민법/제103조", "형법/제250조의2"],
    }
    print("[selftest] synthetic 2-case dict, K=4")
    for mname, ev in EVALS.items():
        accs, eces = solve(data, ev, K=4, repeats=3)
        print(f"  {mname:4s} Acc={accs.mean():.2f}±{accs.std():.2f}  "
              f"ECE={eces.mean():.2f}±{eces.std():.2f}")
    # sanity on the equality function itself
    assert juris_check("민법/제103조", "민법 제103조")
    assert not juris_check("민법/제103조", "민법/제104조")
    assert not juris_equal("", "", "", "")
    print("[selftest] equality assertions passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="jurisnet_<model>.json (omit to run --selftest)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--methods", default="SC,PPL,RPC")
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "results_jurisnet.txt"))
    args = ap.parse_args()

    if args.selftest or not args.json:
        _selftest()
        return

    data = json.load(open(args.json))
    tag = os.path.basename(args.json).replace("jurisnet_", "").replace(".json", "")
    flat = [p for q in data["predict"] for p in q]
    parse_rate = 100 * np.mean([bool(p) for p in flat])
    print(f"[juris] {len(data['predict'])} cases, K={len(data['predict'][0])}, "
          f"non-empty-parse-rate={parse_rate:.1f}%")
    with open(args.out, "a") as f:
        for mname in args.methods.split(","):
            accs, eces = solve(data, EVALS[mname], args.K, args.repeats)
            line = (f"{mname} JurisNet-ko_ver {tag} {args.K} "
                    f"{{'Accuracy': '{accs.mean():.2f} ± {accs.std():.2f}', "
                    f"'ECE': '{eces.mean():.2f} ± {eces.std():.2f}'}}")
            print(line); f.write(line + "\n")


if __name__ == "__main__":
    main()
