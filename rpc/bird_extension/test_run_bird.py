"""Smoke test for run_bird.py: builds a tiny synthetic RPC-format dataset from
real BIRD gold SQL and runs SC/PPL/RPC end-to-end on CPU. Verifies the RPC
evaluator reuse + SQL equal-func injection work and produce sane accuracy
(a path equal to gold should be counted correct)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_bird import EVALUATORS, solve  # noqa: E402
from make_equal_funcs import SQLFuncFactory  # noqa: E402

BIRD = "/mnt/nfs/ssd2/bird_data/dev_20240627"


def main() -> int:
    dev = json.load(open(os.path.join(BIRD, "dev.json")))
    # Pick 3 distinct-db simple items whose gold runs fast.
    picks = []
    seen = set()
    for it in dev:
        if it["db_id"] in seen:
            continue
        if it["question_id"] in (518, 701):
            continue
        picks.append(it)
        seen.add(it["db_id"])
        if len(picks) == 3:
            break

    K = 4
    predict, completion, mean_logprob, answer, meta = [], [], [], [], []
    for it in picks:
        gold = it["SQL"]
        broken = gold.rstrip().rstrip(";") + " LIMIT 0"
        # 3 copies of gold (a correct majority) + 1 broken variant.
        preds = [gold, gold, gold, broken][:K]
        predict.append(preds)
        completion.append(["" for _ in preds])
        # gold paths slightly higher logprob than the broken one.
        mean_logprob.append([-0.2, -0.25, -0.3, -1.0][:K])
        answer.append(gold)
        meta.append({"question_id": it["question_id"], "db_id": it["db_id"]})

    results = {"predict": predict, "completion": completion,
               "mean_logprob": mean_logprob, "answer": answer}
    db_base = os.path.join(BIRD, "dev_databases")
    factory = SQLFuncFactory.from_meta(meta, db_base, timeout=30)

    ok = True
    for method in ("SC", "PPL", "RPC"):
        res = solve(results, factory, EVALUATORS[method], K=K, repeats=3)
        print(f"{method}: {res}")
        acc = float(res["Accuracy"].split("±")[0])
        # With a 3/4 gold majority, accuracy should be high (100% for SC/RPC
        # majority; PPL picks the highest-logprob gold path -> also correct).
        if acc < 99.0:
            print(f"  WARN {method} accuracy unexpectedly low: {acc}")
            ok = False

    print("SMOKE TEST PASSED" if ok else "SMOKE TEST: check warnings")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
