"""Fast fallacy-identification eval for a trained LCF (no generation).

Scores the 4 options of each fallacy_id_test item with the LCF hook OFF
(original) and ON (eta=4.5), reporting Acc + DeltaProb for both. Used for the
multi-model LCF sweep where the slow conclusion-generation eval is skipped and
the headline ΔProb metric is what matters.

Usage: uv run python fallacy_eval.py --model <hf id> --ckpt lcf/checkpoints/<short>
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import LCFInference  # noqa: E402

DATA = "/home/alphabridge/Study/reliableAI_final/lcf/data/fallacy_id_test.jsonl"
FAL_PROMPT = "{premise}\n"


def acc_delta(per_q_logprobs, answers, scale=100.0):
    import torch
    correct, deltas = 0, []
    for lps, ai in zip(per_q_logprobs, answers):
        p = torch.softmax(torch.tensor(lps), 0).numpy()
        if int(np.argmax(lps)) == ai:
            correct += 1
        inc = [p[j] for j in range(len(p)) if j != ai]
        deltas.append(p[ai] - float(np.mean(inc)))
    return 100.0 * correct / len(answers), scale * float(np.mean(deltas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default=None, help="LCF checkpoint dir (default checkpoints/<short>)")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--eta", type=float, default=4.5)
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.data) if l.strip()]
    wrap = LCFInference(args.model, ckpt_dir=args.ckpt)

    def run(enabled):
        wrap._enabled = enabled; wrap._eta = args.eta; wrap._sign = 1
        pq, ans = [], []
        for it in items:
            prompt = FAL_PROMPT.format(premise=it["premise"])
            pq.append([wrap._seq_logprob(prompt, o) for o in it["options"]])
            ans.append(it["answer_idx"])
        return acc_delta(pq, ans)

    short = args.model.split("/")[-1]
    print(f"== fallacy-id {short}  (n={len(items)}, eta={args.eta}) ==")
    a0, d0 = run(False); a1, d1 = run(True)
    print(f"{short},original,Acc={a0:.2f},DeltaProb={d0:.3f}")
    print(f"{short},+LCF,Acc={a1:.2f},DeltaProb={d1:.3f}")
    print(f"RESULT {short}: Acc {a0:.1f}->{a1:.1f}  DeltaProb {d0:.2f}->{d1:.2f}")


if __name__ == "__main__":
    main()
