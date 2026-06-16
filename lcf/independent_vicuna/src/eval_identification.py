"""Fallacy Identification eval (paper Table 1 right half): Accuracy + Delta Prob.

Each test item -> 4 options [valid, invalid_A, invalid_B, "I have no comment."].
We score each option by the LLM's length-normalized log-likelihood given the
premise, take softmax over the 4 -> per-option probability, and report:
  Accuracy   = argmax option is the valid one
  Delta Prob = mean( p(correct) - mean(p(incorrect)) )
both WITHOUT (baseline) and WITH (+LCF) the steering hooks active.

Usage: python src/eval_identification.py --ckpt _scratch/checkpoints/lcf_full.pt
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from model_utils import load_llm
from inference import load_lcf, attach, DEV


@torch.no_grad()
def option_logprob(model, tok, prefix, option):
    """Length-normalized log-likelihood of `option` continuing `prefix`."""
    pre = tok(prefix, return_tensors="pt").input_ids[0]
    opt = tok(option, add_special_tokens=False, return_tensors="pt").input_ids[0]
    ids = torch.cat([pre, opt]).unsqueeze(0).to(model.device)
    logits = model(ids).logits[0]                      # (T, V)
    logp = F.log_softmax(logits.float(), dim=-1)
    # predict opt tokens: positions [len(pre)-1 .. len(ids)-2] -> targets opt
    start = len(pre)
    tgt = ids[0, start:]
    lp = logp[start - 1: -1].gather(1, tgt.unsqueeze(1)).squeeze(1)
    return lp.mean().item()                            # per-token avg log-prob


def eval_split(model, tok, steerer, items, valid_map, eta):
    rng = torch.Generator().manual_seed(CFG.seed)
    out = {"baseline": {"acc": 0, "dprob": 0, "n": 0},
           "lcf": {"acc": 0, "dprob": 0, "n": 0}}
    for it in items:
        valid = valid_map.get(str(it["index"]))
        if not valid:
            continue
        opts = [valid, it["invalid_A"], it["invalid_B"], it["no_comment"]]
        labels = [1, 0, 0, 0]
        perm = torch.randperm(4, generator=rng).tolist()
        opts = [opts[i] for i in perm]
        labels = [labels[i] for i in perm]
        ans = labels.index(1)
        prefix = (f"Premise: {it['premise']}\n"
                  f"Select the logically valid conclusion.\nConclusion:")
        for name, active in (("baseline", False), ("lcf", True)):
            steerer.active = active
            scores = torch.tensor([option_logprob(model, tok, prefix, " " + o) for o in opts])
            probs = F.softmax(scores, dim=0)
            pred = int(scores.argmax())
            dprob = (probs[ans] - (probs.sum() - probs[ans]) / 3).item()
            out[name]["acc"] += int(pred == ans)
            out[name]["dprob"] += dprob
            out[name]["n"] += 1
    for name in out:
        n = max(1, out[name]["n"])
        out[name] = {"accuracy": out[name]["acc"] / n * 100,
                     "delta_prob": out[name]["dprob"] / n, "n": out[name]["n"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CFG.ckpt_dir / "lcf_full.pt"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--invalid", action="store_true",
                    help="reverse steering (paper's 'invalid modification')")
    ap.add_argument("--eta", type=float, default=CFG.eta_identification)
    args = ap.parse_args()
    if args.invalid:
        args.tag += "_inv"

    model, tok = load_llm(CFG)
    lcf, ck = load_lcf(args.ckpt)
    steerer = attach(model, lcf, ck["top_taps"], eta=args.eta,
                     sign=-1.0 if args.invalid else 1.0)
    print(f"eta={args.eta} taps={len(ck['top_taps'])}")

    items = [json.loads(l) for l in open(CFG.data_dir / "identification.jsonl")]
    if args.limit:
        items = items[: args.limit]
    valid_map = json.load(open(CFG.data_dir / "valid_conclusions.json"))

    res = eval_split(model, tok, steerer, items, valid_map, CFG.eta_identification)
    steerer.remove()
    print(json.dumps(res, indent=2))
    out = CFG.results_dir / f"identification_{args.tag}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
