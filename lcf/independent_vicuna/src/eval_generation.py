"""Conclusion Generation eval (paper Table 1 left half): Valid% + Perplexity.

Generates a conclusion for each test premise, baseline vs +LCF, then:
  * Perplexity of the generated text under the base model (fluency check)
  * Valid% via an LLM judge (the paper's "GPT-4 discriminator"); optional.

Usage:
  python src/eval_generation.py --ckpt _scratch/checkpoints/lcf_full.pt --judge anthropic
  python src/eval_generation.py --ckpt ... --judge none   # just save generations
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG
from model_utils import load_llm
from inference import load_lcf, attach

GEN_PROMPT = ("Given the premise, write ONE short logically valid conclusion that "
              "follows from it. Do not overclaim.\nPremise: {premise}\nConclusion:")

JUDGE_PROMPT = ("Premise: {premise}\nConclusion: {conclusion}\n\n"
                "Does the conclusion follow with logical validity from ONLY the premise, "
                "without overclaiming or introducing unsupported certainty? "
                "Answer with a single word: valid or invalid.")


@torch.no_grad()
def perplexity(model, tok, premise, text):
    if not text.strip():
        return float("nan")
    pre = tok(GEN_PROMPT.format(premise=premise), return_tensors="pt").input_ids[0]
    con = tok(" " + text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    ids = torch.cat([pre, con]).unsqueeze(0).to(model.device)
    logits = model(ids).logits[0]
    logp = F.log_softmax(logits.float(), dim=-1)
    start = len(pre)
    tgt = ids[0, start:]
    lp = logp[start - 1: -1].gather(1, tgt.unsqueeze(1)).squeeze(1)
    return math.exp(-lp.mean().item())


@torch.no_grad()
def generate(model, tok, premise, max_new=60):
    msg = [{"role": "user", "content": GEN_PROMPT.format(premise=premise)}]
    try:
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        ids = tok(GEN_PROMPT.format(premise=premise), return_tensors="pt").input_ids
    ids = ids.to(model.device)
    g = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]


def make_local_judge(model, tok):
    """Self-contained validity judge using the base model's own preference for
    'valid' vs 'invalid' (weaker substitute for the paper's GPT-4 discriminator).
    Caller must ensure any steering hooks are inactive so the BASE model judges.
    """
    from eval_identification import option_logprob

    def judge(prem, con):
        prompt = JUDGE_PROMPT.format(premise=prem, conclusion=con)
        sv = option_logprob(model, tok, prompt, " valid")
        si = option_logprob(model, tok, prompt, " invalid")
        return sv > si
    return judge


def make_judge(backend, model_name):
    if backend == "openai":
        from openai import OpenAI
        cli = OpenAI()
        def judge(prem, con):
            r = cli.chat.completions.create(model=model_name or "gpt-4o",
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(premise=prem, conclusion=con)}],
                temperature=0, max_tokens=4)
            # NOTE: "invalid".startswith("valid") is False -> correct; "valid in text" is NOT (substring bug)
            return r.choices[0].message.content.strip().lower().startswith("valid")
        return judge
    if backend == "anthropic":
        import anthropic
        cli = anthropic.Anthropic()
        def judge(prem, con):
            r = cli.messages.create(model=model_name or "claude-sonnet-4-6", max_tokens=4,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(premise=prem, conclusion=con)}])
            return r.content[0].text.strip().lower().startswith("valid")  # not substring (invalid contains valid)
        return judge
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CFG.ckpt_dir / "lcf_full.pt"))
    ap.add_argument("--judge", choices=["none", "openai", "anthropic"], default="none")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--invalid", action="store_true",
                    help="reverse steering (paper's 'invalid modification')")
    ap.add_argument("--eta", type=float, default=CFG.eta_generation)
    args = ap.parse_args()
    if args.invalid:
        args.tag += "_inv"

    model, tok = load_llm(CFG)
    lcf, ck = load_lcf(args.ckpt)
    steerer = attach(model, lcf, ck["top_taps"], eta=args.eta,
                     sign=-1.0 if args.invalid else 1.0)
    print(f"eta={args.eta} taps={len(ck['top_taps'])}")
    judge = make_judge(args.judge, args.judge_model)

    items = [json.loads(l) for l in open(CFG.data_dir / "pairs_test.jsonl")]
    if args.limit:
        items = items[: args.limit]

    rows, agg = [], {"baseline": {"valid": 0, "ppl": [], "nj": 0},
                     "lcf": {"valid": 0, "ppl": [], "nj": 0}}
    for it in items:
        prem = it["premise"]
        row = {"index": it["index"], "premise": prem, "fallacy_type": it["fallacy_type"]}
        for name, active in (("baseline", False), ("lcf", True)):
            steerer.active = active
            gen = generate(model, tok, prem)
            steerer.active = False
            ppl = perplexity(model, tok, prem, gen)
            row[name] = gen
            row[name + "_ppl"] = ppl
            if not math.isnan(ppl):
                agg[name]["ppl"].append(ppl)
            if judge:
                ok = judge(prem, gen)
                row[name + "_valid"] = ok
                agg[name]["valid"] += int(ok)
                agg[name]["nj"] += 1
        rows.append(row)

    steerer.remove()
    summary = {}
    for name in ("baseline", "lcf"):
        ppls = agg[name]["ppl"]
        summary[name] = {
            "perplexity": sum(ppls) / len(ppls) if ppls else None,
            "valid_pct": (agg[name]["valid"] / agg[name]["nj"] * 100) if agg[name]["nj"] else None,
            "n": len(rows),
        }
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "rows": rows},
              open(CFG.results_dir / f"generation_{args.tag}.json", "w"),
              ensure_ascii=False, indent=2)
    print(f"saved -> {CFG.results_dir / f'generation_{args.tag}.json'}")


if __name__ == "__main__":
    main()
