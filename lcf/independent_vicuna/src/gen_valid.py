"""Generate a logically-VALID conclusion for every (premise, invalid) pair.

The paper used GPT-3.5-turbo; we support either an API backend or a local LLM.
We ask the model to REWRITE the invalid conclusion into a valid one with minimal
word changes, which maximizes identical-token overlap (needed for alignment in
extract.py). Output: data/valid_conclusions.json  {index: valid_conclusion}.

Usage:
  python src/gen_valid.py --backend local           # uses cfg.model_name on GPU
  python src/gen_valid.py --backend openai --model gpt-4o-mini   # needs OPENAI_API_KEY
  python src/gen_valid.py --backend anthropic --model claude-haiku-4-5-20251001  # ANTHROPIC_API_KEY
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG


def _retry(fn, tries=4, base=2.0):
    for t in range(tries):
        try:
            return fn()
        except Exception as e:
            if t == tries - 1:
                raise
            print(f"  retry {t+1}/{tries} after error: {str(e)[:80]}")
            time.sleep(base * (t + 1))

PROMPT = (
    "You are a careful logician. Below is a premise and a conclusion that is "
    "LOGICALLY INVALID (it does not follow necessarily from the premise).\n"
    "Rewrite the conclusion so that it becomes LOGICALLY VALID given the premise, "
    "changing as few words as possible and keeping the same topic and wording.\n"
    "Output ONLY the rewritten conclusion sentence.\n\n"
    "Premise: {premise}\nInvalid conclusion: {invalid}\nValid conclusion:"
)


def load_all_pairs():
    pairs = []
    for split in ("train", "val", "test"):
        for line in open(CFG.data_dir / f"pairs_{split}.jsonl"):
            pairs.append(json.loads(line))
    return pairs


def make_openai(model):
    from openai import OpenAI
    client = OpenAI()
    def one(p):
        r = _retry(lambda: client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": p}],
            temperature=0.7, max_tokens=80))
        return r.choices[0].message.content.strip()
    return one


def make_anthropic(model):
    import anthropic
    client = anthropic.Anthropic()
    def one(p):
        r = _retry(lambda: client.messages.create(
            model=model, max_tokens=80, messages=[{"role": "user", "content": p}]))
        return r.content[0].text.strip()
    return one


def gen_local(prompts, model_name):
    import torch
    from model_utils import load_llm
    cfg = CFG
    cfg.model_name = model_name or cfg.model_name
    model, tok = load_llm(cfg)
    out = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        except Exception:
            ids = tok(p, return_tensors="pt").input_ids
        ids = ids.to(model.device)
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=80, do_sample=True, temperature=0.7,
                               top_p=0.9, pad_token_id=tok.pad_token_id)
        text = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
        out.append(text.split("\n")[0].strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "openai", "anthropic"], default="local")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="for a quick smoke test")
    ap.add_argument("--out", default=None, help="output json (default data/valid_conclusions.json)")
    ap.add_argument("--resume", action="store_true", help="skip indices already in --out")
    args = ap.parse_args()

    pairs = load_all_pairs()
    if args.limit:
        pairs = pairs[: args.limit]
    out_path = Path(args.out) if args.out else CFG.data_dir / "valid_conclusions.json"

    if args.backend == "local":  # batch path (one model load)
        prompts = [PROMPT.format(premise=p["premise"], invalid=p["invalid"]) for p in pairs]
        print(f"generating {len(prompts)} valid conclusions via local ...")
        gen = gen_local(prompts, args.model)
        valid = {str(p["index"]): v for p, v in zip(pairs, gen)}
        json.dump(valid, open(out_path, "w"), ensure_ascii=False, indent=2)
        print(f"wrote {len(valid)} -> {out_path}")
        return

    # API path: per-prompt with retry + incremental save + resume
    one = make_anthropic(args.model or "claude-sonnet-4-6") if args.backend == "anthropic" \
        else make_openai(args.model or "gpt-4o-mini")
    valid = json.load(open(out_path)) if (args.resume and out_path.exists()) else {}
    todo = [p for p in pairs if str(p["index"]) not in valid]
    print(f"generating {len(todo)} valid conclusions via {args.backend} (have {len(valid)}) ...")
    for n, p in enumerate(todo, 1):
        valid[str(p["index"])] = one(PROMPT.format(premise=p["premise"], invalid=p["invalid"]))
        if n % 50 == 0:
            json.dump(valid, open(out_path, "w"), ensure_ascii=False, indent=2)
            print(f"  {n}/{len(todo)} saved")
    json.dump(valid, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {len(valid)} -> {out_path}")
    for p in pairs[:3]:
        print(f"\n[{p['index']}] invalid: {p['invalid']}\n     VALID  : {valid[str(p['index'])]}")


if __name__ == "__main__":
    main()
