"""End-to-end LCF evaluation harness.

Given a base model (+ optional LCF checkpoint or baseline adapter), this:
  1. generates conclusions on conclusion_gen_test (LCF infer API, baseline, or plain base)
  2. scores fallacy_id_test options (per-option logprobs)
  3. computes ALL five metrics (metrics.py)
  4. writes lcf/eval/results/<model>_<variant>.json
  5. appends a row to lcf/eval/results/summary.csv (paper Table 1 columns)

variant in {original, +LCF, +SFT, +ITI, +RAHF}

GENERATION/SCORING BACKENDS (selected by --variant):
  original : plain base model (HF generate; option logprobs via teacher-forcing)
  +LCF     : lcf.lcf_impl.infer.generate_with_lcf / score_options_with_lcf
  +SFT     : baselines.sft  (peft adapter; same infer signature)
  +ITI     : baselines.iti  (head-shift intervention at inference)
  +RAHF    : baselines.rahf (representation-control adapter)

All backends expose the SAME two functions so the harness is backend-agnostic:
  generate(prompt, **kw) -> str
  score_options(prompt, options, **kw) -> list[float]   (length-normalized logprob/option)

Use --dry-run to exercise the full pipeline + metric wiring on CPU with a stub
backend (random/heuristic), no GPU and no model download.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parents[1]))  # repo root, for `import lcf...`

import metrics as M  # noqa: E402

RESULTS_DIR = EVAL_DIR / "results"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
SUMMARY_COLS = ["model", "variant", "ValidGPT4", "ValidTrained", "PPL", "Acc", "DeltaProb"]

# Conclusion-generation prompt: premise -> conclusion
GEN_PROMPT = "{premise}\nTherefore,"

# Fallacy-identification: score each option as a continuation of the premise
FALLACY_PROMPT = "{premise}\n"


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class StubBackend:
    """CPU dry-run backend. Deterministic heuristic so metric wiring is testable.

    generate(): echoes a canned 'valid_conclusion' if present else premise tail.
    score_options(): prefers the option whose index matches a planted answer; here
    we just score by negative length so the harness runs and metrics are computed.
    """

    def __init__(self, gold_map=None):
        self.gold_map = gold_map or {}

    def generate(self, prompt, scenario_id=None, **kw):
        return self.gold_map.get(("gen", scenario_id), "Therefore, the claim follows.")

    def score_options(self, prompt, options, scenario_id=None, **kw):
        gold = self.gold_map.get(("ans", scenario_id))
        # give the gold option the highest (least negative) logprob
        import math
        scores = []
        for i, opt in enumerate(options):
            base = -0.01 * len(opt.split())           # length penalty
            bonus = 0.5 if (gold is not None and i == gold) else 0.0
            scores.append(base + bonus + math.log(1.0))
        return scores


def make_backend(variant: str, model_name: str, ckpt: str | None,
                 dry_run: bool, gold_map=None):
    if dry_run:
        return StubBackend(gold_map=gold_map)

    if variant == "+LCF":
        from lcf.lcf_impl.infer import generate_with_lcf, score_options_with_lcf

        class _B:
            def generate(self, prompt, **kw):
                return generate_with_lcf(prompt, eta=kw.get("eta", 0.5))

            def score_options(self, prompt, options, **kw):
                return score_options_with_lcf(prompt, options, eta=kw.get("eta", 4.5))
        return _B()

    if variant == "+SFT":
        import baselines.sft as sft
        return sft.make_backend(model_name, ckpt)

    if variant == "+ITI":
        import baselines.iti as iti
        return iti.make_backend(model_name, ckpt)

    if variant == "+RAHF":
        import baselines.rahf as rahf
        return rahf.make_backend(model_name, ckpt)

    # original: plain base model
    return _PlainBackend(model_name)


class _PlainBackend:
    """Plain (no-LCF) base-model backend: HF generate + teacher-forced option scoring."""

    def __init__(self, model_name: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        self.model.eval()
        self.device = device

    def generate(self, prompt, max_new_tokens=64, **kw):
        import torch
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        return text.strip()

    def score_options(self, prompt, options, **kw):
        """Length-normalized continuation logprob of each option given the prompt."""
        import torch
        scores = []
        for opt in options:
            full = prompt + " " + opt
            p_ids = self.tok(prompt, return_tensors="pt").input_ids
            f_ids = self.tok(full, return_tensors="pt").input_ids.to(self.device)
            n_prompt = p_ids.shape[1]
            with torch.no_grad():
                logits = self.model(f_ids).logits.float()
            logp = torch.log_softmax(logits[0, :-1], dim=-1)
            tgt = f_ids[0, 1:]
            tok_lp = logp[range(tgt.shape[0]), tgt]
            cont = tok_lp[n_prompt - 1:]          # option-continuation tokens
            scores.append((cont.sum() / max(1, cont.numel())).item())
        return scores


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(model_name, variant, data_dir, dry_run=False, ckpt=None,
        discriminator_dir=None, judge_model="gpt-4o", skip_gpt4=False,
        skip_trained=False, skip_ppl=False, delta_scale=1.0):
    data_dir = Path(data_dir)
    gen_items = M.read_jsonl(data_dir / "conclusion_gen_test.jsonl")
    fal_items = M.read_jsonl(data_dir / "fallacy_id_test.jsonl")

    gold_map = {}
    for it in gen_items:
        gold_map[("gen", it["scenario_id"])] = it["valid_conclusion"]
    for it in fal_items:
        gold_map[("ans", it["scenario_id"])] = it["answer_idx"]

    eta_gen = 0.5 if variant == "+LCF" else None
    eta_fal = 4.5 if variant == "+LCF" else None

    backend = make_backend(variant, model_name, ckpt, dry_run, gold_map=gold_map)

    # ---- 1. generate conclusions -----------------------------------------
    gen_records = []
    for it in gen_items:
        prompt = GEN_PROMPT.format(premise=it["premise"])
        kw = {"scenario_id": it["scenario_id"]}
        if eta_gen is not None:
            kw["eta"] = eta_gen
        conclusion = backend.generate(prompt, **kw)
        gen_records.append({"scenario_id": it["scenario_id"],
                            "premise": it["premise"], "conclusion": conclusion})

    # ---- 2. score fallacy-identification options -------------------------
    per_q_logprobs, answers = [], []
    fal_records = []
    for it in fal_items:
        prompt = FALLACY_PROMPT.format(premise=it["premise"])
        kw = {"scenario_id": it["scenario_id"]}
        if eta_fal is not None:
            kw["eta"] = eta_fal
        lps = backend.score_options(prompt, it["options"], **kw)
        per_q_logprobs.append(lps)
        answers.append(it["answer_idx"])
        fal_records.append({"scenario_id": it["scenario_id"],
                            "option_logprobs": lps, "answer_idx": it["answer_idx"]})

    # ---- 3. metrics -------------------------------------------------------
    results = {"model": model_name, "variant": variant, "n_gen": len(gen_records),
               "n_fallacy": len(fal_records)}

    if skip_gpt4 or dry_run:
        results["ValidGPT4"] = None
    else:
        judge_items = [{"premise": r["premise"], "conclusion": r["conclusion"]}
                       for r in gen_records]
        cache = RESULTS_DIR / f"judge_cache_{_slug(model_name)}_{_slug(variant)}.jsonl"
        results["ValidGPT4"] = M.valid_pct_gpt4(judge_items, model=judge_model,
                                                cache_path=cache)

    if skip_trained or dry_run:
        results["ValidTrained"] = None
    else:
        results["ValidTrained"] = M.valid_pct_trained(
            [r["conclusion"] for r in gen_records], discriminator_dir)

    if skip_ppl or dry_run:
        results["PPL"] = None
    else:
        results["PPL"] = M.perplexity([r["conclusion"] for r in gen_records],
                                      model_name=model_name)

    results["Acc"] = M.fallacy_accuracy(per_q_logprobs, answers)
    results["DeltaProb"] = M.delta_probability(per_q_logprobs, answers,
                                               scale=delta_scale)

    # ---- 4 & 5. persist ---------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"{_slug(model_name)}_{_slug(variant)}.json"
    out_json.write_text(json.dumps(
        {**results, "generations": gen_records, "fallacy_scores": fal_records},
        indent=2, ensure_ascii=False))

    _append_summary(results)
    print(json.dumps({k: results[k] for k in
                      ["model", "variant", "ValidGPT4", "ValidTrained",
                       "PPL", "Acc", "DeltaProb"]}, indent=2))
    print(f"[run_eval] wrote {out_json}")
    print(f"[run_eval] appended row to {SUMMARY_CSV}")
    return results


def _slug(s: str) -> str:
    return s.replace("/", "-").replace("+", "").replace(" ", "")


def _append_summary(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exists = SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if not exists:
            w.writeheader()
        w.writerow({k: results.get(k) for k in SUMMARY_COLS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--variant", default="original",
                    choices=["original", "+LCF", "+SFT", "+ITI", "+RAHF"])
    ap.add_argument("--data-dir", default=str(EVAL_DIR.parents[0] / "data"),
                    help="dir with conclusion_gen_test.jsonl & fallacy_id_test.jsonl")
    ap.add_argument("--ckpt", default=None, help="LCF/baseline checkpoint or adapter dir")
    ap.add_argument("--discriminator-dir", default=None)
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU stub backend; skips GPT4/Trained/PPL")
    ap.add_argument("--skip-gpt4", action="store_true")
    ap.add_argument("--skip-trained", action="store_true")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--delta-scale", type=float, default=1.0,
                    help="DeltaProb scale (use 100 to match paper magnitudes)")
    args = ap.parse_args()
    run(args.model, args.variant, args.data_dir, dry_run=args.dry_run, ckpt=args.ckpt,
        discriminator_dir=args.discriminator_dir, judge_model=args.judge_model,
        skip_gpt4=args.skip_gpt4, skip_trained=args.skip_trained,
        skip_ppl=args.skip_ppl, delta_scale=args.delta_scale)


if __name__ == "__main__":
    main()
