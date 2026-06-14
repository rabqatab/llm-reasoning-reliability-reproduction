"""LCF evaluation metrics (AAAI 2025 reproduction).

Implements the five metrics from spec section F:
  - Valid%(GPT-4)   : gpt-4o logical-validity judge over generated conclusions
  - Valid%(Trained) : self-trained LFUD discriminator (see discriminator.py)
  - Perplexity      : PPL of generated conclusions under the base LLM
  - Accuracy        : fallacy-identification argmax(option logprob) == answer_idx
  - Delta Probability: mean over test of  P(correct) - mean(P(incorrect))

All functions operate on plain Python data structures so they can be unit-tested
on CPU with synthetic inputs (no GPU / no API required for the formula tests).

Normalization conventions (documented, see spec key-ambiguities):
  * Per-option logprobs are LENGTH-NORMALIZED: a raw option logprob is the sum of
    token logprobs of the option continuation; we divide by the number of option
    tokens to get an average-token logprob. This is what `score_options_with_lcf`
    is expected to return (one float per option). If a caller passes RAW summed
    logprobs, set `length_normalized=False` and supply `option_token_counts`.
  * Option PROBABILITIES are obtained by softmax over the per-option (normalized)
    logprobs of the 4 options for that question. So sum_i P_i = 1 per question.
  * Delta Probability for one question = P(correct) - mean(P over incorrect options).
    The reported metric is the mean of this over the test set (paper scales x? -> we
    report the raw mean; spec sanity numbers e.g. 6.29 are mean*100, see note below).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# .env loading (no python-dotenv dependency required)
# --------------------------------------------------------------------------- #
def load_env(env_path: str | None = None) -> dict:
    """Parse KEY=VALUE lines from a .env file into a dict (and os.environ)."""
    if env_path is None:
        # default: repo-root .env  (lcf/eval/metrics.py -> ../../.env)
        env_path = Path(__file__).resolve().parents[2] / ".env"
    env: dict[str, str] = {}
    p = Path(env_path)
    if not p.exists():
        return env
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val:
            env[key] = val
            os.environ.setdefault(key, val)
    return env


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ===========================================================================
# 1. Valid%(GPT-4)  — gpt-4o logical-validity judge
# ===========================================================================

# The exact judge prompt (spec F). Premise + conclusion -> VALID/INVALID + fallacy.
JUDGE_SYSTEM = (
    "You are an expert in logic and critical reasoning. You judge whether a "
    "conclusion follows validly from a given premise. A conclusion is VALID only "
    "if it is logically entailed by (or reasonably supported without fallacy by) "
    "the premise. If the conclusion commits any logical fallacy, it is INVALID."
)

JUDGE_USER_TEMPLATE = (
    "Premise: {premise}\n"
    "Conclusion: {conclusion}\n\n"
    "Is this conclusion logically valid given the premise? "
    "Answer VALID or INVALID and the fallacy type if any.\n"
    "Respond on a single line starting with exactly VALID or INVALID."
)


def parse_judge_verdict(text: str) -> dict:
    """Parse a judge response into {'valid': bool|None, 'fallacy_type': str, 'raw': str}."""
    raw = (text or "").strip()
    upper = raw.upper()
    valid: bool | None
    # Look at the first VALID/INVALID token. Check INVALID first (substring of VALID-free check).
    first_invalid = upper.find("INVALID")
    first_valid = upper.find("VALID")
    if first_invalid == -1 and first_valid == -1:
        valid = None
    elif first_invalid == -1:
        valid = True
    elif first_valid == -1 or first_invalid <= first_valid:
        # INVALID appears, and either VALID absent or VALID is just the suffix of INVALID
        valid = False
    else:
        valid = True
    # fallacy type: text after the verdict word, trimmed
    fallacy = ""
    if valid is not None:
        kw = "INVALID" if valid is False else "VALID"
        idx = upper.find(kw)
        fallacy = raw[idx + len(kw):].lstrip(" .:,-").strip()
    return {"valid": valid, "fallacy_type": fallacy, "raw": raw}


def gpt4_judge(
    items: Sequence[dict],
    model: str = "gpt-4o",
    cache_path: str | os.PathLike | None = None,
    env_path: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 60,
) -> list[dict]:
    """Judge a list of {premise, conclusion} dicts with gpt-4o.

    Returns a list of verdict dicts {valid, fallacy_type, raw, premise, conclusion}.
    Results are cached to `cache_path` (jsonl, keyed by (premise, conclusion)) so
    re-runs are cheap and batched. Reads OPENAI_API_KEY from .env.
    """
    load_env(env_path)
    from openai import OpenAI  # imported lazily so formula tests need no openai

    client = OpenAI()  # picks up OPENAI_API_KEY from env

    # load cache
    cache: dict[tuple[str, str], dict] = {}
    if cache_path and Path(cache_path).exists():
        for row in read_jsonl(cache_path):
            cache[(row["premise"], row["conclusion"])] = row

    out: list[dict] = []
    new_rows: list[dict] = []
    for it in items:
        premise, conclusion = it["premise"], it["conclusion"]
        key = (premise, conclusion)
        if key in cache:
            out.append(cache[key])
            continue
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                    premise=premise, conclusion=conclusion)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        verdict = parse_judge_verdict(resp.choices[0].message.content)
        row = {"premise": premise, "conclusion": conclusion, **verdict}
        out.append(row)
        new_rows.append(row)
        cache[key] = row

    if cache_path and new_rows:
        # append-merge: rewrite the whole cache to keep it valid jsonl
        write_jsonl(cache_path, list(cache.values()))
    return out


def valid_pct_gpt4(
    items: Sequence[dict],
    model: str = "gpt-4o",
    cache_path: str | os.PathLike | None = None,
    env_path: str | None = None,
) -> float:
    """Percent of conclusions judged VALID by gpt-4o. Items: {premise, conclusion}."""
    verdicts = gpt4_judge(items, model=model, cache_path=cache_path, env_path=env_path)
    decided = [v["valid"] for v in verdicts if v["valid"] is not None]
    if not decided:
        return 0.0
    return 100.0 * sum(1 for v in decided if v) / len(decided)


# ===========================================================================
# 2. Valid%(Trained)  — self-trained LFUD discriminator
# ===========================================================================
def valid_pct_trained(
    conclusions: Sequence[str],
    discriminator_dir: str | os.PathLike = None,
) -> float:
    """Percent of conclusions classified VALID by the trained discriminator."""
    from discriminator import load_discriminator  # local module
    disc = load_discriminator(discriminator_dir)
    preds = disc.predict_valid(list(conclusions))
    if len(preds) == 0:
        return 0.0
    return 100.0 * sum(1 for p in preds if p) / len(preds)


# ===========================================================================
# 3. Perplexity  — PPL of generated conclusions under the base LLM
# ===========================================================================
def perplexity(
    texts: Sequence[str],
    model=None,
    tokenizer=None,
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda",
    max_length: int = 512,
) -> float:
    """Mean per-token perplexity of `texts` under a causal LM.

    Loads the base model if `model`/`tokenizer` not provided (GPU path).
    PPL = exp( mean over texts of  (sum NLL / num_tokens) ).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
    model.eval()

    nlls = []
    for text in texts:
        if not text.strip():
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc.input_ids.to(model.device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        # out.loss is mean NLL per token already (HF shifts internally)
        nlls.append(out.loss.float().item())
    if not nlls:
        return float("nan")
    return float(math.exp(sum(nlls) / len(nlls)))


# ===========================================================================
# 4 & 5. Fallacy Identification: Accuracy and Delta Probability
# ===========================================================================
def _softmax(xs: Sequence[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def normalize_option_logprobs(
    raw_logprobs: Sequence[float],
    option_token_counts: Sequence[int] | None = None,
    length_normalized: bool = True,
) -> list[float]:
    """Return length-normalized per-option logprobs.

    If `length_normalized` is True, `raw_logprobs` are assumed already averaged
    per token (the contract of `score_options_with_lcf`) and returned unchanged.
    Otherwise they are summed token logprobs and divided by `option_token_counts`.
    """
    if length_normalized or option_token_counts is None:
        return list(raw_logprobs)
    return [lp / max(1, n) for lp, n in zip(raw_logprobs, option_token_counts)]


def option_probabilities(
    option_logprobs: Sequence[float],
    option_token_counts: Sequence[int] | None = None,
    length_normalized: bool = True,
) -> list[float]:
    """Softmax over the (length-normalized) per-option logprobs -> probabilities."""
    norm = normalize_option_logprobs(
        option_logprobs, option_token_counts, length_normalized
    )
    return _softmax(norm)


def fallacy_accuracy(
    per_question_logprobs: Sequence[Sequence[float]],
    answer_idxs: Sequence[int],
) -> float:
    """Accuracy = fraction where argmax(option logprob) == answer_idx."""
    if not per_question_logprobs:
        return 0.0
    correct = 0
    for lps, ans in zip(per_question_logprobs, answer_idxs):
        pred = max(range(len(lps)), key=lambda i: lps[i])
        correct += int(pred == ans)
    return 100.0 * correct / len(per_question_logprobs)


def delta_probability(
    per_question_logprobs: Sequence[Sequence[float]],
    answer_idxs: Sequence[int],
    per_question_token_counts: Sequence[Sequence[int]] | None = None,
    length_normalized: bool = True,
    scale: float = 1.0,
) -> float:
    """Mean over questions of  P(correct) - mean(P(incorrect)).

    Probabilities are softmax over the 4 options' (length-normalized) logprobs.
    `scale` lets callers report the metric x100 to match paper magnitudes; default
    1.0 returns the raw mean in [-1, 1].
    """
    if not per_question_logprobs:
        return 0.0
    deltas = []
    for qi, (lps, ans) in enumerate(zip(per_question_logprobs, answer_idxs)):
        counts = None
        if per_question_token_counts is not None:
            counts = per_question_token_counts[qi]
        probs = option_probabilities(lps, counts, length_normalized)
        p_correct = probs[ans]
        incorrect = [p for i, p in enumerate(probs) if i != ans]
        p_incorrect_mean = sum(incorrect) / len(incorrect) if incorrect else 0.0
        deltas.append(p_correct - p_incorrect_mean)
    return scale * sum(deltas) / len(deltas)


# ===========================================================================
# Self-test (CPU, no GPU/API): run `python metrics.py`
# ===========================================================================
def _selftest() -> None:
    print("== Delta Probability / Accuracy formula verification ==")
    # 3 questions, 4 options each. Higher logprob = model prefers it.
    # Q0: correct option (idx 0) clearly preferred.
    # Q1: correct option (idx 1) preferred.
    # Q2: model WRONG, prefers idx 0 but answer is 2.
    logps = [
        [-0.2, -1.5, -2.0, -3.0],   # ans 0  -> correct
        [-2.0, -0.1, -1.8, -2.5],   # ans 1  -> correct
        [-0.3, -2.0, -1.0, -3.0],   # ans 2  -> WRONG (argmax=0)
    ]
    answers = [0, 1, 2]

    acc = fallacy_accuracy(logps, answers)
    assert abs(acc - (200.0 / 3.0)) < 1e-6, acc  # 2/3 correct
    print(f"  accuracy = {acc:.4f}%  (expected 66.6667, 2/3 correct)")

    # manual delta for Q0
    p0 = option_probabilities(logps[0])
    manual_delta0 = p0[0] - (p0[1] + p0[2] + p0[3]) / 3
    dp_q0_only = delta_probability([logps[0]], [0])
    assert abs(dp_q0_only - manual_delta0) < 1e-9
    print(f"  Q0 softmax probs       = {[round(x,4) for x in p0]} (sum={sum(p0):.4f})")
    print(f"  Q0 delta (P_corr-meanP_incorr) = {manual_delta0:.6f}")

    dp = delta_probability(logps, answers)
    # recompute by hand
    expect = 0.0
    for lp, a in zip(logps, answers):
        pr = option_probabilities(lp)
        inc = [p for i, p in enumerate(pr) if i != a]
        expect += pr[a] - sum(inc) / len(inc)
    expect /= len(logps)
    assert abs(dp - expect) < 1e-9, (dp, expect)
    print(f"  mean DeltaProb (raw)   = {dp:.6f}")
    print(f"  mean DeltaProb (x100)  = {delta_probability(logps, answers, scale=100):.4f}")

    # length-normalization check: raw summed logprobs / token counts
    raw = [-4.0, -0.4]           # summed token logprobs
    counts = [4, 1]              # option 0 has 4 tokens, option 1 has 1 token
    norm = normalize_option_logprobs(raw, counts, length_normalized=False)
    assert norm == [-1.0, -0.4], norm
    print(f"  length-norm: raw {raw} / counts {counts} -> {norm}  OK")

    # judge parser check
    for txt, exp in [
        ("VALID", True), ("INVALID - faulty generalization", False),
        ("The conclusion is INVALID (ad hominem)", False),
        ("VALID. No fallacy.", True), ("unclear", None),
    ]:
        v = parse_judge_verdict(txt)["valid"]
        assert v is exp, (txt, v, exp)
    print("  judge parser: all cases OK")
    print("\nAll metric self-tests PASSED.")


if __name__ == "__main__":
    _selftest()
