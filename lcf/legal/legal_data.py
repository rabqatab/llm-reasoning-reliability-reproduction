"""LEGAL-domain analogue of the LFUD dataset for the LCF method (Paper B).

This mirrors `lcf/lcf_impl/lfud_data.py` but in the LEGAL-REASONING domain, so LCF
(logic-representation editing) can be tested on a SECOND domain beyond fallacies.

Per legal premise (a Korean case / statute fact-pattern `context`) we generate,
with the OpenAI API:
  - a logically VALID legal conclusion   (sound inference from the premise)
  - a logically INVALID legal conclusion (a plausible-sounding but logically
    fallacious inference from the SAME premise -- same legal content, opposite
    validity). Common legal fallacies used as targets: over-generalisation of a
    holding, affirming-the-consequent on statutory conditions, conflating
    necessary vs sufficient conditions, ignoring a stated exception, etc.
We also build a 4-option legal-VALIDITY-identification item:
    1 valid conclusion + 2 invalid conclusions + an "I have no comment" option.

Source: /home/alphabridge/Research/JurisNet-ko/data/benchmark/ko_ver/test.jsonl
  fields used:  prec_seq (unique case id -> scenario_id), context (-> premise).
Premises stay in their source language (Korean); conclusions are generated in
the same language as the premise.

Outputs (EXACT LCF contract, see lcf/data/SCHEMA.md), written into THIS dir so
the existing pipeline can consume them:
  - split_scenarios.json
  - valid_conclusions.jsonl                         (cache, resumable)
  - conclusion_gen_{train,val,test}.jsonl
        {scenario_id, premise, valid_conclusion, invalid_conclusion}
  - fallacy_id_{val,test}.jsonl
        {scenario_id, premise, options:[4 str], answer_idx}

Usage:
  # dry-run: build splits + fallacy_id with BLANK conclusions, no API
  python legal_data.py --no-api --n 200

  # small validation batch with the API (prints 3 example triples)
  python legal_data.py --n 30 --model gpt-4o-mini

  # full run
  python legal_data.py --n 200 --model gpt-4o-mini

The OpenAI call uses only the Python stdlib (urllib) so the `openai` package is
NOT required. The API key is read from $OPENAI_API_KEY or the repo .env, exactly
like lfud_data.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # lcf/legal/
REPO_ROOT = HERE.parents[1]                                  # repo root
DATA_DIR = HERE                                              # emit into lcf/legal/

LEGAL_SRC = Path(
    "/home/alphabridge/Research/JurisNet-ko/data/benchmark/ko_ver/test.jsonl"
)

SEED = 42
# scenario split ratio ~ 70:10:20 (train:val:test), matching LFUD-style proportions.
SPLIT_RATIO = {"train": 0.70, "val": 0.10, "test": 0.20}

# premises shorter / longer than these are skipped / truncated (chars).
MIN_PREMISE_CHARS = 60
MAX_PREMISE_CHARS = 1200

NO_COMMENT = "본 사안만으로는 단정할 수 없다."   # legal "I have no comment"


# --------------------------------------------------------------- load source
def load_premises(n: int):
    """Return up to n (scenario_id, premise) pairs, deduped by prec_seq.

    scenario_id is the JurisNet `prec_seq` (a stable, unique integer case id).
    """
    seen = set()
    items = []
    with open(LEGAL_SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = o.get("prec_seq")
            ctx = (o.get("context") or "").strip()
            if sid is None or not ctx:
                continue
            if sid in seen:
                continue
            if len(ctx) < MIN_PREMISE_CHARS:
                continue
            if len(ctx) > MAX_PREMISE_CHARS:
                ctx = ctx[:MAX_PREMISE_CHARS].rstrip()
            seen.add(sid)
            items.append((int(sid), ctx))
            if len(items) >= n:
                break
    return items


def make_splits(scenario_ids):
    rng = random.Random(SEED)
    order = list(scenario_ids)
    rng.shuffle(order)
    n = len(order)
    n_tr = int(round(n * SPLIT_RATIO["train"]))
    n_va = int(round(n * SPLIT_RATIO["val"]))
    train = order[:n_tr]
    val = order[n_tr:n_tr + n_va]
    test = order[n_tr + n_va:]
    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


# ------------------------------------------------------------------ OpenAI
GEN_SYS = (
    "당신은 한국 법률 추론 전문가입니다. 주어진 법률 사안/법령(전제)으로부터 "
    "두 개의 결론 문장을 작성합니다.\n"
    "1) valid_conclusion: 전제로부터 논리적으로 '타당하게' 도출되는 결론. "
    "논리적 오류가 없고 전제가 실제로 뒷받침하는 한 문장.\n"
    "2) invalid_conclusion: 같은 전제·같은 법률 쟁점을 다루지만 논리적으로 "
    "'오류가 있는'(그럴듯하지만 타당하지 않은) 결론. 예: 판시사항의 과잉 일반화, "
    "후건긍정, 필요조건과 충분조건의 혼동, 명시된 예외의 무시 등.\n"
    "두 결론은 같은 법률 내용을 다루되 논리적 타당성만 정반대여야 합니다. "
    "각 결론은 한 문장이며 전제를 그대로 반복하지 않습니다. "
    "전제와 동일한 언어(한국어)로 작성합니다.\n"
    'JSON으로만 출력: {"valid_conclusion": "...", "invalid_conclusion": "..."}'
)


def _load_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def openai_chat(api_key, model, system, user, temperature=0.7, max_tokens=400,
                timeout=60):
    """Minimal OpenAI chat-completions call via stdlib urllib (no openai pkg)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    msg = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    return msg, usage


def gen_conclusions(api_key, model, premise):
    """Return (valid, invalid, usage_dict). Retries on transient errors."""
    user = (
        f"전제(법률 사안/법령):\n{premise}\n\n"
        "위 전제로부터 valid_conclusion 과 invalid_conclusion 을 JSON으로 작성하세요."
    )
    last_err = None
    for attempt in range(4):
        try:
            content, usage = openai_chat(api_key, model, GEN_SYS, user)
            obj = json.loads(content)
            valid = (obj.get("valid_conclusion") or "").strip()
            invalid = (obj.get("invalid_conclusion") or "").strip()
            if valid and invalid:
                return valid, invalid, usage
            last_err = "empty fields"
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            # other HTTP errors: read body for context then stop
            try:
                last_err += " " + e.read().decode("utf-8")[:200]
            except Exception:
                pass
            break
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
        except (json.JSONDecodeError, KeyError) as e:
            last_err = f"parse: {e}"
            time.sleep(1)
    raise RuntimeError(f"generation failed: {last_err}")


# ----------------------------------------------------------------- cache I/O
def load_cache(path):
    cache = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cache[int(obj["scenario_id"])] = obj
    return cache


# ----------------------------------------------------------------- build
def build(args):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not LEGAL_SRC.exists():
        print(f"[error] legal source not found: {LEGAL_SRC}", file=sys.stderr)
        sys.exit(1)

    premises = load_premises(args.n)
    if not premises:
        print("[error] no premises loaded", file=sys.stderr)
        sys.exit(1)
    premise_of = {sid: p for sid, p in premises}
    scenario_ids = [sid for sid, _ in premises]

    splits = make_splits(scenario_ids)
    with open(DATA_DIR / "split_scenarios.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    split_of = {}
    for name, ids in splits.items():
        for i in ids:
            split_of[i] = name

    # ---- generate / lookup conclusions (resumable cache) ----
    cache_path = DATA_DIR / "valid_conclusions.jsonl"
    cache = load_cache(cache_path)

    api_key = None
    if not args.no_api:
        api_key = _load_api_key()
        if not api_key:
            print("[warn] no OPENAI_API_KEY found; falling back to --no-api",
                  file=sys.stderr)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    n_called = 0
    t0 = time.time()
    cache_f = open(cache_path, "a", encoding="utf-8") if api_key else None
    try:
        for sid, premise in premises:
            if sid in cache and cache[sid].get("valid_conclusion") \
                    and cache[sid].get("invalid_conclusion"):
                continue
            valid = invalid = ""
            if api_key:
                try:
                    valid, invalid, usage = gen_conclusions(
                        api_key, args.model, premise)
                    total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += usage.get(
                        "completion_tokens", 0)
                    n_called += 1
                except Exception as e:  # noqa
                    print(f"[warn] gen failed sid={sid}: {e}", file=sys.stderr)
                    valid = invalid = ""
            obj = {
                "scenario_id": sid,
                "premise": premise,
                "valid_conclusion": valid,
                "invalid_conclusion": invalid,
            }
            cache[sid] = obj
            if cache_f is not None:
                cache_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                cache_f.flush()
    finally:
        if cache_f is not None:
            cache_f.close()
    elapsed = time.time() - t0

    # ---- conclusion_gen_{split}.jsonl ----
    cg = {"train": [], "val": [], "test": []}
    for sid, premise in premises:
        split = split_of[sid]
        c = cache.get(sid, {})
        cg[split].append({
            "scenario_id": sid,
            "premise": premise,
            "valid_conclusion": c.get("valid_conclusion", ""),
            "invalid_conclusion": c.get("invalid_conclusion", ""),
        })
    for split, items in cg.items():
        with open(DATA_DIR / f"conclusion_gen_{split}.jsonl", "w",
                  encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # ---- fallacy_id_{val,test}.jsonl  (legal-validity-identification MCQ) ----
    # 4 options: 1 valid + 2 invalid + "I have no comment". answer_idx points at
    # the VALID option (the logically-sound conclusion). The 2 invalids are this
    # premise's generated fallacy plus one borrowed from another scenario in the
    # same split (a same-domain distractor), mirroring LFUD's 4-option design.
    rng = random.Random(SEED)
    fid_counts = {}
    for split in ("val", "test"):
        # pool of invalid conclusions from OTHER scenarios in this split
        pool = [cache[s]["invalid_conclusion"]
                for s in splits[split]
                if cache.get(s, {}).get("invalid_conclusion")]
        items = []
        for sid in splits[split]:
            c = cache.get(sid, {})
            valid = c.get("valid_conclusion", "")
            invalid = c.get("invalid_conclusion", "")
            if not valid or not invalid:
                continue
            distractors = [x for x in pool if x != invalid]
            extra = rng.choice(distractors) if distractors else NO_COMMENT
            options = [valid, invalid, extra, NO_COMMENT]
            rng.shuffle(options)
            items.append({
                "scenario_id": sid,
                "premise": c.get("premise", premise_of.get(sid, "")),
                "options": options,
                "answer_idx": options.index(valid),
            })
        with open(DATA_DIR / f"fallacy_id_{split}.jsonl", "w",
                  encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        fid_counts[split] = len(items)

    # ---- cost / time estimate (gpt-4o-mini pricing as of 2024-2025) ----
    # $0.150 / 1M input tokens, $0.600 / 1M output tokens.
    in_cost = total_usage["prompt_tokens"] / 1_000_000 * 0.150
    out_cost = total_usage["completion_tokens"] / 1_000_000 * 0.600
    cost = in_cost + out_cost

    # ---- report ----
    filled = sum(1 for v in cache.values()
                 if v.get("valid_conclusion") and v.get("invalid_conclusion"))
    report = {
        "source": str(LEGAL_SRC),
        "n_premises_requested": args.n,
        "n_premises_loaded": len(premises),
        "scenarios": {k: len(v) for k, v in splits.items()},
        "conclusion_gen_rows": {k: len(v) for k, v in cg.items()},
        "conclusions_filled": filled,
        "fallacy_id": fid_counts,
        "api": {
            "model": args.model if api_key else None,
            "calls_this_run": n_called,
            "prompt_tokens": total_usage["prompt_tokens"],
            "completion_tokens": total_usage["completion_tokens"],
            "est_cost_usd": round(cost, 4),
            "elapsed_sec": round(elapsed, 1),
            "per_premise_sec": round(elapsed / n_called, 2) if n_called else None,
            "per_premise_usd": round(cost / n_called, 5) if n_called else None,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # ---- print 3 example triples for quick quality inspection ----
    examples = [v for v in cache.values()
                if v.get("valid_conclusion") and v.get("invalid_conclusion")][:3]
    if examples:
        print("\n===== 3 EXAMPLE TRIPLES (premise / valid / invalid) =====")
        for i, e in enumerate(examples, 1):
            prem = e["premise"]
            print(f"\n--- example {i} (scenario_id={e['scenario_id']}) ---")
            print(f"PREMISE : {prem[:300]}{'...' if len(prem) > 300 else ''}")
            print(f"VALID   : {e['valid_conclusion']}")
            print(f"INVALID : {e['invalid_conclusion']}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200,
                    help="number of legal premises (deduped cases) to use")
    ap.add_argument("--no-api", action="store_true",
                    help="dry-run: build splits + fallacy_id with BLANK conclusions")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="OpenAI model for conclusion generation")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
