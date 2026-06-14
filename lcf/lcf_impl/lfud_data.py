"""LFUD data preparation -- produces the shared data contract under lcf/data/.

Outputs (schema in lcf/data/SCHEMA.md):
  - split_scenarios.json          : {train:[...], val:[...], test:[...]} scenario ids
  - valid_conclusions.jsonl       : cache of GPT-3.5 generated valid conclusions
  - conclusion_gen_{train,val,test}.jsonl
        {scenario_id, premise, valid_conclusion, invalid_conclusion}
  - fallacy_id_{val,test}.jsonl   (from task2)
        {scenario_id, premise, options:[4 str], answer_idx}

Scenario = the `proposition` (67 unique). Split scenarios 45:5:17 -> train/val/test.

Usage:
  python lfud_data.py --no-api          # build splits + fallacy_id, blank valid_conclusion
  python lfud_data.py --model gpt-3.5-turbo   # also call OpenAI for valid conclusions
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # lcf/
LFUD_CSV = ROOT / "LFUD" / "LFUD.csv"
DATA_DIR = ROOT / "data"

SEED = 42
SPLIT = {"train": 45, "val": 5, "test": 17}           # scenarios -> 67 total


def _parse_cell(cell: str):
    """LFUD task cells are python-dict-literal strings."""
    if cell is None or cell.strip() == "":
        return None
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return None


def load_rows():
    with open(LFUD_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def unique_scenarios(rows):
    seen = []
    for r in rows:
        p = r["proposition"]
        if p not in seen:
            seen.append(p)
    return seen


def make_splits(rows):
    scenarios = unique_scenarios(rows)
    assert len(scenarios) == 67, f"expected 67 scenarios, got {len(scenarios)}"
    rng = random.Random(SEED)
    order = scenarios[:]
    rng.shuffle(order)
    n_tr, n_va = SPLIT["train"], SPLIT["val"]
    train = order[:n_tr]
    val = order[n_tr:n_tr + n_va]
    test = order[n_tr + n_va:]
    # map proposition string -> stable scenario id (index in original unique order)
    sid = {p: i for i, p in enumerate(scenarios)}
    return {
        "train": sorted(sid[p] for p in train),
        "val": sorted(sid[p] for p in val),
        "test": sorted(sid[p] for p in test),
    }, sid


# ------------------------------------------------------------------ OpenAI
VALID_GEN_SYS = (
    "You are a logic expert. Given a premise and a fallacious conclusion drawn "
    "from it, write ONE alternative conclusion that follows VALIDLY (no logical "
    "fallacy) from the SAME premise. Keep it short (one sentence), on the same "
    "topic, and do not restate the premise. Output only the conclusion sentence."
)


def gen_valid_conclusion(client, model, premise, invalid):
    prompt = (
        f"Premise: {premise}\n"
        f"Fallacious conclusion: {invalid}\n\n"
        "Write a logically valid conclusion from the same premise:"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": VALID_GEN_SYS},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip()


def load_cache(path):
    cache = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cache[obj["row_index"]] = obj
    return cache


# ----------------------------------------------------------------- main build
def build(args):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    splits, sid = make_splits(rows)

    with open(DATA_DIR / "split_scenarios.json", "w") as f:
        json.dump(splits, f, indent=2)

    # which split a scenario id belongs to
    split_of = {}
    for name, ids in splits.items():
        for i in ids:
            split_of[i] = name

    # ---- valid conclusion cache (resumable) ----
    cache_path = DATA_DIR / "valid_conclusions.jsonl"
    cache = load_cache(cache_path)

    client = None
    if not args.no_api:
        try:
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                env = ROOT.parent / ".env"
                if env.exists():
                    for line in env.read_text().splitlines():
                        if line.startswith("OPENAI_API_KEY="):
                            key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
            client = OpenAI(api_key=key)
        except Exception as e:  # noqa
            print(f"[warn] OpenAI unavailable ({e}); falling back to --no-api", file=sys.stderr)
            client = None

    # generate/lookup valid conclusion per row, append to cache
    cache_f = open(cache_path, "a") if client is not None else None
    for ri, r in enumerate(rows):
        if ri in cache:
            continue
        premise = r["proposition"]
        invalid = r["sentence"]
        valid = ""
        if client is not None:
            try:
                valid = gen_valid_conclusion(client, args.model, premise, invalid)
            except Exception as e:  # noqa
                print(f"[warn] gen failed row {ri}: {e}", file=sys.stderr)
                valid = ""
        if not valid:
            # fallback: task5 has a "correct the fallacy" prompt only (no answer),
            # so there is no ground-truth corrected sentence in the CSV. Leave blank.
            valid = ""
        obj = {"row_index": ri, "scenario_id": sid[premise],
               "premise": premise, "invalid_conclusion": invalid,
               "valid_conclusion": valid}
        cache[ri] = obj
        if cache_f is not None:
            cache_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            cache_f.flush()
    if cache_f is not None:
        cache_f.close()

    # ---- conclusion_gen_{split}.jsonl ----
    cg = {"train": [], "val": [], "test": []}
    for ri, r in enumerate(rows):
        scen = sid[r["proposition"]]
        split = split_of[scen]
        valid = cache.get(ri, {}).get("valid_conclusion", "")
        cg[split].append({
            "scenario_id": scen,
            "premise": r["proposition"],
            "valid_conclusion": valid,
            "invalid_conclusion": r["sentence"],
        })
    for split, items in cg.items():
        with open(DATA_DIR / f"conclusion_gen_{split}.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # ---- fallacy_id_{val,test}.jsonl  (from task2) ----
    fid_counts = {}
    for split in ("val", "test"):
        items = []
        for r in rows:
            scen = sid[r["proposition"]]
            if split_of[scen] != split:
                continue
            t2 = _parse_cell(r["task2"])
            if not t2 or "options" not in t2 or "answer" not in t2:
                continue
            opts = t2["options"]
            if len(opts) != 4:
                continue
            items.append({
                "scenario_id": scen,
                "premise": t2.get("question", ""),
                "options": opts,
                "answer_idx": int(t2["answer"]),
            })
        with open(DATA_DIR / f"fallacy_id_{split}.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        fid_counts[split] = len(items)

    # ---- report ----
    report = {
        "rows": len(rows),
        "scenarios": {k: len(v) for k, v in splits.items()},
        "conclusion_gen_rows": {k: len(v) for k, v in cg.items()},
        "valid_conclusions_filled": sum(
            1 for v in cache.values() if v.get("valid_conclusion")),
        "fallacy_id": fid_counts,
    }
    print(json.dumps(report, indent=2))
    write_schema()
    return report


SCHEMA_MD = """# LCF shared data contract (lcf/data/)

Produced by `lcf/lcf_impl/lfud_data.py`. Scenario = LFUD `proposition` (67 unique).
Scenario split 45:5:17 -> train/val/test (seed 42), mapped to 804 rows.

## split_scenarios.json
`{ "train": [int...], "val": [int...], "test": [int...] }`
Scenario ids are indices into the de-duplicated proposition list (stable order of
first appearance in LFUD.csv).

## valid_conclusions.jsonl  (cache, resumable, one obj per row)
`{ row_index:int, scenario_id:int, premise:str, invalid_conclusion:str,
   valid_conclusion:str }`
`valid_conclusion` may be "" if generated with --no-api or if generation failed.

## conclusion_gen_{train,val,test}.jsonl
`{ scenario_id:int, premise:str, valid_conclusion:str, invalid_conclusion:str }`
One object per LFUD row in that split. `premise` is the proposition;
`invalid_conclusion` is the original fallacious `sentence`.

## fallacy_id_{val,test}.jsonl   (from LFUD task2, 4-option MCQ)
`{ scenario_id:int, premise:str, options:[str,str,str,str], answer_idx:int }`
`premise` is task2.question, `options` is the 4 choices, `answer_idx` is the
0-based index of the correct (fallacy-containing) option per LFUD task2.answer.

The eval/baselines agent reads `conclusion_gen_*` (for generation) and
`fallacy_id_*` (for identification ΔProb/Accuracy). Schemas above are stable.
"""


def write_schema():
    (DATA_DIR / "SCHEMA.md").write_text(SCHEMA_MD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true",
                    help="skip OpenAI; build splits + fallacy_id with blank valid_conclusion")
    ap.add_argument("--model", default="gpt-3.5-turbo",
                    help="OpenAI model for valid-conclusion generation")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
