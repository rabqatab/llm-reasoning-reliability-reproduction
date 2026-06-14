"""Parse LFUD.csv into the splits / files the rest of the pipeline needs.

Outputs (all small JSON, safe to keep in the repo):
  data/splits.json            scenario(proposition) -> {train,val,test}
  data/pairs_{split}.jsonl    {index, proposition, fallacy_type, premise, invalid}
  data/identification.jsonl   test-set 4-option fallacy-identification items

`premise` is the fallacious `sentence` with its (invalid) conclusion removed;
`invalid` is the fallacious conclusion taken from task3's answer option.
The matching VALID conclusion is produced later by gen_valid.py.
"""
from __future__ import annotations
import csv, ast, json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG, FALLACY_TYPES


# task3 questions introduce the premise with varied lead-ins, all ending ":"
#   "Assuming the premise is: X" / "Given this as a premise: X" / "The premise is known: X"
_PREMISE_PAT = re.compile(r"premise[^:\n]*:\s*(.+?)(?:\n|$)", re.I)


def parse_row(row):
    t3 = ast.literal_eval(row["task3"])
    invalid = t3["options"][t3["answer"]].strip()
    sentence = row["sentence"].strip().strip('"')
    m = _PREMISE_PAT.search(t3["question"])
    if m:
        premise = m.group(1).strip()
    else:  # fallback: drop the conclusion clause
        parts = re.split(r"\b[Tt]herefore\b,?|\b[Tt]hus\b,?", sentence)
        premise = parts[0].strip() if len(parts) > 1 else sentence
    return {
        "index": int(row["index"]),
        "proposition": row["proposition"].strip(),
        "fallacy_type": row["fallacy_type"].strip(),
        "premise": premise,
        "invalid": invalid,
    }


def main():
    rows = [parse_row(r) for r in csv.DictReader(open(CFG.lfud_csv))]
    scenarios = sorted({r["proposition"] for r in rows})
    assert len(scenarios) == 67, f"expected 67 scenarios, got {len(scenarios)}"

    rng = random.Random(CFG.seed)
    rng.shuffle(scenarios)
    n_tr, n_va = CFG.n_train_scenarios, CFG.n_val_scenarios
    split_of = {}
    for i, s in enumerate(scenarios):
        split_of[s] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    json.dump(split_of, open(CFG.data_dir / "splits.json", "w"), ensure_ascii=False, indent=2)

    buckets = {"train": [], "val": [], "test": []}
    for r in rows:
        buckets[split_of[r["proposition"]]].append(r)
    for split, items in buckets.items():
        with open(CFG.data_dir / f"pairs_{split}.jsonl", "w") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split:5}: {len(items)} rows")

    # --- build 4-option identification items for the test set -----------------
    # options = [valid(placeholder, filled after gen), invalid_A, invalid_B, "I have no comment."]
    by_prop = {}
    for r in rows:
        by_prop.setdefault(r["proposition"], []).append(r)
    ident = []
    for r in buckets["test"]:
        others = [o for o in by_prop[r["proposition"]] if o["index"] != r["index"]]
        distractor = rng.choice(others)["invalid"] if others else r["invalid"]
        ident.append({
            "index": r["index"], "premise": r["premise"],
            "fallacy_type": r["fallacy_type"],
            "invalid_A": r["invalid"], "invalid_B": distractor,
            "no_comment": "I have no comment.",
            # "valid" added by gen_valid.py; option order/answer fixed at eval time
        })
    with open(CFG.data_dir / "identification.jsonl", "w") as f:
        for it in ident:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"identification (test): {len(ident)} items")
    print(f"\nfallacy types: {len(FALLACY_TYPES)}  | wrote files to {CFG.data_dir}")


if __name__ == "__main__":
    main()
