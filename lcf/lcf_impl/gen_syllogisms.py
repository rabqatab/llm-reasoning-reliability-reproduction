"""Generate a formal categorical-syllogism dataset with a validity x believability
2x2 design — the setup K-CAST / Valentino (AAAI'26) use to study the *content effect*
on formal reasoning.

Each item: two premises + a conclusion; the task is whether the conclusion FORMALLY
follows (Yes/No), independent of whether it is believable in the world.

Crucially, reference (valid vs invalid syllogisms) and task (judge a syllogism) share
the SAME distribution — the condition LCF_model_agnostic.md S7 says is needed for a
fair test of the conditional (kNN) gate that collapsed to ~98% firing on the
out-of-distribution fallacy task.

Design, per term triple a ⊂ b ⊂ c (real subset relation, e.g. roses⊂flowers⊂plants):
  VB valid+believable  : P1 All b are c; P2 All a are b; C All a are c   (Barbara, true)
  VU valid+unbelievable: P1 All b are a; P2 All c are b; C All c are a   (Barbara, false C)
  IB invalid+believable: P1 All a are c; P2 All b are c; C All a are b   (undistributed middle, true C)
  IU invalid+unbelievable: P1 All a are c; P2 All b are c; C All b are a (undistributed middle, false C)

Validity is purely by form (Barbara=valid, undistributed-middle=invalid). Believability
is whether the conclusion is true in the world. Content effect: a model biased by
believability answers Yes for believable / No for unbelievable, which is CORRECT on the
congruent cells (VB, IU) but WRONG on the conflict cells (VU, IB).
"""
from __future__ import annotations
import json, os

# (sub, mid, super) with sub ⊂ mid ⊂ super true in the world.
TRIPLES = [
    ("roses", "flowers", "plants"), ("tulips", "flowers", "plants"),
    ("oaks", "trees", "plants"), ("maples", "trees", "plants"),
    ("sparrows", "birds", "animals"), ("eagles", "birds", "animals"),
    ("salmon", "fish", "animals"), ("sharks", "fish", "animals"),
    ("cobras", "snakes", "reptiles"), ("pythons", "snakes", "reptiles"),
    ("lizards", "reptiles", "animals"), ("poodles", "dogs", "mammals"),
    ("whales", "mammals", "animals"), ("tigers", "cats", "mammals"),
    ("apples", "fruits", "foods"), ("carrots", "vegetables", "foods"),
    ("copper", "metals", "elements"), ("sodium", "metals", "elements"),
    ("rubies", "gems", "minerals"), ("diamonds", "gems", "minerals"),
    ("violins", "instruments", "objects"), ("hammers", "tools", "objects"),
    ("triangles", "polygons", "shapes"), ("squares", "polygons", "shapes"),
    ("ducks", "birds", "animals"), ("trout", "fish", "animals"),
    ("cedars", "trees", "plants"), ("daisies", "flowers", "plants"),
    ("beetles", "insects", "animals"), ("ants", "insects", "animals"),
]


def _item(sid, p1, p2, concl, valid, believable, cell):
    premise = f"Premise 1: All {p1[0]} are {p1[1]}.\nPremise 2: All {p2[0]} are {p2[1]}.\n" \
              f"Conclusion: All {concl[0]} are {concl[1]}."
    return {
        "scenario_id": sid,
        "premise": premise + "\nDoes the conclusion logically follow from the premises?",
        "options": ["Yes", "No"],
        "answer_idx": 0 if valid else 1,       # Yes iff formally valid
        "valid": valid, "believable": believable, "cell": cell,
        "conflict": valid != believable,        # VU / IB are conflict items
    }


def generate():
    rows = []
    sid = 0
    for (a, b, c) in TRIPLES:
        # VB: Barbara, conclusion All a are c (true)
        rows.append(_item(sid, (b, c), (a, b), (a, c), True, True, "VB")); sid += 1
        # VU: Barbara (reversed terms), conclusion All c are a (false)
        rows.append(_item(sid, (b, a), (c, b), (c, a), True, False, "VU")); sid += 1
        # IB: undistributed middle, conclusion All a are b (true)
        rows.append(_item(sid, (a, c), (b, c), (a, b), False, True, "IB")); sid += 1
        # IU: undistributed middle, conclusion All b are a (false)
        rows.append(_item(sid, (a, c), (b, c), (b, a), False, False, "IU")); sid += 1
    return rows


def main():
    out_dir = os.environ.get("LCF_DATA", "/home/alphabridge/Study/reliableAI_final/lcf/data")
    rows = generate()
    # Split by triple so train/test share format but not exact terms.
    n_tri = len(TRIPLES)
    cut = (2 * n_tri // 3) * 4          # 2/3 of triples (x4 cells) -> train(reference)
    train, test = rows[:cut], rows[cut:]
    with open(os.path.join(out_dir, "syllogism_train.jsonl"), "w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "syllogism_test.jsonl"), "w") as f:
        for r in test: f.write(json.dumps(r) + "\n")
    # Sanity summary.
    from collections import Counter
    def summ(rs):
        c = Counter(r["cell"] for r in rs)
        return dict(c), sum(r["valid"] for r in rs), sum(r["conflict"] for r in rs)
    print(f"train n={len(train)} cells={summ(train)[0]} valid={summ(train)[1]} conflict={summ(train)[2]}")
    print(f"test  n={len(test)}  cells={summ(test)[0]} valid={summ(test)[1]} conflict={summ(test)[2]}")
    print("--- example per cell ---")
    seen = set()
    for r in rows:
        if r["cell"] not in seen:
            seen.add(r["cell"])
            print(f"[{r['cell']}] valid={r['valid']} believable={r['believable']} ans={r['options'][r['answer_idx']]}")
            print("   " + r["premise"].replace("\n", " "))


if __name__ == "__main__":
    main()
