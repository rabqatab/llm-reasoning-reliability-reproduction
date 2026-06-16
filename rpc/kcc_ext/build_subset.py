"""Build a class-balanced 4-class subset for the KCC precedent-relevance task.

KCC (`/home/alphabridge/Research/KCC/dataset/*.json`, 20 files, ~2939 (query,
candidate) precedent pairs) is a **GRADED relevance** task with labels **0-3**
(3 = highly relevant; per the authors' `metrics.py`). Raw per-class counts are
imbalanced (0:2217, 1:349, 2:172, 3:201), so we build a class-balanced subset
with an EQUAL number of pairs per grade (seed 0), capped by the smallest class.

NOTE: an earlier version of this script wrongly treated KCC as BINARY
({label==1}=pos vs {label==0}=neg), which dropped grades 2 & 3 and mislabeled
grade 1 (the authors' binary collapse is {2,3}->similar, {0,1}->dissimilar).
This 4-class version measures the real task (chance = 25%).

Output: `kcc_subset.jsonl`, one record per line:
  {query_text, candidate_text, label}
where *_text = caseName + 판시사항(abstract) + 판결요지(note), char-truncated.

Run on CPU (no model needed):
  python3 build_subset.py            # default ~300 pairs, balanced, seed 0
"""
from __future__ import annotations
import argparse, glob, json, os, random

DATA_GLOB = "/home/alphabridge/Research/KCC/dataset/*.json"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def make_text(case_name: str, abstract: str, note: str, max_chars: int) -> str:
    """caseName + 판시사항 + 판결요지, joined and truncated to a sane length."""
    parts = []
    if case_name and case_name.strip():
        parts.append(f"사건명: {case_name.strip()}")
    if abstract and abstract.strip():
        parts.append(f"판시사항: {abstract.strip()}")
    if note and note.strip():
        parts.append(f"판결요지: {note.strip()}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def load_all_pairs():
    pairs = []
    for fp in sorted(glob.glob(DATA_GLOB)):
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        for _key, rec in d.items():
            try:
                label = int(rec.get("label", 0))
            except (TypeError, ValueError):
                continue
            pairs.append((rec, label))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", default=DATA_GLOB)
    ap.add_argument("--total", type=int, default=300,
                    help="target total pairs (balanced: total//2 per class, "
                         "capped by #positives available)")
    ap.add_argument("--max_chars", type=int, default=1500,
                    help="truncate each *_text to this many chars")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "kcc_subset.jsonl"))
    args = ap.parse_args()

    pairs = load_all_pairs()
    total_n = len(pairs)
    # KCC is a GRADED 4-class relevance task (labels 0-3; 3 = highly relevant,
    # per the authors' metrics.py). Earlier this script wrongly binarized to
    # {label==1}=pos vs {label==0}=neg, DROPPING grades 2 & 3 — and grade 1 is
    # actually on the authors' "dissimilar" side. We now build a class-balanced
    # 4-way subset so accuracy/RPC are measured on the real task (chance = 25%).
    by_class = {c: [r for (r, lab) in pairs if lab == c] for c in (0, 1, 2, 3)}
    counts = {c: len(v) for c, v in by_class.items()}
    print(f"[subset] loaded {total_n} pairs from {args.data_glob}: per-class {counts}")

    per_class = min(args.total // 4, min(counts.values()))   # exactly balanced 4-way
    rng = random.Random(args.seed)
    chosen = []
    for c in (0, 1, 2, 3):
        sel = list(by_class[c]); rng.shuffle(sel)
        chosen += [(r, c) for r in sel[:per_class]]
    rng.shuffle(chosen)

    with open(args.out, "w", encoding="utf-8") as f:
        for rec, label in chosen:
            row = {
                "query_text": make_text(
                    rec.get("query_caseName", ""),
                    rec.get("query_precedentAbstract", ""),
                    rec.get("query_precedentNote", ""),
                    args.max_chars,
                ),
                "candidate_text": make_text(
                    rec.get("candidate_caseName", ""),
                    rec.get("candidate_precedentAbstract", ""),
                    rec.get("candidate_precedentNote", ""),
                    args.max_chars,
                ),
                "label": label,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_out = len(chosen)
    from collections import Counter
    dist = Counter(l for _, l in chosen)
    print(f"[subset] wrote {n_out} pairs to {args.out}")
    print(f"[subset] 4-class BALANCE {dict(sorted(dist.items()))} "
          f"({per_class}/class); chance baseline = 25.0% accuracy")


if __name__ == "__main__":
    main()
