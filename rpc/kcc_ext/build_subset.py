"""Build a BALANCED evaluation subset for the KCC civil-precedent-relevance task.

The full KCC dataset (`/home/alphabridge/Research/KCC/dataset/*.json`, 20 files,
~2939 (query, candidate) precedent pairs) is heavily imbalanced — only ~12 %
positive (label=1, "the candidate is a legally related precedent"). A plain
accuracy on the raw distribution would be dominated by the negative class, so we
build a class-balanced subset: ALL (or most) label=1 pairs + an EQUAL number of
randomly sampled label=0 pairs (seed 0). Default target ~300 pairs total
(capped by available positives).

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
    pos = [r for (r, lab) in pairs if lab == 1]
    neg = [r for (r, lab) in pairs if lab == 0]
    total_n = len(pairs)
    print(f"[subset] loaded {total_n} pairs from {args.data_glob}: "
          f"{len(pos)} positive ({100*len(pos)/total_n:.1f}%), {len(neg)} negative")

    per_class = args.total // 2
    n_pos = min(per_class, len(pos))
    n_neg = min(n_pos, len(neg))            # keep exactly balanced
    n_pos = n_neg                           # in case neg were the limiting side

    rng = random.Random(args.seed)
    sel_pos = list(pos)
    rng.shuffle(sel_pos)
    sel_pos = sel_pos[:n_pos]
    sel_neg = list(neg)
    rng.shuffle(sel_neg)
    sel_neg = sel_neg[:n_neg]

    chosen = [(r, 1) for r in sel_pos] + [(r, 0) for r in sel_neg]
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
    n_p = sum(1 for _, l in chosen if l == 1)
    print(f"[subset] wrote {n_out} pairs to {args.out}")
    print(f"[subset] BALANCE: {n_p} positive / {n_out - n_p} negative "
          f"({100*n_p/n_out:.1f}% positive); majority-class baseline ~"
          f"{100*max(n_p, n_out - n_p)/n_out:.1f}% accuracy "
          f"(= 50.0% balanced accuracy)")


if __name__ == "__main__":
    main()
