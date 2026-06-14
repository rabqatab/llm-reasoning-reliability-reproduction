"""Korean-aware canonicalization for JurisNet ko_ver statute-version extraction.

The JurisNet task: given a case `context` (Korean), extract which statute
(law_name) + article (제N조) apply. Evaluated by exact-match on the normalized
set of (law_name, article) pairs.

This module turns either
  * a gold `extractions` list (list of dicts with law_name / article), OR
  * a model's free-text answer (Korean lines like "민법 제103조"),
into a canonical comparable key: a frozenset of (law_name, article) pairs.

`answer_match(a, b)` returns True iff the two inputs canonicalize equal. It is
the `equal_func` / `check_equal` plugged into the RPC evaluators, mirroring how
run_mcq.py uses trivial integer equality.

Korean-normalization notes (gotchas):
  * Articles in gold are always 제N조 or 제N조의M (verified over the dataset);
    model output may add spaces ("제 3 조", "제3조 의2") or omit 제/조, so we
    parse digits and re-emit a canonical "제N조" / "제N조의M".
  * law_name carries meaningful suffixes (시행령 = enforcement decree,
    부칙 = addenda, 시행규칙). These DISTINGUISH different laws, so they must be
    KEPT. Gold law_names sometimes contain spaces ("도시개발법 시행령"); models
    may write them without ("도시개발법시행령"). We therefore strip ALL
    whitespace from law_name — the distinguishing token survives, and this makes
    spaced/un-spaced variants match.
  * Full-width digits / Hanja-style spacing are normalized to ASCII digits.
"""
from __future__ import annotations
import re
import unicodedata

# Full-width -> ASCII digit map (defensive; model can emit them)
_FW = {ord("０") + i: ord("0") + i for i in range(10)}


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").translate(_FW)


def norm_law(law_name) -> str:
    """Canonical law name: NFKC, strip ALL whitespace, drop punctuation noise.

    Keeps Korean suffix tokens (시행령/부칙/시행규칙) that distinguish laws.
    """
    s = _nfkc(str(law_name) if law_name is not None else "")
    s = re.sub(r"\s+", "", s)            # remove all whitespace
    # strip surrounding quotes / brackets a model might add
    s = s.strip("'\"`「」『』<>[](){}·,.")
    return s


def norm_article(article) -> str:
    """Canonical article string -> '제N조' or '제N조의M'.

    Accepts 제3조, 제 3 조, 제3조의2, '3조2', bare '제103조의11', etc.
    Returns '' if no article number is found.
    """
    s = _nfkc(str(article) if article is not None else "")
    # main number followed optionally by 의-sub number
    m = re.search(r"제?\s*(\d+)\s*조(?:\s*의\s*(\d+))?", s)
    if not m:
        # last resort: just digits, treat first as the 조 number
        nums = re.findall(r"\d+", s)
        if not nums:
            return ""
        return f"제{int(nums[0])}조" + (f"의{int(nums[1])}" if len(nums) > 1 else "")
    main, sub = m.group(1), m.group(2)
    out = f"제{int(main)}조"
    if sub is not None:
        out += f"의{int(sub)}"
    return out


def canon_from_extractions(extractions) -> frozenset:
    """Gold path: list[{law_name, article, ...}] -> frozenset[(law, art)]."""
    pairs = set()
    for e in extractions or []:
        law = norm_law(e.get("law_name"))
        art = norm_article(e.get("article"))
        if law or art:
            pairs.add((law, art))
    return frozenset(pairs)


# A model answer line, e.g. "민법 제103조", "도시개발법 시행령 제3조의2",
# "- 형법 제250조 제1항" (paragraph ignored — task scores (law, article)).
_LINE_RE = re.compile(
    r"([가-힣A-Za-z0-9·\s]+?)\s*(제?\s*\d+\s*조(?:\s*의\s*\d+)?)",
)


def canon_from_text(text: str) -> frozenset:
    """Parse a free-text Korean model answer into frozenset[(law, art)].

    Finds every '<law name> 제N조[의M]' span. The law name is the run of
    Korean/alnum chars immediately preceding the 제N조 token (trimmed of
    leading list markers / connective words).
    """
    s = _nfkc(text or "")
    pairs = set()
    for m in _LINE_RE.finditer(s):
        raw_law, raw_art = m.group(1), m.group(2)
        # trim leading list bullets / numbering / connective words from law span
        raw_law = re.sub(r"^[\s\-•*0-9.\)\]]+", "", raw_law)
        # keep only the trailing token sequence (law names have no internal
        # sentence breaks); take last whitespace-joined chunk after a comma/및/와
        raw_law = re.split(r"[,，]|및|와|과|그리고", raw_law)[-1]
        law = norm_law(raw_law)
        art = norm_article(raw_art)
        if law and art:
            pairs.add((law, art))
    return frozenset(pairs)


def canon_str(canon: frozenset) -> str:
    """Stable string form of a canonical frozenset (for storing in JSON)."""
    return " | ".join(f"{l}/{a}" for l, a in sorted(canon))


def parse_canon_str(s: str) -> frozenset:
    """Inverse of canon_str (for evaluators that get the stored string back)."""
    if not s:
        return frozenset()
    out = set()
    for tok in s.split(" | "):
        tok = tok.strip()
        if "/" in tok:
            l, a = tok.rsplit("/", 1)
            out.add((l, a))
    return frozenset(out)


def _as_canon(x) -> frozenset:
    if isinstance(x, frozenset):
        return x
    if isinstance(x, (set, list, tuple)):
        # list of (law, art) pairs or list of extraction dicts
        if x and isinstance(next(iter(x)), dict):
            return canon_from_extractions(list(x))
        return frozenset((norm_law(a), norm_article(b)) for a, b in x)
    if isinstance(x, str):
        # canon_str form uses ' | ' and '/'; raw model text otherwise
        if "/" in x and ("|" in x or "/" in x):
            try:
                return parse_canon_str(x)
            except Exception:
                pass
        return canon_from_text(x)
    return frozenset()


def answer_match(a, b) -> bool:
    """True iff a and b canonicalize to the same (law, article) set.

    a, b may each be: a frozenset, a list of extraction dicts, a list of
    (law, art) pairs, a canon_str string, or raw model text.
    """
    return _as_canon(a) == _as_canon(b)


# ---------------------------------------------------------------------------
# CPU self-test: gold-vs-gold == True, gold-vs-perturbed == False over ~50 rows.
# Run:  python3 normalize.py
# ---------------------------------------------------------------------------
def _selftest(path, n=50):
    import json
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if line.strip():
                rows.append(json.loads(line))

    self_ok = 0
    pert_false = 0
    pert_total = 0
    for r in rows:
        g = r["extractions"]
        gc = canon_from_extractions(g)
        # gold vs gold (via canon_str round-trip to exercise both paths)
        if answer_match(canon_str(gc), gc):
            self_ok += 1
        # perturb: bump the first article number by 1 -> must NOT match
        if g:
            pert = [dict(e) for e in g]
            art = norm_article(pert[0]["article"])
            m = re.search(r"제(\d+)조", art)
            if m:
                newnum = int(m.group(1)) + 1
                pert[0] = dict(pert[0])
                pert[0]["article"] = re.sub(r"제\d+조", f"제{newnum}조", art, count=1)
                pert_total += 1
                if not answer_match(canon_from_extractions(pert), gc):
                    pert_false += 1

    print(f"rows tested:            {len(rows)}")
    print(f"gold self-match rate:   {self_ok}/{len(rows)} = {100*self_ok/len(rows):.1f}%")
    print(f"perturbed-NOT-match:    {pert_false}/{pert_total} = "
          f"{100*pert_false/max(pert_total,1):.1f}%")
    # also exercise text parsing
    sample = rows[0]["extractions"][0]
    txt = f"{sample['law_name']} {norm_article(sample['article'])}"
    print(f"text-parse demo: {txt!r} -> {canon_str(canon_from_text(txt))!r}")
    print(f"  matches gold[0] pair: "
          f"{answer_match(txt, frozenset({(norm_law(sample['law_name']), norm_article(sample['article']))}))}")


if __name__ == "__main__":
    import sys
    p = (sys.argv[1] if len(sys.argv) > 1
         else "/home/alphabridge/Research/JurisNet-ko/data/benchmark/ko_ver/test.jsonl")
    _selftest(p)
