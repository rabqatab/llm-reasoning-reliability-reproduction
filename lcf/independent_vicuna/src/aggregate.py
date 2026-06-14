"""Compile all results/*.json into one results/summary.{json,md} for the report.

Reads identification_<tag>.json and generation_<tag>.json for tags:
  full, no_rec, no_logic, no_content, no_content_proj
and prints/saves the main comparison + ablation tables (markdown).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG

TAGS = ["full", "no_rec", "no_logic", "no_content", "no_content_proj"]


def load(tag, kind):
    p = CFG.results_dir / f"{kind}_{tag}.json"
    return json.load(open(p)) if p.exists() else None


def main():
    out = {"main": {}, "ablation": {}}
    # main table: baseline vs +LCF (full)
    idf = load("full", "identification")
    gen = load("full", "generation")
    if idf:
        out["main"]["identification"] = idf
    if gen:
        out["main"]["generation"] = gen["summary"] if "summary" in gen else gen

    md = ["# LCF reproduction results\n", "## Main: Llama2-7b  Original vs +LCF\n"]
    md.append("| Setting | Valid% | PPL | Accuracy | ΔProb |")
    md.append("|---|---|---|---|---|")
    def row(name, g, i):
        v = (g or {}).get("valid_pct"); p = (g or {}).get("perplexity")
        a = (i or {}).get("accuracy"); d = (i or {}).get("delta_prob")
        f = lambda x, n=2: "—" if x is None else f"{x:.{n}f}"
        return f"| {name} | {f(v)} | {f(p)} | {f(a)} | {f(d)} |"
    if gen or idf:
        gsum = gen["summary"] if gen and "summary" in gen else (gen or {})
        md.append(row("Original", gsum.get("baseline"), (idf or {}).get("baseline")))
        md.append(row("+LCF", gsum.get("lcf"), (idf or {}).get("lcf")))

    # ablation table
    md += ["\n## Ablation (+LCF variants)\n",
           "| Variant | Valid% | Accuracy | ΔProb |", "|---|---|---|---|"]
    for tag in TAGS:
        i = load(tag, "identification"); g = load(tag, "generation")
        if not (i or g):
            continue
        gsum = g["summary"] if g and "summary" in g else (g or {})
        gl = (gsum.get("lcf") or {}); il = (i or {}).get("lcf", {})
        f = lambda x, n=2: "—" if x is None else f"{x:.{n}f}"
        out["ablation"][tag] = {"identification": il, "generation": gl}
        md.append(f"| {tag} | {f(gl.get('valid_pct'))} | {f(il.get('accuracy'))} | {f(il.get('delta_prob'))} |")

    json.dump(out, open(CFG.results_dir / "summary.json", "w"), indent=2)
    (CFG.results_dir / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nsaved -> {CFG.results_dir/'summary.json'} , {CFG.results_dir/'summary.md'}")


if __name__ == "__main__":
    main()
