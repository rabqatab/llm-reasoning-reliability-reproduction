"""Post-hoc fill of ValidGPT4 + ValidTrained for saved eval results.

Run from the LOGIN shell (has internet for OpenAI). Reads the generations already
saved in lcf/eval/results/<model>_<variant>.json, runs the GPT-4 judge and the
(retrained) discriminator, updates the json and rewrites summary.csv.

Usage:
  OPENAI_API_KEY=... uv run python postprocess_judge.py \
      --results-dir results --discriminator-dir discriminator --judge-model gpt-4o
"""
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path

import metrics as M

SUMMARY_COLS = ["model", "variant", "ValidGPT4", "ValidTrained", "PPL", "Acc", "DeltaProb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--discriminator-dir", default="discriminator")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--skip-gpt4", action="store_true")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    # load OPENAI key from project .env if not in env
    if not args.skip_gpt4 and not os.environ.get("OPENAI_API_KEY"):
        env = Path(__file__).resolve().parents[2] / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()

    rows = []
    for jf in sorted(rd.glob("*.json")):
        if jf.name.startswith("judge_cache"):
            continue
        data = json.loads(jf.read_text())
        gens = data.get("generations")
        if not gens:
            continue
        model, variant = data["model"], data["variant"]
        print(f"[pp] {model} {variant}: {len(gens)} generations")

        # Clean: the completion-style prompt ("{premise}\nTherefore,") makes
        # instruct models emit the conclusion on the FIRST line then ramble.
        # Take the first non-empty line as the conclusion. "Therefore," is
        # re-prepended so the judged text is a complete sentence.
        def clean(c):
            for line in c.strip().splitlines():
                line = line.strip()
                if line:
                    return ("Therefore, " + line) if not line.lower().startswith("therefore") else line
            return c.strip()
        for g in gens:
            g["conclusion_clean"] = clean(g["conclusion"])
        texts = [g["conclusion_clean"] for g in gens]

        # ValidTrained with the (retrained) discriminator
        data["ValidTrained"] = M.valid_pct_trained(texts, args.discriminator_dir)

        # ValidGPT4
        if not args.skip_gpt4:
            judge_items = [{"premise": g["premise"], "conclusion": g["conclusion_clean"]}
                           for g in gens]
            cache = rd / f"judge_cache_{_slug(model)}_{_slug(variant)}.jsonl"
            data["ValidGPT4"] = M.valid_pct_gpt4(judge_items, model=args.judge_model,
                                                 cache_path=cache)
        jf.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        rows.append({k: data.get(k) for k in SUMMARY_COLS})
        print(f"     -> ValidGPT4={data.get('ValidGPT4')} ValidTrained={data['ValidTrained']} "
              f"PPL={data.get('PPL'):.3f} Acc={data.get('Acc'):.2f} DeltaProb={data.get('DeltaProb'):.3f}")

    # rewrite summary.csv
    with open(rd / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[pp] wrote {rd/'summary.csv'} with {len(rows)} rows")


def _slug(s):
    return "".join(c if c.isalnum() else "-" for c in str(s))


if __name__ == "__main__":
    main()
