#!/usr/bin/env bash
# Full LCF reproduction pipeline. Run on the H100 box after `uv pip install -r requirements-gpu.txt`.
#   LCF_SCRATCH : where multi-GB artifacts go (NOT the vault).   default: ./_scratch
#   GEN_BACKEND : local | openai | anthropic   (valid-conclusion generation)
#   JUDGE       : none  | openai | anthropic   (Valid% judge for generation eval)
#   HF_TOKEN    : huggingface token (Llama2 is gated)
set -euo pipefail
cd "$(dirname "$0")"
export LCF_SCRATCH="${LCF_SCRATCH:-$PWD/_scratch}"
CK="$LCF_SCRATCH/checkpoints"
echo "scratch = $LCF_SCRATCH"

echo "== 0. data prep ==";            python src/data_prep.py
echo "== 1. valid conclusions ==";    python src/gen_valid.py --backend "${GEN_BACKEND:-local}"
echo "== 2. extract hidden states =="; python src/extract.py
echo "== 3. train (full + ablations) =="
python src/train.py --tag full
python src/train.py --no-content      --tag no_content
python src/train.py --no-logic        --tag no_logic
python src/train.py --no-rec          --tag no_rec
python src/train.py --no-content-proj --tag no_content_proj
echo "== 4. fallacy identification eval =="
for tag in full no_content no_logic no_rec no_content_proj; do
  python src/eval_identification.py --ckpt "$CK/lcf_$tag.pt" --tag "$tag"
done
echo "== 5. conclusion generation eval =="
python src/eval_generation.py --ckpt "$CK/lcf_full.pt" --judge "${JUDGE:-none}" --tag full
echo "== 6. analysis + aggregate =="
python src/analysis_tsne.py --ckpt "$CK/lcf_full.pt"
python src/aggregate.py
echo "DONE. See results/summary.md"
