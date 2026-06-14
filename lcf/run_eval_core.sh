#!/bin/bash
# Core LCF eval on the GPU node (offline): original vs +LCF. Local metrics only
# (ValidTrained/PPL/Acc/DeltaProb); GPT-4 judge is run later from the login shell.
set -uo pipefail
MODEL="${1:-Qwen/Qwen3-8B}"
SAFE=$(echo "$MODEL" | sed 's#.*/##')
ROOT=/home/alphabridge/Study/reliableAI_final
export PYTHONPATH=$ROOT
export LCF_EVAL_MODEL="$MODEL"   # tells infer.py's +LCF singleton which model/ckpt to load
cd "$ROOT/lcf/eval"
DISC="$ROOT/lcf/eval/discriminator"
DATA="$ROOT/lcf/data"
COMMON="--model $MODEL --data-dir $DATA --discriminator-dir $DISC --skip-gpt4 --delta-scale 100"

echo "### eval original"
uv run python run_eval.py $COMMON --variant original || echo "FAIL original"
echo "### eval +LCF"
uv run python run_eval.py $COMMON --variant +LCF --ckpt "$ROOT/lcf/checkpoints/$SAFE/lcf.pt" || echo "FAIL +LCF"
echo "### CORE EVAL DONE"
