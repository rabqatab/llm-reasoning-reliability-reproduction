#!/bin/bash
# Full LCF run for ONE model, ALONE on a node (no concurrency -> no OOM/race on
# GB10 unified memory): extract reps -> train LCF -> eval original -> eval +LCF.
# GPT-4 judge + ValidTrained are filled later by postprocess_judge.py (login shell).
set -uo pipefail
MODEL="${1:-meta-llama/Llama-2-7b-chat-hf}"
SAFE=$(echo "$MODEL" | sed 's#.*/##')
ROOT=/home/alphabridge/Study/reliableAI_final

echo "### [1/4] + [2/4] extract reps + train LCF for $MODEL"
bash "$ROOT/lcf/run_lcf_pipeline.sh" "$MODEL" || { echo "FAIL pipeline"; exit 1; }

export PYTHONPATH=$ROOT
export LCF_EVAL_MODEL="$MODEL"
cd "$ROOT/lcf/eval"
COMMON="--model $MODEL --data-dir $ROOT/lcf/data --discriminator-dir $ROOT/lcf/eval/discriminator --skip-gpt4 --delta-scale 100"

echo "### [3/4] eval original"
uv run python run_eval.py $COMMON --variant original || echo "FAIL original"
echo "### [4/4] eval +LCF"
uv run python run_eval.py $COMMON --variant +LCF --ckpt "$ROOT/lcf/checkpoints/$SAFE/lcf.pt" || echo "FAIL +LCF"
echo "### LCF FULL DONE for $MODEL"
