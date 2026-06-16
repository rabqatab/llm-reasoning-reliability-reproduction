#!/usr/bin/env bash
# Real reproduction run, ordered so the headline (API-free) result lands first.
# Launch: source env.sh && setsid nohup bash run_real.sh > real.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
CK="$LCF_SCRATCH/checkpoints"
ts(){ date +%H:%M:%S; }
step(){ echo; echo "===== [$(ts)] $* ====="; }

step "0 data prep";                 python src/data_prep.py
step "1 valid conclusions (local, all 804)"; python src/gen_valid.py --backend "${GEN_BACKEND:-local}"
step "2 extract hidden states (train+val)";  python src/extract.py
step "3 train FULL";                python src/train.py --tag full
step "4 eval identification FULL (headline)"; python src/eval_identification.py --ckpt "$CK/lcf_full.pt" --tag full
step "5 eval generation FULL (no judge yet)"; python src/eval_generation.py --ckpt "$CK/lcf_full.pt" --judge none --tag full
step "6 t-SNE";                     python src/analysis_tsne.py --ckpt "$CK/lcf_full.pt"
step "7 invalid-modification control (identification)"; python src/eval_identification.py --ckpt "$CK/lcf_full.pt" --tag full --invalid
# --- ablations (identification = API-free Acc/ΔProb) ---
for ab in no_logic no_content_proj no_content no_rec; do
  flag="--${ab//_/-}"   # no_logic -> --no-logic ; no_content_proj -> --no-content-proj
  step "8 train ablation $ab ($flag)";  python src/train.py $flag --tag "$ab"
  step "8 eval ident ablation $ab";     python src/eval_identification.py --ckpt "$CK/lcf_$ab.pt" --tag "$ab"
done
step "9 aggregate";                 python src/aggregate.py
echo; echo "===== [$(ts)] DONE -> results/summary.md ====="
