# LCF — Logic Control Framework (from-scratch AAAI 2025 reproduction)

Trainable adapter inserted after the attention & MLP sub-layers of a **frozen** base
LLM. Disentangles a hidden rep into content + logic, shifts logic toward the valid
region, fuses, and nudges the original rep. See `docs/LCF_implementation_spec.md`.

Default base model: `Qwen/Qwen3-8B` (d=4096, 36 layers, local). Configurable via `--model`.

## Files
| file | role |
|---|---|
| `model.py` | `ContentProjector`, `LogicProjector`, `Decoder` (cross-attn fusion), `LCF` (EMA centroids, modify/forward Eq.3-5) |
| `losses.py` | `rec_loss`, `infonce_logic_pos/neg` (SupCon, τ=0.1), `content_loss`, `total_loss` |
| `lfud_data.py` | builds shared data contract under `lcf/data/` (splits, valid conclusions, fallacy MCQ) |
| `extract_reps.py` | frozen forward pass, hooks attn/mlp at layers 10–30, aligns identical tokens, saves `(R_plus,R_minus,layer,kind)` |
| `train.py` | trains one shared LCF; distinctiveness layer selection; saves checkpoint + config |
| `infer.py` | `LCFInference` wrapper applying LCF as forward hooks; `generate_with_lcf`, `score_options_with_lcf` |
| `smoke_test.py` | CPU smoke tests (run with `uv run python smoke_test.py`) |

## Install
```bash
cd lcf/lcf_impl
uv venv --python 3.12
uv pip install torch transformers datasets accelerate wandb openai numpy scikit-learn
```
Run anything with `uv run python <script>.py` (so the venv is used).

## Data prep
Build splits + fallacy MCQ (no API; valid_conclusion left blank):
```bash
uv run python lfud_data.py --no-api
```
Generate valid conclusions with OpenAI (resumable cache, reads `OPENAI_API_KEY` from
`.env`):
```bash
uv run python lfud_data.py --model gpt-3.5-turbo   # or gpt-4o-mini
```
Outputs in `lcf/data/`: `split_scenarios.json`, `valid_conclusions.jsonl`,
`conclusion_gen_{train,val,test}.jsonl`, `fallacy_id_{val,test}.jsonl`, `SCHEMA.md`.
Split sizes: scenarios 45/5/17 → rows 540/60/204; fallacy_id val=60, test=204.

## GPU steps (run via sparkq — do NOT run 8B forward/train on the login shell)

Extract reps (train + val) for the base model:
```bash
sparkq submit --name lcf-extract-train --gpus 1 -- \
  uv run --project lcf/lcf_impl python lcf/lcf_impl/extract_reps.py \
    --model Qwen/Qwen3-8B --split train

sparkq submit --name lcf-extract-val --gpus 1 -- \
  uv run --project lcf/lcf_impl python lcf/lcf_impl/extract_reps.py \
    --model Qwen/Qwen3-8B --split val
```
Produces `lcf/data/reps_Qwen3-8B_{train,val}.pt`.

Train the shared LCF:
```bash
sparkq submit --name lcf-train --gpus 1 -- \
  uv run --project lcf/lcf_impl python lcf/lcf_impl/train.py \
    --model Qwen/Qwen3-8B \
    --reps lcf/data/reps_Qwen3-8B_train.pt \
    --val-reps lcf/data/reps_Qwen3-8B_val.pt
```
Produces `lcf/checkpoints/Qwen3-8B/lcf.pt` + `config.json`
(d, dims, η defaults {generation:0.5, identification:4.5}, top-10 selected sub-layers).

Set `NVIDIA_DISABLE_REQUIRE=1` and bf16 are handled by the scripts (bf16 auto on CUDA;
FP8 is broken on GB10 so it is not used). Always `sparkq status --all` / `sparkq history`
before submitting.

## Inference API (for the eval agent)
```python
from infer import LCFInference            # run with the lcf_impl venv on PYTHONPATH
w = LCFInference("Qwen/Qwen3-8B")          # loads ckpt from lcf/checkpoints/Qwen3-8B/

# Conclusion generation (eta=0.5, push toward valid logic):
text = w.generate_with_lcf(prompt, eta=0.5, sign=+1)

# Fallacy identification (eta=4.5): per-option sequence logprobs for ΔProb/Accuracy
logprobs = w.score_options_with_lcf(prompt, options, eta=4.5, sign=+1)

# Baseline (original model, hooks off):
w.set_lcf_enabled(False)
```
`sign=-1` performs reverse modification (toward invalid logic). The wrapper applies the
LCF only on `config.selected_sublayers`.

## Notes / resolved ambiguities
- Scenario = `proposition` (67 unique); split seed 42.
- LFUD task5 has no ground-truth corrected sentence, so the `--no-api` fallback leaves
  `valid_conclusion` blank (only OpenAI fills it). Extraction skips rows with empty
  valid/invalid conclusions.
- Centroids: EMA during training, then frozen to the global train mean for inference.
- Cross-attn fusion uses single-head scaled dot-product (per spec default).
- Sub-layer = (layer, kind∈{attn,mlp}); 2 sampled per identical-token occurrence.
