# LCF Baselines + Evaluation Harness

Owner: eval/baselines agent. Consumes the LCF core (`lcf/lcf_impl/`), data
(`lcf/data/`), and checkpoints (`lcf/checkpoints/`) produced by the LCF-core agent.

Reproduces the baselines + metrics for *"Content-free Logical Modification of LLM..."*
(Wu et al., AAAI 2025). See `docs/LCF_implementation_spec.md` (sections E, F, G).

## Layout

```
lcf/eval/
  metrics.py          # Valid%(GPT-4), Valid%(Trained), Perplexity, Accuracy, DeltaProb
  discriminator.py    # self-trained LFUD validity classifier (Valid%(Trained))
  run_eval.py         # end-to-end: generate -> score -> all metrics -> results/*.json + summary.csv
  pyproject.toml      # uv project (torch/transformers/peft/openai/sklearn/...)
  fixtures/           # tiny contract-matching jsonl for CPU/API smoke tests
  results/            # per-run json + summary.csv (paper Table 1 columns)
lcf/baselines/
  sft.py   iti.py   rahf.py   README.md (this file)
```

## Data contract (produced by the LCF-core agent)

- `lcf/data/conclusion_gen_{train,val,test}.jsonl`: `{scenario_id, premise, valid_conclusion, invalid_conclusion}`
- `lcf/data/fallacy_id_{val,test}.jsonl`: `{scenario_id, premise, options[4], answer_idx}`
- `lcf.lcf_impl.infer`: `generate_with_lcf(prompt, eta=0.5)`, `score_options_with_lcf(prompt, options, eta=4.5) -> list[float]` (length-normalized per-option logprob), plus a plain no-LCF path.

Until those files exist, everything is testable against `lcf/eval/fixtures/` and the
StubBackend (`run_eval.py --dry-run`).

## Environment

```bash
cd lcf/eval
uv venv --python 3.12
uv pip install torch transformers datasets accelerate peft wandb openai scikit-learn numpy
```

API keys read from repo-root `.env` (`OPENAI_API_KEY` -> gpt-4o judge). Confirmed working.

## Run order

1. **Discriminator** (Valid%(Trained) judge) — train once on LFUD, reused by all eval runs.
2. **Baselines** — produce adapters/directions: SFT, ITI, RAHF.
3. **Eval** — run `run_eval.py` per (model, variant); rows accumulate in `results/summary.csv`.

## CPU / API smoke tests (no GPU)

```bash
python metrics.py                              # DeltaProb/Accuracy formula self-test
uv run --with openai python smoke_judge.py     # gpt-4o judge on 3 hand-written examples
uv run --with datasets python discriminator.py --smoke    # LFUD data pipeline (45/5/17 split)
python run_eval.py --variant original --dry-run --data-dir fixtures   # full harness wiring
python ../baselines/sft.py  --smoke            # SFT prompt formatting
uv run --with numpy --with scikit-learn python ../baselines/iti.py --smoke   # ITI probe/COM math
uv run --with torch python ../baselines/rahf.py --smoke  # RAHF target-hidden math
```

## sparkq GPU commands

Base model default `Qwen/Qwen3-8B`, bf16, single DGX Spark GB10 node. Always
`mkdir -p` outputs first and set `HF_HOME=/mnt/nfs/ssd1/huggingface_cache`.

```bash
WORKDIR=/home/alphabridge/Study/reliableAI_final/lcf/eval
HF=/mnt/nfs/ssd1/huggingface_cache

# 1. Train the Valid%(Trained) discriminator on LFUD (small backbone; ~20-40 min)
sparkq submit "HF_HOME=$HF uv run python discriminator.py --model microsoft/deberta-v3-small --epochs 3" \
  --workdir $WORKDIR --gpu-mem 8G --cpu-mem 16G --eta 45m --tag lcf-discriminator
#   (paper-faithful large variant: --model Qwen/Qwen3-8B  --gpu-mem 40G --eta 3h)

# 2a. SFT baseline (LoRA on 540 valid conclusions, 10 epochs)
sparkq submit "HF_HOME=$HF uv run python ../baselines/sft.py --model Qwen/Qwen3-8B --epochs 10" \
  --workdir $WORKDIR --gpu-mem 40G --cpu-mem 16G --eta 2h --tag lcf-sft

# 2b. ITI baseline (probe heads on val reps, save top-K directions; fast)
sparkq submit "HF_HOME=$HF uv run python ../baselines/iti.py --model Qwen/Qwen3-8B --k 48 --alpha 15" \
  --workdir $WORKDIR --gpu-mem 40G --cpu-mem 16G --eta 40m --tag lcf-iti

# 2c. RAHF baseline (representation-control LoRA, reduced single-model variant)
sparkq submit "HF_HOME=$HF uv run python ../baselines/rahf.py --model Qwen/Qwen3-8B --epochs 5 --alpha 5" \
  --workdir $WORKDIR --gpu-mem 40G --cpu-mem 16G --eta 2h --tag lcf-rahf

# 3. Evaluation — one job per (model, variant). DeltaProb x100 to match paper magnitudes.
for V in original +SFT +ITI +RAHF +LCF; do
  sparkq submit "HF_HOME=$HF uv run python run_eval.py --model Qwen/Qwen3-8B --variant $V \
      --data-dir ../data --discriminator-dir discriminator --delta-scale 100" \
    --workdir $WORKDIR --gpu-mem 40G --cpu-mem 16G --eta 1h --tag lcf-eval-$V
done
```

`run_eval.py` for `--variant +LCF` imports `lcf.lcf_impl.infer` (uses eta 0.5 for
generation, 4.5 for fallacy scoring per spec A); other variants load the adapters /
directions produced in step 2; `original` uses the plain base model.

## Faithfulness notes (full vs reduced)

| Component | Status | Notes |
|---|---|---|
| Metrics (all 5) | **full** | Exact judge prompt; softmax-over-options length-normalized DeltaProb; PPL via base LLM. |
| GPT-4 judge | **full** | gpt-4o, cached jsonl, parseable VALID/INVALID + fallacy type. Key verified. |
| Discriminator | **full pipeline, configurable backbone** | LFUD task1+task3 mined valid/invalid, scenario-disjoint 45/5/17. Default deberta-v3-small; `--model Qwen/Qwen3-8B` for paper-faithful large classifier. |
| SFT | **full** | LoRA on (premise->valid), loss masked to completion, 10 epochs lr1e-4. |
| ITI | **core faithful, hook reduced** | Reuses honest_llama per-head probe + center-of-mass direction + top-K selection + additive shift (`utils.py`). Re-implements the intervention via architecture-agnostic `o_proj` forward-pre-hooks instead of baukit/pyvene `head_out` (Qwen3-compatible). |
| RAHF | **DUAL loss faithful, direction reduced** | Reproduces RAHF-DUAL target `base + alpha*(good-bad)` + L2 + KL (`RAHF.py:compute_loss_DUAL`). Reduced: preferred/dispreferred hidden states sourced from the frozen base on labeled valid/invalid pairs rather than from two separately-tuned step-1 models. |

## Metric definitions (documented normalization)

- **DeltaProb**: per question, softmax over the 4 options' *length-normalized* logprobs
  -> probabilities (sum=1); delta = P(correct) - mean(P(incorrect)); report mean over
  test. `--delta-scale 100` matches paper magnitudes (e.g. Llama2 +LCF = 6.29).
- **Accuracy**: fraction where argmax(option logprob) == answer_idx.
- **Perplexity**: exp(mean over conclusions of mean-token NLL) under the base LLM.
- **Valid%(GPT-4 / Trained)**: percent of generated conclusions judged valid.
