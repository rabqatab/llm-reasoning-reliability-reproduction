> **Independent cross-check (separate codebase).** This is a *second*, independent
> from-scratch LCF reproduction, distinct from `lcf/lcf_impl/`. It runs on
> **Vicuna-7b** with its own implementation, judge (Claude-Sonnet-4.6), and
> evaluation, to cross-check the team's Qwen3-8B result. **Finding: LCF helps on no
> metric on Vicuna** — the logic/content disentanglement does not form. See
> `docs/LCF_vicuna_independent.md` for the writeup and `report/report.tex` for the
> full ACL report (Korean).

# LCF Reproduction — *Content-free Logical Modification of LLMs*

Reproduction of Wu et al., **AAAI-25**: "Content-free Logical Modification of
Large Language Model by Disentangling and Modifying Logic Representation."
The original paper ships **no code** (only a supplementary PDF), so this is a
from-scratch re-implementation built from the paper + supplementary materials.

Scaled for a **~16 GB VRAM slice of an H100**: one base model
(**Vicuna-7b-v1.5**, 4-bit), both tasks, key ablations, t-SNE, controlled
experiments (fp16 vs 4-bit; label quality), and the bidirectional "invalid
modification" control.

## What LCF does (1 paragraph)
A small module (~projectors + a fusion decoder) is hooked after each Transformer
attention/MLP block. It splits the hidden state into a **content** space and a
**logic** space (two MLP projectors), nudges only the **logic** part toward a
"logically valid" region via a steering vector `V = C_pos − C_neg`, then fuses it
back with the untouched content via a cross-attention decoder. Trained with
reconstruction + supervised-contrastive (logic) + a content-disentangle loss.
See `src/lcf.py` (equations are mapped in the docstring).

## Layout
```
config.py              all hyper-params + paths (artifacts go to $LCF_SCRATCH, NOT the vault)
LFUD.csv               dataset (804 rows = 67 scenarios x 12 fallacies)
src/
  data_prep.py         LFUD -> premise/invalid, 45/5/17 scenario split, identification items
  gen_valid.py         generate the matching VALID conclusions (local LLM or API)
  extract.py           forward-hook hidden-state extraction + identical-token alignment
  lcf.py               the LCF module + losses  (unit-tested, see tests/)
  train.py             train LCF, derive steering vector + top-10 distinctive taps
  inference.py         load ckpt + attach steering hooks to the base model
  eval_identification.py   Accuracy + ΔProb   (deterministic, no judge needed)
  eval_generation.py       Valid% (LLM judge) + Perplexity
  analysis_tsne.py     content vs logic space t-SNE (Fig 3a)
  aggregate.py         compile results/*.json -> results/summary.md
tests/test_lcf_local.py  CPU unit test of the LCF core (no GPU/LLM)
run_all.sh / smoke.sh    full pipeline / quick wiring check
```

## Setup (H100)
```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements-gpu.txt
export HF_TOKEN=...                 # Llama-2 is gated on HF
export LCF_SCRATCH=/path/to/scratch # big files live here, outside the vault
huggingface-cli login              # or rely on HF_TOKEN
```

## Run
```bash
bash smoke.sh           # ~few min: catches wiring bugs end-to-end on tiny slices
bash run_all.sh         # full reproduction
# valid-conclusion gen / Valid% judge backends:
GEN_BACKEND=anthropic JUDGE=anthropic bash run_all.sh   # needs ANTHROPIC_API_KEY
```
Outputs: `results/summary.md` (tables), `results/figures/tsne_content_logic.png`.

## Local sanity check (no GPU)
The hardest, most bug-prone part — the LCF module + losses — is verified on CPU:
```bash
uv venv .venv-local && source .venv-local/bin/activate && uv pip install torch numpy
python tests/test_lcf_local.py     # ALL PASS: rec, content-disentangle, contrastive, steering ±
python src/data_prep.py            # 540/60/204 split
```

## Documented deviations from the paper (report these!)
1. **One model, not six.** Llama-2-7b-chat only (VRAM + time). Code is
   model-agnostic for the Llama family (`model_utils.get_taps`).
2. **4-bit quantization** of the base LLM (paper used full precision). Affects
   absolute numbers slightly; the Original-vs-+LCF *gap* is the claim under test.
3. **Valid conclusions** generated locally (or via API) instead of GPT-3.5-turbo;
   no manual proofreading pass.
4. **Valid% judge** = an LLM judge (Claude/GPT) standing in for the paper's
   "GPT-4 discriminator"; the "trained Llama-2 discriminator" is omitted.
5. **Fallacy-identification options** are constructed from LFUD (1 valid + 2
   invalid + "I have no comment"); the paper's exact option-construction isn't
   fully specified. ΔProb uses length-normalized option log-likelihoods.
6. Global steering vector + top-10 taps by validation **distinctiveness**
   (supplementary's nearest-center separability).

## Expected (paper, Llama2): Original→+LCF
Valid%(GPT4) 70.6→83.8 · Valid%(Trained) 58.8→**96.6** · PPL 21.1→12.1 ·
Acc 51.5→75.0 · ΔProb −1.89→+6.29. We test whether the **direction & magnitude**
of these gaps reproduce under the scaled setup.
```
