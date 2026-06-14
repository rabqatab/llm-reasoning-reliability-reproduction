# RPC extension — KCC (Korean civil-precedent relevance)

A **fourth domain** for the RPC test-time-scaling method (Paper A), after the
paper's math benchmarks, the BIRD text-to-SQL extension, and the JurisNet ko_ver
legal-extraction extension. Here RPC / SC / PPL aggregate K sampled CoT answers to
a **binary legal-relevance classification** task.

## Task

Source: `KoCivSCM` dataset (`/home/alphabridge/Research/KCC/dataset/*.json`, 20
files). Each record is a **(query precedent, candidate precedent) pair** with the
query's and candidate's 사건명 / 판시사항 (`*_precedentAbstract`) / 판결요지
(`*_precedentNote`), and an integer `label`.

We treat it as **binary**: predict whether the candidate is a *legally related
선례 (precedent)* to the query — **1 (관련)** vs **0 (무관)**. The model is prompted
in Korean with both precedents' 판시사항+판결요지 and asked to reason briefly, then
emit `Answer: <0/1>` on the last line. K completions are sampled; each yields a
0/1 vote; aggregation uses **trivial integer (0/1) equality** (the `run_mcq.py`
pattern).

## Class imbalance (important)

The raw dataset is **heavily imbalanced**: of 2939 pairs, only **349 (11.9%) are
label=1**; 2217 are label=0 (and 172 / 201 carry labels 2 / 3 — higher relevance
degrees that we EXCLUDE from the binary 0-vs-1 task). A classifier could win on raw
accuracy simply by always predicting 0.

`build_subset.py` therefore builds a **class-balanced** evaluation subset: it takes
a random sample of label=1 pairs + an **equal** number of randomly sampled label=0
pairs (seed 0), default 300 total → **150 positive / 150 negative (50.0%)**.

Because the subset is 50/50, the **majority-class baseline is ~50% for BOTH plain
accuracy and balanced accuracy**. `run_kcc.py` reports both metrics so an
all-one-class degenerate predictor is visible (it gets ~50% plain *and* 50%
balanced — it cannot hide).

## Files

| file | role | runs on |
|------|------|---------|
| `build_subset.py` | build balanced `kcc_subset.jsonl` ({query_text, candidate_text, label}) | CPU |
| `generate_kcc.py` | HF Qwen3-8B, K sampled CoT completions, RPC-format JSON | **GPU** |
| `run_kcc.py` | reuse RPC `prep/sc/wpc` evaluators with 0/1 equality; SC/PPL/RPC Acc + BalancedAcc + ECE | CPU |

Output JSON shape (mirrors `lfud_mcq/mcq_*.json`): keys `predict` (0/1 int, -1 if
unparseable), `completion` (raw text), `mean_logprob`, `answer` (gold 0/1), each
`[n_pairs][K]`.

## 0. Build the balanced subset (CPU, here)

```bash
python3 /home/alphabridge/Study/reliableAI_final/rpc/kcc_ext/build_subset.py \
    --total 300 --seed 0
# -> kcc_subset.jsonl : 150 positive / 150 negative (50.0%); majority baseline ~50%
```

`*_text` = `사건명 + 판시사항 + 판결요지`, char-truncated (`--max_chars`, default 1500)
to bound prompt length.

## 1. Generate (GPU, login shell)

No GPU on this box — run in the **login shell** of a GPU node (sparkq jobs OOM on
GB10 / hit NFS read-only, so generation is run directly in the login shell,
offline). torch lives in the `lcf/lcf_impl` venv; run via `uv run python` with an
absolute script path.

```bash
cd /home/alphabridge/Study/reliableAI_final/lcf/lcf_impl
HF_HOME=/mnt/nfs/ssd1/huggingface_cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python /home/alphabridge/Study/reliableAI_final/rpc/kcc_ext/generate_kcc.py \
    --model Qwen/Qwen3-8B \
    --data /home/alphabridge/Study/reliableAI_final/rpc/kcc_ext/kcc_subset.jsonl \
    --K 8 --max_new_tokens 256
```

Writes `kcc_Qwen_Qwen3-8B.json` here. **Resumable** via a `.partial.json`
checkpoint (flushed every 5 pairs); re-run the same command to continue.
`enable_thinking=False`; `mean_logprob` is the mean per-token log-prob from
`output_scores`.

## 2. Evaluate (CPU, RPC repo venv)

```bash
cd /home/alphabridge/Study/reliableAI_final/rpc/RPC
uv run python /home/alphabridge/Study/reliableAI_final/rpc/kcc_ext/run_kcc.py \
    --json /home/alphabridge/Study/reliableAI_final/rpc/kcc_ext/kcc_Qwen_Qwen3-8B.json \
    --K 8 --repeats 10
```

Appends `SC / PPL / RPC` lines to `results_kcc.txt`, each reporting
`Accuracy ± std`, `BalancedAccuracy ± std`, `ECE ± std`. The RPC repo venv is
CPU-only and provides `metrics` / `compute_{perp,sc,rpc}`; `run_kcc.py` adds the
RPC dir to `sys.path` and reuses those evaluators **unchanged**, swapping in
integer (0/1) `int_equal` / `int_check` as the `equal_func` / `check_equal`
(exactly the `run_mcq.py` pattern).

**Balanced accuracy** is computed from the per-example *selected* vote: the
max-probability cluster representative the evaluator chose (`chosen_label`), binned
by gold class, averaged over the two classes — so SC/PPL/RPC are each scored both
plainly and class-balanced.

## CPU smoke test (run here)

```bash
# evaluator pipeline on a tiny synthetic RPC-format dict (4 pairs, K=4)
cd /home/alphabridge/Study/reliableAI_final/rpc/RPC
uv run python /home/alphabridge/Study/reliableAI_final/rpc/kcc_ext/run_kcc.py --selftest
```

## Prompt / parse gotchas

- **Last-`Answer:` wins.** `extract_label` takes the *last* `Answer: <0/1>` match,
  so intermediate mentions of 0/1 in the reasoning (e.g. "관련성 0.85") don't
  corrupt the vote. Fallback: last standalone `0`/`1` token; else `-1`
  (unparseable). An unparseable `-1` **never** matches another `-1` in SC/RPC, so
  it cannot inflate consistency.
- The parser only accepts the **binary** chars `0`/`1` after `Answer:` — labels
  `2`/`3` in the raw data are not part of this binary task and are excluded at
  subset-build time, never produced as gold here.
- Korean precedent texts are long; `build_subset.py` truncates each `*_text` to
  1500 chars (`--max_chars`) to keep the two-precedent prompt within a sane token
  budget at K samples.
- Because the subset is exactly 50/50, **read BalancedAccuracy alongside
  Accuracy**: a model biased toward one class shows a gap between the two (and a
  degenerate single-class predictor sits at ~50% on both = the majority baseline).
```
