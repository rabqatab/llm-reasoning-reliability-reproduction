# RPC extension — JurisNet ko_ver (Korean legal statute-version extraction)

A **third domain** for the RPC test-time-scaling method (Paper A), alongside the
paper's math benchmarks and the BIRD text-to-SQL extension. Here RPC / SC / PPL
aggregate K sampled answers to the **JurisNet ko_ver** task.

## Task

Given a Korean case `context` (from `JurisNet-ko/data/benchmark/ko_ver/{test,train,val}.jsonl`),
extract **which statute(s) and article(s) apply**: the set of `(law_name, 제N조)`
pairs. The official metric is exact-match / extraction-F1 on the normalized
`(law_name, article)` set (see `baseline_qwen3_8b_results.json`:
`extraction_f1`, `version_exact_match`).

Each data row: `{prec_seq, decision_date, court, context, extractions: [{law_name, article, paragraph, subpara, ...}]}`.
We score the **(law_name, article) set only** (항/호 = paragraph/subpara are ignored),
matching the `version_exact_match` granularity.

## Files

| file | role | runs on |
|------|------|---------|
| `normalize.py` | Korean-aware canonicalization → `frozenset[(law, article)]`; `answer_match(a,b)` | CPU |
| `generate_jurisnet.py` | HF Qwen3-8B, K sampled completions, RPC-format JSON | **GPU** |
| `run_jurisnet.py` | reuse RPC `prep/sc/wpc` evaluators with `answer_match`; SC/PPL/RPC Acc + ECE | CPU |

Output JSON shape (mirrors `lfud_mcq/mcq_*.json`): keys `predict` (canonical
answer string), `completion` (raw text), `mean_logprob`, `answer` (gold canonical
string), each `[n_cases][K]`.

## 1. Generate (GPU, login shell)

No GPU on this box — run in the **login shell** of a GPU node (sparkq jobs OOM /
hit NFS read-only, so generation is run directly in the login shell, offline).
torch lives in the `lcf/lcf_impl` venv; run via `uv run python` with an absolute
script path.

```bash
cd /home/alphabridge/Study/reliableAI_final/lcf/lcf_impl
HF_HOME=/mnt/nfs/ssd1/huggingface_cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python /home/alphabridge/Study/reliableAI_final/rpc/jurisnet_ext/generate_jurisnet.py \
    --model Qwen/Qwen3-8B \
    --data /home/alphabridge/Research/JurisNet-ko/data/benchmark/ko_ver/test.jsonl \
    --n 150 --K 8 --max_new_tokens 256
```

Writes `jurisnet_Qwen_Qwen3-8B.json` here. **Resumable** via a `.partial.json`
checkpoint (flushed every 5 cases); re-run the same command to continue.
`enable_thinking=False`; `mean_logprob` is the mean per-token log-prob from
`output_scores`. Long Korean case texts are char-truncated (`--max_ctx_chars`,
default 4000) to bound prompt length.

## 2. Evaluate (CPU, RPC repo venv)

```bash
cd /home/alphabridge/Study/reliableAI_final/rpc/RPC
uv run python /home/alphabridge/Study/reliableAI_final/rpc/jurisnet_ext/run_jurisnet.py \
    --json /home/alphabridge/Study/reliableAI_final/rpc/jurisnet_ext/jurisnet_Qwen_Qwen3-8B.json \
    --K 8 --repeats 10
```

Appends `SC / PPL / RPC` lines (`Accuracy ± std`, `ECE ± std`) to
`results_jurisnet.txt`. The RPC repo venv is CPU-only and provides
`metrics` / `compute_{perp,sc,rpc}` — `run_jurisnet.py` adds the RPC dir to
`sys.path` and reuses those evaluators unchanged, swapping in `answer_match` as
the `equal_func` / `check_equal` (exactly the `run_mcq.py` pattern).

## CPU smoke tests (run here, no GPU)

```bash
# normalize self-test: gold-vs-gold == True, gold-vs-perturbed == False (~50 rows)
python3 /home/alphabridge/Study/reliableAI_final/rpc/jurisnet_ext/normalize.py
#   -> gold self-match rate: 50/50 = 100.0% ; perturbed-NOT-match: 50/50 = 100.0%

# evaluator pipeline on a tiny synthetic RPC-format dict
cd /home/alphabridge/Study/reliableAI_final/rpc/RPC
uv run python /home/alphabridge/Study/reliableAI_final/rpc/jurisnet_ext/run_jurisnet.py --selftest
```

## Korean-normalization notes (gotchas)

- **Articles** in the gold data are uniformly `제N조` or `제N조의M` (verified over
  the dataset; no other forms). Models may add spaces (`제 3 조`) or drop the
  `제`/`조` markers, so `norm_article` parses the digits and re-emits the
  canonical `제N조` / `제N조의M`. Full-width digits are NFKC-normalized.
- **Law names** carry meaningful suffixes — `시행령` (enforcement decree),
  `시행규칙` (enforcement rule), `부칙` (addenda). These DISTINGUISH different
  laws and must be kept (e.g. `민법` ≠ `민법 시행령`). Gold law names sometimes
  contain spaces (`도시개발법 시행령`) while models often omit them
  (`도시개발법시행령`), so `norm_law` strips **all** whitespace — the
  distinguishing suffix token survives and spaced/un-spaced variants then match.
- We compare **(law, article) sets**, not paragraphs/subparas, matching the
  `version_exact_match` granularity. An empty parse (model produced no parseable
  pair) never matches anything (incl. another empty), so it cannot inflate SC.
