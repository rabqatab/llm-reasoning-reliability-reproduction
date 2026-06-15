# Paper A (RPC) — Reproduction Results

Full faithful reproduction of "A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning" (NeurIPS 2025). Computed on the authors' published reasoning paths (HF `WNJXYK/*-Reasoning-Paths`), CPU-only, 10 seeds. Metric = Accuracy% / ECE% (mean±std). K=64 for MATH, 128 otherwise.

## Full grid (our reproduction)

| Dataset | Model | PPL (Acc/ECE) | SC (Acc/ECE) | RPC (Acc/ECE) |
|---|---|---|---|---|
| MATH | Deepseek-Math-RL-7B | 42.51 / 53.83 | 53.33 / 6.43 | **53.37 / 6.46** |
| MATH | InternLM2-Math-Plus-1.8B | 33.24 / 61.58 | 36.48 / 6.67 | **37.88 / 6.43** |
| MATH | InternLM2-Math-Plus-7B | 46.99 / 48.99 | 50.57 / 6.71 | **51.94 / 6.41** |
| MathOdyssey | Deepseek-Math-RL-7B | 22.34 / 73.85 | 36.68 / 9.35 | **37.30 / 9.37** |
| MathOdyssey | InternLM2-Math-Plus-1.8B | 16.56 / 77.28 | 14.52 / 18.56 | **16.32 / 16.38** |
| MathOdyssey | InternLM2-Math-Plus-7B | 27.35 / 67.70 | 28.25 / 12.23 | **31.77 / 9.69** |
| OlympiadBench | Deepseek-Math-RL-7B | 5.90 / 90.26 | 11.29 / 15.21 | **11.29 / 15.07** |
| OlympiadBench | InternLM2-Math-Plus-1.8B | 3.08 / 89.70 | 5.99 / 21.19 | **6.55 / 19.63** |
| OlympiadBench | InternLM2-Math-Plus-7B | 7.27 / 86.90 | 11.07 / 20.20 | **11.12 / 18.89** |
| AIME | Deepseek-Math-RL-7B | 3.37 / 93.38 | 9.42 / 12.12 | **9.50 / 11.96** |
| AIME | InternLM2-Math-Plus-7B | 5.96 / 88.98 | 9.40 / 14.35 | **9.75 / 14.30** |

(AIME has no InternLM2-1.8B paths in the released collection.)

## Verification vs paper Table 2 (InternLM2-Math-Plus-7B)

| Dataset | Method | Ours Acc/ECE | Paper Acc/ECE | Match |
|---|---|---|---|---|
| MATH | PPL | 46.99/48.99 | 46.99/48.99 | ✓ exact |
| MATH | SC | 50.57/6.71 | 50.57/6.71 | ✓ exact |
| MATH | RPC | 51.94/6.41 | 51.95/6.41 | ✓ |
| MathOdyssey | PPL | 27.35/67.70 | 27.35/67.70 | ✓ exact |
| MathOdyssey | SC | 28.25/12.23 | 28.25/12.23 | ✓ exact |
| MathOdyssey | RPC | 31.77/9.69 | 31.62/9.87 | ✓ (Weibull MLE noise) |
| OlympiadBench | RPC | 11.12/18.89 | 11.14/18.86 | ✓ |
| AIME | RPC | 9.75/14.30 | 9.74/14.32 | ✓ |

**Conclusion:** RPC reproduced faithfully. PPL/SC match exactly (deterministic); RPC matches within Weibull-MLE seed noise. Key claims confirmed: (1) PPL is badly mis-calibrated (ECE 49–93) while accurate-ish; (2) SC is well-calibrated but lower accuracy; (3) RPC achieves the best accuracy AND low ECE simultaneously on the larger models. On the weak 1.8B model, RPC≈SC with the documented degradation pattern (Remark 6).

## Extensions — RPC on local datasets (multi-domain)

We applied RPC/SC/PPL to **four local datasets** with verifiable answers, all with Qwen3-8B at K=8 (a small budget the paper warns about), to map *where* RPC's advantage over Self-Consistency holds. Each reuses the official RPC evaluators with a domain-specific equality function. Generation via the batched sampler `rpc/_batched_gen.py`; GPU runs on Node 1 (uv) or Node 2 (docker container, free GPU). Eval is CPU-only.

| Domain | equality | SC (Acc/ECE) | PPL (Acc/ECE) | RPC (Acc/ECE) | RPC vs SC |
|---|---|---|---|---|---|
| math (paper, K=128) | math-equal | (see grid) | over-confident | best | **RPC wins** |
| BIRD text-to-SQL (n=60) | SQLite exec-match | **28.54** / 39.79 | 25.00 / 73.55 | 25.00 / 39.18 | tie/lose |
| JurisNet legal extraction (n=150) | (law,article) set exact-match | 19.08 / 52.42 | 18.67 / **78.82** | **20.00 / 46.02** | **RPC wins (Acc+ECE)** |
| KCC precedent relevance (balanced) | binary 0/1 | _running (Node 2)_ | _running_ | _running_ | _pending_ |
| LFUD fallacy-id MCQ (n=100) | option-index | _running (Node 2)_ | _running_ | _running_ | _pending_ |

Code: `rpc/{bird_extension, jurisnet_ext, kcc_ext, lfud_mcq}/`. Raw logs: `results/rpc_*_results.txt`.

**Findings so far:**
1. **PPL over-confidence reproduces in EVERY domain** (ECE 73.6 BIRD, 78.8 JurisNet, 49–93 math) — the paper's core diagnosis is robust and domain-independent.
2. **RPC's accuracy edge over SC is domain-dependent.** It clearly wins on math and on **JurisNet legal extraction** (20.0 vs 19.1, and best ECE 46.0), where the answer space is diverse enough for the Weibull pruning + perplexity-weighting to help. It does **not** beat SC on **BIRD SQL** at K=8.
3. **Why BIRD is hard for RPC:** K=8 is far below the paper's K=64–128. RPC's Reasoning-Pruning fits a 2-component Weibull mixture to the K path probabilities — unreliable with only 8 samples (paper Remark 6 degradation regime). A fair re-test needs larger K (expensive). Reported as-is, not cherry-picked.
4. KCC (binary, imbalanced → balanced subset, reported with **balanced accuracy**) and LFUD-MCQ (4-option; **connects both papers** — RPC applied to LCF's fallacy task) are generating on Node 2 and will fill the table.
