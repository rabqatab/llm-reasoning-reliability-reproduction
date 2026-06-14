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

## Extension — RPC on local BIRD text-to-SQL

RPC/SC/PPL applied to a NEW domain: **BIRD text-to-SQL** (Qwen3-8B generated K SQL paths with mean-logprob; equality = SQLite **execution match** instead of math). Subset n=60 dev questions, K=8 paths. See `rpc/bird_extension/`.

| Method | Accuracy | ECE |
|---|---|---|
| SC | **28.54** | 39.79 |
| PPL | 25.00 | 73.55 |
| RPC | 25.00 | 39.18 |

**Honest finding:** RPC does **not** beat SC here (25.0 vs 28.5). The PPL overconfidence pattern reproduces (ECE 73.6, worst). The likely cause is **K=8 is far below the paper's K=64–128**: RPC's Reasoning-Pruning fits a 2-component Weibull mixture to the path probabilities, which is unreliable with only 8 samples and lands in the degradation regime (paper Remark 6, where RPC→SC or worse). RPC's benefit is sample-size- and domain-dependent; a fair re-test needs larger K (expensive generation). Reported as-is rather than cherry-picking.
