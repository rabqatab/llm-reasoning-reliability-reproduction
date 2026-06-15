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

| Domain (Qwen3-8B, K=8) | answer space | model acc | SC | PPL | RPC | RPC vs SC |
|---|---|---|---|---|---|---|
| math (paper, K=128) | open numeric | mid | — | over-confident | best | **RPC wins** |
| BIRD text-to-SQL (n=60) | open SQL (exec) | low (25–29) | **28.5** / 39.8 | 25.0 / 73.6 | 25.0 / 39.2 | lose |
| JurisNet legal extraction (n=150) | open (law,article) set | low (19–20) | 19.1 / 52.4 | 18.7 / **78.8** | **20.0 / 46.0** | **RPC wins (Acc+ECE)** |
| KCC precedent relevance (n=100, balanced) | binary 0/1 | mid (65 bal) | 64.7 / 31.6 | **66.2 / 21.6** | 65.1 / 33.4 | ~tie |
| LFUD fallacy-id MCQ (n=100) | 4-option | high (88) | 88.0 / 12.1 | 87.0 / **8.3** | 88.0 / 12.1 | ~tie |

(Acc = balanced-acc for KCC; ECE in %. Code: `rpc/{bird_extension,jurisnet_ext,kcc_ext,lfud_mcq}/`; raw logs `results/rpc_*_results.txt`; both generated with the batched sampler + **answer-first** prompt, parse-rate 100%.)

**Findings (complete, 5 domains):**
1. **RPC beats SC only when the model is UNCERTAIN with DIVERSE answers** — math and JurisNet legal extraction (both low-accuracy, open answer space): RPC wins on accuracy *and* calibration. When the model is **confident/accurate** (MCQ 88%) or the **answer space is binary** (KCC), there is little diversity for RPC's Weibull-pruning + perplexity-weighting to exploit, so **RPC ≈ SC**. On BIRD, K=8 is below the paper's K=64–128 (Remark 6 degradation), so RPC loses.
2. **PPL over-confidence is task-difficulty-dependent**, not universal: ECE is terrible on the hard low-accuracy tasks (math 49–93, BIRD 73.6, JurisNet 78.8) but **well-calibrated on the easy tasks** (MCQ 8.3, KCC 21.6) — over-confidence surfaces precisely where the model is wrong a lot.
3. **The headline reproduction (RPC > SC + fixes PPL's calibration) holds where the paper operates** (open-ended math-like reasoning at large K) and degrades gracefully/expectedly off-distribution (small K, binary, or already-easy tasks). A faithful, nuanced characterization rather than a blanket claim.
4. **LFUD-MCQ connects both papers** — applying Paper A's RPC to Paper B's fallacy-identification task; the model is already strong (88%) so test-time scaling adds little here.
