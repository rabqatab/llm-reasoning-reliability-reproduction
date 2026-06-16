# LCF reproduction — final results (Vicuna-7b-v1.5, H100)

**Verdict: the paper's improvements do NOT reproduce in our constrained, from-scratch setting.**
Root cause: the logic/content disentanglement does not emerge — valid vs. invalid
reasoning is not separable in the learned logic space.

## Core mechanism check (synthetic, CPU)
LCF module unit test PASSES: reconstruction ≈0.009, content-disentangle ≈0.0005,
logic separation within +0.999 / across −0.78, bidirectional steering works.
→ The implementation is correct; the issue is empirical, not a code bug.

## Disentanglement on real reps (the crux)
- t-SNE: valid/invalid **intermixed in both content AND logic spaces**
  (`results/figures/tsne_content_logic.png`).
- Top-tap nearest-centroid separability ≈ **0.66** (chance = 0.50).
- Contrastive loss flat throughout training.

## Fallacy Identification (n=204; logprob-based, no judge)
| setting | Accuracy | ΔProb |
|---|---|---|
| Original (Vicuna) | 69.6 | +0.344 |
| +LCF (η=4.5) | 63.7 | +0.300 |

η sweep (n=120): base 71.7 → η0.25–2.0 ≈ 71.7–72.5 → η4.5 **68.3** → η8.0 **65.8**.
*(LCF neutral for small η, harmful at the paper's η.)*

## Conclusion Generation (n=80; Claude-Sonnet-4.6 judge)
| | base | η0.5 | η2.0 | η4.5 |
|---|---|---|---|---|
| Valid% | **36.2** | 35.0 | 31.2 | 35.0 |
| PPL | 1.49 | 1.50 | 1.57 | 2.98 |

*(Fair test — baseline has room — but LCF never improves; high η hurts fluency.)*

## Controlled experiments (rule out confounds)
| config | top separability |
|---|---|
| 4-bit, Vicuna labels (main) | 0.66 |
| fp16, Vicuna labels | 0.67 |
| 4-bit, Claude labels (higher quality) | 0.57 |
| chance | 0.50 |

→ Neither inference precision nor label quality recovers separability. Higher-quality
(minimal-edit) labels are textually closer → *harder* to separate.

## Key deviations from the paper
1 model (Vicuna) not 6 · 4-bit quantization · locally/Claude-generated valid
conclusions (paper: GPT-3.5 + manual review) · Claude judge (paper: GPT-4 + trained
Llama2) · lr 1e-4 + grad-clip (paper's 1e-3 diverged with the cross-attn decoder).

Headline numbers for the report are wired into `report/report.tex`.
