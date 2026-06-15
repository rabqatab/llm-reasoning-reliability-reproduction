# Making LCF model-agnostic — grounded in prior work

This is the constructive half of `LCF_critical_analysis.md`: given that LCF's recipe does not transfer across models (Qwen3 +, Llama2 −), how do we make logic-validity steering **model-agnostic**? We searched the activation-steering / representation-engineering literature and let it guide the redesign.

## 1. Prior work (what we found, and what it says about model-agnosticism)

| Work | Method | Relevance |
|---|---|---|
| **CAA** — Contrastive Activation Addition, Rimsky et al., **ACL 2024** | Steering vector = **mean difference** of residual-stream activations on contrastive (pos/neg) pairs at one layer; added at inference with a coefficient. | The canonical *model-agnostic* steering recipe (just mean-diff + add; no trained module, no architecture assumptions). LCF instead trains projectors+decoder — heavier and, we found, fragile. |
| **RepE** — Representation Engineering, Zou et al., 2023 | Reading/control vectors via PCA (LAT) on contrastive activations. | Foundational; direction from contrastive activations, not a learned autoencoder. |
| **Valentino et al., "Mitigating Content Effects on Reasoning… Activation Steering", AAAI 2026** | Localise formal-vs-plausible layers; contrastive steering on syllogisms; **K-CAST** (kNN conditional). | **Same problem as LCF (content vs formal validity).** Key results we lean on: "contrastive steering supports *linear control* over content biases" BUT **"a static approach is insufficient to debias *all* models"**; a **conditional** (kNN) approach fixes the unresponsive models (+15% formal-reasoning acc). |
| **CAST** — Conditional Activation Steering, Lee et al., ICLR 2025 | Steer only when a condition (probe on the input rep) fires. | Conditional > static for robustness/specificity. |
| **SADI**, Wang et al., ICLR 2025 | Semantics-adaptive dynamic steering vectors (per-input). | Dynamic/per-input beats one fixed vector. |
| **LIMS** — Logical Implication Steering, Kalajdzievski, ICML 2025 | Conditional interventions on transformer generation for logic. | Logic-specific; conditional interventions. |
| **LayerNavigator**, Sun et al., NeurIPS 2025 | Pick promising intervention layer automatically. | Matches our per-layer probe (best single sub-layer). |

## 2. The lesson, and why our v2 was insufficient
The field's consensus, and especially **Valentino et al. (AAAI 2026) on the *identical* content-vs-logic problem, is that STATIC uniform steering is model-dependent** — it works on "responsive" models and fails on others — while **CONDITIONAL / dynamic steering (CAST, K-CAST, SADI, LIMS) is what generalises across models.**

This *exactly* matches our results:
- LCF (static, η fixed): Qwen3 ↑, Llama2 ↓.
- Our v2 (`lcf_v2.py`: best-layer **supervised** direction + norm-relative but still **static/uniform** η): no consistent gain — because it was still static, and used a probe direction rather than the field-standard CAA mean-difference.

So our v2 failed for the reason the literature predicts: **uniform static steering is not model-agnostic.**

## 3. Model-agnostic redesign — v3 = conditional CAA (`lcf/lcf_impl/lcf_caa.py`)
Grounded in CAA + LayerNavigator + CAST/K-CAST:
1. **Direction (CAA):** `v_L = mean(h_valid) − mean(h_invalid)` on the **residual stream** at the auto-localised layer L (no trained projector). Model-agnostic by construction.
2. **Layer (LayerNavigator / our probe):** pick L by held-out valid/invalid separability per model (Qwen3 ≈ L12, Llama2 ≈ L11).
3. **Conditional gate (CAST / K-CAST):** at inference, project the current rep onto `v̂_L`; **only steer tokens whose projection sits on the invalid side** (below the valid/invalid midpoint), and scale the push by how far below — i.e. push *only the inputs that need it, by how much they need it*, instead of a uniform η. This is the piece that the AAAI-2026 result says rescues unresponsive models.
4. **Coefficient α** swept; metric = fallacy-identification ΔProb/Acc, original vs steered, on Qwen3 **and** Llama2 (the responsive + unresponsive pair).

## 4. Success criterion (honest, pre-registered)
v3 is a *model-agnostic improvement* iff conditional CAA steering yields a **consistent (non-negative, ideally positive) ΔProb shift on BOTH** Qwen3 and Llama2 — in particular, it should not degrade Llama2 the way static LCF did, and ideally helps it (à la K-CAST's +15% on unresponsive models). If even conditional CAA fails to steer logic-validity, that is itself a strong, literature-grounded negative (logic-validity is a harder steering target than the content/sentiment/refusal attributes where CAA-family methods succeed).

_Results: filled after the GPU run (Qwen3 + Llama2 coefficient×conditional sweep)._
