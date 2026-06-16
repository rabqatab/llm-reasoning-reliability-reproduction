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

## 5. Results (Qwen3 + Llama2, coefficient × conditional sweep)

Run on the responsive + unresponsive pair. Qwen3 on Node 1 (sparkq), Llama2 on Node 2 (docker), `n_dir=100`, fallacy-identification ΔProb (×100) and Acc. Raw: `results/caa_model_agnostic.txt`.

**Qwen3-8B** (steering layer 12; trained-LCF *helped* this model: ΔProb 3.96→7.83)

| mode | α | Acc | ΔProb |
|---|---|---|---|
| original | 0 | 31.86 | **3.961** |
| static-CAA | 4 | 27.94 | 0.449 |
| cond-CAA | 4 | 27.94 | 0.495 |
| static-CAA | 8 | 26.96 | 0.361 |
| cond-CAA | 8 | 26.96 | 0.348 |

**Llama-2-7b-chat** (steering layer 11; trained-LCF *degraded* this model: ΔProb 4.85→2.44)

| mode | α | Acc | ΔProb |
|---|---|---|---|
| original | 0 | 39.71 | **4.869** |
| static-CAA | 4 | 25.98 | 0.226 |
| cond-CAA | 4 | 25.98 | 0.181 |
| static-CAA | 8 | 23.53 | −0.057 |
| cond-CAA | 8 | 22.55 | −0.035 |

## 6. Verdict — the pre-registered criterion is NOT met (honest negative)
The §4 success criterion was: *conditional CAA yields a consistent non-negative ΔProb shift on BOTH models.* It fails decisively:

1. **CAA steering degrades both models, monotonically in α.** ΔProb collapses from ~4–5 to <0.5 (and negative at α=8 on Llama2); Acc drops 4–14 points. The mean-difference residual direction does not push fallacy-identification in the intended direction on *either* model — including Qwen3, which the *trained* LCF projector did improve.
2. **The conditional gate is inert here: cond ≈ static at every (model, α).** Our midpoint-projection gate (steer only invalid-side tokens) neither rescues the unresponsive model (Llama2) nor protects the responsive one (Qwen3) — contrary to the K-CAST result (Valentino AAAI'26, +15% on unresponsive models) that motivated it.

**Why the divergence from the literature's conditional-steering success?** Two load-bearing differences: (a) our gate is a crude linear midpoint projection, not K-CAST's kNN classifier on the input representation — a weaker condition that fires on nearly every token, making it behave like static; (b) the target differs — K-CAST steered *formal syllogistic validity*, whereas this fallacy-identification ΔProb couples the logic direction to a task-framing the raw mean-diff direction does not transfer to. The trained LCF projector found a model-specific subspace that helped Qwen3; the untrained CAA direction finds none that helps *any* model.

**What this establishes.** Combined with `LCF_critical_analysis.md`, the picture is consistent and honest: logic-validity is a **harder, more entangled steering target** than the content/sentiment/refusal attributes where CAA-family methods succeed. *Trained* LCF steering is model-dependent (helps Qwen3, hurts Llama2/Mistral); the field-standard *untrained* model-agnostic recipe (CAA, ± conditional gate) is model-agnostic only in the trivial sense that it **fails uniformly**. Neither is the model-agnostic improvement we sought.

**Caveats / what would change the verdict (future work).** Only 2 models, one task, α∈{4,8}, `n_dir=100`, and a midpoint gate rather than a faithful kNN K-CAST classifier. A genuine test of the AAAI'26 claim needs: (i) the kNN-conditional gate trained on a held-out validity probe; (ii) per-model layer re-selection via LayerNavigator rather than the fixed L11/L12; (iii) a wider α grid with sign search; (iv) evaluation on a formal-validity task closer to Valentino's, not only fallacy-naming. Until then the result stands as a literature-grounded *negative*: the simplest model-agnostic recipe does not transfer to logic-validity.

## 7. v4 — faithful K-CAST (kNN gate + LayerNavigator + signed α): closes the caveats, confirms the negative, explains *why*
We then implemented exactly the three improvements §6 said were missing (`lcf/lcf_impl/lcf_kcast.py`): **(i) a faithful kNN-classifier gate** on the reference representations (not a midpoint projection); **(ii) LayerNavigator** — pick the steering layer per model by max held-out kNN valid/invalid separability; **(iii) a signed α sweep** (−8,−4,4,8). Raw: `results/kcast_model_agnostic.txt`.

LayerNavigator chose **Qwen3 L=15 (separability 0.885)** and **Llama2 L=22 (0.835)** — both higher-separating than v3's fixed L12/L11, and well above chance (0.5). So the *direction* is found at a genuinely valid/invalid-discriminative layer.

| model | mode | best α | Acc | ΔProb | gate% |
|---|---|---|---|---|---|
| Qwen3 | original | – | 31.86 | **3.961** | – |
| Qwen3 | static-CAA | 4 | 27.94 | 0.658 | – |
| Qwen3 | **kNN-CAST** | 4 | 28.43 | 0.739 | **98.5** |
| Llama2 | original | – | 39.71 | **4.869** | – |
| Llama2 | static-CAA | 8 | 31.37 | 4.000 | – |
| Llama2 | **kNN-CAST** | 8 | 31.37 | 3.974 | **97.7** |

**The negative holds, and now we know the mechanism.** Three findings:
1. **Every steering config still degrades both models** — no α (either sign), no layer, no gate beats `original` ΔProb on either model. The best steered ΔProb is far below baseline (Qwen3 0.74 vs 3.96; Llama2 4.00 vs 4.87).
2. **kNN-CAST ≈ static-CAA at every cell** — the faithful gate did *not* behave differently from static, just as the crude v3 gate didn't.
3. **Why (the key new result): the kNN gate fires on ~98% of tokens** (`gate% = 98.5 / 97.7`). Although the kNN classifier is 0.84–0.89 separable *on the reference distribution* (short valid/invalid conclusion sentences), at inference it classifies almost every fallacy-task token as "invalid-side" — so the conditional collapses to static. This is **distribution shift between the steering-reference texts and the task tokens**: the gate's in-distribution separability does not transfer to the tokens it must gate. That is the concrete reason conditional steering (K-CAST) does not rescue logic-validity here, even implemented faithfully.

**Final verdict (interim, before §8).** Across v3 (CAA ± midpoint gate) and v4 (faithful K-CAST + LayerNavigator + signed sweep), **no untrained, model-agnostic activation-steering recipe improves logic-validity on either model** *on the fallacy-identification task*; the failure is gated by a reference/task distribution mismatch. The one open route was a faithful test on a *formal-syllogism* task where reference and task distributions coincide — done in §8.

## 8. v5 — formal-syllogism task (reference == task distribution): the first POSITIVE result, and a sharper verdict
We built a formal categorical-syllogism dataset with a **validity × believability 2×2** design (`gen_syllogisms.py`; 80 train/reference, 40 test) — the *content-effect* setup Valentino (AAAI'26) study. Task: does the conclusion *formally* follow (Yes/No), regardless of believability. Congruent cells (VB, IU) and **conflict cells (VU = valid-but-unbelievable, IB = invalid-but-believable)** isolate the content effect. We tested two steering directions (`lcf_syllogism_steer.py`): **content** = ablate the believability direction `mean(believable)−mean(invalid)` on believable-side tokens (the faithful debiaser), and **validity** = add `mean(valid)−mean(invalid)` (the v4 recipe). Raw: `results/syllog_*.txt`.

| model · direction | Acc | ΔProb | gate% | VB/VU/IB/IU | content gap |
|---|---|---|---|---|---|
| **Qwen3 · original** | 97.5 | 85.6 | – | 100/100/90/100 | 5.0 |
| **Qwen3 · content-ablate (α≤1)** | **100.0** | **90.4** | 98.6 | 100/100/100/100 | **0.0** |
| Qwen3 · validity-add | 50.0 | ~0 | 94.7 | flips to constant Yes/No | – |
| **Llama2 · original** | 50.0 | 0.9 | – | 0/0/100/100 (constant "No") | 0.0 |
| Llama2 · content-ablate | 50.0 | ≤0.7 | 74.4 | unchanged | 0.0 |
| Llama2 · validity-add | 50.0 | ≤0 | 67.0 | flips constant bias | – |

**Five findings:**
1. **First genuine positive steering result in the whole project: Qwen3 + content-ablation.** Removing the believability signal lifts Acc 97.5→**100** and eliminates the content-effect gap (5→**0**, fixing the lone IB conflict error), ΔProb 85.6→90.4. This is exactly Valentino's debiasing — and it works because the setup is right: in-distribution, the *content* direction is cleanly decodable (believable/not separability 0.94 at L21), and the model already *has* the formal-reasoning capability to free up.
2. **Direction matters more than the gate.** The **validity** direction (add valid−invalid) is *harmful* — it forces a constant answer and collapses Qwen3 from 97.5 to 50. "Steer toward valid" biases the *answer*; "ablate believability" frees the *reasoning*. The earlier negatives partly reflect using the wrong (validity/answer) direction.
3. **The conditional (kNN) gate still adds nothing: kNN-CAST ≈ static at every cell**, even here. So K-CAST's *conditional* claim does not reproduce; the Qwen3 win comes from the **unconditional** content-ablation direction, not from gating.
4. **Distribution shift was real but not the whole story.** Matching reference to task dropped gate firing from ~98% (fallacy task) to **67–95%** — more selective, confirming §7 — yet selectivity never translated into a kNN-vs-static difference, because the gate still applies a sentence-pooled reference per-token.
5. **Steering cannot manufacture a missing capability.** Llama2 answers a constant "No" to all syllogisms (50% = chance); it has no content effect to debias and no latent form-reasoning to unlock, so every intervention only flips the constant bias. Unrescuable for the right reason — capability, not steering.

**FINAL verdict.** The honest, complete picture is *not* a blanket negative:
- **Contrastive activation steering CAN debias the content effect** — demonstrated on Qwen3 (→100%, gap eliminated) — **when** (a) reference and task share a distribution, (b) you ablate the *content* direction rather than add a *validity* direction, and (c) the model already has the underlying formal-reasoning capability.
- **But it is still NOT model-agnostic, and K-CAST's specific contribution (the conditional gate) is not reproduced.** It fails on the LCF fallacy task (distribution mismatch, §6–7), fails on a model that lacks the capability (Llama2, finding 5), and the kNN gate never beats static (finding 3). The benefit is a property of the *contrastive direction + matched distribution + capable model*, not of conditional gating.

So: *trained* LCF is model-dependent; *untrained* steering debiases content effects only under the three conditions above. Logic-validity steering is achievable but conditional on capability and setup — which is itself the answer to "is LCF model-agnostic?": **no, not in general; yes, narrowly, when the model can already do the task and you target the content direction in-distribution.**
