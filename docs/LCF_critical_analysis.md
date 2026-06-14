# Is the LCF paper's claim real? — A critical reproduction analysis

**Scope & stance.** This document rigorously tests whether the central claims of *"Content-free Logical Modification of LLM by Disentangling and Modifying Logic Representation"* (LCF, AAAI 2025) reproduce under a faithful from-scratch re-implementation. We are deliberately careful to separate three very different conclusions: **(A)** "does not reproduce under our setup", **(B)** "claims are overstated / depend on unstated favorable choices / rely on an unauditable metric", and **(C)** "fabricated". We find strong evidence for **(A)** and **(B)**. We do **not** assert **(C)**: we have no direct evidence of fabrication, and reproduction gaps are common. Designed to be falsifiable in our own favor — if a better extraction recovered the paper's gains, that would exonerate the paper (see §5).

## 1. The paper's core claims
1. **Mechanistic premise:** an LLM's hidden state decomposes into a content part and a *logic* part, and there exists a (content-free) "logic-validity direction" V such that moving along it makes outputs more logically valid.
2. **Empirical:** adding LCF improves logical validity across **6 models** (Llama2/3, Vicuna, Mistral, ChatGLM3, Baichuan): e.g. Llama2 Valid%(GPT4) 70.58→83.82, **Valid%(Trained) 58.84→96.56**, ΔProb −1.89→**6.29**, Acc 51.47→75.00.
3. **Implicit:** the effect is general (works across all 6 models with the same recipe).

## 2. What we built
A faithful from-scratch LCF (the official repo `wulidongdong/LCF` ships **no code** — only a README + a supplementary PDF; **claim 0: there is no released implementation to verify against**, itself a reproducibility red flag). Our implementation follows the paper's architecture, losses (Eq. 1–13), η values, layer-10–30 extraction, and distinctiveness layer selection (`docs/LCF_implementation_spec.md`). Data: LFUD with GPT-3.5 valid-conclusion generation, exactly as the paper describes.

## 3. Evidence the central claim does NOT reproduce

**3a. The mechanistic premise is REAL but the paper's recipe destroys it.** We probe whether valid vs. invalid hidden reps are separable (held-out, train→val, scaled + regularized logistic):

| probe on extracted reps | Qwen3-8B | Llama-2-7b-chat |
|---|---|---|
| linear, **pooled** over layers 10–30 | 0.52 | 0.51 |
| MLP / nonlinear, **pooled** | 0.52 | 0.50 |
| mean over single sub-layers | 0.73 | 0.69 |
| **best single sub-layer** | **0.818** (L12 attn) | **0.818** (L11 mlp) |

Chance = 0.50. Two things stand out. **(i)** A logic-validity direction *does* exist — the best single sub-layer separates valid/invalid at **0.82, identically across both models**, which actually *supports* the paper's premise. **(ii)** But it is **destroyed by pooling**: averaging/mixing over the layers 10–30 that the paper modifies collapses the signal to chance (0.52, even nonlinearly). The paper's training samples random layers per token and uses a centroid-difference V, which lands near this diluted regime — explaining why the contrastive plateaus (§3b) and the effect is weak/inconsistent (§3c). So the failure is not "no signal" — it is that **the standard LCF recipe under-exploits a real but layer-localized signal.** (Full per-layer profile: `results/lcf_layer_probe.txt`.) This directly motivates the model-agnostic fix (§6): use the single best sub-layer + a supervised direction.

**3a-bis. The weakness is specific to *logic-validity*, not to representation probing.** As a control we ran the identical probe on a completely different attribute — **suicide risk** — using MoodRisk's pre-extracted Mistral-7B layer reps (8736 user post-sequences, label `trans_730_y`):

| attribute (best single layer) | balanced held-out acc |
|---|---|
| **suicide risk** (MoodRisk, Mistral) | **0.947** (L28; every layer 0.93–0.95) |
| logic-validity (LCF) | 0.82 (L12) — chance (0.52) when pooled |

A semantic/topical attribute is **strongly and robustly linearly encoded across all layers (~0.95)**; logic-validity is only **weakly and layer-fragilely encoded (~0.82 best)**. So representation editing is not broken in general — **logic-validity is simply a much weaker, harder-to-localize direction than typical probing targets.** This is the deeper reason LCF underdelivers. (`results/moodrisk_probe.txt`.)

**3b. The contrastive objective cannot manufacture the signal.** Sweeping the logic contrastive training (lr 1e-3→1e-2, epochs 10→40, τ 0.1→0.05, batch 256→1024) leaves InfoNCE at chance (~ln(batch)≈5.5) and held-out projection separability capped at **~0.66**, with no improvement from stronger training (`results/lcf_contrastive_sweep.txt`). The method is signal-limited, not under-fit.

**3c. The downstream effect is inconsistent and, on the paper's headline model, negative.**

| | Valid%(GPT4) | Acc | ΔProb |
|---|---|---|---|
| Qwen3-8B  Original→+LCF | 47.1→47.1 | 31.9→31.4 | 3.96→**7.83** (helps) |
| Llama2  Original→+LCF (ours) | 35.3→**29.4** | 39.2→**27.0** | 4.85→**2.44** (hurts) |
| _Llama2  Original→+LCF (paper)_ | _70.58→83.82_ | _51.47→75.00_ | _−1.89→6.29_ |

Same code, same hyperparameters: +LCF roughly doubles ΔProb on Qwen3 but **degrades every identification metric on Llama-2-7b-chat** — the opposite of the paper. A method whose published claim is "general across 6 models" should not flip sign on its own headline model under faithful re-implementation.

**3d. The most dramatic metric is unauditable and confoundable.** The paper's flagship number — Valid%(Trained) 58.84→**96.56** — is scored by a **Llama-2 discriminator the authors trained themselves and did not release**. A discriminator trained to recognize "logically valid" text on the same distribution the modification produces can inflate this metric without measuring genuine validity. Our own trained discriminator is **degenerate** on out-of-distribution generations (marks 100% of Llama2 outputs "valid"), demonstrating exactly how fragile/confoundable this metric is. The auditable metric (GPT-4 judge) shows **no** improvement (Qwen3) or a **decline** (Llama2).

**3e. Baseline anomaly.** The paper's Llama2 *original* ΔProb is −1.89 (base model prefers the invalid option); ours is +4.85 (base model already prefers the valid option). Either our MCQ scoring differs from theirs (likely — they did not specify it precisely) or their reported baseline is unusually pessimistic, which mechanically enlarges the headline gain.

## 4. Calibrated verdict
- **(A) Does not reproduce as published — strongly supported.** Across two models and many hyperparameter settings the paper's gains do not reproduce; the effect is weak, model-dependent, and *negative* on the headline model (Llama-2-7b-chat).
- **(B) Overstated / setup-dependent / unauditable — supported.** Crucially, the mechanistic premise itself is **NOT** false — a real logic-validity direction exists (0.82 at the best single sub-layer, consistent across models). What fails is the paper's *recipe* (layer mixing + weak centroid V) and, more seriously, its flagship metric: **Valid%(Trained) 96.56 relies on an unreleased, self-trained discriminator** that we show is easily confoundable (our analogue is degenerate), while the auditable GPT-4 judge shows no gain/decline. The baseline (ΔProb −1.89) may also be unusually pessimistic.
- **(C) Fabrication — explicitly NOT claimed.** The premise is real and the gaps are fully consistent with unstated favorable choices (layer/extraction details, the self-trained discriminator, η and MCQ-scoring tuning), not fraud.

**Bottom line:** the LCF *idea* is sound (a separable logic direction exists), but the paper **as published is not reproducible and not model-agnostic**, and its most dramatic number leans on an unauditable, confoundable discriminator. The honest, constructive result is the model-agnostic redesign (§6) that exploits the *real* 0.82 signal directly.

## 5. Falsification clause (how the paper could still be right)
If, with a better extraction (single best sub-layer, supervised direction) and/or their exact discriminator and η, the gains return *consistently across models*, then the issue was our reproduction, not the paper. We test the extraction half of this in the model-agnostic redesign (§6) — that is the honest next step before any stronger conclusion.

## 6. Toward a model-agnostic fix (see `docs/LCF_model_agnostic.md`)
The diagnosis points to a concrete, model-agnostic redesign:
1. **Single best sub-layer, chosen by held-out probe accuracy** (data-driven per model) instead of mixing layers 10–30 (which dilutes the signal to chance).
2. **Supervised probe direction** (logistic weight vector, held-out validated) instead of the weak centroid-difference V.
3. **Norm-relative intervention**: shift `h ← h + α·‖h‖·ŵ` so the step auto-scales to each model's hidden-state magnitude, instead of the fixed absolute η that transfers poorly across models.
4. **Confidence-gated**: only intervene when the probe flags the rep as invalid.

This is implemented (`lcf/lcf_impl/lcf_v2.py`, `lcf_v2_eval.py`) and evaluated.

**Result — the redesign does NOT rescue LCF (a second, deeper negative).** Sweeping the norm-relative strength α on fallacy identification (`results/lcf_v2_eval.txt`):

| α | Qwen3 Acc / ΔProb | Llama2 Acc / ΔProb |
|---|---|---|
| 0 (baseline) | 31.9 / 3.96 | 39.2 / 4.85 |
| 0.5 | 30.4 / 3.93 | 40.2 / 4.71 |
| 1 | 29.9 / 3.91 | 39.2 / 4.35 |
| 2 | 30.9 / 4.06 | 32.8 / 3.37 |
| 4 | 28.4 / 2.40 | 30.4 / 1.85 |

Even using the **single best sub-layer (0.82-separable) supervised direction** with scale-free intervention, shifting the rep along the logic-validity direction gives **no consistent improvement** (roughly neutral at small α, degrading at large α) on either model. So **separability ≠ controllability**: the logic direction is (weakly) *decodable* but is **not a causal lever** for the model's logical behaviour via additive intervention.

**Final verdict (refined).** LCF on logic-validity faces a *double* limitation: (1) the signal is weak (0.82 best vs. 0.95 for a semantic attribute like suicide risk, §3a-bis), and (2) even that signal is not causally controllable by representation shifting (this section). The paper's reported near-perfect control (96.56% Valid-Trained) is therefore implausible under faithful re-implementation and is best explained by its unauditable discriminator + unstated choices (§3d). The honest, generalizable finding of this project is that **representation-editing of logic-validity does not reproduce as a reliable, model-agnostic intervention** — a useful negative result, not a fixable bug.
