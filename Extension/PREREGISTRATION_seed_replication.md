# Preregistration: Seed Replication of LoRA Fine-Tuning Effects on Diffusion Uncertainty Maps

**File this on OSF BEFORE running anything. Timestamp is the point.**

Author: [NAME]
Date filed: [DATE]
Status: analysis plan fixed in advance; no seed-replication data yet collected

---

## 1. Background and what is already known

A prior single-run experiment (LoRA seed 42) found, on 150 prompt-matched
held-out chest radiographs:

| Quantity | Cardiomegaly (n=107) | Consolidation (n=43) |
|---|---|---|
| Δρ (edge-dominance) | +0.0455 [0.0396, 0.0515] | +0.0613 [0.0508, 0.0726] |
| ΔAUROC observed | −0.0407 [−0.0464, −0.0349] | +0.0153 [0.0038, 0.0265] |
| ΔAUROC predicted | +0.0087 | +0.0176 |
| Residual | **−0.0495**, p<0.0001 | **−0.0023**, p=0.57 |

Predictions came from per-class OLS slopes of AUROC on edge_rho, fitted on
baseline data before any fine-tuning existed and frozen since:

    Cardiomegaly 0.1914   Consolidation 0.2867

**Those slopes are NOT refitted in this study.** They remain the frozen
constants above.

## 2. Why this replication

The prior result rests on ONE LoRA training run. Earlier work in the same
project retracted a downstream finding (−0.084 AUROC, bootstrap CI
excluding zero, surviving multiple-comparison correction) after a five-seed
replication showed every class flipping sign and training-seed SD (0.038)
exceeding test-sampling SD (0.028).

Applying a weaker standard here than the standard that produced that
retraction would be inconsistent. This replication applies the same
protocol.

## 3. Design

**Varied:** LoRA training seed only. Five values: 42 (already run), 7, 123,
2024, 31337 — the same set used in the earlier replication.

`SEED` in the training script controls adapter initialisation, dataloader
shuffle order, timestep sampling and noise draws.

**Held fixed:**
- Training manifest: the same 2,000 train-split images, identical across
  all seeds (selection seed 42). This isolates training stochasticity from
  data selection.
- Caption: `"a chest x-ray"`, label-free, identical for every image.
- LoRA config: rank 8, α 8, `to_q/to_k/to_v/to_out.0`, lr 1e-4, 100-step
  warmup, batch 2 × accum 4, 1,500 steps, fp16.
- Generation: 30 inversion steps, 30 inference steps, N=20, η=1,
  guidance=1.0, `seed_base=0`, same prompt convention, same test split.
- Evaluation: same box masks, same prompt-matched pairing, same frozen
  slopes.

**Explicitly NOT measured, and stated as limitations:** selection noise
(the augmentation/training draw is fixed) and generation-seed variance
(`seed_base` fixed at 0).

## 4. Primary hypotheses

**H1 (edge-dominance).** Δρ > 0 in every seed, for both classes.

**H2 (Cardiomegaly residual).** The residual is negative in every seed, and
|mean residual| > 2 × SD across seeds.

**H3 (Consolidation calibration).** The residual's seed-level 95% CI
includes zero.

**H4 (pointing-hit).** Peak variance falls inside an annotated box in 0 of
150 images, in every seed and both arms.

## 5. Analysis plan

Per seed, compute per-image Δρ, observed ΔAUROC, predicted ΔAUROC
(frozen slope × Δρ), and residual. Then across seeds, per class:

- mean, SD, min, max of each quantity
- 95% CI on the mean, t-based, df = n_seeds − 1
- sign consistency across seeds

The SD across seeds is the empirical training-noise floor and is the
quantity the single-run estimate must be judged against — not zero, and not
the within-run bootstrap CI, which captures image sampling only.

## 6. Decision rules, fixed in advance

The rule differs by hypothesis, because H2 claims an effect exists and H3
claims a prediction holds. Applying a sign-consistency rule to H3 would
reject it for behaving exactly as a true null should.

### H2 — Cardiomegaly residual (claiming an effect)

| Outcome | Conclusion |
|---|---|
| Residual sign consistent across all 5 seeds AND \|mean\| > 2×SD | Effect survives replication; report as a finding |
| Sign flips on any seed | Effect is within training noise; **the single-run result is withdrawn** |
| Sign consistent but \|mean\| < 2×SD | Inconclusive; report as suggestive with the noise floor stated |

A sign flip is disqualifying regardless of the mean. This mirrors the rule
applied in the earlier replication that produced the retraction.

### H3 — Consolidation residual (claiming the prediction holds)

| Outcome | Conclusion |
|---|---|
| Seed-level 95% CI on the mean residual includes zero | Prediction holds; calibration confirmed |
| CI excludes zero | Prediction fails; the frozen slope does not transfer |

Sign flipping across seeds is **expected** here and is not disqualifying —
it is what a residual genuinely centred on zero looks like.

### H1 and H4

H1 (Δρ > 0) and H4 (pointing-hit = 0) are reported per seed. Any seed
violating either is reported explicitly rather than pooled away.

## 7. What is reported regardless of outcome

All five seeds, per class, whatever they show. The seed-level SD is
reported alongside every effect estimate. No seed is excluded post hoc for
any reason.

If H2 fails, the Cardiomegaly claim is withdrawn and the paper reports the
null with the measured noise floor.

## 8. Deviations

Any departure from this plan will be reported as a deviation, with the
reason, in the final write-up.
