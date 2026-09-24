# Post-Thesis Extension: Does Domain Adaptation Explain Edge-Dominated Uncertainty?

> **Status:** Phase 1 complete. Phase 2 (seed replication) in progress —
> the Phase 1 result should not be cited until Phase 2 confirms it. See
> [Status and caveats](#status-and-caveats).

Work carried out **after** the thesis was submitted. Not part of the
examined document.

Full write-up: [`EXTENSION_REPORT.md`](EXTENSION_REPORT.md)

---

## The question

The thesis reports that diffusion-ensemble variance localises pathology
only weakly, that peak variance never falls inside an annotated lesion box
(0 of 547 images), and that Cardiomegaly is anti-correlated with its
annotation (AUROC 0.450). It attributes this to **edge dominance**:
variance concentrates on anatomical boundaries rather than pathology.

The obvious objection was never tested. RoentGen-v2 was used zero-shot on
VinDr-CXR, so a critic could say:

> The maps may be edge-dominated because the generator was out of domain,
> not because generative variance is intrinsically edge-driven.

This extension fine-tunes the generator on in-domain data and asks whether
edge dominance survives.

**Scope note.** §4.2 of the thesis frames zero-shot use as a deliberate
design choice, and the Limitations section does not list domain adaptation
as an unmeasured factor. This tests an assumption the thesis *defended*,
rather than patching an admitted weakness.

---

## Phase 1 — complete

### What was built

LoRA fine-tuning of the diffusion UNet: rank 8, α 8, attention projections
only, 1,659,904 trainable parameters (**0.191%** of the UNet), 2,000
train-split images, 1,500 steps, ~16 min on one T4.

**Training captions are label-free** (`"a chest x-ray"`, identical for every
image) while generation prompts keep the thesis convention
(`"a chest x-ray showing {finding}"`). This is deliberate: label-carrying
captions would encode pathology information in the weights, confounding
domain adaptation with newly injected label supervision. Because generation
prompts are identical across arms, the prompt convention cancels in the
paired comparison. Verified: 0 of 2,000 training images appear in the test
split.

### A new metric

Per-image Spearman correlation between a Sobel edge map and the variance
map, turning the thesis's qualitative edge-dominance claim into a number.
Validated against known cases (variance = edges → ρ 1.00; noise → 0.00;
lesion-dominated → 0.38).

**Baseline: pooled ρ = 0.466 across all 547 held-out images.**

### The finding that inverted the experiment

Edge-dominance **positively** predicts localisation AUROC, in all five
classes, surviving control for lesion size (partial ρ +0.24 to +0.75, four
of five p < 0.05).

Images whose variance tracks anatomical edges more strongly localise
*better*, not worse. The reading: edges act as a **thoracic bulk prior**.
Boxes sit on anatomy; background and soft tissue are edge-poor and box-free,
so an edge-tracking map earns above-chance AUROC without detecting any
lesion.

This inverted the hypothesis and turned the study into a **calibration
test** with per-class slopes fitted on baseline data before the
intervention existed, then frozen.

### Results (n = 150 prompt-matched pairs)

| | Cardiomegaly (n=107) | Consolidation (n=43) |
|---|---|---|
| Δρ | **+0.046** [0.040, 0.052] | **+0.061** [0.051, 0.073] |
| ΔAUROC observed | −0.041 [−0.046, −0.035] | +0.015 [0.004, 0.027] |
| ΔAUROC predicted | +0.009 | +0.018 |
| **Residual** | **−0.050**, p<0.0001 | **−0.002**, p=0.57 |
| Pointing-hit | 0/150 both arms | 0/150 both arms |

**Fine-tuning increased edge-dominance in both classes** — the opposite of
the domain-gap hypothesis. Adapting to VinDr made variance track anatomical
boundaries *more*.

**Consolidation: the prediction held.** A slope fitted before the
intervention predicted its effect to within 0.002 AUROC.

**Cardiomegaly: the prediction failed.** Predicted +0.009, observed −0.041.
Mechanistically coherent: its box encloses the smooth cardiac silhouette
with strong edges *outside* it, so pushing variance onto edges pushes it
further out of the box.

The headline: **an observational slope did not transfer to an
intervention.** The within-class slopes were fitted on natural variation
between images; fine-tuning imposed a systematic shift in map character.
The intervention moved along the *between*-class relationship (ρ = −0.40),
not the within-class one.

**Peak variance still never landed in a lesion.** The thesis's core
negative result survives its hardest available test.

---

## Phase 2 — in progress

### Why

The Phase 1 result rests on **one LoRA training run** (seed 42). Earlier
work in this same project retracted a downstream finding (−0.084 AUROC,
bootstrap CI excluding zero, surviving multiple-comparison correction)
after a five-seed replication showed every class flipping sign, with
training-seed SD (0.038) exceeding test-sampling SD (0.028).

Applying a weaker standard here than the one that produced that retraction
would be inconsistent. Phase 2 applies the same protocol.

### Design

Five LoRA training seeds {42, 7, 123, 2024, 31337} — the same set as the
earlier replication. Only `SEED` varies: adapter init, shuffle order,
timestep sampling, noise draws. The **same 2,000-image manifest** is reused
across all seeds, isolating training stochasticity from data selection.

**Preregistered on OSF before running.**
See [`PREREGISTRATION_seed_replication.md`](PREREGISTRATION_seed_replication.md).

Decision rules differ by hypothesis, because one claims an effect and one
claims a prediction holds:

| | Cardiomegaly (H2, effect claimed) | Consolidation (H3, null claimed) |
|---|---|---|
| Pass | consistent sign across all seeds AND \|mean\| > 2·SD | seed-level 95% CI includes zero |
| Fail | sign flips on any seed → **withdrawn** | CI excludes zero |

A sign flip is disqualifying for H2 regardless of the mean. It is
*expected* for H3 and not disqualifying — applying the effect rule there
would reject the hypothesis for behaving exactly as a true null should.

### Cost

~7.7 GPU-hours: 4 × 16 min training, 4 × 99 min generation (150 images
each), ~30 min CPU analysis. Fits inside a week of Kaggle's free quota
across two sessions.

---

## Status and caveats

**Do not cite the Phase 1 Cardiomegaly result until Phase 2 completes.**
It is a single training run, and this project has already retracted one
single-run finding for exactly that reason.

Other limitations, in full in [`EXTENSION_REPORT.md`](EXTENSION_REPORT.md) §7:

- **Two classes, 150 images.** Consolidation n=43 after prompt-matched
  filtering, below the 54 the frozen slope was fitted on.
- **One LoRA configuration.** Rank 8, one learning rate, 1,500 steps. No
  evidence on whether adaptation *capacity* changes the conclusion — full
  fine-tuning and a rank sweep are planned (report §8).
- **Modest adaptation.** Loss fell 8.6% by step 250; RoentGen-v2 was
  already chest-X-ray-adapted, so this is refinement, not transfer.
- **Generation-seed variance unmeasured.** `seed_base=0` throughout.
- **Prompt conditioning.** Generation conditions on the class name. The
  *weights* are label-free; the estimator is not entirely. Thesis §4.2
  describes the estimator as label-free, which is stronger than the prompt
  convention supports.

---

## Reproducing

| Notebook | Scripts | Output |
|---|---|---|
| A (GPU) | `step2_caption_manifest.py`, `step3_lora_training.py` | `train_manifest.csv`, `lora_cxr/` |
| B (CPU) | `step0_edge_correlation_baseline.py` | `edge_correlation_baseline.csv` |
| C (GPU) | `metric_reproducibility_check.py`, `cell6_scope_targets.py`, `cell7_generate_finetuned.py` | `finetuned_maps/` |
| D (CPU) | `notebookD_analysis.py` | `analysis/` |
| E (GPU) | `multiseed_train_and_generate.py` | `lora_seeds/`, `finetuned_maps_seeds/` |
| F (CPU) | `multiseed_analysis.py` | `multiseed/` |

### Methodological controls, all asserted in code

- **Label-free training caption** — asserted `nunique() == 1`
- **Zero test-split leakage** — asserted against `vindr_cxr_test.csv`
- **Prompt-matched pairing** — 207 → 150 pairs; baseline maps live in
  per-class datasets with per-class prompts, so multi-label images can
  differ in prompt between arms, which would masquerade as a LoRA effect
- **Frozen slopes** — fitted on baseline only, never refitted on combined
  data
- **Identical generation parameters** across arms; only UNet weights differ

### On reproducibility checking

An initial check thresholded on max relative pixel difference and reported
"does not reproduce" (worst 0.4538) — while map correlations in the same
output were 0.9897–0.9999. A worst-pixel statistic on a map spanning
5e-06 to 0.103 is dominated by fp16 jitter in the tail and says nothing
about rank-based metrics over 262k pixels.

Re-checked on the quantities that actually enter the comparison: worst
|ΔAUROC| = **0.0003**, ~100× below the smallest predicted effect. This is
what licensed comparing new fine-tuned maps against existing baseline maps
instead of regenerating both arms.

*Caveat:* `seed_base` is fixed, so this bounds environment nondeterminism,
not generation-seed variance.

---

## Data and licensing

VinDr-CXR is credentialed PhysioNet data. **This repository does not
redistribute it.** Obtain access from the original provider. Check the data
use agreement before publishing figures containing radiographs, and check
RoentGen-v2's licence before redistributing derived LoRA weights.
