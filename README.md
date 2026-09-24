# Quantifying Epistemic Uncertainty in Diffusion Models for Chest Radiography

Code and analysis for an MSc thesis investigating whether pixel-wise
variance across inversion-anchored diffusion reconstructions can serve as
a spatially meaningful uncertainty signal for chest radiograph pathology
localisation, and whether that signal has downstream utility for training
data curation.

**Dataset:** VinDr-CXR (5 of 22 annotated local findings: Pneumothorax,
Consolidation, Nodule/Mass, Cardiomegaly, Atelectasis)
**Method:** DDIM-inverted latent anchor, N=20 stochastic reverse-diffusion
reconstructions, pixel-wise variance as the uncertainty map
**Headline result:** a real but modest and edge-dominated spatial signal;
no downstream benefit from uncertainty-based data curation; a pre-registered
replication that retracted one of the study's own significant findings

> **A note on reproducibility before anything else:** this repository
> documents a real research process, including a dead end. Stage 4's
> original MC-Dropout comparison and one Stage 7 augmentation result were
> both later shown to be invalid or unsupported by replication (see
> [Known Issues and Retractions](#known-issues-and-retractions) below).
> Both are kept in the repository and in the pipeline history rather than
> deleted, because the correction process is itself a result the thesis
> reports on (Chapter 6, Chapter 7 §7.4). If you are here to reproduce a
> specific number from the thesis, check that table's provenance against
> that section before trusting a script's default output.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Data Requirements](#data-requirements)
- [Pipeline Overview](#pipeline-overview)
- [Stage-by-Stage Guide](#stage-by-stage-guide)
- [Reproducing the Figures and Tables](#reproducing-the-figures-and-tables)
- [Known Issues and Retractions](#known-issues-and-retractions)
- [Citation](#citation)
- [License](#license)

---

## Repository Structure

```
.
├── README.md                          <- this file
│
├──Stage+ path to notebooks/                         <- Kaggle notebooks, run in order
│   ├── Data cleaning + Transform_data.py           
│   ├── stage2_baseline/ablation_uncertainty + CXRay_Generation/Main-Inversion-generator.ipynb           
│   ├── Stage 3 evaluations + Calibration and spatial Evaluation\stage3_cross_class_aggregator.py
         
│   ├── Mc Dropout logit/GradCam + DenseNet MC Dropout uncertainty generation/stage4-mc-dropout-logits.ipynb 
│   ├── Seed Rep/Thresholdponse + Seed Replication and threshold dose study/seed-replication-and-threshold-response-stage-6.ipynb
│   └── stage-7-densenet-evaluation + DenseNet evaluations/stage-7-densenet-evaluation.ipynb

├── extensions/                         <- POST-THESIS work, not examined
│   ├── README.md                       # overview + status
│   ├── PREREGISTRATION_seed_replication.md
│   ├── ext-dfinet-stage0.ipynb         #edge_correlation_baseline code
│   ├── edge_correlation_baseline.csv   #stage 0 output
│   ├── ext-dfint-step1-3.ipynb         #lora_training +scope_targets +generate_finetuned
│   ├── ext-step9-11-evaluation.ipynb   #notebookD_analysis
│   ├──# Below to be added#
│   ├── metric_reproducibility_check.py
│   ├── notebookD_analysis.py
│   ├── multiseed_train_and_generate.py
│   └── multiseed_analysis.py

──────────



---

## Environment Setup

Developed and run on Kaggle notebooks (P100/T4 GPU). To reproduce locally or on another platform:

```bash
pip install torch torchvision diffusers transformers \
            numpy pandas scipy scikit-learn statsmodels \
            matplotlib h5py tqdm --break-system-packages
```

Or via `requirements.txt`:

```bash
pip install -r requirements.txt --break-system-packages
```

**Kaggle-specific note:** every script in `drivers/` and `plotting/` was
written assuming Kaggle's `/kaggle/input/` and `/kaggle/working/`
convention, with `CHANGE-ME` placeholders in dataset paths that must be
updated to match your actual dataset mount points (visible in the Kaggle
notebook sidebar after attaching a dataset). Each script's `preflight()`
or `inspect_*()` function checks paths exist before spending GPU time —
run that step alone first, always.

**Bibliography compiler:** the thesis requires `biber`, not classic
`bibtex`. In Overleaf: Menu → Settings → Bibliography compiler → Biber.

---

## Data Requirements

- **VinDr-CXR** — 18,000 images (15,000 train / 3,000 test), 22 local
  finding classes + 6 global diagnoses. Available via PhysioNet or the
  Kaggle VinBigData Chest X-ray Abnormalities Detection competition.
  This project uses 5 of the 22 local findings.
- **Image archive**: DICOM images windowed and resized to a common
  resolution, stored as a single HDF5 file, keyed by `image_id`.
  `plotting/plot_variance_heatmap_examples.py` includes an `inspect_h5()`
  function to verify the archive's actual key structure before use — the
  assumed `h5file[image_id]` convention has not been independently
  confirmed against every possible export of this dataset.
- **Annotation CSV**: one row per image, with all boxes for that image
  packed into a `boxes` column as a Python-literal list of dicts
  (`class_name`, `x_min`, `y_min`, `x_max`, `y_max`, `rad_id`, `class_id`).
  Also carries `rows`/`columns` (original DICOM height/width — note this
  naming, not `height`/`width`) and `split`. `drivers/stage9_final.py`
  includes an `explode_boxes()` function that unpacks this into the
  long format the rest of the pipeline expects.
- **Pretrained generative backbone**: a latent diffusion model fine-tuned
  for chest radiograph synthesis (used zero-shot, no further fine-tuning).
- **DenseNet-121**: ImageNet-pretrained weights, pinned to a fixed local
  copy in a Kaggle dataset rather than downloaded at run time (Kaggle
  batch-commit notebooks run with internet disabled by default; see
  [Known Issues](#known-issues-and-retractions)).

---

## Pipeline Overview

```
                    ┌─────────────────────────────────────────┐
                    │  Stage 1-2: Data prep, LDM setup,        │
                    │  baseline uncertainty map (train split)  │
                    └───────────────────┬───────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────────┐
                    │  Stage 3: Stochasticity ablation           │
                    │  (η, step count, N — dose-response)        │
                    └───────────────────┬───────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
┌───────▼────────┐          ┌───────────▼────────────┐      ┌───────────▼──────────┐
│ Stage 4:        │          │ Stage 6: Downstream     │      │ Approach A / Stage 9: │
│ MC-Dropout +     │          │ classifier training      │      │ HELD-OUT map          │
│ Grad-CAM         │          │ (none/all/filtered/      │      │ evaluation — resolves │
│ (train split —   │          │  random_matched arms,    │      │ Stage 4's image-level │
│ CONFOUNDED,       │          │ seed & threshold sweeps) │      │ confound (but NOT the │
│ see below)        │          └───────────┬────────────┘      │ label-supervision one)│
└──────────────────┘                       │                   └───────────────────────┘
                                            │
                            ┌───────────────▼───────────────┐
                            │ Stage 7: Classifier evaluation │
                            │ (AUROC/AUPRC, DeLong, McNemar, │
                            │  BH correction)                │
                            └───────┬───────────────┬───────┘
                                    │               │
                    ┌───────────────▼───┐   ┌───────▼────────────┐
                    │ Approach C:         │   │ Approach D:         │
                    │ 5-seed replication  │   │ threshold dose-     │
                    │ → RETRACTS a Stage  │   │ response sweep      │
                    │ 7 finding           │   │ (25/50/75/90 pct)   │
                    └─────────────────────┘   └─────────────────────┘
```

---

## Stage-by-Stage Guide

### Stages 1-3 — Generation and ablation (train split)
`CXRay_Generation/Main-Inversion-generator.ipynb` (Select class to generate)
Dataset preparation, DDIM-inversion-anchored ensemble generation (N=20 per
image, five classes), and the stochasticity configuration ablation (η ∈
{0, 0.5, 1.0}; reduced step count; N=10 negative control). Produces the
dose-response results reported in Thesis §5.3.



### Stage 4 — MC-Dropout and Grad-CAM comparators (train split)
`DenseNet MC Dropout uncertainty generation/stage4-mc-dropout-logits.ipynb` (Set logit or Grad-Cam in input driver)
Generates MC-Dropout logit variance and Grad-CAM-under-dropout maps from
the DenseNet-121 classifier, on the **training** split.

> ⚠️ **This comparison is confounded and its original conclusion
> (MC-Dropout beats diffusion "nearly everywhere") is superseded.** The
> classifier had been trained on the exact images used to evaluate it.
> See Approach A / Stage 9 below, and Thesis §3.4.3 / §5.4.

### Stage 6 — Downstream classifier training

`Seed Replication and threshold dose study/seed-replication-and-threshold-response-stage-6.ipynb`

Trains DenseNet-121 classifiers across four augmentation arms
(`none`, `all`, `filtered`, `random_matched`), producing checkpoints for:

- The original single run (`RUN_TAG="weighted"`, `TRAINING_SEED=42`,
  `VARIANCE_PERCENTILE=75.0`)
- **Approach C** — 5 training seeds `{42, 7, 123, 2024, 31337}` ×
  `{none, all}` arms
- **Approach D** — 4 filtering percentiles `{25, 50, 75, 90}` ×
  `{filtered, random_matched}` arms

Checkpoints save as `{arm}_{RUN_TAG}/best.pt`.

**⚠️ Notebook cells default to executing on run** (`if __name__ ==
"__main__": main()` fires immediately in a notebook context). The driver
scripts below set `DRIVER_MODE = True` to suppress this — always define
that flag *before* re-running this cell if pasting drivers after it.

### Stage 7 — Classifier evaluation

`DenseNet evaluations/stage-7-densenet-evaluation.ipynb`

Evaluates trained checkpoints on the held-out **test** split. Computes
per-class AUROC/AUPRC/F1, DeLong tests and stratified bootstrap CIs on
AUC differences, McNemar tests on thresholded predictions, and BH
correction. Outputs `stage7_per_class_metrics.csv` and
`stage7_statistical_tests.csv`, plus cached probability arrays
(`stage7_probs_{arm}.npz`) used by the plotting scripts.

### Approach C — Seed replication

`DenseNet evaluations/stage-7-densenet-evaluation.ipynb` (Set approach C as driver)

Evaluates the 5-seed checkpoints, builds the per-seed delta matrix and
cross-seed summary (mean, SD, t-based 95% CI at df=4), and generates the
strip-plot and mean-CI figures (Thesis Fig 6.2, 6.3).

**Compute:** 10 models × ~20 min ≈ 3.3 GPU-hours. Publishes CSVs
incrementally after each seed — 

**This is the analysis that retracts a Stage 7 finding.** See
[Known Issues](#known-issues-and-retractions).

### Approach D — Threshold dose-response

`DenseNet evaluations/stage-7-densenet-evaluation.ipynb` Set Approach D as driver
Evaluates `filtered` vs `random_matched` across four filtering
percentiles, differencing out the dataset-size effect at every point
(the control is size-matched at each threshold). Generates the
per-class + macro dose-response figure (Thesis Fig 6.1) and a per-class
linear trend test — **read the trend p-values with the same scepticism
Chapter 6 applies to everything else**: with 4 points per class, one
nominal `p < 0.05` out of 5 independent tests is what chance predicts, and
the Pneumothorax slope in this study is exactly that (BH-corrected
p = 0.200, not significant; see Thesis §6.3).

**Compute:** 8 models × ~20 min ≈ 2.7 GPU-hours.

### Approach A / Stage 9 — Held-out map evaluation

`DenseNet evaluations/testset-map-evaluation+visualizations.ipynb`

Evaluates all three uncertainty methods (diffusion variance, MC-Dropout
logit variance, MC-Dropout-Grad-CAM variance) on the **held-out test
split**, resolving Stage 4's image-level confound. Handles VinDr-CXR's
chunked `.npz` map format (`{image_id}__{variance,mean,z_T}` keys, many
images per file) and the wide-to-long annotation unpacking
(`explode_boxes()`).

**Metrics:** per-image pixel AUROC, AUPRC (with in-box pixel fraction as
baseline), Dice at map-derived percentile thresholds, pointing-hit rate.
**Statistics:** Friedman omnibus across the three methods, BH-corrected
post-hoc paired Wilcoxon tests, rank-biserial effect sizes.

**Key finding this stage produced:** removing the image-level confound did
**not** close the gap between MC-Dropout and diffusion variance — MC-Dropout
still wins in 4/5 classes at large effect sizes (Cardiomegaly:
rank-biserial = -1.00, every single held-out image favoured MC-Dropout).
The residual explanation is label supervision, not image exposure (Thesis
§3.4.3, §5.4, §7.1). This is a *correction to the original hypothesis*,
not a confirmation of it — worth reading carefully before citing this
stage's result as "the confound was fixed."

**Also confirms:** Cardiomegaly's localisation AUROC on held-out data is
**0.450, below chance** (CI [0.442, 0.459], 331 images) — not merely a null
result but a systematic anti-signal. And the single highest-variance
pixel fell inside an annotated box in **zero** of 547 held-out test images
across all five classes (Thesis §5.2).

---

## Post-Thesis Extension

Work carried out after submission, testing whether the edge-dominated
uncertainty reported in the thesis is a domain-gap artifact or intrinsic
to DDIM sampling. LoRA fine-tuning of the generator **increased**
edge-dominance rather than reducing it, and peak variance still landed
inside a lesion box in 0 of 150 images.

> **Phase 2 (seed replication) in progress.** The Phase 1 Cardiomegaly
> result rests on one LoRA training run and should not be cited until
> replication completes.

See [`Extension/Extension_README.md`](Extension/Extension_README.md) for the summary.


## Reproducing the Figures and Tables

| Thesis item | Script | Depends on |
|---|---|---|
| Table 5.1, Fig 5.2 | `plotting/plot_localisation_auroc.py` | `stage9_summary.csv` |
| Fig 5.1 | `plotting/plot_variance_heatmap_examples.py` | H5 image archive, `DenseNet evaluations/testset-map-evaluation+visualizations.ipynb`'s loaders, annotation CSV |
| Table 5.2, 5.3 (method comparison) | `DenseNet evaluations/testset-map-evaluation+visualizations.ipynb` (`main()` output) | Stage 9 `.npz` maps |
| Table 6.1 (per-arm metrics) | Stage 7 notebook | Stage 6 checkpoints |
| Fig 6.1, Table 6.dose-response | `DenseNet evaluations/stage-7-densenet-evaluation.ipynb` | Approach D checkpoints |
| Fig 6.2, 6.3, Table 6.2 (cross-seed) | `DenseNet evaluations/stage-7-densenet-evaluation.ipynb` | Approach C checkpoints |
| Fig A.1 (ROC/PR by arm) | `plotting/plot_roc_pr_by_arm.py` | `stage7_probs_{arm}.npz` |
| Fig A.2 (AUC bars by arm) | `plotting/plot_auc_bars_by_arm.py` | `stage7_probs_{arm}.npz` |
| Table A.3 (resolution audit) | `DenseNet evaluations/testset-map-evaluation+visualizations.ipynb` (`native_res` output) | Stage 9 `.npz` maps |

**Provenance note for Fig A.1/A.2:** the "original single run" probability
arrays were never uploaded as their own file — they don't need to be.
Approach C's `seed42` checkpoint and Approach D's `pct75` checkpoint are
bit-identical to the original run (verified to 6 decimal places against
`stage7_per_class_metrics.csv` before either plotting script trusted
them). If you retrain from scratch rather than reusing these archives,
re-verify this before assuming the same file-reuse shortcut holds.

---





## Known Issues and Retractions

This section exists because the thesis argues, at length, that
single-run results without a measured noise floor should not be trusted —
and it would be inconsistent to hide the two places in this repository's
own history where that happened.

1. **Stage 4's MC-Dropout comparison (train split) is confounded and
   superseded.** The classifier had trained on the images it was
   evaluated on. Fixed by Approach A / Stage 9, which evaluates on the
   held-out split — but the fix changed the explanation, not the
   direction of the result (MC-Dropout still wins).

2. **A Stage 7 single-run finding was retracted by Approach C.** The
   original run reported unfiltered augmentation degrading Atelectasis
   AUROC by 0.084, with a bootstrap CI excluding zero and significance
   surviving BH correction. Five-seed replication showed: (a) every
   class's sign flips across seeds, (b) seed-to-seed SD (0.038) exceeds
   the bootstrap-implied test-sampling SD (0.028), and (c) the specific
   seed that produced the original estimate was itself a statistical
   outlier — 2.86 SD above the other four seeds on the `none` arm alone.
   **Do not cite the −0.084 figure as a finding of this study.**

3. **Grad-CAM-under-dropout is not a clean upper bound.** It is the
   variance of Grad-CAM attribution across MC-Dropout passes, not a
   single deterministic saliency map — a subtly different and noisier
   quantity than classical Grad-CAM. It beats diffusion in some classes
   and loses in others, and loses to MC-Dropout logit variance
   everywhere. Do not describe it as a "skyline" that diffusion is
   expected to lose to by design.

4. **Kaggle batch-commit notebooks run with internet disabled by
   default.** `torchvision.models.densenet121(pretrained=True)` will fail
   with `gaierror: Temporary failure in name resolution` unless either
   internet is enabled in notebook settings, or ImageNet weights are
   pre-cached from a pinned dataset (recommended — see Environment Setup).

5. **Generation conditions on the class name.** The prompt is
   `f"a chest x-ray showing {finding.lower()}"`, so the generator receives
   class-name conditioning at inference. The *weights* are label-free
   (zero-shot, never fine-tuned on VinDr labels); the estimator is not
   entirely. Thesis §4.2 describes the estimator as label-free, which is
   stronger than the prompt convention supports.

---

---

## Citation

```bibtex
@mastersthesis{P.Chandratre2026,
  author = {Pranav Ganesh Chandratre},
  title  = {Quantifying Epistemic Uncertainty in Diffusion Models for Chest Radiography},
  school = {University of Birmingham},
  year   = {2026}
}
```

## License

MIT License
