"""
stage2_wilcoxon_variance_test.py
============================================================================
Kaggle Notebook, CPU-only.

Stage 2 statistical validation: for each abnormality class with ablation
data available, runs a paired Wilcoxon signed-rank test comparing per-image
MEAN VARIANCE at baseline settings vs. each ablation setting, on the
identical image_id set.

This is deliberately narrow in scope: it never touches radiologist
bounding boxes or ground truth — that comparison is Stage 3
(calibration_spatial_evaluation_pipeline.py), a separate script. This
script only asks: "does changing eta / N / steps shift the distribution of
per-image mean variance, within the same class, on the same images?"

INPUTS (edit CONFIG in Cell 1):
  For each class you want to test, a "baseline" dataset directory plus one
  directory per ablation setting you have available for that class. Each
  directory holds that setting's variance maps for ONE class, in EITHER of
  two supported layouts (auto-detected per directory, see
  `index_variance_dataset()`):
    (a) chunked .npz files (e.g. uncertainty_chunk_000.npz) containing keys
        named "{image_id}__variance", OR
    (b) one .npy file per image, named "{image_id}.npy" or
        "{image_id}__variance.npy"
  Mixing both layouts within a single directory is not supported.

OUTPUT:
  /kaggle/working/stage2_wilcoxon_results.csv — one row per
  (class, ablation_setting) comparison: n_images, Wilcoxon statistic,
  p-value, mean-of-means for each arm, median paired difference.

Ablation settings this script is parameterized for (edit CONFIG to match
whichever classes actually have these datasets ready):
  baseline  : eta=1.0, N=20, G=1, steps=30
  eta_0.5   : eta=0.5, N=20, G=1, steps=30
  n_10      : eta=1.0, N=10, G=1, steps=30
  steps_15  : eta=1.0, N=20, G=1, steps=15
"""

# ============================================================================
# CELL 0 — Imports (no installs needed: scipy/numpy/pandas/tqdm are
# preinstalled on Kaggle)
# ============================================================================
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm.auto import tqdm


# ============================================================================
# CELL 1 — CONFIG (edit this to match your actual dataset/directory layout)
# ============================================================================
# One entry per class. Only include the ablation-setting keys you actually
# have data for — the script skips any (class, setting) pair that isn't
# present rather than erroring, so partially-completed classes are fine.
#
# EXAMPLE — replace these paths with your real Kaggle dataset paths:
CLASS_ABLATION_DATASETS = {
    "Pneumothorax": {
        "baseline": "/kaggle/input/ddim-variance-pneumothorax/baseline",
        "eta_0.5":  "/kaggle/input/ddim-variance-pneumothorax/eta_0.5",
        "n_10":     "/kaggle/input/ddim-variance-pneumothorax/n_10",
        "steps_15": "/kaggle/input/ddim-variance-pneumothorax/steps_15",
    },
    # Add more classes here as their ablation runs complete, e.g.:
    # "Consolidation": {
    #     "baseline": "/kaggle/input/ddim-variance-consolidation/baseline",
    #     "eta_0.5":  "/kaggle/input/ddim-variance-consolidation/eta_0.5",
    #     ...
    # },
}

# Ablation settings to test against "baseline", in order. A (class, label)
# pair is only run if CLASS_ABLATION_DATASETS[class] actually has that key.
ABLATION_SETTINGS = ["eta_0.5", "n_10", "steps_15"]

# Human-readable config strings, stamped into the output CSV for traceability.
ABLATION_CONFIG_DESCRIPTIONS = {
    "baseline": "eta=1.0, N=20, G=1, steps=30",
    "eta_0.5":  "eta=0.5, N=20, G=1, steps=30",
    "n_10":     "eta=1.0, N=10, G=1, steps=30",
    "steps_15": "eta=1.0, N=20, G=1, steps=15",
}

OUTPUT_DIR = "/kaggle/working"


# ============================================================================
# CELL 2 — Dataset indexing + per-image mean variance
# ============================================================================
def index_variance_dataset(dataset_dir):
    """
    Build {image_id: (path, key_or_None)} for ONE dataset directory
    (one class, one ablation setting), auto-detecting the storage layout:
      (a) chunked .npz files containing keys "{image_id}__variance"
      (b) one .npy file per image: "{image_id}.npy" or
          "{image_id}__variance.npy"
    Raises if the directory is missing, empty, or contains neither
    recognized layout.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    index = {}
    npz_files = sorted(dataset_dir.glob("*.npz"))
    npy_files = sorted(dataset_dir.glob("*.npy"))

    if npz_files:
        for p in npz_files:
            with np.load(p) as data:
                for key in data.files:
                    if key.endswith("__variance"):
                        image_id = key[: -len("__variance")]
                        index[image_id] = (p, key)
        if not index:
            raise ValueError(
                f"{dataset_dir} contains .npz files but none have keys "
                f"ending in '__variance' — check the file format matches "
                f"what ddim_epistemic_uncertainty_pipeline.py produces."
            )
    elif npy_files:
        for p in npy_files:
            stem = p.stem
            image_id = stem[: -len("__variance")] if stem.endswith("__variance") else stem
            index[image_id] = (p, None)
    else:
        raise FileNotFoundError(
            f"No .npz or .npy files found in {dataset_dir} — nothing to index."
        )

    return index


def load_variance_array(image_id, index):
    """Load a single image's (512,512) variance array given its index entry."""
    path, key = index[image_id]
    if key is not None:  # chunked .npz layout
        with np.load(path) as data:
            return data[key].astype(np.float32)
    else:  # one .npy file per image
        return np.load(path).astype(np.float32)


def compute_per_image_mean_variance(index, desc="Computing per-image mean variance"):
    """Returns {image_id: scalar mean variance} for every image_id in index."""
    means = {}
    for image_id in tqdm(sorted(index.keys()), desc=desc):
        arr = load_variance_array(image_id, index)
        means[image_id] = float(arr.mean())
    return means


# ============================================================================
# CELL 3 — Paired image_id validation
# ============================================================================
def validate_matching_image_ids(baseline_means, ablation_means, baseline_label, ablation_label):
    """
    Wilcoxon signed-rank requires a fully paired, identical sample set.
    Raises a descriptive ValueError (rather than silently intersecting)
    if the two datasets don't cover exactly the same image_ids.
    """
    baseline_ids = set(baseline_means.keys())
    ablation_ids = set(ablation_means.keys())

    if baseline_ids != ablation_ids:
        only_baseline = sorted(baseline_ids - ablation_ids)
        only_ablation = sorted(ablation_ids - baseline_ids)

        def _preview(ids, limit=10):
            shown = ids[:limit]
            suffix = f" ... (+{len(ids) - limit} more)" if len(ids) > limit else ""
            return f"{shown}{suffix}"

        raise ValueError(
            f"image_id mismatch between '{baseline_label}' "
            f"({len(baseline_ids)} images) and '{ablation_label}' "
            f"({len(ablation_ids)} images) — cannot run a paired test.\n"
            f"  Only in '{baseline_label}': {len(only_baseline)} image(s): "
            f"{_preview(only_baseline)}\n"
            f"  Only in '{ablation_label}': {len(only_ablation)} image(s): "
            f"{_preview(only_ablation)}\n"
            f"Resolve this (regenerate the missing images, or deliberately "
            f"restrict to the intersection and document that decision) "
            f"before re-running — do not silently drop mismatched images."
        )

    return sorted(baseline_ids)


# ============================================================================
# CELL 4 — Paired Wilcoxon signed-rank test
# ============================================================================
def run_paired_wilcoxon(baseline_means, ablation_means, image_ids):
    """
    Runs scipy.stats.wilcoxon as a paired test between baseline and
    ablation per-image mean variance, on the given (already-validated,
    identical) image_id set.
    """
    baseline_vec = np.array([baseline_means[i] for i in image_ids], dtype=np.float64)
    ablation_vec = np.array([ablation_means[i] for i in image_ids], dtype=np.float64)
    diffs = ablation_vec - baseline_vec

    if np.all(diffs == 0):
        raise ValueError(
            "All paired differences are exactly zero — the Wilcoxon test is "
            "undefined here (baseline and ablation per-image mean variance "
            "are identical for every image). This usually means the two "
            "dataset directories point at the same underlying files by "
            "mistake — check CLASS_ABLATION_DATASETS."
        )

    try:
        statistic, p_value = wilcoxon(baseline_vec, ablation_vec)
    except ValueError as e:
        # scipy raises if, after its default zero-difference handling, too
        # few non-zero-difference pairs remain to compute a statistic.
        raise ValueError(
            f"scipy.stats.wilcoxon could not compute a statistic for this "
            f"comparison (n={len(image_ids)} paired images): {e}"
        ) from e

    return {
        "n_images": len(image_ids),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "baseline_mean_of_means": float(baseline_vec.mean()),
        "ablation_mean_of_means": float(ablation_vec.mean()),
        "median_paired_diff": float(np.median(diffs)),
        "mean_paired_diff": float(diffs.mean()),
    }


# ============================================================================
# CELL 5 — Orchestration
# ============================================================================
def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for class_name, datasets in CLASS_ABLATION_DATASETS.items():
        if "baseline" not in datasets:
            warnings.warn(f"No 'baseline' dataset configured for {class_name} — skipping class.")
            continue

        print(f"\n=== {class_name} ===")
        print(f"Indexing baseline: {datasets['baseline']}")
        baseline_index = index_variance_dataset(datasets["baseline"])
        baseline_means = compute_per_image_mean_variance(
            baseline_index, desc=f"{class_name} / baseline"
        )
        print(f"  {len(baseline_means)} images found in baseline")

        for ablation_label in ABLATION_SETTINGS:
            if ablation_label not in datasets:
                print(f"  [skip] no '{ablation_label}' dataset configured for {class_name} yet")
                continue

            print(f"Comparing baseline vs '{ablation_label}'...")
            try:
                ablation_index = index_variance_dataset(datasets[ablation_label])
                ablation_means = compute_per_image_mean_variance(
                    ablation_index, desc=f"{class_name} / {ablation_label}"
                )
                print(f"  {len(ablation_means)} images found in {ablation_label}")

                matched_ids = validate_matching_image_ids(
                    baseline_means, ablation_means, "baseline", ablation_label
                )

                test_result = run_paired_wilcoxon(baseline_means, ablation_means, matched_ids)
                test_result.update({
                    "class_name": class_name,
                    "ablation_setting": ablation_label,
                    "baseline_config": ABLATION_CONFIG_DESCRIPTIONS.get("baseline", ""),
                    "ablation_config": ABLATION_CONFIG_DESCRIPTIONS.get(ablation_label, ""),
                    "error": "",
                })
                print(
                    f"  n={test_result['n_images']}, "
                    f"W={test_result['statistic']:.2f}, "
                    f"p={test_result['p_value']:.4g}, "
                    f"median_diff={test_result['median_paired_diff']:.6f}"
                )
                results.append(test_result)

            except (FileNotFoundError, ValueError) as e:
                # A bad/missing/mismatched dataset for ONE (class, ablation)
                # pair should not discard results already computed for
                # other pairs — log it as a failed row and keep going.
                print(f"  [ERROR] {class_name} / {ablation_label}: {e}")
                results.append({
                    "class_name": class_name,
                    "ablation_setting": ablation_label,
                    "n_images": np.nan,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "baseline_mean_of_means": np.nan,
                    "ablation_mean_of_means": np.nan,
                    "median_paired_diff": np.nan,
                    "mean_paired_diff": np.nan,
                    "baseline_config": ABLATION_CONFIG_DESCRIPTIONS.get("baseline", ""),
                    "ablation_config": ABLATION_CONFIG_DESCRIPTIONS.get(ablation_label, ""),
                    "error": str(e),
                })

    if not results:
        raise RuntimeError(
            "No comparisons were run — check that CLASS_ABLATION_DATASETS has "
            "at least one class with a 'baseline' entry plus at least one "
            "matching ABLATION_SETTINGS entry."
        )

    results_df = pd.DataFrame(results)
    column_order = [
        "class_name", "ablation_setting", "n_images", "statistic", "p_value",
        "baseline_mean_of_means", "ablation_mean_of_means",
        "median_paired_diff", "mean_paired_diff",
        "baseline_config", "ablation_config", "error",
    ]
    results_df = results_df[column_order]

    out_path = output_dir / "stage2_wilcoxon_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(results_df[["class_name", "ablation_setting", "n_images", "statistic", "p_value"]])

    return results_df


if __name__ == "__main__":
    main()
