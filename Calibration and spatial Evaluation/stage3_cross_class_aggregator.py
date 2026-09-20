"""
stage3_cross_class_aggregator.py
============================================================================
Kaggle Notebook, CPU-only.

Aggregates per-class Stage 3 outputs (results_summary.csv +
results_per_image.csv, one pair per abnormality class, produced by
calibration_spatial_evaluation_pipeline.py) into cross-class comparisons:

  - A single cross-class summary table (one row per class) with AUROC,
    Dice, mean box size, and a base-rate-normalized "lift ratio" so
    classes with very different box sizes/prevalence are comparable.
  - A scatter plot: per-image AUROC vs. box size (n_positive_pixels),
    colored by class — the main diagnostic for "does localization
    strength track pathology size/conspicuousness?"
  - A bar chart: mean AUROC per class with std-dev error bars.
  - A bar chart: lift ratio per class.

IMPORTANT — before running this, make sure each class's Stage 3 outputs
were copied to their OWN directory. calibration_spatial_evaluation_pipeline.py
always writes to the same OUTPUT_DIR, so re-running it for a new class
without copying the previous class's results out first will silently
overwrite them.

INPUT (edit CONFIG in Cell 1): a dict mapping class_name -> paths to that
class's results_summary.csv and results_per_image.csv.

OUTPUT (written to OUTPUT_DIR):
  - cross_class_summary.csv
  - auroc_vs_boxsize_scatter.png
  - auroc_by_class_bar.png
  - lift_ratio_by_class_bar.png
"""

# ============================================================================
# CELL 0 — Imports (all preinstalled on Kaggle; no pip install needed)
# ============================================================================
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# CELL 1 — CONFIG (edit to point at each class's Stage 3 output directory)
# ============================================================================
# Only include classes whose Stage 3 run has actually completed — the
# script skips (with a warning) any class whose files aren't found, so
# partial completion across your 5 classes is fine.
CLASS_RESULTS = {
    "Pneumothorax": {
        "summary_csv":   "/kaggle/working/pneumothorax_results/results_summary.csv",
        "per_image_csv": "/kaggle/working/pneumothorax_results/results_per_image.csv",
    },
    "Consolidation": {
        "summary_csv":   "/kaggle/working/consolidation_results/results_summary.csv",
        "per_image_csv": "/kaggle/working/consolidation_results/results_per_image.csv",
    },
    "Nodule/Mass": {
        "summary_csv":   "/kaggle/working/nodulemass_results/results_summary.csv",
        "per_image_csv": "/kaggle/working/nodulemass_results/results_per_image.csv",
    },
    "Cardiomegaly": {
        "summary_csv":   "/kaggle/working/cardiomegaly_results/results_summary.csv",
        "per_image_csv": "/kaggle/working/cardiomegaly_results/results_per_image.csv",
    },
    "Atelectasis": {
        "summary_csv":   "/kaggle/working/atelectasis_results/results_summary.csv",
        "per_image_csv": "/kaggle/working/atelectasis_results/results_per_image.csv",
    },
}

OUTPUT_DIR = "/kaggle/working/aggregate_results"


# ============================================================================
# CELL 2 — Load + combine per-class results
# ============================================================================
def load_all_results(class_results):
    """
    Loads every class's results_summary.csv and results_per_image.csv,
    tags each row with class_name, and concatenates across classes.
    Classes with missing/unreadable files are skipped with a warning
    rather than failing the whole run.
    """
    summary_frames, per_image_frames = [], []

    for class_name, paths in class_results.items():
        summary_path = Path(paths["summary_csv"])
        per_image_path = Path(paths["per_image_csv"])

        if not summary_path.exists() or not per_image_path.exists():
            warnings.warn(
                f"Skipping '{class_name}': missing "
                f"{'summary_csv' if not summary_path.exists() else 'per_image_csv'} "
                f"at the configured path. Stage 3 for this class may not be complete yet."
            )
            continue

        summary_df = pd.read_csv(summary_path)
        summary_df.insert(0, "class_name", class_name)
        summary_frames.append(summary_df)

        per_image_df = pd.read_csv(per_image_path)
        per_image_df.insert(0, "class_name", class_name)
        per_image_frames.append(per_image_df)

    if not summary_frames:
        raise RuntimeError(
            "No classes loaded — check CLASS_RESULTS paths, and confirm each "
            "class's Stage 3 outputs were copied to their own directory "
            "(calibration_spatial_evaluation_pipeline.py overwrites OUTPUT_DIR "
            "on every run, so results must be moved out between classes)."
        )

    summary_all = pd.concat(summary_frames, ignore_index=True)
    per_image_all = pd.concat(per_image_frames, ignore_index=True)
    return summary_all, per_image_all


# ============================================================================
# CELL 3 — Derived cross-class table (box size, base rate, lift ratio)
# ============================================================================
def build_cross_class_table(summary_all, per_image_all):
    """
    Combines each class's aggregate summary row with per-image-derived
    stats computed directly from results_per_image.csv:
      - mean_box_size_pixels: average n_positive_pixels per image
      - eval_positive_rate: average per-image fraction of interior pixels
        that are positive (n_positive_pixels / (n_positive_pixels + n_negative_pixels))
        — computed on the EVAL split, since positive rate isn't persisted
        in results_summary.csv (that's the calibration split's rate,
        printed at runtime but not saved to file)
      - lift_ratio: max_calibrated_prob_overall / eval_positive_rate — how
        many times higher than base rate the calibrator's ceiling reaches,
        the fair cross-class comparison since raw ceiling isn't comparable
        across classes with very different prevalence/box sizes
    """
    per_image_all = per_image_all.copy()
    per_image_all["total_interior_pixels"] = (
        per_image_all["n_positive_pixels"] + per_image_all["n_negative_pixels"]
    )
    per_image_all["per_image_positive_rate"] = (
        per_image_all["n_positive_pixels"] / per_image_all["total_interior_pixels"]
    )

    derived = per_image_all.groupby("class_name").agg(
        n_images_per_image_csv=("image_id", "nunique"),
        mean_box_size_pixels=("n_positive_pixels", "mean"),
        median_box_size_pixels=("n_positive_pixels", "median"),
        eval_positive_rate=("per_image_positive_rate", "mean"),
    ).reset_index()

    cross_class = summary_all.merge(derived, on="class_name", how="left")

    if "max_calibrated_prob_overall" in cross_class.columns:
        cross_class["lift_ratio"] = (
            cross_class["max_calibrated_prob_overall"] / cross_class["eval_positive_rate"]
        )
    else:
        warnings.warn(
            "max_calibrated_prob_overall not found in results_summary.csv — "
            "lift_ratio cannot be computed. This column requires the version "
            "of calibration_spatial_evaluation_pipeline.py that logs the "
            "calibrator-ceiling diagnostic; re-run Stage 3 with the updated "
            "script if this matters for your comparison."
        )
        cross_class["lift_ratio"] = np.nan

    return cross_class


# ============================================================================
# CELL 4 — Visualizations
# ============================================================================
def _class_color_map(class_names):
    palette = plt.get_cmap("tab10")
    return {name: palette(i % 10) for i, name in enumerate(sorted(class_names))}


def plot_auroc_vs_boxsize(per_image_all, output_dir):
    """
    Scatter of per-image AUROC vs. box size (n_positive_pixels), colored
    by class, log-x since box size spans orders of magnitude across
    pathology types. The main diagnostic for whether localization
    strength tracks pathology conspicuousness rather than being a fixed
    property of the method.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    color_map = _class_color_map(per_image_all["class_name"].unique())

    for class_name, group in per_image_all.groupby("class_name"):
        valid = group.dropna(subset=["auroc", "n_positive_pixels"])
        valid = valid[valid["n_positive_pixels"] > 0]  # log-scale requires > 0
        ax.scatter(
            valid["n_positive_pixels"], valid["auroc"],
            label=class_name, color=color_map[class_name], alpha=0.65, s=35,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (AUROC=0.5)")
    ax.set_xscale("log")
    ax.set_xlabel("Ground-truth box size (positive pixels, log scale)")
    ax.set_ylabel("Per-image AUROC")
    ax.set_title("Localization strength (AUROC) vs. pathology box size, by class")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    out_path = Path(output_dir) / "auroc_vs_boxsize_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_auroc_by_class_bar(cross_class, output_dir):
    """Bar chart of mean AUROC per class, with std-dev error bars."""
    ordered = cross_class.sort_values("auroc_mean", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        ordered["class_name"], ordered["auroc_mean"],
        yerr=ordered["auroc_std"], capsize=4,
        color="steelblue", alpha=0.85,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (AUROC=0.5)")
    ax.set_ylabel("Mean AUROC (± std across eval images)")
    ax.set_title("Spatial localization strength by pathology class")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    out_path = Path(output_dir) / "auroc_by_class_bar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_lift_ratio_by_class(cross_class, output_dir):
    """Bar chart of lift ratio (calibrator ceiling / base rate) per class."""
    if cross_class["lift_ratio"].isna().all():
        warnings.warn("lift_ratio is unavailable for every class — skipping this plot.")
        return None

    ordered = cross_class.dropna(subset=["lift_ratio"]).sort_values("lift_ratio", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ordered["class_name"], ordered["lift_ratio"], color="darkorange", alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Lift ratio = 1 (no enrichment)")
    ax.set_ylabel("Lift ratio (calibrator ceiling / base rate)")
    ax.set_title("How much the calibrator's peak probability exceeds base rate, by class")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    out_path = Path(output_dir) / "lift_ratio_by_class_bar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================================
# CELL 5 — Orchestration
# ============================================================================
def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading per-class Stage 3 results...")
    summary_all, per_image_all = load_all_results(CLASS_RESULTS)
    print(f"      loaded {summary_all['class_name'].nunique()} class(es): "
          f"{sorted(summary_all['class_name'].unique())}")

    print("[2/4] Building cross-class summary table...")
    cross_class = build_cross_class_table(summary_all, per_image_all)
    table_path = Path(output_dir) / "cross_class_summary.csv"
    cross_class.to_csv(table_path, index=False)
    print(f"      saved: {table_path}")
    display_cols = [c for c in [
        "class_name", "n_images_evaluated", "auroc_mean", "auroc_std",
        "dice_best_threshold_mean", "mean_box_size_pixels",
        "eval_positive_rate", "max_calibrated_prob_overall", "lift_ratio",
    ] if c in cross_class.columns]
    print(cross_class[display_cols].to_string(index=False))

    print("[3/4] Plotting AUROC vs. box size scatter...")
    plot_auroc_vs_boxsize(per_image_all, output_dir)

    print("[4/4] Plotting per-class bar charts...")
    plot_auroc_by_class_bar(cross_class, output_dir)
    plot_lift_ratio_by_class(cross_class, output_dir)

    print(f"\nDone. All outputs in {output_dir}")
    return cross_class


if __name__ == "__main__":
    main()
