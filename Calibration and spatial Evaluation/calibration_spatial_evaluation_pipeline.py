"""
calibration_spatial_evaluation_pipeline.py
============================================================================
Kaggle Notebook, CPU-only.

Calibrates raw DDIM epistemic-uncertainty variance maps into [0, 1]
probabilities (netcal), then evaluates whether the calibrated uncertainty
spatially aligns with radiologist-annotated pathology boxes on VinBigData
Chest X-rays, via pixel-level AUROC and Dice.

INPUTS (edit CONFIG in Cell 1):
  - `uncertainty_chunk_NNN.npz` (+ companion `_meta.csv`) files, produced by
    ddim_epistemic_uncertainty_pipeline.py. Each npz holds, per image_id,
    a "{image_id}__mean" and "{image_id}__variance" array, both (512,512)
    float32.
  - VinBigData `train.csv` (radiologist bounding boxes, ORIGINAL DICOM pixel
    coordinates — NOT rescaled to 512x512).
  - (optional but recommended) `vindr_cxr_metadata.csv`, produced by
    vinbigdata_diffusion_pipeline.py, used to look up each image's original
    `rows`/`columns` so boxes can be correctly rescaled to 512x512.
  - (optional) `vindr_cxr_512.h5`, produced by vinbigdata_diffusion_pipeline.py,
    used only to render the original X-ray under the overlay visualizations.
    If missing, overlays are still produced on a blank background.

OUTPUTS (written to /kaggle/working/):
  - reliability_diagram_calibrated.png       (in-sample, calibration split)
  - reliability_diagram_raw_normalized.png   (before/after comparison, in-sample)
  - reliability_diagram_holdout.png          (out-of-sample, evaluation split — the trustworthy one)
  - overlay_<image_id>.png                   (N_OVERLAY_EXAMPLES of these)
  - results_summary.csv                      (aggregate metrics)
  - results_per_image.csv                    (per-image AUROC / Dice)

Known project caveats this script bakes in (see prior session's project
notes — do not "fix" these without updating both places):
  - Outer ~20px border ring of every variance map carries ~44% inflated
    variance (border mean 0.0112 vs interior 0.0078), likely a VAE-decode
    edge effect -> excluded from BOTH calibrator fitting and spatial
    scoring (known issue #3).
  - Box coordinates in the raw VinBigData CSV are in ORIGINAL DICOM
    resolution, not 512x512 -> rescaled here via each image's orig
    rows/columns from vindr_cxr_metadata.csv (known issue #4).
  - Multiple radiologists annotate the same image independently; this
    script takes the UNION of all matching-class boxes per image as the
    ground-truth positive region.
  - `ece_calibrated_in_sample` (in results_summary.csv) is computed on the
    SAME pixels the calibrator was fit on — isotonic regression fits its
    own training labels closely by construction, so a near-zero value
    here is close to guaranteed and is NOT evidence of good calibration.
    `ece_holdout_out_of_sample` is the trustworthy number: it's computed
    on an independent pixel sample from the held-out evaluation split,
    never seen during calibrator fitting.
"""

# ============================================================================
# CELL 0 — Install dependencies (CPU-only Kaggle environment)
# ============================================================================
# netcal is not preinstalled on Kaggle; everything else (scipy, h5py,
# scikit-learn, tqdm, pandas, matplotlib) normally is, but we install
# quietly/idempotently so this cell is safe to re-run and safe on any base
# image variant. Uses IPython's `!` shell magic, matching Cell 0 in the
# earlier pipelines in this project — run this as a notebook cell.
!pip install -q netcal h5py

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib

matplotlib.use("Agg")  # headless-safe: required for Kaggle background/batch runs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage

from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from netcal.binning import IsotonicRegression
from netcal.scaling import TemperatureScaling
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# CELL 1 — CONFIG (edit these paths for your Kaggle setup)
# ============================================================================
NPZ_DIR = "/kaggle/input/ddim-uncertainty-maps"                       # dir containing uncertainty_chunk_*.npz
TRAIN_CSV_PATH = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv"
METADATA_CSV_PATH = "/kaggle/input/vindr-cxr-512/vindr_cxr_metadata.csv"  # optional; None to disable
H5_PATH = "/kaggle/input/vindr-cxr-512/vindr_cxr_512.h5"               # optional; None to disable

OUTPUT_DIR = "/kaggle/working"

IMG_SIZE = 512
BORDER_EXCLUDE_PX = 20            # known issue #3: exclude outer ring from fit + scoring

# Must match (a subset of) the `finding` values used when generating the
# variance maps (see RARE_FINDINGS in ddim_epistemic_uncertainty_pipeline.py).
TARGET_CLASSES = ["Pneumothorax"]

CALIBRATION_METHOD = "isotonic"   # "isotonic" (recommended, non-parametric,
                                   # handles unbounded raw variance directly)
                                   # or "temperature"
CALIBRATION_FRACTION = 0.4        # fraction of usable images used to FIT the
                                   # calibrator; remainder is held out for
                                   # spatial evaluation (no leakage)
MAX_PIXELS_PER_IMAGE_FOR_FIT = 20_000  # subsample cap per image when
                                        # building the calibration pixel set

DICE_THRESHOLD = 0.5              # fixed threshold reported alongside the
                                   # per-image best-threshold Dice
N_OVERLAY_EXAMPLES = 6
RANDOM_SEED = 42
N_RELIABILITY_BINS = 10

np.random.seed(RANDOM_SEED)


# ============================================================================
# CELL 2 — Bounding-box loading, rescaling, and rasterization to masks
# ============================================================================
def load_and_prepare_boxes(train_csv_path, metadata_csv_path, target_classes):
    """
    Load VinBigData train.csv radiologist boxes and attach each image's
    original width/height (needed to rescale box coordinates to 512x512).

    Returns a DataFrame: image_id, class_name, x_min, y_min, x_max, y_max,
    orig_w, orig_h — one row per (image, radiologist, box).
    """
    boxes = pd.read_csv(train_csv_path)

    # Normalize column names (strip whitespace, lowercase) so trivial
    # export differences ("X_min ", "Xmin") don't cause a silent schema
    # mismatch — but keep the ORIGINAL names for the final error message
    # so the user can see exactly what they have.
    original_columns = boxes.columns.tolist()
    normalized = {c: c.strip().lower().replace(" ", "_") for c in boxes.columns}
    boxes = boxes.rename(columns=normalized)

    required = {"image_id", "class_name", "x_min", "y_min", "x_max", "y_max"}
    missing = required - set(boxes.columns)
    if missing:
        raise ValueError(
            f"TRAIN_CSV_PATH ({train_csv_path}) is missing required "
            f"column(s): {sorted(missing)}.\n"
            f"Columns actually found in this file: {original_columns}\n"
            f"This usually means TRAIN_CSV_PATH is pointing at the wrong "
            f"file — e.g. your own vindr_cxr_metadata.csv (which stores "
            f"boxes as a 'boxes' list-of-dicts column, not flat "
            f"x_min/y_min/x_max/y_max columns) or sample_submission.csv, "
            f"rather than the raw VinBigData competition train.csv."
        )

    # "No finding" rows (a radiologist's negative read) carry NaN box
    # coordinates in VinBigData's format and contribute no positive area.
    boxes = boxes.dropna(subset=["x_min", "y_min", "x_max", "y_max"]).copy()

    if target_classes is not None:
        boxes = boxes[boxes["class_name"].isin(target_classes)].copy()

    if metadata_csv_path is not None and Path(metadata_csv_path).exists():
        meta = pd.read_csv(metadata_csv_path, usecols=["image_id", "rows", "columns"])
        meta = meta.drop_duplicates(subset="image_id")
        boxes = boxes.merge(
            meta.rename(columns={"rows": "orig_h", "columns": "orig_w"}),
            on="image_id", how="left",
        )
        n_missing = int(boxes["orig_h"].isna().sum())
        if n_missing:
            warnings.warn(
                f"{n_missing} box rows had no matching original width/height "
                f"in {metadata_csv_path}; dropping them."
            )
            boxes = boxes.dropna(subset=["orig_h", "orig_w"])
    else:
        warnings.warn(
            "METADATA_CSV_PATH not found/unset — assuming box coordinates "
            "are ALREADY at 512x512. This is almost certainly wrong for raw "
            "VinBigData boxes; supply vindr_cxr_metadata.csv to rescale "
            "correctly (known issue #4)."
        )
        boxes["orig_h"] = IMG_SIZE
        boxes["orig_w"] = IMG_SIZE

    return boxes


def rescale_box(x_min, y_min, x_max, y_max, orig_w, orig_h, target_size=IMG_SIZE):
    """Rescale a box from original DICOM resolution to target_size x target_size."""
    scale_x = target_size / orig_w
    scale_y = target_size / orig_h
    return x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y


def boxes_to_mask(image_boxes, img_size=IMG_SIZE):
    """
    Rasterize all (rescaled) boxes for ONE image into a single binary
    ground-truth mask, shape (img_size, img_size) uint8. Union across all
    radiologists / boxes for that image.
    """
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for _, row in image_boxes.iterrows():
        x0, y0, x1, y1 = rescale_box(
            row["x_min"], row["y_min"], row["x_max"], row["y_max"],
            row["orig_w"], row["orig_h"], img_size,
        )
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, img_size), min(y1, img_size)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    return mask


# ============================================================================
# CELL 3 — Variance npz indexing + lazy per-image loading
# ============================================================================
def index_variance_npz(npz_dir):
    """
    Scan npz_dir for uncertainty_chunk_*.npz and build {image_id: npz_path}
    without loading arrays into memory yet.
    """
    npz_dir = Path(npz_dir)
    npz_paths = sorted(npz_dir.glob("uncertainty_chunk_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No uncertainty_chunk_*.npz files found under {npz_dir}")

    index = {}
    for p in npz_paths:
        with np.load(p) as data:
            for key in data.files:
                if key.endswith("__variance"):
                    image_id = key[: -len("__variance")]
                    index[image_id] = p
    return index


def load_variance_map(image_id, npz_index):
    with np.load(npz_index[image_id]) as data:
        return data[f"{image_id}__variance"].astype(np.float32)


# ============================================================================
# CELL 4 — Interior (border-excluded) mask + calibration/eval split
# ============================================================================
def make_interior_mask(img_size=IMG_SIZE, border=BORDER_EXCLUDE_PX):
    m = np.zeros((img_size, img_size), dtype=bool)
    m[border: img_size - border, border: img_size - border] = True
    return m


INTERIOR_MASK = make_interior_mask()


def split_image_ids(image_ids, calibration_fraction, seed):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(image_ids))
    rng.shuffle(ids)
    n_calib = max(1, int(round(len(ids) * calibration_fraction)))
    n_calib = min(n_calib, len(ids) - 1) if len(ids) > 1 else len(ids)
    return ids[:n_calib].tolist(), ids[n_calib:].tolist()


# ============================================================================
# CELL 5 — Pixel sampling + calibrator fitting
# ============================================================================
def sample_pixels_for_calibration(image_ids, npz_index, gt_masks, max_per_image, seed,
                                   desc="Sampling calibration pixels"):
    """
    Build a flat (X, y) pixel-level dataset: X = raw variance value,
    y = 0/1 ground-truth pathology label. Interior-only (border ring
    excluded, issue #3). Subsamples up to `max_per_image` pixels per image
    so the operation stays tractable while still drawing from every image
    in `image_ids`.

    Reused for two purposes in this pipeline: (1) building the calibrator's
    FITTING set from the calibration split, and (2) building an independent
    held-out pixel sample from the evaluation split for out-of-sample ECE
    (see compute_holdout_ece below) — `desc` just customizes the progress
    bar label so it's clear which one is running.
    """
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for image_id in tqdm(image_ids, desc=desc):
        if image_id not in npz_index or image_id not in gt_masks:
            continue
        var_map = load_variance_map(image_id, npz_index)
        mask = gt_masks[image_id]

        flat_var = var_map[INTERIOR_MASK]
        flat_gt = mask[INTERIOR_MASK]

        n_pixels = flat_var.shape[0]
        if n_pixels > max_per_image:
            idx = rng.choice(n_pixels, size=max_per_image, replace=False)
            flat_var, flat_gt = flat_var[idx], flat_gt[idx]

        X_parts.append(flat_var)
        y_parts.append(flat_gt)

    if not X_parts:
        raise RuntimeError("No calibration pixels collected — check that "
                            "calibration image_ids exist in both npz_index "
                            "and gt_masks.")

    X = np.concatenate(X_parts).astype(np.float64)
    y = np.concatenate(y_parts).astype(np.int64)
    return X, y


def fit_calibrator(X, y, method=CALIBRATION_METHOD):
    if method == "isotonic":
        calibrator = IsotonicRegression()
    elif method == "temperature":
        calibrator = TemperatureScaling()
    else:
        raise ValueError(f"Unknown CALIBRATION_METHOD: {method!r}")
    calibrator.fit(X, y)
    return calibrator


def get_calibrated_map(image_id, npz_index, calibrator):
    """Apply the fitted calibrator to the FULL 512x512 variance map."""
    var_map = load_variance_map(image_id, npz_index)
    calibrated_flat = calibrator.transform(var_map.flatten().astype(np.float64))
    return np.asarray(calibrated_flat, dtype=np.float64).reshape(IMG_SIZE, IMG_SIZE)


# ============================================================================
# CELL 6 — Reliability diagrams (before / after calibration)
# ============================================================================
def plot_reliability_diagrams(X, y, calibrator, output_dir, n_bins=N_RELIABILITY_BINS):
    """
    Saves two PNGs: the calibrated reliability diagram (the one that
    matters), and a min-max-normalized "raw" diagram purely for visual
    before/after contrast. Raw DDIM variance is unbounded, so it cannot be
    fed to ReliabilityDiagram/ECE directly (both expect confidences in
    [0, 1]) — min-max normalization is ONLY used for this visual, never
    for the actual calibration or spatial scoring.

    IMPORTANT: X, y here are the CALIBRATION split — the same pixels the
    calibrator was just fit on. The resulting ece_calibrated is therefore
    an IN-SAMPLE number: isotonic regression is a monotonic step function
    fit directly against these labels, so near-zero in-sample ECE is close
    to guaranteed by construction, not evidence the calibration
    generalizes. See compute_holdout_ece() below for the number that
    actually tests generalization, on the held-out evaluation split.
    """
    calibrated = np.asarray(calibrator.transform(X), dtype=np.float64)
    x_norm = (X - X.min()) / (np.ptp(X) + 1e-12)

    ece_metric = ECE(n_bins)
    ece_calibrated = float(ece_metric.measure(calibrated, y))
    ece_raw_norm = float(ece_metric.measure(x_norm, y))

    diagram = ReliabilityDiagram(n_bins)

    fig_cal = diagram.plot(calibrated, y)
    fig_cal.savefig(Path(output_dir) / "reliability_diagram_calibrated.png",
                     dpi=150, bbox_inches="tight")
    plt.close(fig_cal)

    fig_raw = diagram.plot(x_norm, y)
    fig_raw.savefig(Path(output_dir) / "reliability_diagram_raw_normalized.png",
                     dpi=150, bbox_inches="tight")
    plt.close(fig_raw)

    return {"ece_calibrated": ece_calibrated, "ece_raw_minmax_normalized": ece_raw_norm}


def compute_holdout_ece(eval_ids, npz_index, gt_masks, calibrator, output_dir,
                         max_per_image=MAX_PIXELS_PER_IMAGE_FOR_FIT,
                         seed=RANDOM_SEED, n_bins=N_RELIABILITY_BINS):
    """
    Out-of-sample ECE: applies the ALREADY-FITTED calibrator (fit only on
    the calibration split, never touched here) to a fresh pixel sample
    drawn from the held-out evaluation split, and measures ECE there.

    This is the trustworthy calibration-quality number for reporting.
    Unlike plot_reliability_diagrams()'s ece_calibrated (computed on the
    calibrator's own fitting data), this pixel sample was never seen
    during fitting, so a low value here is real evidence the calibration
    curve generalizes to new images rather than an artifact of isotonic
    regression fitting its own training labels closely.

    Uses a seed offset (seed + 1) from the calibration-sampling seed so
    the two pixel draws are independent even if the same seed value is
    passed through CONFIG.
    """
    X_eval, y_eval = sample_pixels_for_calibration(
        eval_ids, npz_index, gt_masks, max_per_image, seed + 1,
        desc="Sampling held-out pixels for out-of-sample ECE",
    )

    calibrated_eval = np.asarray(calibrator.transform(X_eval), dtype=np.float64)

    ece_metric = ECE(n_bins)
    ece_holdout = float(ece_metric.measure(calibrated_eval, y_eval))

    diagram = ReliabilityDiagram(n_bins)
    fig_holdout = diagram.plot(calibrated_eval, y_eval)
    fig_holdout.savefig(Path(output_dir) / "reliability_diagram_holdout.png",
                         dpi=150, bbox_inches="tight")
    plt.close(fig_holdout)

    return {"ece_holdout": ece_holdout, "n_holdout_pixels_for_ece": int(len(X_eval))}


# ============================================================================
# CELL 7 — Spatial evaluation: pixel-level AUROC + Dice
# ============================================================================
def dice_coefficient(pred_binary, gt_binary):
    pred_binary = np.asarray(pred_binary, dtype=bool)
    gt_binary = np.asarray(gt_binary, dtype=bool)
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    denom = pred_binary.sum() + gt_binary.sum()
    if denom == 0:
        return np.nan  # no positive pixels in prediction OR ground truth -> undefined
    return 2.0 * intersection / denom


def best_dice_threshold(y_score, y_true, thresholds=np.arange(0.05, 1.0, 0.05)):
    best_t, best_d = 0.5, -1.0
    for t in thresholds:
        d = dice_coefficient(y_score >= t, y_true)
        if not np.isnan(d) and d > best_d:
            best_d, best_t = d, float(t)
    return best_t, best_d


def evaluate_spatial_alignment(image_ids, npz_index, gt_masks, calibrator,
                                fixed_threshold=DICE_THRESHOLD):
    """
    For each held-out image: calibrate the full variance map, restrict to
    the interior (border ring excluded, issue #3), and compute pixel-level
    AUROC + Dice (at a fixed threshold and at the image's best threshold)
    against the radiologist box mask.
    """
    rows = []
    for image_id in tqdm(image_ids, desc="Spatial evaluation"):
        if image_id not in npz_index or image_id not in gt_masks:
            continue
        calibrated_map = get_calibrated_map(image_id, npz_index, calibrator)
        gt_mask = gt_masks[image_id]

        y_true = gt_mask[INTERIOR_MASK]
        y_score = calibrated_map[INTERIOR_MASK]

        n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
        auroc = roc_auc_score(y_true, y_score) if (n_pos > 0 and n_neg > 0) else np.nan

        dice_fixed = dice_coefficient(y_score >= fixed_threshold, y_true)
        best_t, dice_best = best_dice_threshold(y_score, y_true)

        rows.append({
            "image_id": image_id,
            "n_positive_pixels": n_pos,
            "n_negative_pixels": n_neg,
            "auroc": auroc,
            f"dice_at_{fixed_threshold}": dice_fixed,
            "dice_best_threshold": dice_best,
            "best_threshold": best_t,
            "max_calibrated_prob": float(y_score.max()),
        })

    return pd.DataFrame(rows)


# ============================================================================
# CELL 8 — Visualization overlays (original X-ray + GT box + heatmap)
# ============================================================================
def load_original_image(image_id, h5_path):
    """Returns a (512,512) float array in [0,1], or None if unavailable."""
    if h5_path is None or not Path(h5_path).exists():
        return None
    with h5py.File(h5_path, "r") as f:
        if image_id not in f:
            return None
        arr = f[image_id][()].astype(np.float32)
    # vinbigdata_diffusion_pipeline.py normalizes to [-1, 1] float32
    arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
    return arr


def draw_box_outlines(ax, mask, color="lime"):
    """
    Outline each connected component of a binary mask with a rectangle.
    Note: overlapping boxes from different radiologists merge into one
    connected component and are drawn as a single rectangle — a known
    simplification of the true per-radiologist boxes.
    """
    if mask.sum() == 0:
        return
    labeled, n_components = ndimage.label(mask)
    for i in range(1, n_components + 1):
        ys, xs = np.where(labeled == i)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        rect = patches.Rectangle(
            (x0, y0), x1 - x0 + 1, y1 - y0 + 1,
            linewidth=1.5, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)


def save_overlay_figure(image_id, npz_index, gt_masks, calibrator, h5_path, output_dir):
    calibrated_map = get_calibrated_map(image_id, npz_index, calibrator)
    gt_mask = gt_masks[image_id]
    xray = load_original_image(image_id, h5_path)
    base = xray if xray is not None else np.zeros((IMG_SIZE, IMG_SIZE))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.imshow(base, cmap="gray", vmin=0, vmax=1)
        ax.set_xlim(0, IMG_SIZE)
        ax.set_ylim(IMG_SIZE, 0)
        ax.axis("off")

    axes[0].set_title(f"{image_id}\nOriginal X-ray" + ("" if xray is not None else " (unavailable)"))

    axes[1].set_title("Radiologist ground-truth box(es)")
    draw_box_outlines(axes[1], gt_mask, color="lime")

    axes[2].set_title("Calibrated uncertainty overlay")
    heat_display = np.ma.masked_where(~INTERIOR_MASK, calibrated_map)  # hide border ring, issue #3
    heat = axes[2].imshow(heat_display, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
    draw_box_outlines(axes[2], gt_mask, color="lime")
    fig.colorbar(heat, ax=axes[2], fraction=0.046, pad=0.04, label="Calibrated P(pathology)")

    fig.tight_layout()
    out_path = Path(output_dir) / f"overlay_{image_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================================
# CELL 9 — results_summary.csv
# ============================================================================
def save_results_summary(eval_df, ece_dict, ece_holdout_dict, config, output_dir):
    max_prob_overall = eval_df["max_calibrated_prob"].max()
    fixed_threshold_reachable = bool(max_prob_overall >= config["DICE_THRESHOLD"])

    summary = {
        "n_images_evaluated": int(eval_df["image_id"].nunique()),
        "auroc_mean": eval_df["auroc"].mean(skipna=True),
        "auroc_std": eval_df["auroc"].std(skipna=True),
        "auroc_median": eval_df["auroc"].median(skipna=True),
        "n_images_auroc_undefined": int(eval_df["auroc"].isna().sum()),
        f"dice_at_{config['DICE_THRESHOLD']}_mean": eval_df[f"dice_at_{config['DICE_THRESHOLD']}"].mean(skipna=True),
        "dice_best_threshold_mean": eval_df["dice_best_threshold"].mean(skipna=True),
        "mean_best_threshold": eval_df["best_threshold"].mean(skipna=True),
        "max_calibrated_prob_overall": max_prob_overall,
        "mean_per_image_max_calibrated_prob": eval_df["max_calibrated_prob"].mean(),
        "fixed_threshold_reachable": fixed_threshold_reachable,
        "ece_calibrated_in_sample": ece_dict["ece_calibrated"],
        "ece_raw_minmax_normalized": ece_dict["ece_raw_minmax_normalized"],
        "ece_holdout_out_of_sample": ece_holdout_dict["ece_holdout"],
        "n_holdout_pixels_for_ece": ece_holdout_dict["n_holdout_pixels_for_ece"],
        "calibration_method": config["CALIBRATION_METHOD"],
        "border_exclude_px": config["BORDER_EXCLUDE_PX"],
        "calibration_fraction": config["CALIBRATION_FRACTION"],
        "target_classes": ",".join(config["TARGET_CLASSES"]),
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
    }

    summary_path = Path(output_dir) / "results_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    per_image_path = Path(output_dir) / "results_per_image.csv"
    eval_df.to_csv(per_image_path, index=False)

    return summary_path, per_image_path


# ============================================================================
# CELL 10 — Orchestration
# ============================================================================
def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] Indexing variance npz files...")
    npz_index = index_variance_npz(NPZ_DIR)
    print(f"      found variance maps for {len(npz_index)} images")

    print("[2/8] Loading + rescaling radiologist bounding boxes...")
    boxes = load_and_prepare_boxes(TRAIN_CSV_PATH, METADATA_CSV_PATH, TARGET_CLASSES)
    image_ids_with_boxes = set(boxes["image_id"].unique())

    usable_ids = sorted(set(npz_index.keys()) & image_ids_with_boxes)
    print(f"      {len(usable_ids)} images have both a variance map and "
          f">=1 box in target classes {TARGET_CLASSES}")
    if not usable_ids:
        raise RuntimeError(
            "No overlap between variance-map image_ids and bounding-box "
            "image_ids — check TARGET_CLASSES and paths in CONFIG."
        )

    print("[3/8] Rasterizing ground-truth masks...")
    gt_masks = {
        image_id: boxes_to_mask(boxes[boxes["image_id"] == image_id])
        for image_id in tqdm(usable_ids)
    }

    calib_ids, eval_ids = split_image_ids(usable_ids, CALIBRATION_FRACTION, RANDOM_SEED)
    print(f"      calibration split: {len(calib_ids)} images | "
          f"evaluation split: {len(eval_ids)} images")

    print(f"[4/8] Sampling pixels + fitting calibrator ({CALIBRATION_METHOD})...")
    X_calib, y_calib = sample_pixels_for_calibration(
        calib_ids, npz_index, gt_masks, MAX_PIXELS_PER_IMAGE_FOR_FIT, RANDOM_SEED
    )
    print(f"      {len(X_calib):,} pixels sampled, positive rate = {y_calib.mean():.4%}")
    calibrator = fit_calibrator(X_calib, y_calib, CALIBRATION_METHOD)

    print("[5/8] Plotting reliability diagrams (in-sample, calibration split)...")
    ece_dict = plot_reliability_diagrams(X_calib, y_calib, calibrator, output_dir)
    print(f"      ECE in-sample = {ece_dict['ece_calibrated']:.4f} "
          f"(raw, min-max normalized = {ece_dict['ece_raw_minmax_normalized']:.4f})")

    print("[6/8] Computing out-of-sample ECE (held-out evaluation split)...")
    ece_holdout_dict = compute_holdout_ece(eval_ids, npz_index, gt_masks, calibrator, output_dir)
    print(f"      ECE out-of-sample = {ece_holdout_dict['ece_holdout']:.4f} "
          f"({ece_holdout_dict['n_holdout_pixels_for_ece']:,} held-out pixels) "
          f"<- this is the number that reflects true generalization")

    print("[7/8] Evaluating spatial alignment on held-out images...")
    eval_df = evaluate_spatial_alignment(eval_ids, npz_index, gt_masks, calibrator)
    if eval_df.empty:
        raise RuntimeError("Evaluation produced no rows — check eval_ids overlap "
                            "with npz_index / gt_masks.")
    print(eval_df[["auroc", f"dice_at_{DICE_THRESHOLD}", "dice_best_threshold"]]
          .describe().loc[["mean", "std", "min", "max"]])

    max_calibrated_prob = eval_df["max_calibrated_prob"].max()
    if max_calibrated_prob < DICE_THRESHOLD:
        print(
            f"      WARNING: the calibrator never outputs a probability >= "
            f"DICE_THRESHOLD ({DICE_THRESHOLD}) on ANY held-out image — max "
            f"calibrated probability seen was {max_calibrated_prob:.4f}. "
            f"This means dice_at_{DICE_THRESHOLD} is 0.0 for every image by "
            f"construction (no pixel is ever classified positive), not "
            f"because localization failed. Use dice_best_threshold instead, "
            f"or lower DICE_THRESHOLD. This is usually driven by low "
            f"positive-pixel prevalence in the calibration set (isotonic "
            f"regression can't output a probability higher than the "
            f"highest LOCAL positive frequency it observed)."
        )

    print("[8/8] Saving overlay visualizations + results_summary.csv...")
    n_examples = min(N_OVERLAY_EXAMPLES, len(eval_ids))
    example_ids = np.random.default_rng(RANDOM_SEED).choice(eval_ids, size=n_examples, replace=False)
    for image_id in example_ids:
        save_overlay_figure(image_id, npz_index, gt_masks, calibrator, H5_PATH, output_dir)

    config_dict = {
        "CALIBRATION_METHOD": CALIBRATION_METHOD,
        "BORDER_EXCLUDE_PX": BORDER_EXCLUDE_PX,
        "CALIBRATION_FRACTION": CALIBRATION_FRACTION,
        "TARGET_CLASSES": TARGET_CLASSES,
        "DICE_THRESHOLD": DICE_THRESHOLD,
    }
    summary_path, per_image_path = save_results_summary(
        eval_df, ece_dict, ece_holdout_dict, config_dict, output_dir
    )
    print(f"Saved: {summary_path}")
    print(f"Saved: {per_image_path}")
    print(f"Saved: {n_examples} overlay PNGs to {output_dir}")


if __name__ == "__main__":
    main()
