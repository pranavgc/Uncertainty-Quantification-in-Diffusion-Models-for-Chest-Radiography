"""
vinbigdata_uncertainty_classification.py
============================================================================
Kaggle Notebook, GPU T4 x2.

Downstream classification stage of the epistemic-uncertainty project. Prior
stages in this project:

  1. vinbigdata_diffusion_pipeline.py       -> vindr_cxr_512.h5 + metadata csv
  2. ms-the-inversion-generator.ipynb        -> uncertainty_chunk_NNN.npz
     (== ddim_epistemic_uncertainty_pipeline.py)  (+ companion _meta.csv), via DDIM inversion +
                                                20-sample stochastic ensemble
  3. calibration_spatial_evaluation_pipeline.py -> calibrates variance maps,
                                                checks spatial alignment with
                                                radiologist boxes

THIS script asks a different question: does filtering synthetic augmentation
images by their epistemic-uncertainty score actually improve a downstream
Normal-vs-Abnormal *classifier* trained on real + synthetic data, evaluated
on a clean real-only test set?

INPUTS (edit CONFIG in Cell 1):
  - Real data: a "modified" VinBigData Kaggle dataset containing `train.csv`
    (official box-level annotations; class_id 14 == "No finding") and the
    real images themselves. Real images are located either as loose image
    files under REAL_DATA_DIR (searched across a few common subfolder/
    extension combinations), or, as a fallback, inside the project's own
    vindr_cxr_512.h5 (produced by vinbigdata_diffusion_pipeline.py) if
    REAL_H5_PATH is set.
  - Synthetic data: the SAME `uncertainty_chunk_NNN.npz` (+ `_meta.csv`)
    files produced by ms-the-inversion-generator.ipynb (stage 2 above).
    There is no separate "synthetic image folder" — the pipeline's
    "{image_id}__mean" array (the mean of the 20-sample DDIM ensemble) IS
    the synthetic augmentation image, and "{image_id}__variance" (or the
    meta.csv's pre-computed `variance_mean`) is the source of each image's
    global epistemic-uncertainty score.

OUTPUTS (written to /kaggle/working/):
  - roc_comparison.png — ROC curves for unfiltered vs. uncertainty-filtered
    synthetic augmentation, with AUC in the legend.

Known project caveats this script bakes in (see prior stages' notes — do
not "fix" these without updating the other pipeline scripts too):
  - Outer ~20px border ring of every variance map runs ~44% hotter than the
    interior (VAE-decode edge effect, not signal — known issue #3 from the
    generator stage, already handled the same way in
    calibration_spatial_evaluation_pipeline.py). By default this script
    re-derives each synthetic image's global uncertainty score from the
    raw "{image_id}__variance" array, averaged over the INTERIOR region
    only, rather than trusting the generator's own `variance_mean` column
    in `_meta.csv` (which averages over the full image, border included).
    Set APPLY_BORDER_CORRECTION = False to use the raw meta.csv value
    instead. Keep BORDER_EXCLUDE_PX in sync with
    calibration_spatial_evaluation_pipeline.py's constant of the same name.
  - Every synthetic image is a DDIM reconstruction of ONE specific real
    image_id (RARE_FINDINGS-targeted, per ms-the-inversion-generator.ipynb)
    — synthetic image_ids are literally shared with the real dataset's
    image_ids. This means a synthetic reconstruction of a real image that
    landed in the TEST split would leak test information into training if
    included. This script guards against that: any synthetic image whose
    source image_id falls in the held-out test split is excluded from
    training regardless of its uncertainty score.
  - As of the current generator run, RARE_FINDINGS effectively yields a
    Pneumothorax-only synthetic pool (a known, separately-tracked bug —
    "Lung cyst" isn't a real class in this CSV version). This script
    doesn't assume a specific pathology mix; it just reflects whatever
    `finding` values are actually present in the synthetic meta files.
"""

# ============================================================================
# CELL 0 — Install / verify dependencies
# ============================================================================
# torch/torchvision/sklearn/pandas/matplotlib are Kaggle-preinstalled;
# h5py is used for the optional real-image .h5 fallback and is usually
# preinstalled too, but we install quietly/idempotently just in case (safe
# to re-run), matching Cell 0's convention in
# calibration_spatial_evaluation_pipeline.py.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "h5py"], check=False)

import os
import re
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt


# ============================================================================
# CELL 1 — CONFIG (edit these paths for your Kaggle setup)
# ============================================================================
# --- Real data ------------------------------------------------------------
REAL_DATA_DIR = "/kaggle/input/vinbigdata-modified"          # real images + train.csv
REAL_TRAIN_CSV = os.path.join(REAL_DATA_DIR, "train.csv")

# Loose real image files are searched here first (a few common layouts):
REAL_IMAGE_SEARCH_DIRS = [
    os.path.join(REAL_DATA_DIR, "train"),
    os.path.join(REAL_DATA_DIR, "train_images"),
    os.path.join(REAL_DATA_DIR, "images"),
    REAL_DATA_DIR,
]
REAL_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".dicom", ".dcm")

# Fallback if a real image_id isn't found as a loose file above: read it
# from this project's own preprocessed HDF5 (produced by
# vinbigdata_diffusion_pipeline.py, [-1, 1]-normalized float32, 512x512).
# Set to None to disable this fallback entirely.
REAL_H5_PATH = "/kaggle/input/datasets/pgc17ms072/vindr-cxr-512-h5/vindr_cxr_512.h5"

# --- Synthetic data ---------------------------------------------------------
# Directory containing uncertainty_chunk_NNN.npz (+ companion
# uncertainty_chunk_NNN_meta.csv) produced by ms-the-inversion-generator.ipynb.
SYNTHETIC_DATA_DIR = "/kaggle/input/ddim-uncertainty-maps"

BORDER_EXCLUDE_PX = 20                # keep in sync with
                                       # calibration_spatial_evaluation_pipeline.py
APPLY_BORDER_CORRECTION = True        # see module docstring
SYNTHETIC_ARRAY_SIZE = 512            # npz mean/variance arrays are 512x512

# --- Labeling ---------------------------------------------------------------
NO_FINDING_CLASS_ID = 14              # VinBigData: class_id 14 == "No finding"
LABEL_NORMAL, LABEL_ABNORMAL = 0, 1
# Fallback label for a synthetic image with no `finding` in its meta AND no
# match in the real label table. The generator only ever targets rare
# pathology classes (never "No finding"), so Abnormal is the safe default.
DEFAULT_SYNTHETIC_LABEL = LABEL_ABNORMAL

# --- Training hyperparameters ------------------------------------------------
IMG_SIZE = 224            # classifier input resolution (resize target)
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
TEST_SIZE = 0.2           # fraction of REAL labeled data held out as the clean test set
RANDOM_SEED = 42
NUM_WORKERS = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

OUTPUT_DIR = "/kaggle/working"

# The two experiments this script compares.
THRESHOLD_UNFILTERED = 1.0    # effectively "include all synthetic images"
THRESHOLD_FILTERED = 0.5      # only synthetic images with uncertainty <= 0.5


# ============================================================================
# CELL 2 — Reproducibility
# ============================================================================
def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed every RNG we touch so both experiments start from identical
    model init and data shuffling — isolating the uncertainty threshold as
    the only variable that differs between the two runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# CELL 3 — train.csv label parsing + real image lookup (file or .h5)
# ============================================================================
# VinBigData's `train.csv` has one row PER bounding-box annotation, so a
# single image_id can appear many times (once per finding, or once with a
# dummy full-image box for class_id 14 "No finding"). We collapse this to
# one binary label per image: Normal (0) iff every annotation for that
# image is class_id 14, else Abnormal (1).
def parse_real_labels(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns or "class_id" not in df.columns:
        raise ValueError(
            f"{csv_path} is missing expected columns 'image_id'/'class_id'. "
            f"Found columns: {list(df.columns)}"
        )
    labels = {}
    grouped = df.groupby("image_id")["class_id"].apply(lambda s: set(s.tolist()))
    for image_id, class_ids in grouped.items():
        labels[image_id] = (
            LABEL_NORMAL if class_ids == {NO_FINDING_CLASS_ID} else LABEL_ABNORMAL
        )
    return labels


def find_real_image_file(image_id: str):
    """Search the candidate real-image directories/extensions for this
    image_id as a loose file. Returns None if not found."""
    for directory in REAL_IMAGE_SEARCH_DIRS:
        for ext in REAL_IMAGE_EXTENSIONS:
            candidate = os.path.join(directory, image_id + ext)
            if os.path.exists(candidate):
                return candidate
    return None


def get_h5_keys(h5_path) -> set:
    """One-time read of every image_id available in the fallback .h5, so
    dataset construction can report accurate missing-image counts instead
    of discovering failures mid-training."""
    if h5_path is None or not os.path.exists(h5_path):
        return set()
    with h5py.File(h5_path, "r") as f:
        return set(f.keys())


# ============================================================================
# CELL 4 — Image decoding helpers (real file / real .h5 / synthetic npz)
# ============================================================================
def array01_to_pil(arr: np.ndarray) -> Image.Image:
    """Grayscale float array assumed already in [0, 1] -> RGB PIL image
    (3 identical channels, matching the standard trick for feeding
    single-channel medical images into ImageNet-pretrained 3-channel
    models)."""
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
    return img.convert("RGB")


def h5_pm1_array_to_pil(arr: np.ndarray) -> Image.Image:
    """vinbigdata_diffusion_pipeline.py normalizes real images to [-1, 1]
    float32; rescale to [0, 1] before the shared uint8/PIL conversion."""
    arr01 = np.clip((arr.astype(np.float32) + 1.0) / 2.0, 0.0, 1.0)
    return array01_to_pil(arr01)


def load_real_file_as_pil(path: str) -> Image.Image:
    if path.lower().endswith((".dcm", ".dicom")):
        import pydicom  # imported lazily; only needed if DICOMs are present

        dcm = pydicom.dcmread(path)
        arr = dcm.pixel_array.astype(np.float32)
        arr01 = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return array01_to_pil(arr01)
    return Image.open(path).convert("RGB")


def load_h5_image_as_pil(h5_path: str, image_id: str) -> Image.Image:
    with h5py.File(h5_path, "r") as f:
        arr = f[image_id][()]
    return h5_pm1_array_to_pil(arr)


def load_npz_mean_as_pil(npz_path, image_id: str) -> Image.Image:
    """The "{image_id}__mean" array IS the synthetic augmentation image —
    the mean reconstruction across the 20-sample DDIM ensemble, already in
    [0, 1] (see decode_latents_to_grayscale in ms-the-inversion-
    generator.ipynb)."""
    with np.load(npz_path) as data:
        arr = data[f"{image_id}__mean"].astype(np.float32)
    return array01_to_pil(arr)


# ============================================================================
# CELL 5 — Synthetic npz/meta indexing + border-corrected global scores
# ============================================================================
def make_interior_mask(size: int = SYNTHETIC_ARRAY_SIZE, border: int = BORDER_EXCLUDE_PX) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    m[border: size - border, border: size - border] = True
    return m


INTERIOR_MASK = make_interior_mask()


def index_synthetic_npz(npz_dir: str):
    """Mirrors calibration_spatial_evaluation_pipeline.py's
    index_variance_npz: scans uncertainty_chunk_*.npz for every image_id
    present (via the "{image_id}__mean" key) and reads each chunk's
    companion _meta.csv for the `finding` label and the generator's own
    (uncorrected) variance_mean/variance_max.

    Returns:
        npz_index: {image_id: npz_path}
        meta:      {image_id: {"finding": str|None, "variance_mean_raw": float, "variance_max": float}}
    """
    npz_dir = Path(npz_dir)
    npz_paths = sorted(npz_dir.glob("uncertainty_chunk_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No uncertainty_chunk_*.npz files found under {npz_dir}")

    npz_index = {}
    for p in npz_paths:
        with np.load(p) as data:
            for key in data.files:
                if key.endswith("__mean"):
                    npz_index[key[: -len("__mean")]] = p

    meta = {}
    for p in npz_paths:
        meta_path = p.with_name(p.stem + "_meta.csv")
        if not meta_path.exists():
            warnings.warn(
                f"[index_synthetic_npz] Missing companion meta file {meta_path} "
                f"for {p.name}; its images will lack a 'finding' label and a "
                f"pre-computed variance_mean fallback."
            )
            continue
        meta_df = pd.read_csv(meta_path)
        for _, row in meta_df.iterrows():
            meta[row["image_id"]] = {
                "finding": row.get("finding"),
                "variance_mean_raw": float(row["variance_mean"]),
                "variance_max": float(row["variance_max"]) if "variance_max" in row else np.nan,
            }
    return npz_index, meta


def compute_global_uncertainty_scores(npz_index: dict, meta: dict,
                                       apply_border_correction: bool = APPLY_BORDER_CORRECTION) -> dict:
    """Builds {image_id: global_uncertainty_score}, used for threshold
    filtering. See module docstring for the border-correction rationale.
    Chunk files are opened once each (grouped by path), not once per image.
    """
    if not apply_border_correction:
        return {image_id: m["variance_mean_raw"] for image_id, m in meta.items()}

    by_path = {}
    for image_id, path in npz_index.items():
        by_path.setdefault(path, []).append(image_id)

    scores = {}
    for path, image_ids in tqdm(by_path.items(), desc="Computing border-corrected uncertainty scores"):
        with np.load(path) as data:
            for image_id in image_ids:
                var_key = f"{image_id}__variance"
                if var_key in data.files:
                    var_map = data[var_key].astype(np.float32)
                    scores[image_id] = float(var_map[INTERIOR_MASK].mean())
                elif image_id in meta:
                    # No raw variance array (shouldn't normally happen) —
                    # fall back to the generator's own uncorrected score.
                    scores[image_id] = meta[image_id]["variance_mean_raw"]
    return scores


# ============================================================================
# CELL 6 — `VinBigDataset`
# ============================================================================
# Real images are ALWAYS included regardless of `max_uncertainty_threshold`
# (that parameter only ever filters synthetic images). Set
# `include_synthetic=False` (used for the held-out test set) to build a
# real-image-only dataset regardless of what synthetic args are passed.
class VinBigDataset(Dataset):
    def __init__(
        self,
        real_labels: dict,
        real_ids: list,
        synthetic_npz_index: dict = None,
        synthetic_meta: dict = None,
        synthetic_scores: dict = None,
        max_uncertainty_threshold: float = 1.0,
        leakage_guard_ids: set = None,
        transform=None,
        include_synthetic: bool = True,
        real_h5_path: str = REAL_H5_PATH,
    ):
        self.transform = transform
        self.real_h5_path = real_h5_path
        self.samples = []  # list of (ref_tuple, label); ref_tuple dispatched in __getitem__

        # ---- Real images: always included ----
        h5_keys = get_h5_keys(real_h5_path)
        missing_real = 0
        for image_id in real_ids:
            path = find_real_image_file(image_id)
            if path is not None:
                self.samples.append((("file", path), real_labels[image_id]))
            elif image_id in h5_keys:
                self.samples.append((("h5", image_id), real_labels[image_id]))
            else:
                missing_real += 1
        if missing_real:
            warnings.warn(
                f"[VinBigDataset] {missing_real} real image_ids from train.csv "
                f"were not found as loose files or in REAL_H5_PATH, and were skipped."
            )

        # ---- Synthetic images: filtered by uncertainty threshold + leakage guard ----
        self.n_synthetic_included = 0
        self.n_synthetic_excluded_threshold = 0
        self.n_synthetic_excluded_missing_score = 0
        self.n_synthetic_excluded_leakage = 0

        if include_synthetic and synthetic_npz_index:
            synthetic_meta = synthetic_meta or {}
            synthetic_scores = synthetic_scores or {}
            leakage_guard_ids = leakage_guard_ids or set()

            for image_id, npz_path in synthetic_npz_index.items():
                # Leakage guard: every synthetic image is a DDIM
                # reconstruction of ONE specific real image_id. If that
                # source image_id is part of the held-out test split,
                # training on its reconstruction leaks test information —
                # exclude it regardless of uncertainty score.
                if image_id in leakage_guard_ids:
                    self.n_synthetic_excluded_leakage += 1
                    continue

                score = synthetic_scores.get(image_id)
                if score is None:
                    # No verifiable uncertainty score for this image —
                    # exclude rather than assume it's safe to use.
                    self.n_synthetic_excluded_missing_score += 1
                    continue
                if score > max_uncertainty_threshold:
                    self.n_synthetic_excluded_threshold += 1
                    continue

                finding = synthetic_meta.get(image_id, {}).get("finding")
                if finding:
                    # The generator only ever targets rare pathology
                    # classes (never "No finding"), so a present finding
                    # is a reliable Abnormal signal.
                    label = LABEL_ABNORMAL
                else:
                    label = real_labels.get(image_id, DEFAULT_SYNTHETIC_LABEL)

                self.samples.append((("npz", npz_path, image_id), label))
                self.n_synthetic_included += 1

        print(
            f"[VinBigDataset] threshold={max_uncertainty_threshold} | "
            f"real_included={len(real_ids) - missing_real} | "
            f"synthetic_included={self.n_synthetic_included} | "
            f"synthetic_excluded_by_threshold={self.n_synthetic_excluded_threshold} | "
            f"synthetic_excluded_missing_score={self.n_synthetic_excluded_missing_score} | "
            f"synthetic_excluded_leakage={self.n_synthetic_excluded_leakage} | "
            f"total_samples={len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ref, label = self.samples[idx]
        kind = ref[0]
        if kind == "file":
            image = load_real_file_as_pil(ref[1])
        elif kind == "h5":
            image = load_h5_image_as_pil(self.real_h5_path, ref[1])
        else:  # "npz"
            image = load_npz_mean_as_pil(ref[1], ref[2])
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================================
# CELL 7 — Transforms
# ============================================================================
# No horizontal flip: flipping a chest X-ray reverses left/right anatomical
# laterality, which is not a valid augmentation for this modality. Only
# small rotation + mild brightness/contrast jitter are used for training;
# the test/eval transform is deterministic (resize + normalize only).
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ============================================================================
# CELL 8 — Model
# ============================================================================
def build_model() -> nn.Module:
    try:
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    except AttributeError:
        # Older torchvision (<0.13) uses the `pretrained=True` kwarg instead
        # of the `weights` enum.
        model = models.densenet121(pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)  # 2-way softmax: [Normal, Abnormal]
    return model


# ============================================================================
# CELL 9 — Train / evaluate loops (AMP)
# ============================================================================
def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch, num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # autocast runs the forward pass (and loss) in float16 where it's
        # numerically safe, falling back to float32 for ops that need it
        # (e.g. batchnorm reductions). GradScaler then scales the loss
        # before backward() so small float16 gradients don't underflow to
        # zero, and unscales them again before the optimizer step.
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"  Epoch [{epoch + 1}/{num_epochs}] train_loss={epoch_loss:.4f} train_acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(Abnormal)
        all_probs.append(probs.float().cpu())
        all_labels.append(labels)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    return fpr, tpr, auc


# ============================================================================
# CELL 10 — One full experiment (build dataset at a given threshold, train, evaluate)
# ============================================================================
def run_experiment(
    name: str,
    threshold: float,
    real_labels: dict,
    train_ids: list,
    synthetic_npz_index: dict,
    synthetic_meta: dict,
    synthetic_scores: dict,
    leakage_guard_ids: set,
    test_dataset: Dataset,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
):
    print(f"\n{'=' * 70}\nExperiment: {name}  (max_uncertainty_threshold={threshold})\n{'=' * 70}")

    # Re-seed so both experiments get identical model init and identical
    # data shuffling order — the ONLY thing that should differ between runs
    # is which synthetic images survive the uncertainty threshold.
    set_seed(RANDOM_SEED)

    train_dataset = VinBigDataset(
        real_labels=real_labels,
        real_ids=train_ids,
        synthetic_npz_index=synthetic_npz_index,
        synthetic_meta=synthetic_meta,
        synthetic_scores=synthetic_scores,
        max_uncertainty_threshold=threshold,
        leakage_guard_ids=leakage_guard_ids,
        transform=train_transform,
        include_synthetic=True,
        real_h5_path=REAL_H5_PATH,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = build_model().to(device)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs via nn.DataParallel")
        model = nn.DataParallel(model)

    # Class-imbalance handling: weight CE loss inversely to class frequency
    # actually observed in THIS training set (real + surviving synthetic),
    # rather than assuming a 50/50 split — the two thresholds can produce
    # meaningfully different class balances.
    labels_arr = np.array([lbl for _, lbl in train_dataset.samples])
    class_counts = np.bincount(labels_arr, minlength=2).astype(np.float32)
    class_weights = torch.tensor(
        class_counts.sum() / (2.0 * np.maximum(class_counts, 1.0)), dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, epoch, num_epochs)

    fpr, tpr, auc = evaluate(model, test_loader, device)
    print(f"  [{name}] Test AUC = {auc:.4f}")

    del model
    torch.cuda.empty_cache()
    return fpr, tpr, auc


# ============================================================================
# CELL 11 — main(): runs both experiments and produces the comparison plot
# ============================================================================
def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | visible GPUs: {torch.cuda.device_count()}")

    # ---- Labels from train.csv ----
    real_labels = parse_real_labels(REAL_TRAIN_CSV)
    all_ids = list(real_labels.keys())
    all_lbls = [real_labels[i] for i in all_ids]

    # ---- Clean, label-stratified hold-out test split ----
    # VinBigData's public Kaggle test set has no released labels, so we
    # carve a fixed stratified hold-out out of the labeled real pool to
    # serve as the "clean official test set". This split is computed ONCE
    # and reused identically for both threshold experiments below, so any
    # AUC difference is attributable to the synthetic-data filtering and
    # not to differing test data.
    train_ids, test_ids = train_test_split(
        all_ids, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=all_lbls,
    )
    test_ids_set = set(test_ids)
    print(f"Real images: {len(all_ids)} total -> {len(train_ids)} train / {len(test_ids)} test")

    # ---- Synthetic data: index the DDIM-generator's npz/meta outputs ----
    print("Indexing synthetic uncertainty_chunk_*.npz files...")
    synthetic_npz_index, synthetic_meta = index_synthetic_npz(SYNTHETIC_DATA_DIR)
    print(f"      found {len(synthetic_npz_index)} synthetic images across "
          f"{len(set(synthetic_npz_index.values()))} chunk file(s)")

    print("Computing global uncertainty scores "
          f"(border-corrected={APPLY_BORDER_CORRECTION})...")
    synthetic_scores = compute_global_uncertainty_scores(synthetic_npz_index, synthetic_meta)

    n_leak = sum(1 for image_id in synthetic_npz_index if image_id in test_ids_set)
    print(f"      {n_leak} synthetic image(s) derive from a real image in the "
          f"TEST split; these will be excluded from BOTH training runs "
          f"regardless of threshold (leakage guard).")

    # ---- Clean test dataset: real images ONLY, no synthetic, no augmentation ----
    test_dataset = VinBigDataset(
        real_labels=real_labels,
        real_ids=test_ids,
        synthetic_npz_index=None,
        transform=eval_transform,
        include_synthetic=False,
        real_h5_path=REAL_H5_PATH,
    )

    # ---- Run A: unfiltered synthetic data ----
    fpr_a, tpr_a, auc_a = run_experiment(
        "Dataset A (unfiltered synthetic)", THRESHOLD_UNFILTERED,
        real_labels, train_ids, synthetic_npz_index, synthetic_meta, synthetic_scores,
        test_ids_set, test_dataset, device,
    )

    # ---- Run B: uncertainty-filtered synthetic data ----
    fpr_b, tpr_b, auc_b = run_experiment(
        "Dataset B (uncertainty-filtered synthetic)", THRESHOLD_FILTERED,
        real_labels, train_ids, synthetic_npz_index, synthetic_meta, synthetic_scores,
        test_ids_set, test_dataset, device,
    )

    # ---- Comparative ROC plot ----
    plt.figure(figsize=(7, 7))
    plt.plot(fpr_a, tpr_a, linewidth=2,
              label=f"Unfiltered synthetic (thr={THRESHOLD_UNFILTERED}), AUC={auc_a:.3f}")
    plt.plot(fpr_b, tpr_b, linewidth=2,
              label=f"Uncertainty-filtered synthetic (thr={THRESHOLD_FILTERED}), AUC={auc_b:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Effect of Uncertainty-Based Synthetic Data Filtering\non VinBigData Normal/Abnormal Classification")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, "roc_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparative ROC plot to {out_path}")
    plt.show()

    print("\nSummary:")
    print(f"  Dataset A (threshold={THRESHOLD_UNFILTERED}, unfiltered): AUC = {auc_a:.4f}")
    print(f"  Dataset B (threshold={THRESHOLD_FILTERED}, filtered):     AUC = {auc_b:.4f}")
    delta = auc_b - auc_a
    verdict = "filtering helped" if delta > 0 else "filtering did not help"
    print(f"  Delta (B - A): {delta:+.4f}  ({verdict})")


if __name__ == "__main__":
    main()
