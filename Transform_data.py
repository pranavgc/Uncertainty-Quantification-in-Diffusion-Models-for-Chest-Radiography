"""
VinBigData CXR (VinDr-CXR) — Patient-Leakage-Free Preprocessing Pipeline
for Diffusion Model Training
==========================================================================

Designed to run as a Kaggle Notebook. Paste each "# %% [CELL N]" block into
its own notebook cell, or run the whole script as-is.

Pipeline stages:
  1. Extract & cache DICOM metadata (PatientID, ViewPosition, etc.)
  2. Merge metadata with bounding-box annotations; group boxes -> 1 row/image
  3. De-duplicate at the image level; filter to PA/AP views; isolate abnormal
     cases ("No finding" removed) for pathology-conditioned generation
  4. Patient-disjoint Train/Val/Test split via GroupShuffleSplit + leakage assert
  5. Stream DICOM -> resize 512x512 -> normalize to [-1, 1] float32 -> HDF5
     (gzip-compressed, keyed by image_id), processed in chunks to respect
     Kaggle's 20GB working-directory limit
  6. Save split metadata CSVs to /kaggle/working/

Assumes the standard Kaggle input layout:
  /kaggle/input/vinbigdata-chest-xray-abnormalities-detection/
      train/*.dicom
      train.csv
"""

# %% [CELL 1] --------------------------------------------------------------
# Imports & configuration
# ---------------------------------------------------------------------------
import os
import gc
import h5py
import cv2
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

try:
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    HAS_VOI_LUT = True
except ImportError:
    HAS_VOI_LUT = False

# ---- Paths ------------------------------------------------------------
INPUT_DIR = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection")
DICOM_DIR = INPUT_DIR / "train"
ANNOTATIONS_CSV = INPUT_DIR / "train.csv"

WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

H5_PATH = WORKING_DIR / "vindr_cxr_512.h5"
METADATA_CSV = WORKING_DIR / "vindr_cxr_metadata.csv"
TRAIN_CSV_OUT = WORKING_DIR / "vindr_cxr_train.csv"
VAL_CSV_OUT = WORKING_DIR / "vindr_cxr_val.csv"
TEST_CSV_OUT = WORKING_DIR / "vindr_cxr_test.csv"

# ---- Params -------------------------------------------------------------
IMG_SIZE = 512                 # target resolution for the diffusion model
VALID_VIEWS = ["PA", "AP"]     # standard frontal chest projections
NO_FINDING_LABEL = "No finding"
RANDOM_STATE = 42
TEST_SIZE = 0.15               # fraction of patients held out for test
VAL_SIZE = 0.15                # fraction of remaining patients held out for val
HDF5_COMPRESSION = "gzip"
HDF5_COMPRESSION_OPTS = 4      # 1 (fast/large) .. 9 (slow/small); 4 is a good tradeoff
CHUNK_SIZE = 500                # images processed per streaming batch


# %% [CELL 2] --------------------------------------------------------------
# Step 1: Load annotation CSV
# ---------------------------------------------------------------------------
print("Loading annotations...")
annotations = pd.read_csv(ANNOTATIONS_CSV)
print(f"  {len(annotations):,} annotation rows, "
      f"{annotations['image_id'].nunique():,} unique images")


# %% [CELL 3] --------------------------------------------------------------
# Step 2: Extract DICOM metadata (header-only, no pixel decode -> fast)
# ---------------------------------------------------------------------------
def extract_dicom_metadata(dicom_dir: Path) -> pd.DataFrame:
    """Read DICOM headers only (stop_before_pixels=True) and build a
    per-image metadata table. Corrupted / unreadable files are skipped
    and logged rather than crashing the whole pass."""
    records = []
    failed = []
    dicom_files = sorted(dicom_dir.glob("*.dicom"))
    print(f"Scanning {len(dicom_files):,} DICOM files for metadata...")

    for file in tqdm(dicom_files, desc="Reading DICOM headers"):
        try:
            ds = pydicom.dcmread(file, stop_before_pixels=True)
            records.append({
                "image_id": file.stem,
                "patient_id": getattr(ds, "PatientID", None),
                "study_uid": getattr(ds, "StudyInstanceUID", None),
                "series_uid": getattr(ds, "SeriesInstanceUID", None),
                "view": getattr(ds, "ViewPosition", None),
                "rows": getattr(ds, "Rows", None),
                "columns": getattr(ds, "Columns", None),
                "age": getattr(ds, "PatientAge", None),
                "sex": getattr(ds, "PatientSex", None),
                "filepath": str(file),
            })
        except Exception as e:
            failed.append((file.name, str(e)))

    if failed:
        print(f"  WARNING: {len(failed)} DICOM files failed to read and were skipped.")
        pd.DataFrame(failed, columns=["file", "error"]).to_csv(
            WORKING_DIR / "corrupted_dicom_files.csv", index=False
        )

    return pd.DataFrame(records)


metadata = extract_dicom_metadata(DICOM_DIR)
print(f"  Extracted metadata for {len(metadata):,} images, "
      f"{metadata['patient_id'].nunique():,} unique patients")

# Drop rows with missing critical identifiers (patient_id or view)
metadata = metadata.dropna(subset=["patient_id", "image_id"]).reset_index(drop=True)


# %% [CELL 4] --------------------------------------------------------------
# Step 3: Group bounding boxes -> one row per image, then merge with metadata
# ---------------------------------------------------------------------------
def group_boxes(annotations: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-radiologist / multi-row annotations into a single
    JSON-like list-of-dicts column, one row per image_id."""
    box_cols = ["class_name", "class_id", "rad_id",
                "x_min", "y_min", "x_max", "y_max"]
    box_cols = [c for c in box_cols if c in annotations.columns]

    grouped = (
        annotations
        .groupby("image_id")
        .apply(lambda x: x[box_cols].to_dict("records"))
        .reset_index(name="boxes")
    )

    # Convenience flags/counts used later for filtering
    grouped["n_boxes"] = grouped["boxes"].apply(len)
    grouped["labels"] = annotations.groupby("image_id")["class_name"].apply(
        lambda s: sorted(set(s))
    ).reset_index(drop=True)
    grouped["is_normal"] = grouped["labels"].apply(
        lambda labs: labs == [NO_FINDING_LABEL]
    )
    return grouped


boxes = group_boxes(annotations)
print(f"  Grouped into {len(boxes):,} image-level rows "
      f"(1 row per image, boxes nested as a list)")

# Step 3b: de-duplicate metadata at the IMAGE level (a given image_id should
# only appear once in the header scan; this guards against re-runs / dupes)
metadata = metadata.drop_duplicates(subset="image_id", keep="first")

dataset = metadata.merge(boxes, on="image_id", how="inner")
print(f"  Merged dataset: {len(dataset):,} images, "
      f"{dataset['patient_id'].nunique():,} unique patients")


# %% [CELL 5] --------------------------------------------------------------
# Step 4: Filter to standard PA/AP views, isolate abnormal cases
# ---------------------------------------------------------------------------
before = len(dataset)
dataset = dataset[dataset["view"].isin(VALID_VIEWS)].reset_index(drop=True)
print(f"  View filter (PA/AP): {before:,} -> {len(dataset):,} images")

# NOTE on "patient-level de-duplication":
# We deliberately do NOT collapse a patient down to a single image here.
# For diffusion-model training, retaining all of a patient's distinct
# studies/images maximizes usable data and is standard practice; the
# leakage risk that matters is a patient's images being split ACROSS
# train/val/test, not a patient contributing >1 image to the SAME split.
# That risk is eliminated in Step 6 via GroupShuffleSplit on patient_id.
#
# If your protocol instead requires strict one-image-per-patient (e.g. for
# an i.i.d. assumption in some downstream statistical test), uncomment:
#
# dataset = (
#     dataset.sort_values("study_uid")
#            .drop_duplicates("patient_id", keep="first")
#            .reset_index(drop=True)
# )
# print(f"  Strict patient-level de-duplication -> {len(dataset):,} images, "
#       f"{dataset['patient_id'].nunique():,} patients")

# Isolate abnormal cases for pathology-conditioned generation
abnormal_dataset = dataset[~dataset["is_normal"]].reset_index(drop=True)
normal_dataset = dataset[dataset["is_normal"]].reset_index(drop=True)
print(f"  Abnormal (pathology) images: {len(abnormal_dataset):,}")
print(f"  Normal ('No finding') images: {len(normal_dataset):,}")

# For this pipeline we train the diffusion model on abnormal (pathology)
# cases, matching the stated project objective. Swap `dataset = normal_dataset`
# or `dataset = pd.concat([...])` if a different mix is desired.
dataset = abnormal_dataset.copy()


# %% [CELL 6] --------------------------------------------------------------
# Step 5: Patient-disjoint Train / Val / Test split (GroupShuffleSplit)
# ---------------------------------------------------------------------------
def group_shuffle_split_three(df, group_col, test_size, val_size, random_state):
    """Two-stage GroupShuffleSplit -> disjoint train/val/test by group."""
    # Stage 1: carve out test set
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(df, groups=df[group_col]))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Stage 2: carve val out of remaining train_val, val_size expressed
    # relative to the ORIGINAL dataset, so rescale relative to train_val_df
    relative_val_size = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size,
                              random_state=random_state)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df[group_col]))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


train_df, val_df, test_df = group_shuffle_split_three(
    dataset, group_col="patient_id",
    test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE
)

print(f"  Train: {len(train_df):,} images / {train_df['patient_id'].nunique():,} patients")
print(f"  Val:   {len(val_df):,} images / {val_df['patient_id'].nunique():,} patients")
print(f"  Test:  {len(test_df):,} images / {test_df['patient_id'].nunique():,} patients")

# ---- Hard assertion: zero patient overlap across any pair of splits ----
train_p, val_p, test_p = (
    set(train_df.patient_id), set(val_df.patient_id), set(test_df.patient_id)
)
assert len(train_p & val_p) == 0, "LEAKAGE: patients shared between train and val!"
assert len(train_p & test_p) == 0, "LEAKAGE: patients shared between train and test!"
assert len(val_p & test_p) == 0, "LEAKAGE: patients shared between val and test!"
print("  ✔ Verified: zero patient leakage across train/val/test splits.")

for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    split_df["split"] = split_name

full_split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)


# %% [CELL 7] --------------------------------------------------------------
# Step 6: Save split metadata as CSV (drop the heavy `boxes` list-of-dicts
# to a JSON string so it round-trips cleanly through CSV)
# ---------------------------------------------------------------------------
def serialize_boxes_column(df):
    df = df.copy()
    df["boxes"] = df["boxes"].apply(lambda b: str(b))
    df["labels"] = df["labels"].apply(lambda l: str(l))
    return df

serialize_boxes_column(full_split_df).to_csv(METADATA_CSV, index=False)
serialize_boxes_column(train_df).to_csv(TRAIN_CSV_OUT, index=False)
serialize_boxes_column(val_df).to_csv(VAL_CSV_OUT, index=False)
serialize_boxes_column(test_df).to_csv(TEST_CSV_OUT, index=False)

print(f"  Saved metadata CSVs to {WORKING_DIR}")


# %% [CELL 8] --------------------------------------------------------------
# Step 7: DICOM -> 512x512 float32 [-1,1] tensor -> HDF5 (gzip), streamed
# ---------------------------------------------------------------------------
def dicom_to_normalized_array(filepath: str, size: int = IMG_SIZE) -> np.ndarray:
    """Read a DICOM file, apply VOI LUT / rescale if present, resize, and
    normalize pixel intensities to [-1, 1] as float32 (diffusion-model
    convention)."""
    ds = pydicom.dcmread(filepath)
    pixels = ds.pixel_array.astype(np.float32)

    # Apply VOI LUT / windowing when available for correct contrast
    if HAS_VOI_LUT:
        try:
            pixels = apply_voi_lut(pixels, ds).astype(np.float32)
        except Exception:
            pass  # fall back to raw pixel array

    # MONOCHROME1 images have inverted intensity relative to MONOCHROME2
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    # Min-max scale to [0, 1] using this image's own range
    p_min, p_max = pixels.min(), pixels.max()
    if p_max > p_min:
        pixels = (pixels - p_min) / (p_max - p_min)
    else:
        pixels = np.zeros_like(pixels)

    # Resize to target resolution
    pixels = cv2.resize(pixels, (size, size), interpolation=cv2.INTER_AREA)

    # Normalize [0, 1] -> [-1, 1] for diffusion-model training
    pixels = pixels * 2.0 - 1.0

    return pixels.astype(np.float32)


def process_and_write_h5(df: pd.DataFrame, h5_path: Path,
                          size: int = IMG_SIZE, chunk_size: int = CHUNK_SIZE):
    """Stream DICOM -> normalized array -> HDF5 dataset keyed by image_id,
    processed in chunks to bound peak memory and keep working-dir usage
    predictable (gzip compression keeps the .h5 well under the 20GB cap)."""
    n_total = len(df)
    n_failed = 0

    with h5py.File(h5_path, "a") as h5f:
        for start in tqdm(range(0, n_total, chunk_size), desc="Writing HDF5 chunks"):
            chunk = df.iloc[start:start + chunk_size]
            for _, row in chunk.iterrows():
                image_id = row["image_id"]
                if image_id in h5f:
                    continue  # resume-safe: skip already-written keys
                try:
                    arr = dicom_to_normalized_array(row["filepath"], size=size)
                    h5f.create_dataset(
                        image_id,
                        data=arr,
                        compression=HDF5_COMPRESSION,
                        compression_opts=HDF5_COMPRESSION_OPTS,
                        dtype="float32",
                    )
                    # Store split/patient as attrs for convenient downstream filtering
                    h5f[image_id].attrs["patient_id"] = str(row["patient_id"])
                    h5f[image_id].attrs["split"] = str(row["split"])
                    h5f[image_id].attrs["view"] = str(row["view"])
                except Exception as e:
                    n_failed += 1
                    print(f"  Failed on {image_id}: {e}")
            # Free memory between chunks
            gc.collect()

    print(f"  Done. {n_total - n_failed:,}/{n_total:,} images written to {h5_path}")
    if n_failed:
        print(f"  WARNING: {n_failed} images failed during pixel processing.")


print("Processing images to HDF5 (this streams in chunks; safe to interrupt/resume)...")
process_and_write_h5(full_split_df, H5_PATH, size=IMG_SIZE, chunk_size=CHUNK_SIZE)


# %% [CELL 9] --------------------------------------------------------------
# Sanity check + storage summary
# ---------------------------------------------------------------------------
h5_size_gb = os.path.getsize(H5_PATH) / (1024 ** 3)
print(f"\nFinal HDF5 size: {h5_size_gb:.2f} GB  (limit: 20 GB working-dir cap)")

with h5py.File(H5_PATH, "r") as h5f:
    sample_key = next(iter(h5f.keys()))
    sample = h5f[sample_key][:]
    print(f"Sample tensor '{sample_key}': shape={sample.shape}, dtype={sample.dtype}, "
          f"min={sample.min():.3f}, max={sample.max():.3f}")
    print(f"Total keys stored: {len(h5f.keys()):,}")

print("\nOutputs in /kaggle/working/:")
for f in [H5_PATH, METADATA_CSV, TRAIN_CSV_OUT, VAL_CSV_OUT, TEST_CSV_OUT]:
    if f.exists():
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")

print("\nPipeline complete. Publish /kaggle/working/ as a Kaggle Dataset to persist it.")