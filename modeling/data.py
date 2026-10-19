"""Labels, data splits, augmentation and datasets for the cropped .npy volumes (crop.py output)."""

import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch.utils.data import Dataset

# DDx: 1 pleomorphic adenoma, 2 Warthin tumour, 3 other benign, 4 malignant.
MULTICLASS_MAP = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = {
    "binary": ["Benign", "Malignant"],
    "multiclass": ["Pleomorphic Adenoma", "Warthin's Tumor", "Other Benign"],
}
SEQUENCE_MODALITY = {
    "T1": "structural", "T2": "structural", "T1c": "structural",
    "B0": "diffusion", "DWI": "diffusion", "ADC": "diffusion",
}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

REQUIRED_COLUMNS = ("Scan_ID", "DDx", "Sex", "Age", "Laterality")
_SEX = {"m": "M", "male": "M", "f": "F", "female": "F"}
_LATERALITY = {"r": "R", "right": "R", "l": "L", "left": "L"}
_AGE_UNITS = {"D": 365.25, "W": 52.1775, "M": 12.0, "Y": 1.0}


def num_classes(task: str) -> int:
    return 2 if task == "binary" else 3


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def parse_age(value) -> float:
    """Years from 57, '57', '057Y' (DICOM age string) or '6M'. NaN if missing."""
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([DWMY]?)", str(value).strip().upper())
    if m is None:
        raise ValueError(f"Unrecognised age '{value}'")
    return float(m.group(1)) / _AGE_UNITS[m.group(2) or "Y"]


def _normalize(series: pd.Series, mapping: dict, column: str) -> pd.Series:
    def one(v):
        if pd.isna(v) or str(v).strip() == "":
            return np.nan
        key = str(v).strip().lower()
        if key not in mapping:
            raise ValueError(f"Unrecognised {column} value '{v}' (expected one of {sorted(set(mapping))})")
        return mapping[key]

    out = series.map(one)
    if out.isna().any():
        warnings.warn(f"{int(out.isna().sum())} rows have no {column}; encoded as 0 in the clinical vector.")
    return out


def load_labels(csv_path) -> pd.DataFrame:
    """Reads and validates a label table. Accepts R/L or Right/Left, M/F, ages as numbers or
    DICOM age strings; drops empty rows; derives Malignant from DDx."""
    df = pd.read_csv(csv_path, dtype={"Scan_ID": str}, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    df = df.dropna(how="all")
    df = df[df["Scan_ID"].notna() & (df["Scan_ID"].str.strip() != "")].copy()
    df["Scan_ID"] = df["Scan_ID"].str.strip()
    dupes = df["Scan_ID"][df["Scan_ID"].duplicated()].unique()
    if len(dupes):
        raise ValueError(f"Duplicate Scan_ID values: {list(dupes)[:5]}")

    df["DDx"] = pd.to_numeric(df["DDx"], errors="raise").astype(int)
    unknown = sorted(set(df["DDx"]) - {1, 2, 3, 4})
    if unknown:
        raise ValueError(f"DDx must be 1-4, found {unknown}")
    df["Malignant"] = (df["DDx"] == 4).astype(int)
    df["Age"] = df["Age"].map(parse_age)
    df["Sex"] = _normalize(df["Sex"], _SEX, "Sex")
    df["Laterality"] = _normalize(df["Laterality"], _LATERALITY, "Laterality")
    return df.reset_index(drop=True)


def clinical_tensor(row) -> torch.Tensor:
    """[age / 100, sex (M=1), laterality (R=1)]; missing age -> 0.5."""
    age = row["Age"] / 100.0 if pd.notna(row["Age"]) else 0.5
    sex = 1.0 if row["Sex"] == "M" else 0.0
    lat = 1.0 if row["Laterality"] == "R" else 0.0
    return torch.tensor([age, sex, lat], dtype=torch.float32)


def task_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """The multiclass task covers benign tumours only."""
    return df[df["DDx"] != 4].reset_index(drop=True) if task == "multiclass" else df.reset_index(drop=True)


def task_label(row, task: str) -> int:
    return int(row["Malignant"]) if task == "binary" else MULTICLASS_MAP[int(row["DDx"])]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def patient_groups(df: pd.DataFrame) -> pd.Series:
    """Scan IDs with a numeric suffix (P0001_2) are further studies of one patient."""
    return df["Scan_ID"].map(lambda s: re.sub(r"_\d+$", "", str(s)))


def has_repeat_patients(df: pd.DataFrame) -> bool:
    return bool(patient_groups(df).duplicated().any())


def stratified_holdout(df: pd.DataFrame, fraction: float, seed: int):
    """Stratified (on DDx) train/holdout split. Repeat studies of one patient stay together."""
    if has_repeat_patients(df):
        n_splits = max(2, int(round(1 / fraction)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr, ho = next(sgkf.split(df, df["DDx"], patient_groups(df)))
        return df.iloc[tr], df.iloc[ho]
    return train_test_split(df, test_size=fraction, stratify=df["DDx"], random_state=seed)


def cv_splits(pool: pd.DataFrame, n_splits: int, seed: int) -> list:
    """Stratified (on DDx) K-fold indices; grouped by patient when repeat studies exist."""
    if has_repeat_patients(pool):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(sgkf.split(pool, pool["DDx"], patient_groups(pool)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(pool, pool["DDx"]))


def check_disjoint(a: pd.DataFrame, b: pd.DataFrame, what: str) -> None:
    overlap = set(patient_groups(a)) & set(patient_groups(b))
    if overlap:
        raise RuntimeError(f"{what}: {len(overlap)} patients on both sides, e.g. {sorted(overlap)[:3]}")


def load_split(results_dir) -> tuple:
    results_dir = Path(results_dir)
    pool = load_labels(results_dir / "train_cv_split.csv")
    test = load_labels(results_dir / "test_split.csv")
    check_disjoint(pool, test, "CV pool / test set")
    return pool, test


def load_train_pool(cfg) -> pd.DataFrame:
    """The CV pool; under fast_run a small stratified subsample. Never includes the test set."""
    pool, _ = load_split(cfg.paths.results_dir)
    if not cfg.fast_run:
        return pool
    parts = [g.sample(n=min(len(g), cfg.fast_run_settings.n_per_class), random_state=cfg.seed)
             for _, g in pool.groupby("DDx")]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Intensity helpers
# ---------------------------------------------------------------------------
def fit_adc_scaler(df: pd.DataFrame, data_dir) -> dict:
    """1st/99th percentile of ADC over the centre slices of `df` (subsampled 4x in-plane).
    Fit on training patients only."""
    values = []
    for scan_id in df["Scan_ID"]:
        path = Path(data_dir) / f"{scan_id}_ADC.npy"
        if path.exists():
            arr = np.load(path)
            values.append(arr[arr.shape[0] // 2, ::4, ::4].flatten())
    if not values:
        return {"p1": 0.0, "p99": 1.0}
    all_adc = np.concatenate(values)
    return {"p1": np.percentile(all_adc, 1), "p99": np.percentile(all_adc, 99)}


def scale_adc(arr: np.ndarray, scaler) -> np.ndarray:
    """ADC is stored in physical units; map [p1, p99] to [-1, 1]."""
    if scaler is None:
        return arr
    p1, p99 = scaler["p1"], scaler["p99"]
    arr = np.clip((arr - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    return (arr - 0.5) * 2.0


def to_rgb(arr: np.ndarray, axis: int, imagenet_norm: bool = False) -> np.ndarray:
    """Replicate a grey image to 3 channels for the ImageNet encoder."""
    rgb = np.stack([arr, arr, arr], axis=axis)
    if imagenet_norm:
        shape = [1] * rgb.ndim
        shape[axis] = 3
        rgb = (rgb - IMAGENET_MEAN.reshape(shape)) / IMAGENET_STD.reshape(shape)
    return rgb


def center_slices(arr: np.ndarray, num_slices: int):
    """Keep the central `num_slices` slices along Z."""
    total = arr.shape[0]
    actual = min(num_slices, total)
    if actual < total:
        start = (total - actual) // 2
        arr = arr[start:start + actual]
    return arr, actual


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
class ConcordantTransform:
    """One random flip/rotation/shift per patient, applied identically to every sequence
    and slice. A horizontal flip swaps sides, so the laterality feature is flipped too."""

    def __init__(self, p_flip=0.5, max_rot=15.0, max_trans=0.1):
        self.p_flip = p_flip
        self.max_rot = max_rot
        self.max_trans = max_trans

    def __call__(self, sample):
        do_flip = random.random() < self.p_flip
        angle = random.uniform(-self.max_rot, self.max_rot)
        img_size = 224
        tx = random.uniform(-self.max_trans, self.max_trans) * img_size
        ty = random.uniform(-self.max_trans, self.max_trans) * img_size

        for seq, img in sample["images"].items():
            if do_flip:
                img = TF.hflip(img)
            img = TF.affine(img, angle=angle, translate=(int(tx), int(ty)), scale=1.0, shear=0.0, fill=0.0)
            sample["images"][seq] = img

        if do_flip:
            sample["clinical"][2] = 1.0 - sample["clinical"][2]
        return sample


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ParotidDataset(Dataset):
    """One item per patient: {'images': {seq: tensor}, 'label', 'clinical', 'pid'
    [, 'slice_indices']}.

    multi_slice=False: the centre slice, (3, H, W) per sequence.
    multi_slice=True:  the central `num_slices` slices, (S, 3, H, W) per sequence.
    """

    def __init__(self, df, data_dir, task="binary", sequences=("T1",), multi_slice=False, num_slices=5,
                 adc_scaler=None, transform=None, imagenet_norm=False):
        self.df = task_frame(df, task)
        self.data_dir = Path(data_dir)
        self.task = task
        self.sequences = list(sequences)
        self.multi_slice = multi_slice
        self.num_slices = num_slices
        self.adc_scaler = adc_scaler
        self.transform = transform
        self.imagenet_norm = imagenet_norm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row["Scan_ID"])

        images, counts = {}, []
        for seq in self.sequences:
            arr = np.load(self.data_dir / f"{pid}_{seq}.npy")  # (Z, H, W)
            if self.multi_slice:
                arr, actual = center_slices(arr, self.num_slices)
                counts.append(actual)
            else:
                arr = arr[arr.shape[0] // 2]
            if seq == "ADC":
                arr = scale_adc(arr, self.adc_scaler)
            images[seq] = torch.tensor(to_rgb(arr, axis=1 if self.multi_slice else 0, imagenet_norm=self.imagenet_norm),
                                       dtype=torch.float32)

        sample = {"images": images, "label": torch.tensor(task_label(row, self.task), dtype=torch.long),
                  "clinical": clinical_tensor(row), "pid": pid}

        if self.multi_slice:
            # Co-registered sequences share one Z window, so the counts must agree.
            if len(set(counts)) != 1:
                raise ValueError(f"Sequences disagree on slice count for {pid}: {dict(zip(self.sequences, counts))}")
            sample["slice_indices"] = torch.arange(counts[0], dtype=torch.long)

        if self.transform:
            if self.multi_slice:
                # Fold slices into the channel axis so every slice gets the same transform.
                shapes = {s: t.shape for s, t in sample["images"].items()}
                for s, (n, c, h, w) in shapes.items():
                    sample["images"][s] = sample["images"][s].view(n * c, h, w)
                sample = self.transform(sample)
                for s, shape in shapes.items():
                    sample["images"][s] = sample["images"][s].view(shape)
            else:
                sample = self.transform(sample)
        return sample
