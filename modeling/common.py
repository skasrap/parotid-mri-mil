"""Configuration, logging and seeding shared by train.py and evaluate.py."""

import logging
import os
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml


class Config(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e


def _to_config(obj):
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def load_config(path, fast_run=None) -> Config:
    """Reads the YAML config. Relative paths are taken relative to the config file. With
    fast_run, the values under `fast_run_settings` replace the regular ones and the work and
    results folders get a `_fast_run` suffix, so a smoke test never mixes with a real run."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key, value in raw["paths"].items():
        p = Path(value)
        raw["paths"][key] = str(p if p.is_absolute() else (path.parent / p).resolve())
    if fast_run is not None:
        raw["fast_run"] = fast_run
    if raw.get("fast_run"):
        fr = raw["fast_run_settings"]
        raw["cv"]["n_splits"] = fr["n_splits"]
        raw["cv"]["epochs"] = fr["epochs"]
        raw["cv"]["n_bootstraps"] = fr["n_bootstraps"]
        raw["training"]["batch_size"] = fr["batch_size"]
        raw["data"]["num_workers"] = fr["num_workers"]
        for key in ("work_dir", "results_dir"):
            raw["paths"][key] = raw["paths"][key].rstrip("/\\") + "_fast_run"
    cfg = _to_config(raw)
    for key in ("work_dir", "results_dir"):
        Path(cfg.paths[key]).mkdir(parents=True, exist_ok=True)
    return cfg


def set_tf32(enabled: bool) -> None:
    """TF32 matmuls and convolutions on Ampere or newer GPUs: faster, slightly less precise.
    No effect on older GPUs or the CPU."""
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")


def get_device(cfg) -> torch.device:
    if cfg.get("device", "auto") == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def setup_logging(log_path) -> logging.Logger:
    """Console + file logging; the file is appended to across runs."""
    logger = logging.getLogger("parotid")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


log = logging.getLogger("parotid")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id) -> None:
    """DataLoader workers get a torch seed but not a Python/NumPy one; the augmentation
    draws from `random`, so seed those too."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def experiment_seed(experiment_name: str, task: str, fold: int, base: int) -> int:
    """Seed for one (experiment, task, fold), independent of what else ran in the process.
    CRC32 rather than hash(): Python randomizes string hashes per process."""
    h = zlib.crc32(f"{experiment_name}|{task}|{int(fold)}".encode("utf-8"))
    return (base + h) % 2 ** 31


def worker_count(cfg) -> int:
    n = cfg.data.num_workers
    return min(n, os.cpu_count() or 1) if n else 0
