"""Cross-validated training shared by all four steps.

One function, run_experiment, handles every architecture: K-fold CV over the training pool,
optional warm-starting from earlier steps, epoch selection, decision-threshold calibration,
fold checkpoints, per-patient out-of-fold (OOF) predictions and pooled bootstrap CIs.
"""

import copy
import gc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import data
import metrics
from common import experiment_seed, get_device, log, make_generator, seed_everything, seed_worker, worker_count
from models import STEP_COMBINED, STEP_FUSION, STEP_MIL, STEP_SINGLE, build_model, model_kwargs, set_bn_eval

RESULT_FILES = {
    STEP_SINGLE: "Stage1_SingleSlice_Results.csv",
    STEP_MIL: "Stage2_MIL_Results.csv",
    STEP_FUSION: "Stage3_Fusion_Results.csv",
    STEP_COMBINED: "Stage4_Final_Results.csv",
}
BOOTSTRAP_FILE = "Bootstrap_Summary.csv"

RESULT_COLUMNS = [
    "Timestamp", "Experiment", "Task", "Fold", "Seq",
    "Step", "Arch", "Model_Type", "Fusion_Type", "MIL_Type", "Use_Clinical",
    "Epoch_Selection", "Best_Epoch", "Threshold_Source", "N_Train", "N_Val", "Train_Params",
    "t_star", "AUC", "Bal_Acc", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV",
    "F1_Macro", "Precision_Macro", "Recall_Macro",
]
# Architecture codes written by the original notebooks (the "Arch" column).
ARCH_CODE = {STEP_SINGLE: 1, STEP_MIL: 3, STEP_FUSION: 2, STEP_COMBINED: 4}

BOOTSTRAP_COLUMNS = [
    "Timestamp", "Experiment", "Task", "N_Pooled", "N_Bootstraps",
    "AUC_Pooled", "BalAcc_Pooled", "Sens_Pooled", "Spec_Pooled", "PPV_Pooled", "NPV_Pooled",
    "F1_Pooled", "Acc_Pooled",
    "AUC_Mean", "AUC_CI_Low", "AUC_CI_High",
    "BalAcc_Mean", "BalAcc_CI_Low", "BalAcc_CI_High",
    "Sens_Mean", "Sens_CI_Low", "Sens_CI_High",
    "Spec_Mean", "Spec_CI_Low", "Spec_CI_High",
    "PPV_Mean", "PPV_CI_Low", "PPV_CI_High",
    "NPV_Mean", "NPV_CI_Low", "NPV_CI_High",
    "F1_Mean", "F1_CI_Low", "F1_CI_High",
]


# ---------------------------------------------------------------------------
# Paths and result tables
# ---------------------------------------------------------------------------
def results_dir(cfg) -> Path:
    return Path(cfg.paths.results_dir)


def results_csv(cfg, step) -> Path:
    return results_dir(cfg) / RESULT_FILES[step]


def bootstrap_csv(cfg) -> Path:
    return results_dir(cfg) / BOOTSTRAP_FILE


def checkpoint_path(cfg, experiment, task, fold) -> Path:
    """fold is 1-based."""
    return Path(cfg.paths.work_dir) / f"{experiment}_{task}_fold{fold}.pt"


def oof_path(cfg, experiment, task) -> Path:
    d = results_dir(cfg) / "oof"
    d.mkdir(exist_ok=True)
    return d / f"{experiment}_{task}_oof.csv"


def replace_rows(csv_path: Path, new_rows: list, columns: list, experiment: str, task: str) -> None:
    """Write rows for (experiment, task), replacing any from an earlier run of the same experiment
    so that re-runs are never averaged together with old folds."""
    new = pd.DataFrame(new_rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        old = old[~((old["Experiment"] == experiment) & (old["Task"] == task))]
        new = pd.concat([old, new], ignore_index=True)
    for col in columns:
        if col not in new.columns:
            new[col] = np.nan
    new[columns].to_csv(csv_path, index=False)


def save_oof(cfg, experiment, task, pids, folds, y_true, y_prob, y_pred, meta) -> None:
    y_prob = np.asarray(y_prob)
    df = pd.DataFrame({"pid": np.asarray(pids).astype(str), "fold": folds, "y_true": y_true, "y_pred": y_pred})
    if y_prob.ndim == 1:
        df["y_prob"] = y_prob
    else:
        for c in range(y_prob.shape[1]):
            df[f"y_prob_{c}"] = y_prob[:, c]
    for k, v in meta.items():
        df[k] = v
    df.to_csv(oof_path(cfg, experiment, task), index=False)


def load_oof(cfg, experiment, task):
    path = oof_path(cfg, experiment, task)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    prob_cols = sorted(c for c in df.columns if c.startswith("y_prob_"))
    return {"pid": df["pid"].astype(str).to_numpy(), "fold": df["fold"].to_numpy(),
            "y_true": df["y_true"].to_numpy(), "y_pred": df["y_pred"].to_numpy(),
            "y_prob": df[prob_cols].to_numpy() if prob_cols else df["y_prob"].to_numpy()}


def log_bootstrap_summary(cfg, experiment, task, y_true, y_prob, y_pred) -> dict:
    """Pooled-OOF point estimates and bootstrap 95% CIs. AUC from probabilities, the
    confusion metrics from the thresholded predictions."""
    n_boot = cfg.cv.n_bootstraps
    pooled = metrics.compute_confusion_metrics(y_true, y_pred, task)
    auc_pooled = metrics.safe_auc(y_true, y_prob, task)
    auc_mean, auc_lo, auc_hi = metrics.bootstrap_auc(y_true, y_prob, task, n_boot)
    ci = metrics.bootstrap_confusion_metrics(y_true, y_pred, task, n_boot)

    record = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Experiment": experiment, "Task": task,
              "N_Pooled": len(y_true), "N_Bootstraps": n_boot,
              "AUC_Pooled": round(auc_pooled, 4), "BalAcc_Pooled": round(pooled["bal_acc"], 4),
              "Sens_Pooled": round(pooled["sensitivity"], 4), "Spec_Pooled": round(pooled["specificity"], 4),
              "PPV_Pooled": round(pooled["ppv"], 4), "NPV_Pooled": round(pooled["npv"], 4),
              "F1_Pooled": round(pooled["f1_macro"], 4), "Acc_Pooled": round(pooled["acc"], 4),
              "AUC_Mean": round(auc_mean, 4), "AUC_CI_Low": round(auc_lo, 4), "AUC_CI_High": round(auc_hi, 4)}
    for key, name in [("bal_acc", "BalAcc"), ("sensitivity", "Sens"), ("specificity", "Spec"),
                      ("ppv", "PPV"), ("npv", "NPV"), ("f1_macro", "F1")]:
        m, lo, hi = ci[key]
        record.update({f"{name}_Mean": round(m, 4), f"{name}_CI_Low": round(lo, 4), f"{name}_CI_High": round(hi, 4)})
    replace_rows(bootstrap_csv(cfg), [record], BOOTSTRAP_COLUMNS, experiment, task)

    log.info(f"[pooled OOF] {experiment} ({task}): AUC {auc_pooled:.4f} [{auc_lo:.4f}, {auc_hi:.4f}] | "
             f"Bal_Acc {pooled['bal_acc']:.4f} | Sens {pooled['sensitivity']:.4f} | "
             f"Spec {pooled['specificity']:.4f} (n={len(y_true)})")
    return record


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) with per-class weights alpha. gamma=0 is weighted CE."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha.to(logits.device).gather(0, targets) * loss
        return loss.mean()


def fold_class_weights(train_df, task) -> torch.Tensor:
    """Inverse class frequency in the training fold, normalized to sum to the number of classes present.
    A class absent from the fold gets weight 0 (it has no samples to weight)."""
    if task == "binary":
        counts = train_df["Malignant"].value_counts().sort_index()
    else:
        counts = train_df["DDx"].map(data.MULTICLASS_MAP).dropna().astype(int).value_counts().sort_index()
    total = counts.sum()
    alpha = torch.tensor([total / counts[c] for c in sorted(counts.keys())], dtype=torch.float32)
    alpha = alpha / alpha.sum() * len(counts)
    full = torch.zeros(data.num_classes(task), dtype=torch.float32)
    full[sorted(counts.keys())] = alpha
    return full


# ---------------------------------------------------------------------------
# Datasets, models, forward pass
# ---------------------------------------------------------------------------
def build_dataset(cfg, step, df, task, sequences, adc_scaler, transform=None):
    return data.ParotidDataset(df, cfg.paths.crops_dir, task=task, sequences=sequences,
                               multi_slice=step in (STEP_MIL, STEP_COMBINED), num_slices=cfg.data.num_slices,
                               adc_scaler=adc_scaler, transform=transform, imagenet_norm=cfg.data.imagenet_norm)


def make_model(cfg, step, task, sequences, model_type="single", fusion_type="concat", mil_type="mean",
               use_clinical=False):
    return build_model(step, data.num_classes(task), sequences, model_type, fusion_type, mil_type, use_clinical,
                       **model_kwargs(cfg))


def forward_pass(model, batch, step, sequences, device):
    clinical = batch["clinical"].to(device, non_blocking=True)
    if step in (STEP_SINGLE, STEP_MIL):
        x = batch["images"][sequences[0]].to(device, non_blocking=True)
        if step == STEP_SINGLE:
            return model(x, clinical)[0]
        return model(x, clinical, batch["slice_indices"].to(device, non_blocking=True))[0]
    images = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
    if step == STEP_FUSION:
        return model(images, clinical)[0]
    return model(images, clinical=clinical, slice_indices=batch["slice_indices"].to(device, non_blocking=True))[0]


def apply_bn_mode(model, mode, freeze_depth) -> None:
    """'frozen': all BatchNorm in eval mode. 'selective': BN updates only inside the trainable
    block(s). 'unfrozen': all BN updates. Call after model.train()."""
    if mode == "unfrozen":
        return
    if mode == "frozen":
        model.apply(set_bn_eval)
        return
    trainable = {"none": ["features"], "layer4": ["features.7"], "layer3": ["features.6", "features.7"]}[freeze_depth]
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.train(any(s in name for s in trainable))


# ---------------------------------------------------------------------------
# Warm-starting
# ---------------------------------------------------------------------------
def _transfer(module: nn.Module, state: dict, prefixes, rename=None) -> int:
    """Load every shape-matching tensor under `prefixes` into `module`. Returns the count."""
    rename = rename or {}
    target = module.state_dict()
    picked = {}
    for prefix in prefixes:
        for k, v in state.items():
            if k.startswith(prefix):
                sub = rename.get(k[len(prefix):], k[len(prefix):])
                if sub in target and v.shape == target[sub].shape:
                    picked[sub] = v
    module.load_state_dict(picked, strict=False)
    return len(picked)


def _load_state(path):
    if path is None or not Path(path).exists():
        return None
    return torch.load(path, map_location="cpu")


def warm_start(model, cfg, task, fold, encoder_from=None, mil_from=None, seqattn_from=None) -> None:
    """Initialize parts of `model` from fold-matched checkpoints of earlier steps:
    encoder <- encoder_from, slice-attention head <- mil_from (Step 2 MIL model),
    sequence-attention head <- seqattn_from (Step 3 fusion model). A joint-attention head
    takes slice_emb and the gating MLP from the MIL head, and seq_emb from the sequence-
    attention head's positional embedding. Missing checkpoints leave the default init."""
    m = cfg.model
    if not m.warm_start:
        return
    ckpt = lambda name: checkpoint_path(cfg, name, task, fold) if name else None  # noqa: E731

    if encoder_from:
        state = _load_state(ckpt(encoder_from))
        attr = "encoder" if hasattr(model, "encoder") else "shared_encoder"
        if state is None:
            log.info(f"  [warm-start] {ckpt(encoder_from).name} not found; encoder stays at ImageNet init")
        else:
            n = _transfer(getattr(model, attr), state, ("encoder.", "shared_encoder."))
            log.info(f"  [warm-start] encoder: {n} tensors from {encoder_from}")

    if hasattr(model, "joint_attention"):
        target, n = model.joint_attention, 0
        mil_state = _load_state(ckpt(mil_from))
        if mil_state is not None:
            n += _transfer(target, mil_state, ("mil.", "shared_mil."), rename={"pos_emb.weight": "slice_emb.weight"})
        sa_state = _load_state(ckpt(seqattn_from))
        v = sa_state.get("seq_attention.pos_emb.weight") if sa_state is not None else None
        if v is not None and v.shape == target.seq_emb.weight.shape:
            target.seq_emb.weight.data.copy_(v)
            n += 1
        log.info(f"  [warm-start] joint attention: {n} tensors")
        return

    mil_attr = "mil" if hasattr(model, "mil") else ("shared_mil" if hasattr(model, "shared_mil") else None)
    if m.warm_start_mil and mil_from and mil_attr:
        state = _load_state(ckpt(mil_from))
        if state is not None:
            n = _transfer(getattr(model, mil_attr), state, ("mil.", "shared_mil."))
            log.info(f"  [warm-start] slice attention: {n} tensors from {mil_from}")

    if m.warm_start_seq_attn and seqattn_from and hasattr(model, "seq_attention"):
        state = _load_state(ckpt(seqattn_from))
        if state is not None:
            n = _transfer(model.seq_attention, state, ("seq_attention.",))
            log.info(f"  [warm-start] sequence attention: {n} tensors from {seqattn_from}")


# ---------------------------------------------------------------------------
# One pass over a loader
# ---------------------------------------------------------------------------
def run_epoch(model, loader, step, sequences, task, device, criterion, optimizer=None, scaler=None,
              accum_steps=1, bn_mode="selective", freeze_depth="layer4", use_amp=False):
    """Trains when an optimizer is given, otherwise evaluates. Returns (loss, auc, y_true, y_prob, pids)."""
    train = optimizer is not None
    if train:
        model.train()
        apply_bn_mode(model, bn_mode, freeze_depth)
        optimizer.zero_grad()
    else:
        model.eval()

    total_loss, y_true, y_prob, pids = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for i, batch in enumerate(loader):
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = forward_pass(model, batch, step, sequences, device)
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss / accum_steps).backward()
                if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
            y_prob.extend(probs[:, 1] if task == "binary" else probs)
            y_true.extend(labels.cpu().numpy())
            pids.extend(batch["pid"])

    y_true, y_prob = np.array(y_true), np.array(y_prob)
    return total_loss / max(len(loader.dataset), 1), metrics.safe_auc(y_true, y_prob, task), y_true, y_prob, pids


def make_inner_split(cfg, train_df, task, seed):
    """Inner train/validation split of an outer training fold, for epoch_selection='inner'.
    Returns (None, None) if the inner validation set would be too small to be useful."""
    strat = data.task_frame(train_df, task)
    if len(strat) < 10:
        return None, None
    try:
        inner_tr, inner_va = data.stratified_holdout(train_df, cfg.cv.inner_val_fraction, seed)
    except ValueError:
        return None, None
    if task == "binary":
        counts, needed = inner_va["Malignant"].value_counts(), [0, 1]
    else:
        counts, needed = inner_va[inner_va["DDx"] != 4]["DDx"].value_counts(), [1, 2, 3]
    if any(counts.get(c, 0) < cfg.cv.min_inner_val_per_class for c in needed):
        return None, None
    return inner_tr.reset_index(drop=True), inner_va.reset_index(drop=True)


# ---------------------------------------------------------------------------
# The cross-validation loop
# ---------------------------------------------------------------------------
def run_experiment(cfg, step, experiment_name, task, sequences, model_type="single", fusion_type="concat",
                   mil_type="mean", use_clinical=False, warm_start_from=None, warm_start_mil_from=None,
                   warm_start_seqattn_from=None, csv_path=None, cv_seed=None) -> dict:
    """K-fold CV of one configuration on the training pool.

    Epoch selection (cfg.cv.epoch_selection):
      outer_val  keep the epoch with the best AUC on the validation fold, which is also the
                 fold reported. Uses all training patients; the reported CV AUC is optimistic.
      swa        average the weights of the last swa_last_n epochs.
      last       keep the last epoch.
      inner      select on an inner split of the training fold.
    Decision threshold (binary; cfg.cv.threshold_source):
      pooled_oof one Youden threshold on the pooled OOF probabilities of all folds.
      train_fold per fold, from the fold's own (in-sample) training predictions.
      inner      per fold, from the inner validation split (requires epoch_selection='inner').
    """
    device = get_device(cfg)
    cv_seed = cfg.seed if cv_seed is None else cv_seed
    selection = cfg.cv.epoch_selection
    threshold_source = cfg.cv.threshold_source
    epochs = cfg.cv.epochs
    batch_size = cfg.training.batch_size[step]
    accum_steps = cfg.training.accum_steps[step]
    full_name = experiment_name + ("_Clinical" if use_clinical else "")
    csv_path = Path(csv_path) if csv_path else results_csv(cfg, step)
    pool = data.load_train_pool(cfg)
    n_workers = worker_count(cfg)

    log.info(f"\n{'=' * 74}\nStep {step} | {full_name} | {task} | seqs {sequences} | "
             f"{model_type}/{fusion_type}/{mil_type} | selection {selection}\n{'=' * 74}")

    fold_buffer = []
    for fold, (train_idx, val_idx) in enumerate(data.cv_splits(pool, cfg.cv.n_splits, cv_seed)):
        fold_seed = experiment_seed(experiment_name, task, fold, base=cv_seed)
        seed_everything(fold_seed)
        train_df = pool.iloc[train_idx].reset_index(drop=True)
        val_df = pool.iloc[val_idx].reset_index(drop=True)

        fold_mode, inner_va_df = selection, None
        fit_df = train_df
        if selection == "inner":
            inner_tr_df, inner_va_df = make_inner_split(cfg, train_df, task, fold_seed)
            if inner_va_df is None:
                log.info(f"  [fold {fold + 1}] inner split too small; using the last epoch")
                fold_mode = "last"
            else:
                fit_df = inner_tr_df

        adc_scaler = data.fit_adc_scaler(fit_df, cfg.paths.crops_dir) if "ADC" in sequences else None
        t = cfg.training.augment
        transform = data.ConcordantTransform(p_flip=t.p_flip, max_rot=t.max_rot, max_trans=t.max_trans)

        def loader(df, transform=None, shuffle=False):
            ds = build_dataset(cfg, step, df, task, sequences, adc_scaler, transform)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers,
                              pin_memory=device.type == "cuda", persistent_workers=n_workers > 0,
                              worker_init_fn=seed_worker, generator=make_generator(fold_seed) if shuffle else None)

        loader_train = loader(fit_df, transform, shuffle=True)
        loader_fit_eval = loader(fit_df)  # same patients, no augmentation (for 'train_fold' thresholds)
        loader_val = loader(val_df)
        loader_inner = loader(inner_va_df) if fold_mode == "inner" else None

        model = make_model(cfg, step, task, sequences, model_type, fusion_type, mil_type, use_clinical)
        # Attention heads are only carried into Step 4; the sequence-attention head defaults to
        # the same Step 3 checkpoint the encoder comes from.
        combined = step == STEP_COMBINED
        warm_start(model, cfg, task, fold + 1, encoder_from=warm_start_from,
                   mil_from=warm_start_mil_from if combined else None,
                   seqattn_from=(warm_start_seqattn_from or warm_start_from) if combined else None)
        multi_gpu = torch.cuda.device_count() > 1
        if multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(device)
        bare = model.module if multi_gpu else model

        criterion = FocalLoss(alpha=fold_class_weights(fit_df, task).to(device),
                              gamma=cfg.training.focal_gamma[task])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr[step],
                                      weight_decay=cfg.training.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler(device.type, enabled=cfg.training.use_amp)
        epoch_kw = dict(step=step, sequences=sequences, task=task, device=device, criterion=criterion,
                        bn_mode=cfg.model.bn_mode, freeze_depth=cfg.model.freeze_depth,
                        use_amp=cfg.training.use_amp)

        select_loader = {"inner": loader_inner, "outer_val": loader_val}.get(fold_mode)
        best_auc, best_weights, best_epoch = -1.0, None, epochs
        swa_buffer, history = [], []
        for epoch in range(epochs):
            tr_loss, tr_auc, *_ = run_epoch(model, loader_train, optimizer=optimizer, scaler=scaler,
                                            accum_steps=accum_steps, **epoch_kw)
            scheduler.step()

            sel_loss = sel_auc = float("nan")
            if select_loader is not None:
                sel_loss, sel_auc, *_ = run_epoch(model, select_loader, **epoch_kw)
                if np.isfinite(sel_auc) and sel_auc > best_auc:
                    best_auc, best_epoch = sel_auc, epoch + 1
                    best_weights = copy.deepcopy(bare.state_dict())
            if fold_mode == "swa" and epoch >= epochs - cfg.cv.swa_last_n:
                swa_buffer.append({k: v.detach().cpu().clone() for k, v in bare.state_dict().items()})

            # Monitoring only; written to a separate file and never used for selection.
            mon_loss = mon_auc = float("nan")
            if cfg.cv.monitor_val_curve and select_loader is not loader_val:
                mon_loss, mon_auc, *_ = run_epoch(model, loader_val, **epoch_kw)
            history.append({"Experiment": full_name, "Task": task, "Fold": fold + 1, "Epoch": epoch + 1,
                            "Train_Loss": round(tr_loss, 5), "Train_AUC": round(tr_auc, 5),
                            "Select_Loss": round(sel_loss, 5), "Select_AUC": round(sel_auc, 5),
                            "Monitor_Val_Loss": round(mon_loss, 5), "Monitor_Val_AUC": round(mon_auc, 5)})

        if fold_mode == "swa" and swa_buffer:
            # Integer buffers (BatchNorm num_batches_tracked) are not averaged.
            best_weights = {k: (torch.stack([sd[k].float() for sd in swa_buffer]).mean(0).to(swa_buffer[-1][k].dtype)
                                if swa_buffer[-1][k].is_floating_point() else swa_buffer[-1][k])
                            for k in swa_buffer[0]}
            best_epoch = epochs
        elif best_weights is None:
            best_weights = copy.deepcopy(bare.state_dict())
            best_epoch = epochs
        bare.load_state_dict(best_weights)
        hist_dir = results_dir(cfg) / "epoch_history"
        hist_dir.mkdir(exist_ok=True)
        hist_path = hist_dir / f"{full_name}_{task}_epochs.csv"
        prior = pd.read_csv(hist_path) if hist_path.exists() and fold > 0 else None
        pd.concat([prior, pd.DataFrame(history)]).to_csv(hist_path, index=False)

        y_true_cal = y_prob_cal = None
        if task == "binary" and threshold_source != "pooled_oof":
            cal_loader = loader_inner if (threshold_source == "inner" and loader_inner is not None) else loader_fit_eval
            _, _, y_true_cal, y_prob_cal, _ = run_epoch(model, cal_loader, **epoch_kw)

        _, val_auc, y_true_val, y_prob_val, pid_val = run_epoch(model, loader_val, **epoch_kw)
        log.info(f"Fold {fold + 1}: kept epoch {best_epoch}/{epochs} ({fold_mode}); validation AUC {val_auc:.4f}")

        torch.save(best_weights, checkpoint_path(cfg, full_name, task, fold + 1))
        fold_buffer.append({"fold": fold + 1, "y_true": y_true_val, "y_prob": y_prob_val, "pid": pid_val,
                            "y_true_cal": y_true_cal, "y_prob_cal": y_prob_cal, "best_epoch": best_epoch,
                            "fold_mode": fold_mode, "n_train": len(fit_df),
                            "trainable_params": sum(p.numel() for p in bare.parameters() if p.requires_grad)})

        del model, bare, optimizer, scheduler, loader_train, loader_fit_eval, loader_val, loader_inner
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Thresholding happens after all folds: 'pooled_oof' needs every fold's OOF predictions.
    shared_thr, thr_label = 0.5, "n/a (multiclass uses argmax)"
    if task == "binary":
        if threshold_source == "pooled_oof":
            shared_thr = metrics.get_youden_threshold(np.concatenate([f["y_true"] for f in fold_buffer]),
                                                      np.concatenate([f["y_prob"] for f in fold_buffer]))
            thr_label = "pooled-OOF"
        else:
            thr_label = "inner-val" if threshold_source == "inner" else "train-fold (in-sample)"

    rows, oof = [], {"pid": [], "fold": [], "y_true": [], "y_prob": [], "y_pred": []}
    for fb in fold_buffer:
        thr = shared_thr
        if task == "binary" and threshold_source != "pooled_oof":
            thr = metrics.get_youden_threshold(fb["y_true_cal"], fb["y_prob_cal"])
        m, y_pred = metrics.compute_metrics(fb["y_true"], fb["y_prob"], task, thr)
        log.info(f"Fold {fb['fold']}: AUC {m['auc']:.4f} | Bal_Acc {m['bal_acc']:.4f} (t*={thr:.3f}, {thr_label})")
        oof["pid"] += list(fb["pid"])
        oof["fold"] += [fb["fold"]] * len(fb["y_true"])
        oof["y_true"].append(fb["y_true"])
        oof["y_prob"].append(fb["y_prob"])
        oof["y_pred"].append(y_pred)
        rows.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Experiment": full_name, "Task": task,
            "Fold": fb["fold"], "Seq": "+".join(sequences), "Step": step, "Arch": ARCH_CODE[step],
            "Model_Type": model_type, "Fusion_Type": fusion_type, "MIL_Type": mil_type,
            "Use_Clinical": bool(use_clinical), "Epoch_Selection": fb["fold_mode"], "Best_Epoch": fb["best_epoch"],
            "Threshold_Source": thr_label, "N_Train": fb["n_train"], "N_Val": len(fb["y_true"]),
            "Train_Params": fb["trainable_params"],
            "t_star": round(float(thr), 4) if task == "binary" else "N/A",
            "AUC": round(m["auc"], 4), "Bal_Acc": round(m["bal_acc"], 4), "Accuracy": round(m["acc"], 4),
            "Sensitivity": round(m["sensitivity"], 4), "Specificity": round(m["specificity"], 4),
            "PPV": round(m["ppv"], 4), "NPV": round(m["npv"], 4), "F1_Macro": round(m["f1_macro"], 4),
            "Precision_Macro": round(m["precision_macro"], 4), "Recall_Macro": round(m["recall_macro"], 4)})
    replace_rows(csv_path, rows, RESULT_COLUMNS, full_name, task)

    y_true = np.concatenate(oof["y_true"])
    y_prob = np.concatenate(oof["y_prob"], axis=0)
    y_pred = np.concatenate(oof["y_pred"])
    save_oof(cfg, full_name, task, oof["pid"], oof["fold"], y_true, y_prob, y_pred,
             meta={"Step": step, "Seq": "+".join(sequences), "Model_Type": model_type, "Fusion_Type": fusion_type,
                   "Epoch_Selection": "|".join(sorted({f["fold_mode"] for f in fold_buffer}))})
    log_bootstrap_summary(cfg, full_name, task, y_true, y_prob, y_pred)
    return {"oof_y_true": y_true, "oof_y_prob": y_prob, "oof_y_pred": y_pred, "oof_pid": np.array(oof["pid"])}
