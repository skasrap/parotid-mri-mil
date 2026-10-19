"""Evaluate the selected models once on the held-out test set.

Per task, four configurations are read from the training results (ranked by pooled OOF AUC):
  primary              best step-4 model without clinical features
  secondary            best step-4 model on a different sequence set
  clinical             the primary configuration with clinical features
  baseline_single_seq  best single-sequence multi-slice (step 2) model

Each is applied as the ensemble of its five fold checkpoints. A temperature and, for the binary
task, a decision threshold are fitted on out-of-fold predictions of the CV pool (re-computed by
inference); the test set is not used for any fitting. Outputs go to <results_dir>/evaluation.

Usage: python evaluate.py --config config.yaml [--no-gallery]
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import auc as area_under, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

import data
import engine
import metrics
from common import get_device, load_config, log, setup_logging
from models import STEP_COMBINED, STEP_FUSION, STEP_MIL, STEP_SINGLE

ROLES = ["primary", "secondary", "clinical", "baseline_single_seq"]
ROLE_DESCRIPTION = {"primary": "Primary model", "secondary": "Alternative sequence combination",
                    "clinical": "With clinical features (age, sex, laterality)",
                    "baseline_single_seq": "Single-sequence comparator"}
MAIN_METRICS = [("auc", "AUC"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                ("ppv", "PPV"), ("npv", "NPV"), ("acc", "Accuracy")]
EXTRA_METRICS = [("bal_acc", "Balanced accuracy"), ("f1_macro", "F1 (macro)")]
FUSION_DISPLAY = {"early": "Early fusion", "late": "Late fusion", "mid": "Mid fusion",
                  "joint": "Joint attention", "single": "Single sequence"}


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------
def pretty_seq(sequences, sep=" + "):
    if isinstance(sequences, str):
        sequences = re.split(r"[+_]", sequences)
    return sep.join(s for s in sequences if s)


def pretty_model(c) -> str:
    base = "Single sequence, multi-slice" if c["kind"] == "mil" else FUSION_DISPLAY.get(c["model_type"], c["model_type"])
    if c["fusion_type"] == "seq_attention":
        base += ", sequence attention"
    if c["use_clinical"]:
        base += " + clinical"
    return f"{base} ({pretty_seq(c['sequences'])})"


def pretty_experiment(name) -> str:
    name = str(name)
    clinical, name = name.endswith("_Clinical"), name.removesuffix("_Clinical")
    seq_attn, name = name.endswith("_SeqAttn"), name.removesuffix("_SeqAttn")
    m = re.match(r"Stage4_(Early|Late|Mid)Fusion_(.+)$", name)
    if m:
        label = f"{m.group(1)} fusion ({pretty_seq(m.group(2))})"
    elif name.startswith("Stage4_JointAttention_"):
        label = f"Joint attention ({pretty_seq(name[len('Stage4_JointAttention_'):])})"
    else:
        m = re.match(r"Stage[1-4][A-Za-z]*_?(?:Early|LateConcat|SeqAttn)?_?(.+)$", name)
        label = pretty_seq(m.group(1)) if m else name.replace("_", " ")
    return label + (", sequence attention" if seq_attn else "") + (" + clinical" if clinical else "")


def fmt(value, ci=None, show_ci=True, decimals=2) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    txt = f"{value:.{decimals}f}"
    if show_ci and ci is not None and all(np.isfinite(c) for c in ci):
        txt += f" ({ci[0]:.{decimals}f}-{ci[1]:.{decimals}f})"
    return txt


# ---------------------------------------------------------------------------
# Which configurations to evaluate
# ---------------------------------------------------------------------------
def pooled_cv_auc(cfg, experiment, task):
    path = engine.bootstrap_csv(cfg)
    if not path.exists():
        return np.nan, np.nan, np.nan
    b = pd.read_csv(path)
    b = b[(b["Experiment"] == experiment) & (b["Task"] == task)]
    if b.empty:
        return np.nan, np.nan, np.nan
    r = b.iloc[-1]
    return r["AUC_Pooled"], r["AUC_CI_Low"], r["AUC_CI_High"]


def discover(cfg, step, task) -> list:
    """Every experiment in a step's results CSV, best pooled OOF AUC first, with its architecture
    taken from the logged columns (or, for older CSVs, from the experiment name)."""
    path = engine.results_csv(cfg, step)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df[df["Task"] == task]
    out = []
    for name, g in df.groupby("Experiment", sort=False):
        first = lambda col, default=None: g[col].dropna().iloc[0] if col in g and g[col].notna().any() else default  # noqa: E731
        model_type = first("Model_Type")
        if model_type is None:
            m = re.match(r"Stage4_(Early|Late|Mid)Fusion_", str(name))
            model_type = m.group(1).lower() if m else ("joint" if "JointAttention" in str(name) else "late")
        fusion_type = first("Fusion_Type", "seq_attention" if "_SeqAttn" in str(name) else "concat")
        auc_p, lo, hi = pooled_cv_auc(cfg, name, task)
        out.append({"task": task, "experiment": name, "kind": "mil" if step == STEP_MIL else "stage4",
                    "step": step, "sequences": str(g["Seq"].iloc[0]).split("+"),
                    "use_clinical": bool(first("Use_Clinical", str(name).endswith("_Clinical"))),
                    "model_type": model_type, "fusion_type": fusion_type,
                    "mil_type": first("MIL_Type", "attention"),
                    "cv_pooled_auc": auc_p, "cv_auc_lo": lo, "cv_auc_hi": hi})
    return sorted(out, key=lambda c: -(c["cv_pooled_auc"] if np.isfinite(c["cv_pooled_auc"]) else -1))


def choose_configs(cfg) -> dict:
    chosen = {}
    for task in cfg.tasks:
        s4 = discover(cfg, STEP_COMBINED, task)
        mil = [c for c in discover(cfg, STEP_MIL, task) if not c["use_clinical"]]
        plain = [c for c in s4 if not c["use_clinical"]]
        clin = [c for c in s4 if c["use_clinical"]]
        roles = {}
        if plain:
            roles["primary"] = plain[0]
            first_set = frozenset(plain[0]["sequences"])
            alt = next((c for c in plain[1:] if frozenset(c["sequences"]) != first_set), None)
            if alt:
                roles["secondary"] = alt
        if clin:
            primary_name = roles["primary"]["experiment"] if "primary" in roles else ""
            roles["clinical"] = next((c for c in clin if c["experiment"] == primary_name + "_Clinical"), clin[0])
        if mil:
            roles["baseline_single_seq"] = mil[0]
        chosen[task] = roles
        for role, c in roles.items():
            log.info(f"[{task}] {role:<20s} {c['experiment']} (CV pooled AUC {c['cv_pooled_auc']:.4f})")
        if not plain:
            log.info(f"[{task}] no step-4 results found")
    return chosen


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def build(cfg, c):
    model = engine.make_model(cfg, c["step"], c["task"], c["sequences"], c["model_type"], c["fusion_type"],
                              c["mil_type"], c["use_clinical"])
    return model


def load_fold_model(cfg, c, fold, device):
    path = engine.checkpoint_path(cfg, c["experiment"], c["task"], fold)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    model = build(cfg, c)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval().to(device)


def loader_for(cfg, c, df, scaler):
    ds = engine.build_dataset(cfg, STEP_COMBINED if c["kind"] == "stage4" else STEP_MIL, df, c["task"],
                              c["sequences"], scaler)
    return DataLoader(ds, batch_size=cfg.evaluation.batch_size, shuffle=False, num_workers=0)


def predict_logits(model, loader, c, device):
    logits, labels, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits.append(engine.forward_pass(model, batch, c["step"], c["sequences"], device).float().cpu())
            labels.append(batch["label"])
            pids += list(batch["pid"])
    return torch.cat(logits), torch.cat(labels), pids


def fit_temperature(logits, labels, device, max_iter=100) -> float:
    """Temperature scaling (Guo et al., 2017), one parameter fitted by LBFGS on the NLL."""
    logits, labels = logits.to(device), labels.to(device)
    temperature = nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / temperature.clamp(min=1e-2), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.clamp(min=1e-2).item())


def calibrate(cfg, c, pool, device) -> dict:
    """Each fold model on its own validation fold -> pooled OOF logits -> temperature and
    (binary) Youden threshold on the temperature-scaled probabilities."""
    all_logits, all_labels = [], []
    for fold, (tr, va) in enumerate(data.cv_splits(pool, cfg.cv.n_splits, cfg.seed), start=1):
        tr_df, va_df = pool.iloc[tr].reset_index(drop=True), pool.iloc[va].reset_index(drop=True)
        scaler = data.fit_adc_scaler(tr_df, cfg.paths.crops_dir) if "ADC" in c["sequences"] else None
        model = load_fold_model(cfg, c, fold, device)
        lg, lb, _ = predict_logits(model, loader_for(cfg, c, va_df, scaler), c, device)
        all_logits.append(lg)
        all_labels.append(lb)
        del model
    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    T = fit_temperature(logits, labels, device)
    probs = torch.softmax(logits / T, dim=1).numpy()
    t_star = metrics.get_youden_threshold(labels.numpy(), probs[:, 1]) if c["task"] == "binary" else None
    log.info(f"  calibration on {len(labels)} OOF predictions: temperature {T:.3f}"
             + (f", threshold {t_star:.3f}" if t_star is not None else ""))
    return {"temperature": T, "t_star": t_star}


def evaluate_on_test(cfg, c, calib, pool, test, device) -> dict:
    """Fold-ensemble predictions on the test set.

    Binary decision (evaluation.binary_decision):
      vote       each fold model applies the threshold to its own probability and the ensemble
                 takes the majority. The threshold was fitted on single-model OOF outputs, so it
                 is applied to single-model outputs.
      mean_prob  threshold applied to the averaged probability. Averaging
                 pulls probabilities towards the middle, which shifts the operating point.
    The AUC uses the averaged probability either way.
    """
    ev = cfg.evaluation
    T, n_folds = calib["temperature"], cfg.cv.n_splits
    splits = data.cv_splits(pool, n_folds, cfg.seed)
    pooled_scaler = data.fit_adc_scaler(pool, cfg.paths.crops_dir) if "ADC" in c["sequences"] else None

    per_model, y_true, pids = [], None, None
    for fold in range(1, n_folds + 1):
        scaler = pooled_scaler
        if "ADC" in c["sequences"] and ev.adc_scaler == "per_fold":
            # The scaler this fold model was trained with.
            scaler = data.fit_adc_scaler(pool.iloc[splits[fold - 1][0]], cfg.paths.crops_dir)
        model = load_fold_model(cfg, c, fold, device)
        lg, lb, p = predict_logits(model, loader_for(cfg, c, test, scaler), c, device)
        per_model.append(torch.softmax(lg / T, dim=1).numpy())
        if y_true is None:
            y_true, pids = lb.numpy(), p
        del model

    per_model = np.stack(per_model)  # (folds, N, C)
    ensemble = per_model.mean(axis=0)
    task = c["task"]
    if task == "binary":
        y_prob = ensemble[:, 1]
        if ev.binary_decision == "vote":
            votes = (per_model[:, :, 1] >= calib["t_star"]).sum(axis=0)
            y_pred = np.where(votes * 2 == n_folds, (y_prob >= calib["t_star"]).astype(int),
                              (votes * 2 > n_folds).astype(int))
        else:
            y_pred = (y_prob >= calib["t_star"]).astype(int)
    else:
        y_prob = ensemble
        y_pred = ensemble.argmax(axis=1)
    m = metrics.compute_confusion_metrics(y_true, y_pred, task, data.num_classes(task))
    m["auc"] = metrics.safe_auc(y_true, y_prob, task)
    return {"config": c, "calib": calib, "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred, "pids": pids,
            "ensemble": ensemble, "metrics": m}


def with_intervals(cfg, r) -> dict:
    """{metric: (estimate, ci_low, ci_high)}. The estimate is the full-test-set value
    (evaluation.point_estimate=full_sample) or the bootstrap mean (bootstrap_mean)."""
    ev, task = cfg.evaluation, r["config"]["task"]
    auc_mean, lo, hi = metrics.bootstrap_auc(r["y_true"], r["y_prob"], task, ev.n_bootstraps)
    ci = metrics.bootstrap_confusion_metrics(r["y_true"], r["y_pred"], task, ev.n_bootstraps,
                                             n_classes=data.num_classes(task))
    out = {"auc": (auc_mean, lo, hi), **ci}
    if ev.point_estimate == "full_sample":
        out = {k: (float(r["metrics"][k]), v[1], v[2]) for k, v in out.items()}
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def head_to_head(cfg, results, task, a, b, why):
    if (task, a) not in results or (task, b) not in results:
        return None
    ra, rb = results[(task, a)], results[(task, b)]
    y, pa, pb, _ = metrics.align_by_pid(ra["pids"], ra["y_true"], ra["y_prob"], rb["pids"], rb["y_true"], rb["y_prob"])
    res = {"task": task, "comparison": why, "a": ra["config"]["experiment"], "b": rb["config"]["experiment"],
           **metrics.compare_predictions(y, pa, pb, task)}
    log.info(f"  [{task}] {why}: AUC {res['auc_a']:.3f} vs {res['auc_b']:.3f}, diff {res['diff']:+.3f} "
             f"[{res['diff_lo']:+.3f}, {res['diff_hi']:+.3f}], bootstrap p={res['boot_p']:.3f}"
             + (f", DeLong p={res['delong_p']:.3f}" if np.isfinite(res["delong_p"]) else "") + f", n={res['n']}")
    return res


def metric_row(cfg, results, intervals, key, metric_list=MAIN_METRICS, role=None):
    c = results[key]["config"]
    row = ({"Role": ROLE_DESCRIPTION[role]} if role else {}) | {"Model": pretty_model(c), "n": len(results[key]["y_true"])}
    for mkey, label in metric_list:
        est, lo, hi = intervals[key][mkey]
        row[label] = fmt(est, (lo, hi), cfg.evaluation.show_ci)
    return row


def write_tables(cfg, results, intervals, out_dir: Path) -> dict:
    per_class = {}
    for task in cfg.tasks:
        keys = [(task, r) for r in ROLES if (task, r) in results]
        if not keys:
            continue
        if (task, "primary") in results:
            main = pd.DataFrame([metric_row(cfg, results, intervals, (task, "primary"))])
            main.to_csv(out_dir / f"Table_Primary_{task}.csv", index=False)
            (out_dir / f"Table_Primary_{task}.tex").write_text(main.to_latex(index=False), encoding="utf-8")
        comp = pd.DataFrame([metric_row(cfg, results, intervals, k, role=k[1]) for k in keys])
        comp.to_csv(out_dir / f"Table_AllConfigurations_{task}.csv", index=False)
        (out_dir / f"Table_AllConfigurations_{task}.tex").write_text(comp.to_latex(index=False), encoding="utf-8")

        full = []
        for k in keys:
            c = results[k]["config"]
            row = {"Role": k[1], "Model": pretty_model(c), "Experiment": c["experiment"],
                   "CV_pooled_AUC": c["cv_pooled_auc"], "n": len(results[k]["y_true"])}
            if task == "binary":
                row["Threshold"] = results[k]["calib"]["t_star"]
            row["Temperature"] = results[k]["calib"]["temperature"]
            for mkey, label in MAIN_METRICS + EXTRA_METRICS:
                est, lo, hi = intervals[k][mkey]
                row.update({label: est, f"{label}_CI_low": lo, f"{label}_CI_high": hi})
            full.append(row)
        pd.DataFrame(full).round(4).to_csv(out_dir / f"Metrics_Full_{task}.csv", index=False)

        if task == "multiclass":
            names = data.CLASS_NAMES[task]
            for k in keys:
                r = results[k]
                pc = metrics.per_class_metrics(r["y_true"], r["y_pred"], r["ensemble"], names)
                cis = metrics.bootstrap_per_class(r["y_true"], r["y_pred"], r["ensemble"], names,
                                                  cfg.evaluation.n_bootstraps)
                table = pd.DataFrame([{"Tumour type": row["Class"], "n": int(row["n"]),
                                       **{lab: fmt(row[m], cis[(i, m)], cfg.evaluation.show_ci)
                                          for m, lab in [("auc", "AUC"), ("sensitivity", "Sensitivity"),
                                                         ("specificity", "Specificity"), ("ppv", "PPV"),
                                                         ("npv", "NPV")]}}
                                      for i, row in pc.iterrows()])
                table.to_csv(out_dir / f"PerClass_{task}_{k[1]}.csv", index=False)
                per_class[k] = table
    return per_class


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def step_ranking(cfg, step, task, group_col):
    path = engine.results_csv(cfg, step)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[(df["Task"] == task) & ~df["Experiment"].astype(str).str.endswith("_Clinical")]
    rows = []
    for key, g in df.groupby(group_col):
        auc_p, lo, hi = pooled_cv_auc(cfg, g["Experiment"].iloc[0], task)
        rows.append({"label": key, "AUC": auc_p, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    return out.dropna(subset=["AUC"]).sort_values("AUC") if not out.empty else out


def plot_selection_funnel(cfg, task, fig_dir):
    specs = [(STEP_SINGLE, "Seq", "Step 1: single-slice screen"), (STEP_MIL, "Seq", "Step 2: multi-slice MIL screen"),
             (STEP_FUSION, "Experiment", "Step 3: fusion configurations"),
             (STEP_COMBINED, "Experiment", "Step 4: final combined")]
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    for (step, col, title), ax in zip(specs, axes):
        r = step_ranking(cfg, step, task, col)
        if r.empty:
            ax.set_title(f"{title}\n(no results)")
            ax.axis("off")
            continue
        labels = [pretty_seq(l) if col == "Seq" else pretty_experiment(l) for l in r["label"]]
        err = [r["AUC"] - r["lo"], r["hi"] - r["AUC"]] if r["lo"].notna().all() else None
        ax.barh(labels, r["AUC"], xerr=err, color=["#2b5c8f"] * (len(r) - 1) + ["#d95f02"],
                error_kw=dict(ecolor="0.35", lw=1, capsize=2))
        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("Cross-validated AUC (95% CI)")
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"Model selection, {task} task", fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig(fig_dir / f"selection_funnel_{task}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc(cfg, r, intervals, path):
    c, task = r["config"], r["config"]["task"]
    est, lo, hi = intervals["auc"]
    ci = f" ({lo:.2f}-{hi:.2f})" if cfg.evaluation.show_ci else ""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    if task == "binary":
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"AUC = {est:.2f}{ci}")
        m = r["metrics"]
        ax.scatter(1 - m["specificity"], m["sensitivity"], color="#d95f02", s=95, zorder=5, edgecolors="black",
                   label=f"Operating point: sensitivity {m['sensitivity']:.2f}, specificity {m['specificity']:.2f}")
        subtitle = "Benign vs. malignant, held-out test set"
    else:
        names = data.CLASS_NAMES[task]
        y_bin = label_binarize(r["y_true"], classes=list(range(len(names))))
        for i, color in enumerate(["#1b9e77", "#d95f02", "#7570b3"]):
            if y_bin[:, i].min() == y_bin[:, i].max():
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], r["ensemble"][:, i])
            ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{names[i]} (AUC = {area_under(fpr, tpr):.2f})")
        subtitle = "Benign subtypes (one-vs-rest), held-out test set"
    ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--")
    ax.set(xlim=(-0.02, 1.0), ylim=(0.0, 1.03))
    ax.set_xlabel("1 - Specificity", fontweight="bold")
    ax.set_ylabel("Sensitivity", fontweight="bold")
    ax.set_title(f"{pretty_model(c)}\n{subtitle}", fontweight="bold", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(r, path):
    cm, names = r["metrics"]["cm"], data.CLASS_NAMES[r["config"]["task"]]
    rows = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows != 0)
    annot = np.array([f"{v}\n({p:.0%})" for v, p in zip(cm.flatten(), pct.flatten())]).reshape(cm.shape)
    sns.set_theme(style="white", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(5.4 + 0.45 * len(names), 4.4 + 0.4 * len(names)))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, xticklabels=names, yticklabels=names,
                annot_kws={"size": 12, "weight": "bold"}, ax=ax, linewidths=0.6, linecolor="white")
    ax.set_title(f"{pretty_model(r['config'])}\nHeld-out test set (n = {cm.sum()})", fontweight="bold", fontsize=11)
    ax.set_ylabel("Reference diagnosis", fontweight="bold")
    ax.set_xlabel("Model prediction", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(r, path, n_bins=8):
    """Confidence of the predicted class vs observed accuracy, with expected calibration error."""
    conf = r["ensemble"][np.arange(len(r["y_pred"])), r["y_pred"]]
    correct = (r["y_pred"] == r["y_true"]).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    acc = np.array([correct[ids == b].mean() if (ids == b).any() else np.nan for b in range(n_bins)])
    mean_conf = np.array([conf[ids == b].mean() if (ids == b).any() else np.nan for b in range(n_bins)])
    count = np.array([(ids == b).sum() for b in range(n_bins)])
    ok = ~np.isnan(acc)
    ece = float(np.sum(np.abs(acc[ok] - mean_conf[ok]) * count[ok] / len(conf)))
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax.bar(bins[:-1][ok], acc[ok], width=1.0 / n_bins, align="edge", edgecolor="black", color="#2b5c8f",
           alpha=0.75, label="Observed accuracy")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlabel("Predicted confidence (predicted class)", fontweight="bold")
    ax.set_ylabel("Observed accuracy", fontweight="bold")
    ax.set_title(f"{pretty_model(r['config'])}\nTemperature {r['calib']['temperature']:.2f}, ECE {ece:.3f}",
                 fontweight="bold", fontsize=10)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_forest(results, intervals, task, path):
    keys = [(task, r) for r in ROLES if (task, r) in intervals]
    if not keys:
        return
    colors = sns.color_palette("Set2", len(keys))
    step = 0.70 / len(keys)
    fig, ax = plt.subplots(figsize=(8.6, len(MAIN_METRICS) + 2.0))
    for ki, key in enumerate(keys):
        for mi, (mkey, _) in enumerate(MAIN_METRICS):
            est, lo, hi = intervals[key][mkey]
            if not np.isfinite(est):
                continue
            ax.errorbar(est, mi + ki * step, xerr=[[max(est - lo, 0)], [max(hi - est, 0)]], fmt="o",
                        color=colors[ki], capsize=4, markersize=7,
                        label=pretty_model(results[key]["config"]) if mi == 0 else None)
    ax.set_yticks([mi + (len(keys) - 1) * step / 2 for mi in range(len(MAIN_METRICS))])
    ax.set_yticklabels([label for _, label in MAIN_METRICS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("Value (95% CI), held-out test set", fontweight="bold")
    ax.set_title("Benign vs. malignant" if task == "binary" else "Benign subtypes", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grad-CAM and attention gallery
# ---------------------------------------------------------------------------
class GradCAM:
    """Hooks the last ResNet block and keeps one activation/gradient pair per encoder call:
    late, mid and joint fusion call the shared encoder once per sequence in a single forward.
    Backward hooks fire in reverse call order, so gradients are prepended."""

    def __init__(self, layer):
        self.activations, self.gradients = [], []
        self._f = layer.register_forward_hook(lambda m, i, o: self.activations.append(o))
        self._b = layer.register_full_backward_hook(lambda m, gi, go: self.gradients.insert(0, go[0]))

    def remove(self):
        self._f.remove()
        self._b.remove()

    def heatmap(self, idx, size):
        act, grad = self.activations[idx], self.gradients[idx]
        b, k = grad.shape[:2]
        sal = F.relu((grad.view(b, k, -1).mean(2).view(b, k, 1, 1) * act).sum(1, keepdim=True))
        sal = F.interpolate(sal, size=size, mode="bilinear", align_corners=False).squeeze(1)
        lo = sal.view(b, -1).min(1)[0].view(b, 1, 1)
        hi = sal.view(b, -1).max(1)[0].view(b, 1, 1)
        return ((sal - lo) / (hi - lo + 1e-8)).detach().cpu().numpy()


def case_interpretability(models_and_samples, c, device):
    """Grad-CAM per sequence and slice, slice-attention and sequence-attention weights, averaged
    over the fold models. Each model gets the sample prepared with its own ADC scaler."""
    seqs = c["sequences"]
    cams, weights, seq_attn, probs = {s: [] for s in seqs}, {s: [] for s in seqs}, [], []
    for model, sample in models_and_samples:
        images = {k: v.unsqueeze(0).to(device) for k, v in sample["images"].items()}
        clinical = sample["clinical"].unsqueeze(0).to(device)
        idx = sample["slice_indices"].unsqueeze(0).to(device)
        model.zero_grad()
        enc = model.encoder if hasattr(model, "encoder") else model.shared_encoder
        cam = GradCAM(enc.features[-1])
        if c["kind"] == "mil":
            logits, aux = model(images[seqs[0]], clinical, idx)
        else:
            logits, aux = model(images, clinical=clinical, slice_indices=idx)
        p = torch.softmax(logits, dim=1)
        logits[0, int(p.argmax())].backward()
        probs.append(p.detach().cpu().numpy()[0])

        size = tuple(images[seqs[0]].shape[-2:])
        if len(cam.activations) == len(seqs) > 1:
            for i, s in enumerate(seqs):
                cams[s].append(cam.heatmap(i, size))
        else:
            for s in seqs:
                cams[s].append(cam.heatmap(0, size))
        if isinstance(aux, dict):
            for s in seqs:
                w = aux.get(s, aux.get("fused_attention"))
                weights[s].append(w.detach().cpu().numpy().squeeze() if w is not None else None)
            if aux.get("__seq_attention__") is not None:
                seq_attn.append(aux["__seq_attention__"].detach().cpu().numpy().squeeze())
        else:
            for s in seqs:
                weights[s].append(aux.detach().cpu().numpy().squeeze() if aux is not None else None)
        cam.remove()

    n_slices = models_and_samples[0][1]["slice_indices"].numel()
    ens = np.mean(probs, axis=0)
    return {"gradcam": {s: np.mean(v, axis=0) for s, v in cams.items()},
            "weights": {s: (np.mean([w for w in v if w is not None], axis=0) if any(w is not None for w in v)
                            else np.ones(n_slices) / n_slices) for s, v in weights.items()},
            "seq_attention": np.mean(seq_attn, axis=0) if seq_attn else None,
            "pred_class": int(ens.argmax()), "pred_prob": float(ens.max())}


def _display(img, mask, seq):
    """Percentile window inside the tissue mask; ADC keeps its fixed [-1, 1] training scale."""
    tissue = mask > 0
    if seq == "ADC":
        vmin, vmax = -1.0, 1.0
    elif tissue.any():
        vmin, vmax = np.percentile(img[tissue], [1, 99])
    else:
        vmin, vmax = float(img.min()), float(img.max())
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1) * mask


def plot_case(cfg, interp, sample, c, pid, true_label, case_label):
    """Each Grad-CAM overlay next to the image it was computed on; rows are slices, and the
    slice-attention weight is printed beside each slice index."""
    seqs, names = c["sequences"], data.CLASS_NAMES[c["task"]]
    n_slices = sample["slice_indices"].numel()
    fig, axes = plt.subplots(n_slices, 2 * len(seqs), figsize=(2.9 * len(seqs), 1.45 * n_slices), squeeze=False,
                             gridspec_kw=dict(wspace=0.04, hspace=0.10))
    status = "Correct" if interp["pred_class"] == true_label else "Misclassified"
    title = (f"{status} | Patient {pid}\nReference: {names[true_label]} | "
             f"Model: {names[interp['pred_class']]} ({interp['pred_prob']:.0%})")
    if interp["seq_attention"] is not None:
        title += "\nSequence weighting: " + "  ".join(
            f"{s} {float(w):.2f}" for s, w in zip(seqs, np.atleast_1d(interp["seq_attention"])))
    fig.suptitle(f"{title}\n{case_label}", fontsize=10.5, y=0.995, fontweight="bold")

    for qi, s in enumerate(seqs):
        mask_name = "StructMask" if data.SEQUENCE_MODALITY[s] == "structural" else "DiffMask"
        mask_path = Path(cfg.paths.crops_dir) / f"{pid}_{mask_name}.npy"
        masks = (data.center_slices(np.load(mask_path), cfg.data.num_slices)[0] if mask_path.exists()
                 else np.ones((n_slices,) + tuple(sample["images"][s].shape[-2:])))
        raw = sample["images"][s][:, 0].numpy()
        for si in range(n_slices):
            disp = _display(raw[si], masks[si], s)
            ax_raw, ax_cam = axes[si, 2 * qi], axes[si, 2 * qi + 1]
            ax_raw.imshow(disp, cmap="gray")
            ax_cam.imshow(disp, cmap="gray")
            ax_cam.imshow(plt.get_cmap("jet")(interp["gradcam"][s][si] * masks[si])[..., :3], alpha=0.42)
            for ax in (ax_raw, ax_cam):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if si == 0:
                ax_raw.set_title(s, fontsize=9, fontweight="bold", pad=5)
                ax_cam.set_title("Grad-CAM", fontsize=8, color="0.35", pad=5)
            if qi == 0:
                w = np.atleast_1d(interp["weights"][s])
                ax_raw.set_ylabel(f"slice {si}\n{w[si] if si < len(w) else np.nan:.2f}", fontsize=7.5,
                                  rotation=0, ha="right", va="center", labelpad=16)
    return fig


def select_cases(r, n_errors):
    y, yp, pids, task = r["y_true"], r["y_pred"], r["pids"], r["config"]["task"]
    cases = {}
    if task == "binary":
        tp = [i for i in range(len(y)) if y[i] == 1 and yp[i] == 1]
        tn = [i for i in range(len(y)) if y[i] == 0 and yp[i] == 0]
        cases["True positive"] = tp[0] if tp else None
        cases["True negative"] = tn[0] if tn else None
        for j, i in enumerate([i for i in range(len(y)) if y[i] != yp[i]][:n_errors]):
            cases[f"{'False negative' if y[i] == 1 else 'False positive'} #{j + 1}"] = i
    else:
        names = data.CLASS_NAMES[task]
        for k, name in enumerate(names):
            hit = [i for i in range(len(y)) if y[i] == k and yp[i] == k]
            cases[f"{name}, correct"] = hit[0] if hit else None
        for j, i in enumerate([i for i in range(len(y)) if y[i] != yp[i]][:n_errors]):
            cases[f"Misclassified #{j + 1} (true {names[y[i]]}, predicted {names[yp[i]]})"] = i
    return {label: (pids[i], int(y[i])) for label, i in cases.items() if i is not None}


def case_gallery(cfg, r, pool, test, device, path):
    c = r["config"]
    splits = data.cv_splits(pool, cfg.cv.n_splits, cfg.seed)
    pooled = data.fit_adc_scaler(pool, cfg.paths.crops_dir) if "ADC" in c["sequences"] else None
    folds = []
    for fold in range(1, cfg.cv.n_splits + 1):
        scaler = pooled
        if "ADC" in c["sequences"] and cfg.evaluation.adc_scaler == "per_fold":
            scaler = data.fit_adc_scaler(pool.iloc[splits[fold - 1][0]], cfg.paths.crops_dir)
        model = load_fold_model(cfg, c, fold, device)
        for p in model.parameters():  # Grad-CAM needs gradients through the last block
            p.requires_grad_(True)
        ds = engine.build_dataset(cfg, STEP_COMBINED, test, c["task"], c["sequences"], scaler)
        folds.append((model, ds))

    with PdfPages(path) as pdf:
        for label, (pid, true_label) in select_cases(r, cfg.evaluation.gallery_errors).items():
            i = next(i for i in range(len(folds[0][1])) if str(folds[0][1].df.iloc[i]["Scan_ID"]) == str(pid))
            pairs = [(m, ds[i]) for m, ds in folds]
            interp = case_interpretability(pairs, c, device)
            fig = plot_case(cfg, interp, pairs[0][1], c, pid, true_label, label)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(cfg, results, intervals, per_class, comparisons, out_dir):
    lines, summary = ["PRIMARY RESULTS (held-out test set)", "=" * 78,
                      "One model per task, chosen by cross-validation before the test set was used.", ""], []
    for task in cfg.tasks:
        key = (task, "primary")
        if key not in results:
            continue
        r, c = results[key], results[key]["config"]
        lines += ["-" * 78, ("Benign vs. malignant" if task == "binary" else "Benign subtypes").upper(), "-" * 78,
                  f"Model:         {pretty_model(c)}", f"Experiment:    {c['experiment']}",
                  f"Test set n:    {len(r['y_true'])}",
                  f"Calibration:   temperature {r['calib']['temperature']:.3f}"
                  + (f", threshold {r['calib']['t_star']:.3f} ({cfg.evaluation.binary_decision})"
                     if task == "binary" else " (argmax)"), ""]
        for mkey, label in MAIN_METRICS:
            est, lo, hi = intervals[key][mkey]
            lines.append(f"{label:<14s}{fmt(est, (lo, hi), cfg.evaluation.show_ci)}")
        if key in per_class:
            lines += ["", "Per class (one-vs-rest):", per_class[key].to_string(index=False)]
        names, cm = data.CLASS_NAMES[task], r["metrics"]["cm"]
        width = max(len(n) for n in names) + 2
        lines += ["", "Confusion matrix (rows: reference, columns: prediction):",
                  " " * width + "".join(f"{n[:12]:>14s}" for n in names)]
        lines += [f"{n:<{width}s}" + "".join(f"{v:>14d}" for v in cm[i]) for i, n in enumerate(names)]
        lines.append("")

    lines += ["", "SUPPLEMENTARY", "=" * 78, "Cross-validation vs. test AUC:"]
    for (task, role), r in results.items():
        c = r["config"]
        est = intervals[(task, role)]["auc"][0]
        if np.isfinite(c["cv_pooled_auc"]):
            lines.append(f"  [{task}/{role}] {pretty_model(c)}: CV {c['cv_pooled_auc']:.3f} -> test {est:.3f} "
                         f"({est - c['cv_pooled_auc']:+.3f})")
        for mkey, label in MAIN_METRICS + EXTRA_METRICS:
            e, lo, hi = intervals[(task, role)][mkey]
            summary.append({"Task": task, "Role": role, "Model": pretty_model(c), "Experiment": c["experiment"],
                            "Metric": label, "Estimate": e, "CI_Low": lo, "CI_High": hi})
    if comparisons:
        lines += ["", "Paired comparisons on the test set (same patients):"]
        for res in comparisons:
            lines.append(f"  [{res['task']}] {res['comparison']}: {pretty_experiment(res['a'])} AUC {res['auc_a']:.3f}"
                         f" vs {pretty_experiment(res['b'])} AUC {res['auc_b']:.3f}; difference {res['diff']:+.3f} "
                         f"[{res['diff_lo']:+.3f}, {res['diff_hi']:+.3f}], p={res['boot_p']:.3f}"
                         + (f" (DeLong p={res['delong_p']:.3f})" if np.isfinite(res["delong_p"]) else "")
                         + f", n={res['n']}")
    lines += ["", f"Settings: binary_decision={cfg.evaluation.binary_decision}, "
                  f"adc_scaler={cfg.evaluation.adc_scaler}, point_estimate={cfg.evaluation.point_estimate}, "
                  f"{cfg.evaluation.n_bootstraps} bootstrap resamples (percentile 95% CI)."]
    text = "\n".join(lines)
    (out_dir / "Final_Report.txt").write_text(text, encoding="utf-8")
    pd.DataFrame(summary).round(4).to_csv(out_dir / "Summary.csv", index=False)
    log.info("\n" + text)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--fast-run", action="store_true", help="evaluate models trained with --fast-run")
    p.add_argument("--no-gallery", action="store_true", help="skip the Grad-CAM case gallery")
    args = p.parse_args()

    cfg = load_config(args.config, fast_run=True if args.fast_run else None)
    out_dir = engine.results_dir(cfg) / "evaluation"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "evaluation.log")
    device = get_device(cfg)

    pool = data.load_train_pool(cfg)
    _, test = data.load_split(cfg.paths.results_dir)
    log.info(f"Device {device} | CV pool {len(pool)} | test set {len(test)} patients")

    results, intervals = {}, {}
    for task, roles in choose_configs(cfg).items():
        plot_selection_funnel(cfg, task, fig_dir)
        for role, c in roles.items():
            log.info(f"\n{c['experiment']} ({task} / {role})")
            try:
                calib = calibrate(cfg, c, pool, device)
                r = evaluate_on_test(cfg, c, calib, pool, test, device)
            except FileNotFoundError as e:
                log.info(f"  skipped: {e}")
                continue
            results[(task, role)] = r
            intervals[(task, role)] = with_intervals(cfg, r)
            m = r["metrics"]
            log.info(f"  test (n={len(r['y_true'])}): AUC {m['auc']:.4f} | Sens {m['sensitivity']:.4f} | "
                     f"Spec {m['specificity']:.4f} | PPV {m['ppv']:.4f} | NPV {m['npv']:.4f}")

    comparisons = []
    for task in cfg.tasks:
        for a, b, why in [("primary", "baseline_single_seq", "fused vs. best single sequence"),
                          ("primary", "secondary", "the two step-4 sequence sets"),
                          ("primary", "clinical", "image only vs. with clinical features")]:
            res = head_to_head(cfg, results, task, a, b, why)
            if res:
                comparisons.append(res)
    if comparisons:
        pd.DataFrame(comparisons).to_csv(out_dir / "HeadToHead_TestSet.csv", index=False)

    per_class = write_tables(cfg, results, intervals, out_dir)
    for task in cfg.tasks:
        key = (task, "primary")
        if key in results:
            plot_roc(cfg, results[key], intervals[key], fig_dir / f"ROC_{task}.pdf")
            plot_confusion(results[key], fig_dir / f"ConfusionMatrix_{task}.pdf")
            plot_reliability(results[key], fig_dir / f"Reliability_{task}.pdf")
            if not args.no_gallery:
                case_gallery(cfg, results[key], pool, test, device, fig_dir / f"CaseGallery_{task}.pdf")
        plot_forest(results, intervals, task, fig_dir / f"Forest_{task}.pdf")
    write_report(cfg, results, intervals, per_class, comparisons, out_dir)


if __name__ == "__main__":
    main()
