"""Classification metrics, bootstrap confidence intervals and paired model comparisons."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

CONFUSION_METRIC_KEYS = ["bal_acc", "acc", "f1_macro", "sensitivity", "specificity", "ppv", "npv"]


def safe_auc(y_true, y_prob, task="binary") -> float:
    try:
        if task == "binary":
            return float(roc_auc_score(y_true, y_prob))
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:  # a class is missing from y_true
        return float("nan")


def get_youden_threshold(y_true, y_prob) -> float:
    """Threshold maximizing sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr + (1 - fpr) - 1  # kept in this form: tpr - fpr can round differently and break ties elsewhere
    return float(thresholds[np.argmax(j_scores)])


def _sens_spec(cm, task):
    """Binary: from the 2x2 matrix. Multiclass: one-vs-rest, macro-averaged."""
    if task == "binary" and cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        return {"sensitivity": tp / (tp + fn) if tp + fn else 0.0,
                "specificity": tn / (tn + fp) if tn + fp else 0.0,
                "ppv": tp / (tp + fp) if tp + fp else 0.0,
                "npv": tn / (tn + fn) if tn + fn else 0.0}
    spec, npv = [], []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec.append(tn / (tn + fp) if tn + fp else 0.0)
        npv.append(tn / (tn + fn) if tn + fn else 0.0)
    return {"specificity": float(np.mean(spec)), "npv": float(np.mean(npv))}


def compute_confusion_metrics(y_true, y_pred, task="binary", n_classes=None) -> dict:
    """Metrics from hard predictions. For multiclass, n_classes fixes the class set (used on the
    test set); None macro-averages over the classes present in y_true or y_pred (used in CV,
    where a small fold can miss a class)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if task == "binary":
        labels = [0, 1]
    elif n_classes:
        labels = list(range(n_classes))
    else:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    m = {"bal_acc": balanced_accuracy_score(y_true, y_pred),
         "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
         "acc": accuracy_score(y_true, y_pred),
         "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
         "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
         "cm": cm}
    if task != "binary":
        m["sensitivity"] = m["recall_macro"]
        m["ppv"] = m["precision_macro"]
    m.update(_sens_spec(cm, task))
    return m


def compute_metrics(y_true, y_prob, task="binary", threshold=0.5):
    """AUC from probabilities plus the confusion metrics at `threshold` (binary) or argmax.
    Returns (metrics, y_pred)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int) if task == "binary" else np.argmax(y_prob, axis=1)
    m = compute_confusion_metrics(y_true, y_pred, task)
    m["auc"] = safe_auc(y_true, y_prob, task)
    return m, y_pred


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def _n_needed(y_true, task):
    return 2 if task == "binary" else len(np.unique(y_true))


def bootstrap_auc(y_true, y_prob, task="binary", n_bootstraps=2000, seed=42):
    """Percentile 95% CI of the AUC over patient resamples. Returns (mean, low, high)."""
    rng = np.random.RandomState(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    n, need = len(y_true), _n_needed(y_true, task)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < need:
            continue
        s = safe_auc(y_true[idx], y_prob[idx], task)
        if np.isfinite(s):
            scores.append(s)
    if not scores:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def bootstrap_confusion_metrics(y_true, y_pred, task="binary", n_bootstraps=2000, seed=42, n_classes=None):
    """Percentile 95% CIs for the confusion metrics. Returns {key: (mean, low, high)}."""
    rng = np.random.RandomState(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n, need = len(y_true), _n_needed(y_true, task)
    collected = {k: [] for k in CONFUSION_METRIC_KEYS}
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < need:
            continue
        m = compute_confusion_metrics(y_true[idx], y_pred[idx], task, n_classes)
        for k in CONFUSION_METRIC_KEYS:
            collected[k].append(m[k])
    out = {}
    for k, vals in collected.items():
        if not vals:
            out[k] = (float("nan"),) * 3
        else:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[k] = (float(np.mean(vals)), float(lo), float(hi))
    return out


def per_class_metrics(y_true, y_pred, y_prob, class_names) -> pd.DataFrame:
    """One-vs-rest AUC / sensitivity / specificity / PPV / NPV per class, with support n."""
    y_true, y_pred, y_prob = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
    rows = []
    for i, name in enumerate(class_names):
        t, p = (y_true == i).astype(int), (y_pred == i).astype(int)
        tp, fn = int(((t == 1) & (p == 1)).sum()), int(((t == 1) & (p == 0)).sum())
        fp, tn = int(((t == 0) & (p == 1)).sum()), int(((t == 0) & (p == 0)).sum())
        rows.append({"Class": name, "n": int(t.sum()),
                     "auc": safe_auc(t, y_prob[:, i]) if y_prob.ndim > 1 else np.nan,
                     "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                     "specificity": tn / (tn + fp) if tn + fp else np.nan,
                     "ppv": tp / (tp + fp) if tp + fp else np.nan,
                     "npv": tn / (tn + fn) if tn + fn else np.nan})
    return pd.DataFrame(rows)


def bootstrap_per_class(y_true, y_pred, y_prob, class_names, n_bootstraps=2000, seed=42) -> dict:
    """{(class index, metric): (low, high)}."""
    rng = np.random.RandomState(seed)
    y_true, y_pred, y_prob = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
    keys = ["auc", "sensitivity", "specificity", "ppv", "npv"]
    collected = {(c, k): [] for c in range(len(class_names)) for k in keys}
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < len(class_names):
            continue
        df_b = per_class_metrics(y_true[idx], y_pred[idx], y_prob[idx], class_names)
        for c, row in df_b.iterrows():
            for k in keys:
                if np.isfinite(row[k]):
                    collected[(c, k)].append(row[k])
    return {key: (tuple(float(x) for x in np.percentile(v, [2.5, 97.5])) if v else (np.nan, np.nan))
            for key, v in collected.items()}


# ---------------------------------------------------------------------------
# Paired comparisons of two models on the same patients
# ---------------------------------------------------------------------------
def _delong_placements(y_true, y_prob):
    pos, neg = y_prob[y_true == 1], y_prob[y_true == 0]
    tx = np.array([((neg < p).sum() + 0.5 * (neg == p).sum()) / len(neg) for p in pos])
    ty = np.array([((pos > n).sum() + 0.5 * (pos == n).sum()) / len(pos) for n in neg])
    return tx, ty


def delong_paired_test(y_true, prob_a, prob_b):
    """Paired DeLong test for equal binary AUCs (Sun & Xu, 2014). Returns (auc_a, auc_b, z, p)."""
    y_true, prob_a, prob_b = np.asarray(y_true), np.asarray(prob_a), np.asarray(prob_b)
    auc_a, auc_b = safe_auc(y_true, prob_a), safe_auc(y_true, prob_b)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos < 2 or n_neg < 2:
        return auc_a, auc_b, float("nan"), float("nan")
    tx_a, ty_a = _delong_placements(y_true, prob_a)
    tx_b, ty_b = _delong_placements(y_true, prob_b)
    s = np.cov(np.vstack([tx_a, tx_b])) / n_pos + np.cov(np.vstack([ty_a, ty_b])) / n_neg
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:  # near-identical predictions
        return auc_a, auc_b, float("nan"), float("nan")
    z = (auc_a - auc_b) / np.sqrt(var)
    return auc_a, auc_b, float(z), float(2 * (1 - norm.cdf(abs(z))))


def paired_bootstrap_auc_diff(y_true, prob_a, prob_b, task="binary", n_bootstraps=5000, seed=0):
    """AUC(A) - AUC(B) resampling patients jointly. Returns (mean, low, high, two-sided p)."""
    rng = np.random.RandomState(seed)
    y_true, prob_a, prob_b = np.asarray(y_true), np.asarray(prob_a), np.asarray(prob_b)
    n, need = len(y_true), _n_needed(y_true, task)
    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < need:
            continue
        d = safe_auc(y_true[idx], prob_a[idx], task) - safe_auc(y_true[idx], prob_b[idx], task)
        if np.isfinite(d):
            diffs.append(d)
    if not diffs:
        return (float("nan"),) * 4
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diffs.mean()), float(lo), float(hi), float(min(p, 1.0))


def align_by_pid(pids_a, y_a, p_a, pids_b, y_b, p_b):
    """Restrict two prediction sets to common patients, in the same order."""
    ia = {p: i for i, p in enumerate(map(str, pids_a))}
    ib = {p: i for i, p in enumerate(map(str, pids_b))}
    common = sorted(set(ia) & set(ib))
    if not common:
        raise ValueError("No patients in common.")
    a_idx, b_idx = [ia[p] for p in common], [ib[p] for p in common]
    y = np.asarray(y_a)[a_idx]
    if not np.array_equal(y, np.asarray(y_b)[b_idx]):
        raise ValueError("Label mismatch between the two prediction sets.")
    return y, np.asarray(p_a)[a_idx], np.asarray(p_b)[b_idx], common


def compare_predictions(y_true, prob_a, prob_b, task, n_bootstraps=5000) -> dict:
    """DeLong (binary only) and paired bootstrap of the AUC difference."""
    out = {"n": len(y_true)}
    if task == "binary":
        out["auc_a"], out["auc_b"], out["delong_z"], out["delong_p"] = delong_paired_test(y_true, prob_a, prob_b)
    else:
        out["auc_a"], out["auc_b"] = safe_auc(y_true, prob_a, task), safe_auc(y_true, prob_b, task)
        out["delong_z"] = out["delong_p"] = float("nan")
    d, lo, hi, p = paired_bootstrap_auc_diff(y_true, prob_a, prob_b, task, n_bootstraps)
    out.update(diff=d, diff_lo=lo, diff_hi=hi, boot_p=p)
    return out
