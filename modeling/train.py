"""Train and select models by cross-validation on the training pool (the test set is never read).

Steps (each can be run on its own; later steps read the results of earlier ones from disk):
  split        stratified train/test split, written once and reused afterwards
  1            single-slice baseline per sequence
  2            multi-slice attention MIL per sequence (encoder warm-started from step 1)
  3            multi-sequence fusion on sequence sets nominated from steps 1 and 2
  4            multi-slice + multi-sequence models for the best step-3 configurations
  diagnostics  paired tests (best single sequence vs best fused) and sequence-attention weights
Optional, after step 4:
  challengers  every other step-4 fusion architecture on the same sequence sets
  extended     further 2/3-sequence combinations at step 3
  promote      step 4 for a configuration found by 'extended'
  repeat-cv    the step-4 models on additional independent fold splits

Usage: python train.py --config config.yaml [--steps split 1 2 3 4 diagnostics] [--fast-run]
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import data
import engine
import metrics
from common import get_device, load_config, log, set_tf32, setup_logging
from models import STEP_COMBINED, STEP_FUSION, STEP_MIL, STEP_SINGLE

DEFAULT_STEPS = ["split", "1", "2", "3", "4", "diagnostics"]
FUSION_VARIANTS = [
    ("Early", dict(model_type="early", fusion_type="concat")),
    ("LateConcat", dict(model_type="late", fusion_type="concat")),
    ("SeqAttn", dict(model_type="late", fusion_type="seq_attention")),
]


def state_path(cfg, name) -> Path:
    return engine.results_dir(cfg) / name


def save_json(cfg, name, obj):
    with open(state_path(cfg, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(cfg, name):
    path = state_path(cfg, name)
    if not path.exists():
        raise RuntimeError(f"{path.name} not found; run the step that writes it first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def make_split(cfg):
    """Stratified (on DDx) train/test split. If split files exist they are reused, so the
    held-out set never changes between runs."""
    rdir = engine.results_dir(cfg)
    tr_path, te_path = rdir / "train_cv_split.csv", rdir / "test_split.csv"
    if tr_path.exists() and te_path.exists():
        pool, test = data.load_split(rdir)
        log.info(f"Using existing split: {len(pool)} CV / {len(test)} test patients")
        return

    df = data.load_labels(cfg.paths.labels_csv)
    crops = Path(cfg.paths.crops_dir)
    missing = [f"{sid}_{seq}" for sid in df["Scan_ID"] for seq in cfg.sequences
               if not (crops / f"{sid}_{seq}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} crops missing in {crops}, e.g. {missing[:3]}")
    if data.has_repeat_patients(df):
        log.info("Repeat studies found (IDs with _2, _3 ...): splits are grouped by patient.")

    pool, test = data.stratified_holdout(df, cfg.data.test_fraction, cfg.seed)
    data.check_disjoint(pool, test, "CV pool / test set")
    pool.to_csv(tr_path, index=False)
    test.to_csv(te_path, index=False)
    log.info(f"Split written: {len(pool)} CV / {len(test)} test patients")
    log.info(f"Class counts (DDx) CV pool: {pool['DDx'].value_counts().sort_index().to_dict()} | "
             f"test: {test['DDx'].value_counts().sort_index().to_dict()}")


# ---------------------------------------------------------------------------
# Ranking (by pooled out-of-fold AUC)
# ---------------------------------------------------------------------------
def merge_pooled_auc(cfg, ranked, task):
    path = engine.bootstrap_csv(cfg)
    if not path.exists():
        ranked["AUC_Pooled"] = ranked["AUC_mean"]
        ranked["BalAcc_Pooled"] = ranked["BalAcc_mean"]
        return ranked
    boot = pd.read_csv(path)
    boot = boot[boot["Task"] == task][["Experiment", "AUC_Pooled", "BalAcc_Pooled", "AUC_CI_Low", "AUC_CI_High"]]
    merged = ranked.merge(boot.drop_duplicates("Experiment", keep="last"), on="Experiment", how="left")
    merged["AUC_Pooled"] = merged["AUC_Pooled"].fillna(merged["AUC_mean"])
    merged["BalAcc_Pooled"] = merged["BalAcc_Pooled"].fillna(merged["BalAcc_mean"])
    return merged


def rank_sequences(cfg, step, task) -> pd.DataFrame:
    """Single-sequence experiments (steps 1 and 2), best first. Clinical variants excluded."""
    path = engine.results_csv(cfg, step)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[(df["Task"] == task) & ~df["Experiment"].astype(str).str.endswith("_Clinical")]
    if df.empty:
        return pd.DataFrame()
    ranked = (df.groupby(["Experiment", "Seq"])
              .agg(AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"), BalAcc_mean=("Bal_Acc", "mean"),
                   n_folds=("Fold", "count"))
              .reset_index())
    return merge_pooled_auc(cfg, ranked, task).sort_values("AUC_Pooled", ascending=False).reset_index(drop=True)


def rank_experiments(cfg, csv_path, task) -> pd.DataFrame:
    """Multi-sequence experiments (steps 3 and 4), best first."""
    if not Path(csv_path).exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df = df[df["Task"] == task]
    if df.empty:
        return pd.DataFrame()
    agg = {"AUC_mean": ("AUC", "mean"), "AUC_std": ("AUC", "std"), "BalAcc_mean": ("Bal_Acc", "mean"),
           "Seq": ("Seq", "first"), "n_folds": ("Fold", "count")}
    for col in ("Model_Type", "Fusion_Type", "Use_Clinical"):
        if col in df.columns:
            agg[col] = (col, "first")
    ranked = df.groupby("Experiment").agg(**agg).reset_index()
    return merge_pooled_auc(cfg, ranked, task).sort_values("AUC_Pooled", ascending=False).reset_index(drop=True)


def consensus_ranking(cfg, task) -> pd.DataFrame:
    """Per-sequence average of the step-1 and step-2 ranks. Ranks rather than AUCs, because
    multi-slice pooling lifts some sequences far more than others; ties go to step 2, the
    closer proxy for the final multi-slice model."""
    r1, r2 = rank_sequences(cfg, STEP_SINGLE, task), rank_sequences(cfg, STEP_MIL, task)
    if r1.empty or r2.empty:
        raise RuntimeError("Steps 1 and 2 must be complete before nominating sequences.")
    m = (r1[["Seq", "AUC_Pooled"]].rename(columns={"AUC_Pooled": "AUC_SingleSlice"})
         .merge(r2[["Seq", "AUC_Pooled"]].rename(columns={"AUC_Pooled": "AUC_MultiSlice"}), on="Seq", how="outer"))
    m["Rank_SingleSlice"] = m["AUC_SingleSlice"].rank(ascending=False, method="min")
    m["Rank_MultiSlice"] = m["AUC_MultiSlice"].rank(ascending=False, method="min")
    m["Consensus_Rank"] = m[["Rank_SingleSlice", "Rank_MultiSlice"]].mean(axis=1)
    m["MIL_Gain"] = (m["AUC_MultiSlice"] - m["AUC_SingleSlice"]).round(4)
    m["Modality"] = m["Seq"].map(data.SEQUENCE_MODALITY)
    return m.sort_values(["Consensus_Rank", "AUC_MultiSlice"], ascending=[True, False]).reset_index(drop=True)


def nominate_fusion_configs(cfg, consensus) -> dict:
    """A = top-k sequences by consensus rank. B replaces A's weakest member with the best
    sequence of the other modality if A is single-modality, otherwise with the next best
    sequence overall, so step 3 always compares at least two distinct sets."""
    size, n_configs = cfg.selection.fusion_combo_size, cfg.selection.n_fusion_configs
    seqs = consensus["Seq"].tolist()
    configs = {"A": seqs[:size]}
    A = configs["A"]
    mods = {data.SEQUENCE_MODALITY[s] for s in A}
    B = None
    if len(mods) == 1:
        other = "diffusion" if mods.pop() == "structural" else "structural"
        cand = [s for s in seqs if data.SEQUENCE_MODALITY[s] == other]
        if cand:
            B = A[:size - 1] + [cand[0]]
    if B is None:
        rest = [s for s in seqs if s not in A]
        if rest:
            B = A[:size - 1] + [rest[0]]
    if B is not None and set(B) != set(A):
        configs["B"] = B
    if len(configs) < 2:
        log.info("WARNING: only one distinct fusion configuration could be derived.")
    return {k: configs[k] for k in list(configs)[:max(n_configs, 1)]}


def decide_stage4_arch(model_type, fusion_type):
    """Step-3 winner -> step-4 architecture. Early stays early; a sequence-attention winner
    stays late so the selected mechanism is kept; a late-concat winner becomes mid fusion
    (sequences are co-registered, so fusing per slice before slice attention is natural)."""
    if model_type == "early":
        return "early", "concat"
    if fusion_type == "seq_attention":
        return "late", "seq_attention"
    return "mid", "concat"


def stage4_name(arch_name, fusion_type, seqs):
    if arch_name == "joint":
        return f"Stage4_JointAttention_{'_'.join(seqs)}"
    suffix = "_SeqAttn" if fusion_type == "seq_attention" else ""
    return f"Stage4_{arch_name.capitalize()}Fusion_{'_'.join(seqs)}{suffix}"


def best_step2_prefix(cfg, task, seqs):
    r2 = rank_sequences(cfg, STEP_MIL, task)
    hit = r2[r2["Seq"].isin(seqs)] if not r2.empty else r2
    return f"Stage2_{hit.iloc[0]['Seq']}" if not hit.empty else None


def parse_fusion_row(row) -> dict:
    name = str(row["Experiment"])
    model_type = row.get("Model_Type") if pd.notna(row.get("Model_Type")) else (
        "early" if "_Early_" in name else "late")
    fusion_type = row.get("Fusion_Type") if pd.notna(row.get("Fusion_Type")) else (
        "seq_attention" if "_SeqAttn" in name else "concat")
    return {"experiment_name": name, "sequences": str(row["Seq"]).split("+"), "model_type": model_type,
            "fusion_type": fusion_type, "pooled_auc": float(row["AUC_Pooled"])}


def pick_fusion_winners(cfg, task, csv_path=None):
    """Best fusion run for each distinct sequence set, until n_stage4_configs sets are picked."""
    ranked = rank_experiments(cfg, csv_path or engine.results_csv(cfg, STEP_FUSION), task)
    ranked = ranked[~ranked["Experiment"].astype(str).str.endswith("_Clinical")]
    picked, seen = [], set()
    for _, row in ranked.iterrows():
        key = frozenset(str(row["Seq"]).split("+"))
        if key in seen:
            continue
        seen.add(key)
        picked.append(parse_fusion_row(row))
        if len(picked) >= cfg.selection.n_stage4_configs:
            break
    return picked, ranked


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def run_step1(cfg):
    for task in cfg.tasks:
        for seq in cfg.sequences:
            engine.run_experiment(cfg, STEP_SINGLE, f"Stage1_{seq}", task, [seq])


def run_step2(cfg):
    # Each sequence warm-starts from its own step-1 model, so steps 1 and 2 differ only in slice pooling.
    for task in cfg.tasks:
        for seq in cfg.sequences:
            engine.run_experiment(cfg, STEP_MIL, f"Stage2_{seq}", task, [seq], mil_type="attention",
                                  warm_start_from=f"Stage1_{seq}")


def nominate(cfg) -> dict:
    out = {}
    for task in cfg.tasks:
        consensus = consensus_ranking(cfg, task)
        consensus.to_csv(engine.results_dir(cfg) / f"Consensus_Ranking_{task}.csv", index=False)
        log.info(f"\nConsensus sequence ranking ({task}):\n" + consensus[
            ["Seq", "Modality", "AUC_SingleSlice", "AUC_MultiSlice", "MIL_Gain", "Consensus_Rank"]].to_string(index=False))
        out[task] = nominate_fusion_configs(cfg, consensus)
        for label, seqs in out[task].items():
            log.info(f"  fusion config {label} ({task}): {seqs}")
    save_json(cfg, "fusion_configs.json", out)
    return out


def run_step3(cfg):
    configs = nominate(cfg)
    for task in cfg.tasks:
        ranked1 = rank_sequences(cfg, STEP_SINGLE, task)
        for label, seqs in configs[task].items():
            # Encoder from the best single sequence of this configuration.
            warm = f"Stage1_{ranked1[ranked1['Seq'].isin(seqs)].iloc[0]['Seq']}"
            for variant, kw in FUSION_VARIANTS:
                if kw["model_type"] == "early" and len(seqs) != 3:
                    continue
                engine.run_experiment(cfg, STEP_FUSION, f"Stage3{label}_{variant}_{'_'.join(seqs)}", task, seqs,
                                      warm_start_from=warm, **kw)


def run_stage4_config(cfg, task, base_name, seqs, model_type, fusion_type, encoder_from, mil_from,
                      seqattn_from=None, clinical=(False, True), csv_path=None, cv_seed=None):
    for use_clinical in clinical:
        engine.run_experiment(cfg, STEP_COMBINED, base_name, task, seqs, model_type=model_type,
                              fusion_type=fusion_type, mil_type="attention", use_clinical=use_clinical,
                              warm_start_from=encoder_from, warm_start_mil_from=mil_from,
                              warm_start_seqattn_from=seqattn_from, csv_path=csv_path, cv_seed=cv_seed)


def run_step4(cfg):
    stage4_runs = {}
    for task in cfg.tasks:
        winners, fusion_ranked = pick_fusion_winners(cfg, task)
        log.info(f"\nStep 3 ranking ({task}):\n" + fusion_ranked[
            ["Experiment", "Seq", "AUC_Pooled", "AUC_CI_Low", "AUC_CI_High"]].to_string(index=False))
        runs = []
        for w in winners:
            seqs = w["sequences"]
            arch, fusion = decide_stage4_arch(w["model_type"], w["fusion_type"])
            name = stage4_name(arch, fusion, seqs)
            mil_from = best_step2_prefix(cfg, task, seqs)
            log.info(f"\n[{task}] {name}: encoder <- {w['experiment_name']}, slice attention <- {mil_from}")
            run_stage4_config(cfg, task, name, seqs, arch, fusion, w["experiment_name"], mil_from)
            runs.append({"base_name": name, "sequences": seqs, "model_type": arch, "fusion_type": fusion,
                         "source": w["experiment_name"]})

        # Joint slice x sequence attention on the top configuration: the one step-4 family
        # that does not factorise the two attention axes.
        if cfg.selection.run_joint_attention and winners:
            seqs = winners[0]["sequences"]
            same_set = fusion_ranked[fusion_ranked["Seq"] == "+".join(seqs)]
            sa = same_set[same_set["Experiment"].astype(str).str.contains("_SeqAttn")]
            sa_from = sa.iloc[0]["Experiment"] if not sa.empty else winners[0]["experiment_name"]
            name = stage4_name("joint", "joint_attention", seqs)
            run_stage4_config(cfg, task, name, seqs, "joint", "joint_attention", winners[0]["experiment_name"],
                              best_step2_prefix(cfg, task, seqs), seqattn_from=sa_from)
            runs.append({"base_name": name, "sequences": seqs, "model_type": "joint",
                         "fusion_type": "joint_attention", "source": winners[0]["experiment_name"]})
        stage4_runs[task] = runs
        ranked = rank_experiments(cfg, engine.results_csv(cfg, STEP_COMBINED), task)
        log.info(f"\nStep 4 ranking ({task}):\n" + ranked[
            ["Experiment", "Seq", "AUC_Pooled", "AUC_CI_Low", "AUC_CI_High"]].to_string(index=False))
    save_json(cfg, "stage4_runs.json", stage4_runs)


# ---------------------------------------------------------------------------
# Diagnostics (cross-validation pool only; the confirmatory comparison is in evaluate.py)
# ---------------------------------------------------------------------------
def compare_experiments(cfg, name_a, name_b, task):
    a, b = engine.load_oof(cfg, name_a, task), engine.load_oof(cfg, name_b, task)
    if a is None or b is None:
        log.info(f"  [{task}] no saved OOF predictions for {name_a if a is None else name_b}; skipped")
        return None
    y, pa, pb, _ = metrics.align_by_pid(a["pid"], a["y_true"], a["y_prob"], b["pid"], b["y_true"], b["y_prob"])
    res = {"task": task, "a": name_a, "b": name_b, **metrics.compare_predictions(y, pa, pb, task)}
    log.info(f"  [{task}] {name_a} (AUC {res['auc_a']:.4f}) vs {name_b} (AUC {res['auc_b']:.4f}), n={res['n']}: "
             f"diff {res['diff']:+.4f} [{res['diff_lo']:+.4f}, {res['diff_hi']:+.4f}], bootstrap p={res['boot_p']:.4f}"
             + (f", DeLong p={res['delong_p']:.4f}" if np.isfinite(res["delong_p"]) else ""))
    return res


def inspect_sequence_attention(cfg, name, task, sequences, model_type):
    """Mean sequence-attention weight on each fold's own validation patients, split by whether
    the prediction was correct. Uniform weights or a weight pinned to one sequence mean the
    head is not doing per-patient weighting."""
    device = get_device(cfg)
    use_clinical = name.endswith("_Clinical")
    pool = data.load_train_pool(cfg)
    W, correct = [], []
    for fold, (tr_idx, va_idx) in enumerate(data.cv_splits(pool, cfg.cv.n_splits, cfg.seed)):
        path = engine.checkpoint_path(cfg, name, task, fold + 1)
        if not path.exists():
            continue
        tr_df = pool.iloc[tr_idx].reset_index(drop=True)
        scaler = data.fit_adc_scaler(tr_df, cfg.paths.crops_dir) if "ADC" in sequences else None
        ds = engine.build_dataset(cfg, STEP_COMBINED, pool.iloc[va_idx].reset_index(drop=True), task, sequences, scaler)
        model = engine.make_model(cfg, STEP_COMBINED, task, sequences, model_type,
                                  "joint_attention" if model_type == "joint" else "seq_attention", "attention",
                                  use_clinical)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval().to(device)
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=cfg.training.batch_size[STEP_COMBINED]):
                images = {k: v.to(device) for k, v in batch["images"].items()}
                logits, aux = model(images, clinical=batch["clinical"].to(device),
                                    slice_indices=batch["slice_indices"].to(device))
                W.append(aux["__seq_attention__"].squeeze(-1).cpu().numpy())
                correct.append(logits.argmax(1).cpu().numpy() == batch["label"].numpy())
        del model
    if not W:
        return None
    W, correct = np.concatenate(W), np.concatenate(correct)
    rows = [{"Sequence": s, "Weight_when_correct": W[correct, i].mean() if correct.any() else np.nan,
             "Weight_when_wrong": W[~correct, i].mean() if (~correct).any() else np.nan}
            for i, s in enumerate(sequences)]
    df = pd.DataFrame(rows)
    df["Delta"] = df["Weight_when_correct"] - df["Weight_when_wrong"]
    mean_w = W.mean(axis=0)
    verdict = ("near-uniform (head reduces to averaging)" if mean_w.max() - mean_w.min() < 0.05 else
               "collapsed onto one sequence" if mean_w.max() > 0.85 else "non-uniform without collapse")
    log.info(f"\n  {name} ({task}), {int(correct.sum())} correct / {int((~correct).sum())} wrong: {verdict}\n"
             + df.round(4).to_string(index=False))
    df.insert(0, "Experiment", name)
    df.insert(1, "Task", task)
    return df


def run_diagnostics(cfg):
    rows = []
    stage4_runs = load_json(cfg, "stage4_runs.json")
    for task in cfg.tasks:
        r2 = rank_sequences(cfg, STEP_MIL, task)
        r4 = rank_experiments(cfg, engine.results_csv(cfg, STEP_COMBINED), task)
        r4 = r4[~r4["Experiment"].astype(str).str.endswith("_Clinical")] if not r4.empty else r4
        if r2.empty or r4.empty:
            continue
        log.info(f"\n[{task}] best single sequence vs best fused (CV pool, diagnostic only)")
        res = compare_experiments(cfg, r2.iloc[0]["Experiment"], r4.iloc[0]["Experiment"], task)
        if res:
            rows.append({**res, "comparison": "best single sequence vs best fused"})
        runs = stage4_runs.get(task, [])
        if len(runs) >= 2:
            res = compare_experiments(cfg, runs[0]["base_name"], runs[1]["base_name"], task)
            if res:
                rows.append({**res, "comparison": "step-4 configuration A vs B"})
    if rows:
        pd.DataFrame(rows).to_csv(engine.results_dir(cfg) / "CV_Paired_Comparisons.csv", index=False)

    frames = []
    for task in cfg.tasks:
        for run in stage4_runs.get(task, []):
            if run["fusion_type"] not in ("seq_attention", "joint_attention"):
                continue
            for name in (run["base_name"], run["base_name"] + "_Clinical"):
                if engine.checkpoint_path(cfg, name, task, 1).exists():
                    df = inspect_sequence_attention(cfg, name, task, run["sequences"], run["model_type"])
                    if df is not None:
                        frames.append(df)
    if frames:
        pd.concat(frames).to_csv(engine.results_dir(cfg) / "Sequence_Attention_Diagnostics.csv", index=False)


# ---------------------------------------------------------------------------
# Optional steps
# ---------------------------------------------------------------------------
def run_challengers(cfg):
    """Every step-4 architecture not chosen by decide_stage4_arch, on the same sequence sets."""
    stage4_runs = load_json(cfg, "stage4_runs.json")
    all_archs = [("early", "concat"), ("late", "concat"), ("mid", "concat"), ("late", "seq_attention"),
                 ("mid", "seq_attention"), ("joint", "joint_attention")]
    for task in cfg.tasks:
        fusion_ranked = rank_experiments(cfg, engine.results_csv(cfg, STEP_FUSION), task)
        done = {(r["model_type"], r["fusion_type"], "+".join(r["sequences"])) for r in stage4_runs.get(task, [])}
        for run in [r for r in stage4_runs.get(task, []) if r["model_type"] != "joint"]:
            seqs = run["sequences"]
            same_set = fusion_ranked[fusion_ranked["Seq"] == "+".join(seqs)]
            for arch, fusion in all_archs:
                if (arch, fusion, "+".join(seqs)) in done or (arch == "early" and len(seqs) != 3):
                    continue
                done.add((arch, fusion, "+".join(seqs)))
                want_sa = fusion == "seq_attention"
                cand = same_set[same_set["Experiment"].astype(str).str.contains("_SeqAttn") == want_sa]
                source = cand.iloc[0]["Experiment"] if not cand.empty else run["source"]
                run_stage4_config(cfg, task, stage4_name(arch, fusion, seqs), seqs, arch, fusion, source,
                                  best_step2_prefix(cfg, task, seqs), seqattn_from=source, clinical=(False,))


def run_extended(cfg):
    """Step 3 on further sequence combinations, ranked by summed consensus AUC and capped."""
    ext_csv = engine.results_dir(cfg) / "Stage3_Fusion_Extended_Results.csv"
    best = {}
    for task in cfg.tasks:
        tested = set()
        for path in (engine.results_csv(cfg, STEP_FUSION), ext_csv):
            if path.exists():
                df = pd.read_csv(path)
                tested |= {frozenset(str(s).split("+")) for s in df.loc[df["Task"] == task, "Seq"].unique()}
        consensus = consensus_ranking(cfg, task)
        score = consensus.set_index("Seq")[["AUC_SingleSlice", "AUC_MultiSlice"]].mean(axis=1).to_dict()
        cands = sorted(((sum(score.get(s, 0.0) for s in combo), list(combo))
                        for size in cfg.extended.combo_sizes
                        for combo in itertools.combinations(cfg.sequences, size)
                        if frozenset(combo) not in tested), key=lambda x: x[0], reverse=True)
        ranked1 = rank_sequences(cfg, STEP_SINGLE, task)
        for _, seqs in cands[:cfg.extended.max_combos]:
            warm = f"Stage1_{ranked1[ranked1['Seq'].isin(seqs)].iloc[0]['Seq']}"
            for variant, kw in FUSION_VARIANTS:
                if kw["model_type"] == "early" and len(seqs) != 3:
                    continue
                engine.run_experiment(cfg, STEP_FUSION, f"Stage3Ext_{variant}_{'_'.join(seqs)}", task, seqs,
                                      warm_start_from=warm, csv_path=ext_csv, **kw)
        frames = [pd.read_csv(p) for p in (engine.results_csv(cfg, STEP_FUSION), ext_csv) if p.exists()]
        combined = engine.results_dir(cfg) / f"_combined_fusion_{task}.csv"
        pd.concat(frames, ignore_index=True).to_csv(combined, index=False)
        top = rank_experiments(cfg, combined, task).iloc[0]
        combined.unlink()
        best[task] = parse_fusion_row(top)
        log.info(f"[{task}] best fusion run across original + extended search: {top['Experiment']} "
                 f"(pooled OOF AUC {top['AUC_Pooled']:.4f})")
    save_json(cfg, "extended_best.json", best)


def run_promote(cfg):
    """Step 4 for the best extended-search configuration, if it is not already a step-4 source."""
    best = load_json(cfg, "extended_best.json")
    stage4_runs = load_json(cfg, "stage4_runs.json")
    for task in cfg.tasks:
        cand = best.get(task)
        if cand is None or cand["experiment_name"] in {r["source"] for r in stage4_runs.get(task, [])}:
            log.info(f"[{task}] nothing to promote")
            continue
        arch, fusion = decide_stage4_arch(cand["model_type"], cand["fusion_type"])
        name = stage4_name(arch, fusion, cand["sequences"])
        run_stage4_config(cfg, task, name, cand["sequences"], arch, fusion, cand["experiment_name"],
                          best_step2_prefix(cfg, task, cand["sequences"]))
        stage4_runs.setdefault(task, []).append({"base_name": name, "sequences": cand["sequences"],
                                                 "model_type": arch, "fusion_type": fusion,
                                                 "source": cand["experiment_name"]})
    save_json(cfg, "stage4_runs.json", stage4_runs)


def fusion_source(cfg, name, task) -> dict:
    """Sequences and architecture of a step-3 run, from the main or the extended results."""
    for path in (engine.results_csv(cfg, STEP_FUSION), engine.results_dir(cfg) / "Stage3_Fusion_Extended_Results.csv"):
        if path.exists():
            df = pd.read_csv(path)
            hit = df[(df["Experiment"] == name) & (df["Task"] == task)]
            if not hit.empty:
                return {"sequences": str(hit.iloc[0]["Seq"]).split("+"), "model_type": hit.iloc[0]["Model_Type"],
                        "fusion_type": hit.iloc[0]["Fusion_Type"]}
    raise RuntimeError(f"Step-3 run {name} ({task}) not found")


def replicate_chain(cfg, task, run, seed, csv_path):
    """Retrain the warm-start sources of one step-4 model on the fold split `seed`.

    Warm-start checkpoints are fold-matched: fold k of the original split was trained on
    patients that may sit in fold k's validation set under another split, so they cannot be
    reused there. The sequences and architectures chosen on the main split are kept."""
    suffix = f"_rep{seed}"
    src = fusion_source(cfg, run["source"], task)
    ranked1 = rank_sequences(cfg, STEP_SINGLE, task)
    enc_seq = ranked1[ranked1["Seq"].isin(src["sequences"])].iloc[0]["Seq"]
    mil_prefix = best_step2_prefix(cfg, task, run["sequences"])
    mil_seq = mil_prefix.split("_", 1)[1] if mil_prefix else None

    def done(name):  # shared by several step-4 models; train once per split
        return all(engine.checkpoint_path(cfg, name, task, k).exists() for k in range(1, cfg.cv.n_splits + 1))

    for seq in sorted({enc_seq, mil_seq} - {None}):
        if not done(f"Stage1_{seq}{suffix}"):
            engine.run_experiment(cfg, STEP_SINGLE, f"Stage1_{seq}{suffix}", task, [seq], csv_path=csv_path,
                                  cv_seed=seed)
    if mil_seq and not done(f"Stage2_{mil_seq}{suffix}"):
        engine.run_experiment(cfg, STEP_MIL, f"Stage2_{mil_seq}{suffix}", task, [mil_seq], mil_type="attention",
                              warm_start_from=f"Stage1_{mil_seq}{suffix}", csv_path=csv_path, cv_seed=seed)
    if not done(run["source"] + suffix):
        engine.run_experiment(cfg, STEP_FUSION, run["source"] + suffix, task, src["sequences"],
                              model_type=src["model_type"], fusion_type=src["fusion_type"],
                              warm_start_from=f"Stage1_{enc_seq}{suffix}", csv_path=csv_path, cv_seed=seed)
    return run["source"] + suffix, (f"Stage2_{mil_seq}{suffix}" if mil_seq else None)


def run_repeat_cv(cfg):
    """The step-4 models (image only) on further independent fold splits. The spread of the
    pooled OOF AUC across splits shows how much depends on which patients share a fold."""
    stage4_runs = load_json(cfg, "stage4_runs.json")
    out_csv = engine.results_dir(cfg) / "Stage4_RepeatCV_Results.csv"
    summary = []
    for task in cfg.tasks:
        for run in stage4_runs.get(task, []):
            boot = pd.read_csv(engine.bootstrap_csv(cfg))
            aucs = boot[(boot["Experiment"] == run["base_name"]) & (boot["Task"] == task)]["AUC_Pooled"].tolist()[-1:]
            for seed in cfg.repeat_cv.seeds:
                encoder_from, mil_from = replicate_chain(cfg, task, run, seed, out_csv)
                name = f"{run['base_name']}_rep{seed}"
                run_stage4_config(cfg, task, name, run["sequences"], run["model_type"], run["fusion_type"],
                                  encoder_from, mil_from, clinical=(False,), csv_path=out_csv, cv_seed=seed)
                boot = pd.read_csv(engine.bootstrap_csv(cfg))
                aucs += boot[(boot["Experiment"] == name) & (boot["Task"] == task)]["AUC_Pooled"].tolist()[-1:]
            aucs = np.array(aucs, dtype=float)
            sd = aucs.std(ddof=1) if len(aucs) > 1 else np.nan
            summary.append({"Experiment": run["base_name"], "Task": task, "N_Splits": len(aucs),
                            "AUC_Mean": aucs.mean(), "AUC_SD": sd, "Per_Split": " ".join(f"{a:.4f}" for a in aucs)})
            log.info(f"[{task}] {run['base_name']}: pooled OOF AUC {aucs.mean():.4f} +/- {sd:.4f} "
                     f"over {len(aucs)} splits")
    pd.DataFrame(summary).to_csv(engine.results_dir(cfg) / "Stage4_RepeatCV_Summary.csv", index=False)


STEPS = {"split": make_split, "1": run_step1, "2": run_step2, "3": run_step3, "4": run_step4,
         "diagnostics": run_diagnostics, "challengers": run_challengers, "extended": run_extended,
         "promote": run_promote, "repeat-cv": run_repeat_cv}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--steps", nargs="+", default=DEFAULT_STEPS, choices=list(STEPS))
    p.add_argument("--fast-run", action="store_true", help="smoke test: small subsample, 2 folds, 2 epochs")
    args = p.parse_args()

    cfg = load_config(args.config, fast_run=True if args.fast_run else None)
    setup_logging(engine.results_dir(cfg) / "training.log")
    set_tf32(cfg.training.allow_tf32)
    log.info(f"Device: {get_device(cfg)} | fast_run: {cfg.fast_run} | steps: {args.steps}")
    log.info(f"Epoch selection: {cfg.cv.epoch_selection} | threshold: {cfg.cv.threshold_source}")
    if cfg.cv.epoch_selection == "outer_val":
        log.info("Note: 'outer_val' selects the epoch on the fold it reports, so CV numbers are optimistic; "
                 "the held-out test set (evaluate.py) is the primary result.")
    for step in args.steps:
        STEPS[step](cfg)


if __name__ == "__main__":
    main()
