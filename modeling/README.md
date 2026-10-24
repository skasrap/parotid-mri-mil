# Modeling

Cross-validated model selection (`train.py`) and held-out evaluation (`evaluate.py`) on the
crops written by `preprocessing/crop.py`. All settings are in `config.yaml`; relative paths in
it are resolved from the folder of the config file.

| File | Contents |
|---|---|
| `data.py` | label loading and validation, splits, ADC scaling, augmentation, dataset |
| `models.py` | ResNet-18 encoder, gated-attention MIL, CBAM, all step 1-4 models |
| `engine.py` | the cross-validation loop shared by all steps, warm-starting, result files |
| `metrics.py` | metrics, Youden threshold, bootstrap CIs, DeLong and paired bootstrap tests |
| `train.py` | the steps below |
| `evaluate.py` | test-set evaluation, tables, figures, Grad-CAM gallery |

## Input

- `labels.csv` with `Scan_ID`, `DDx`, `Sex`, `Age`, `Laterality` (see the main README). Values
  are validated on load: unknown codes stop the run instead of being silently encoded as 0.
  `Malignant` is derived from `DDx == 4`.
- `<Scan_ID>_<sequence>.npy` crops, shape (Z, 224, 224), for T1, T2, T1c, B0, DWI, ADC.

Single-slice models use the centre slice; multi-slice models the 5 central slices. Each grey
slice is copied into the three input channels. ADC, which keeps its physical units through
preprocessing, is mapped to [-1, 1] by its 1st/99th percentiles on the training fold.
Augmentation (flip, rotation up to 15 degrees, shift up to 10%) is drawn once per patient and
applied identically to all slices and sequences; a flip also swaps the laterality feature.

## Training: `python train.py --config config.yaml [--steps ...]`

| Step | Name | What happens |
|---|---|---|
| `split` | | stratified 85/15 split; written once to `results/`, reused afterwards |
| `1` | `Stage1_<seq>` | single-slice baseline for each sequence and task |
| `2` | `Stage2_<seq>` | 5-slice attention MIL per sequence, encoder from `Stage1_<seq>` |
| `3` | `Stage3<A/B>_<variant>_<seqs>` | two sequence sets nominated from the step 1 and 2 ranks; early, late-concat and sequence-attention fusion |
| `4` | `Stage4_<Arch>Fusion_<seqs>` | best fusion run per sequence set, extended to 5 slices, with and without clinical features; joint slice x sequence attention on the best set |
| `diagnostics` | | paired tests on the CV pool and sequence-attention weights by prediction correctness |

Each experiment runs 5-fold CV and writes:

- fold checkpoints `work/<experiment>_<task>_fold<k>.pt` (state dicts);
- one row per fold in `results/Stage<n>_..._Results.csv`, with the selection settings used;
- pooled out-of-fold predictions per patient in `results/oof/`;
- the pooled OOF AUC and confusion metrics with bootstrap CIs in `results/Bootstrap_Summary.csv`;
- per-epoch training and validation curves in `results/epoch_history/`.

Re-running an experiment replaces its earlier rows. Rankings between experiments use the
pooled OOF AUC. Later steps read earlier results from disk, so steps can be run one at a time.

Warm-starting is fold-matched: fold k of a step is initialized from fold k of the earlier
step, which never saw fold k's validation patients. The step-4 sequence-attention head starts
from the step-3 head, and the slice-attention head from the best step-2 model of the set.

Optional steps, run after step 4: `challengers` (every other step-4 architecture on the same
sequence sets), `extended` and `promote` (further sequence combinations at step 3, and step 4
for a better one), `repeat-cv` (the step-4 models on four more fold splits; the warm-start
chain is retrained for each split so no checkpoint has seen its validation fold).

`--fast-run` uses a subsample of 8 patients per diagnosis, 2 folds and 2 epochs, and writes to
`<work_dir>_fast_run` and `<results_dir>_fast_run` instead of the real folders. Use
`evaluate.py --fast-run` to evaluate such a run.

### Choices worth knowing

- `epoch_selection: outer_val` keeps the epoch with the best AUC on the validation fold, which
  is also the reported fold. This uses every training patient but makes the CV AUC optimistic.
  `swa` (average of the last 3 epochs) and `last` do not touch the validation fold and are the
  natural sensitivity check; `inner` selects on an inner split.
- `threshold_source: pooled_oof` fits one Youden threshold on all out-of-fold probabilities.
- Equal epoch budgets for all steps, so no configuration gets more chances at a good epoch.
- `use_cbam: false`: CBAM attention after the encoder was ablated and increased overfitting.
- Mixed precision is off: fp16 with the focal loss produced NaN AUCs. TF32 (`allow_tf32`) is on
  for training; it only has an effect on Ampere or newer GPUs.

## Evaluation: `python evaluate.py --config config.yaml`

Reads the step 2 and 4 results, picks four models per task (primary, second sequence set,
clinical variant, best single sequence) and evaluates each once on the test set as the
ensemble of its fold checkpoints. Calibration uses only the CV pool: each fold model predicts
its own validation fold, and a temperature (and, for binary, a Youden threshold) is fitted on
those pooled predictions.

| `evaluation:` option | Default | Alternative |
|---|---|---|
| `binary_decision` | `vote`: each fold model applies the threshold, majority decides | `mean_prob`: threshold on the averaged probability |
| `adc_scaler` | `per_fold`: each fold model gets the scaler it was trained with | `pooled`: one scaler fitted on the whole pool |
| `point_estimate` | `full_sample`: metric on the full test set, CI from the bootstrap | `bootstrap_mean`: mean of the bootstrap distribution |

The threshold is fitted on single-model probabilities, while averaging five models pulls
probabilities towards the middle; voting applies the threshold to the outputs it was fitted
on. AUC is threshold-free and identical under both.

Outputs in `results/evaluation/`: `Table_Primary_<task>`, `Table_AllConfigurations_<task>`
(CSV and LaTeX), `Metrics_Full_<task>.csv`, per-class tables for the multiclass task,
`HeadToHead_TestSet.csv` (paired DeLong and bootstrap tests), `Final_Report.txt`, and in
`figures/` the selection funnel, ROC, confusion matrix, reliability diagram, forest plot and a
Grad-CAM / attention case gallery (`--no-gallery` to skip). The gallery shows patient images and
identifiers: do not publish it as is.
