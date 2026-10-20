"""Co-register, resample, mask, bias-correct and normalize multi-sequence parotid MRI.

Input: one folder per patient with T1, T1c, T2, B0, DWI and ADC volumes (NIfTI).
T1c is the fixed anchor. T1 and T2 are registered rigidly to T1c; B0 is registered
affinely to T2 and the composite transform is applied to B0, DWI and ADC.

Output, per patient: <out>/<id>/<id>_{T1c,T1,T2,StructMask,B0,DWI,ADC,DiffMask}.nii.gz,
QA figures in <out>/QA_Plots and an audit CSV with registration metrics.

Brain-oriented pipelines assume the anatomy is centred in the field of view and
skull-strip around it. Parotid tumours often sit at the lateral edge of the FOV, so
masks here come from intensity thresholds plus morphology, and crops are padded
rather than centred on an assumed head centre.
"""

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

SEQUENCES = ("T1c", "T1", "T2", "B0", "ADC", "DWI")
OUTPUTS = ("T1c", "T1", "T2", "StructMask", "B0", "DWI", "ADC", "DiffMask")

# Common 1 x 1 x 5 mm grid for both structural and diffusion outputs.
TARGET_STRUCT_SPACING = (1.0, 1.0, 5.000)
TARGET_DIFF_SPACING = (1.0, 1.0, 5.000)
STRUCT_XY = 320
DIFF_XY = 320


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def standardize_to_ras(image: sitk.Image) -> sitk.Image:
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation('RAS')
    return orient_filter.Execute(image)


def create_reference_grid(anchor_image: sitk.Image, target_spacing: tuple) -> sitk.Image:
    """Grid with the anchor's origin and direction, resampled to target_spacing."""
    orig_size = np.array(anchor_image.GetSize())
    orig_spacing = np.array(anchor_image.GetSpacing())
    new_spacing = np.array(target_spacing, dtype=float)
    new_size = np.round(orig_size * (orig_spacing / new_spacing)).astype(int)

    grid = sitk.Image(new_size.tolist(), sitk.sitkFloat32)
    grid.SetOrigin(anchor_image.GetOrigin())
    grid.SetDirection(anchor_image.GetDirection())
    grid.SetSpacing(new_spacing.tolist())
    return grid


def resample_to_grid(moving_image, reference_grid, transform, interpolator) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_grid)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    return resampler.Execute(moving_image)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def _clean_mask(mask: sitk.Image, spacing, closing_radius_mm: float) -> sitk.Image:
    """Largest connected component, in-plane closing, hole filling."""
    fg_cc = sitk.RelabelComponent(sitk.ConnectedComponent(mask), sortByObjectSize=True)
    mask = sitk.Cast(sitk.Equal(fg_cc, 1), sitk.sitkUInt8)

    radius_x = max(1, int(round(closing_radius_mm / spacing[0])))
    radius_y = max(1, int(round(closing_radius_mm / spacing[1])))

    closing = sitk.BinaryMorphologicalClosingImageFilter()
    closing.SetKernelType(sitk.sitkBall)
    closing.SetKernelRadius([radius_x, radius_y, 1])
    closing.SetForegroundValue(1)
    mask = closing.Execute(mask)

    fill_holes = sitk.BinaryFillholeImageFilter()
    fill_holes.SetForegroundValue(1)
    return fill_holes.Execute(mask)


def generate_struct_mask(image: sitk.Image, closing_radius_mm: float = 20.0) -> sitk.Image:
    """Tissue mask for structural images (Li threshold on a smoothed image)."""
    smoothed = sitk.SmoothingRecursiveGaussian(image, sigma=1.0)

    li_filter = sitk.LiThresholdImageFilter()
    li_filter.SetInsideValue(0)
    li_filter.SetOutsideValue(1)
    mask = sitk.Cast(li_filter.Execute(smoothed), sitk.sitkUInt8)
    return _clean_mask(mask, image.GetSpacing(), closing_radius_mm)


def generate_diff_mask(image: sitk.Image, closing_radius_mm: float = 20.0) -> sitk.Image:
    """Tissue mask for EPI images.

    Diffusion intensities are heavy-tailed: a few bright voxels push Otsu above the
    dimmer peripheral tissue. Clipping at the 99.5th percentile and log-compressing
    brings both into range of a single Otsu threshold.
    """
    arr = sitk.GetArrayViewFromImage(image)
    valid = arr[arr > 1e-5]
    if len(valid) > 0:
        p_hi = np.percentile(valid, 99.5)
        clipped_img = sitk.IntensityWindowing(
            image, windowMinimum=0.0, windowMaximum=float(p_hi),
            outputMinimum=0.0, outputMaximum=float(p_hi)
        )
    else:
        clipped_img = image

    clipped_float = sitk.Cast(clipped_img, sitk.sitkFloat32)
    log_img = sitk.Log10(clipped_float + 1.0)
    smoothed = sitk.SmoothingRecursiveGaussian(log_img, sigma=2.0)

    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    mask = sitk.Cast(otsu.Execute(smoothed), sitk.sitkUInt8)
    return _clean_mask(mask, image.GetSpacing(), closing_radius_mm)


# ---------------------------------------------------------------------------
# Intensity
# ---------------------------------------------------------------------------
def apply_fast_n4_bias_correction(image: sitk.Image, mask: sitk.Image, shrink_factor: int = 3) -> sitk.Image:
    """N4 fitted on a downsampled copy; the bias field is evaluated at full resolution."""
    image_float = sitk.Cast(image, sitk.sitkFloat32)
    shrink = [shrink_factor] * image.GetDimension()
    shrunk_image = sitk.Shrink(image_float, shrink)
    shrunk_mask = sitk.Shrink(mask, shrink)

    n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
    n4_filter.Execute(shrunk_image, shrunk_mask)
    log_bias_field = n4_filter.GetLogBiasFieldAsImage(image_float)
    return sitk.Divide(image_float, sitk.Exp(log_bias_field))


def robust_normalize(img: sitk.Image) -> sitk.Image:
    """Window to [1st, 99.5th] percentile of non-zero voxels -> [0, 1]. Used for registration only."""
    arr = sitk.GetArrayViewFromImage(img)
    valid = arr[arr > 1e-5]
    if len(valid) == 0:
        return sitk.Cast(img, sitk.sitkFloat32)
    p1, p99 = np.percentile(valid, [1, 99.5])
    if p99 - p1 < 1e-5:
        return sitk.Cast(img, sitk.sitkFloat32)
    windowed = sitk.IntensityWindowing(img, windowMinimum=float(p1), windowMaximum=float(p99),
                                       outputMinimum=0.0, outputMaximum=1.0)
    return sitk.Cast(windowed, sitk.sitkFloat32)


def z_score_normalize(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """Z-score with mean/std from the mask (0.5-99.5 percentile clipped); background set to 0."""
    img_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)

    tissue_voxels = img_arr[mask_arr > 0]
    p_low, p_high = np.percentile(tissue_voxels, [0.5, 99.5])
    clipped_tissue = np.clip(tissue_voxels, p_low, p_high)

    mean, std = np.mean(clipped_tissue), np.std(clipped_tissue)
    normalized_arr = (img_arr - mean) / (std + 1e-8)
    normalized_arr[mask_arr == 0] = 0.0

    out_img = sitk.GetImageFromArray(normalized_arr)
    out_img.CopyInformation(image)
    return out_img


def apply_unscaled_mask(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """Zero the background but keep physical units (used for ADC)."""
    img_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)
    img_arr[mask_arr == 0] = 0.0
    out_img = sitk.GetImageFromArray(img_arr)
    out_img.CopyInformation(image)
    return out_img


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def compute_registration(fixed: sitk.Image, moving: sitk.Image, is_b0: bool, label: str = "",
                         seed: int | None = 42):
    """Mattes MI registration: affine for B0 -> T2, rigid for structural -> T1c.

    The metric is computed on a random subset of voxels. `seed` fixes that subset so
    runs are reproducible; seed=None uses SimpleITK's wall-clock seed.
    """
    fixed_f = robust_normalize(fixed)
    moving_f = robust_normalize(moving)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if is_b0:
        sampling = 0.2
        reg.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
        reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0])
        initial_tx = sitk.AffineTransform(3)
    else:
        sampling = 0.15
        reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
        initial_tx = sitk.Euler3DTransform()

    if seed is None:
        reg.SetMetricSamplingPercentage(sampling)
    else:
        reg.SetMetricSamplingPercentage(sampling, seed)
    reg.SetOptimizerAsGradientDescentLineSearch(
        learningRate=0.1, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_tx, inPlace=False)

    transform = reg.Execute(fixed_f, moving_f)
    mi_score = reg.GetMetricValue()

    # Negative MI close to zero means the images share almost no information after alignment.
    if mi_score > -0.05:
        raise ValueError(f"[{label}] Registration failed: metric {mi_score:.4f} indicates no alignment.")

    print(f"    [{label}] neg. MI {mi_score:.4f} | {reg.GetOptimizerStopConditionDescription()}")
    return transform, mi_score


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------
def get_tissue_bounding_box(mask: sitk.Image):
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(sitk.Cast(mask, sitk.sitkUInt8))
    if not label_stats.HasLabel(1):
        return None
    return label_stats.GetBoundingBox(1)


def crop_pad_to_bbox(image: sitk.Image, bbox: tuple, target_xy: int, pad_value: float) -> sitk.Image:
    """Crop to the tissue bounding box (all three axes), then pad or centre-crop in-plane to target_xy."""
    start_x, start_y, start_z, size_x, size_y, size_z = bbox
    tight_img = sitk.RegionOfInterest(image, size=[size_x, size_y, size_z], index=[start_x, start_y, start_z])

    diff_x = target_xy - size_x
    diff_y = target_xy - size_y
    pad_x_lower = max(0, diff_x // 2)
    pad_x_upper = max(0, diff_x - pad_x_lower)
    pad_y_lower = max(0, diff_y // 2)
    pad_y_upper = max(0, diff_y - pad_y_lower)
    if pad_x_lower > 0 or pad_x_upper > 0 or pad_y_lower > 0 or pad_y_upper > 0:
        tight_img = sitk.ConstantPad(tight_img, [pad_x_lower, pad_y_lower, 0],
                                     [pad_x_upper, pad_y_upper, 0], pad_value)

    current_x, current_y, _ = tight_img.GetSize()
    crop_diff_x = current_x - target_xy
    crop_diff_y = current_y - target_xy
    if crop_diff_x > 0 or crop_diff_y > 0:
        crop_x_lower = max(0, crop_diff_x // 2)
        crop_y_lower = max(0, crop_diff_y // 2)
        tight_img = sitk.RegionOfInterest(tight_img, size=[target_xy, target_xy, size_z],
                                          index=[crop_x_lower, crop_y_lower, 0])
    return tight_img


# ---------------------------------------------------------------------------
# Per-patient pipeline
# ---------------------------------------------------------------------------
def output_paths(out_base: Path, patient_id: str) -> dict:
    return {name: out_base / patient_id / f"{patient_id}_{name}.nii.gz" for name in OUTPUTS}


def process_patient(patient_id: str, inputs: dict, out_base: Path, seed: int | None = 42,
                    qa: bool = True) -> dict:
    """inputs maps each name in SEQUENCES to a NIfTI path."""
    nat = {seq: standardize_to_ras(sitk.ReadImage(str(inputs[seq]))) for seq in SEQUENCES}

    xfm_t2_to_t1c, mi_t2_t1c = compute_registration(nat["T1c"], nat["T2"], is_b0=False, label="T2 -> T1c", seed=seed)
    xfm_t1_to_t1c, _ = compute_registration(nat["T1c"], nat["T1"], is_b0=False, label="T1 -> T1c", seed=seed)
    xfm_b0_to_t2, mi_b0_t2 = compute_registration(nat["T2"], nat["B0"], is_b0=True, label="B0 -> T2", seed=seed)
    # B0 is T2-weighted, so it is registered to T2 and then carried to T1c through T2.
    xfm_diff = sitk.CompositeTransform([xfm_b0_to_t2, xfm_t2_to_t1c])

    struct_grid = create_reference_grid(nat["T1c"], TARGET_STRUCT_SPACING)
    diff_grid = create_reference_grid(nat["T1c"], TARGET_DIFF_SPACING)

    t1c_res = resample_to_grid(nat["T1c"], struct_grid, sitk.Transform(), sitk.sitkLinear)
    t1_res = resample_to_grid(nat["T1"], struct_grid, xfm_t1_to_t1c, sitk.sitkLinear)
    t2_res = resample_to_grid(nat["T2"], struct_grid, xfm_t2_to_t1c, sitk.sitkLinear)
    b0_res = resample_to_grid(nat["B0"], diff_grid, xfm_diff, sitk.sitkLinear)
    adc_res = resample_to_grid(nat["ADC"], diff_grid, xfm_diff, sitk.sitkLinear)
    dwi_res = resample_to_grid(nat["DWI"], diff_grid, xfm_diff, sitk.sitkLinear)

    struct_mask = generate_struct_mask(t1c_res)
    # Diffusion mask = structural anatomy AND EPI signal, so EPI distortion and
    # FOV edges outside the anatomy are excluded.
    struct_mask_on_diff = resample_to_grid(struct_mask, diff_grid, sitk.Transform(), sitk.sitkNearestNeighbor)
    b0_mask = generate_diff_mask(b0_res)
    diff_mask = sitk.And(sitk.Cast(struct_mask_on_diff, sitk.sitkUInt8), sitk.Cast(b0_mask, sitk.sitkUInt8))

    t1c_n4 = apply_fast_n4_bias_correction(t1c_res, struct_mask)
    t1_n4 = apply_fast_n4_bias_correction(t1_res, struct_mask)
    t2_n4 = apply_fast_n4_bias_correction(t2_res, struct_mask)

    # N4 is applied to the structural sequences only. ADC keeps its physical units
    # and is rescaled at training time.
    normed = {
        "T1c": z_score_normalize(t1c_n4, struct_mask),
        "T1": z_score_normalize(t1_n4, struct_mask),
        "T2": z_score_normalize(t2_n4, struct_mask),
        "B0": z_score_normalize(b0_res, diff_mask),
        "DWI": z_score_normalize(dwi_res, diff_mask),
        "ADC": apply_unscaled_mask(adc_res, diff_mask),
    }

    s_bbox = get_tissue_bounding_box(struct_mask)
    if s_bbox is None:
        raise ValueError("Structural mask is empty.")
    d_bbox = get_tissue_bounding_box(diff_mask)
    if d_bbox is None:
        raise ValueError("Diffusion mask is empty.")

    final = {}
    for name in ("T1c", "T1", "T2"):
        final[name] = crop_pad_to_bbox(normed[name], s_bbox, STRUCT_XY, pad_value=0.0)
    final["StructMask"] = crop_pad_to_bbox(struct_mask, s_bbox, STRUCT_XY, pad_value=0.0)
    for name in ("B0", "DWI", "ADC"):
        final[name] = crop_pad_to_bbox(normed[name], d_bbox, DIFF_XY, pad_value=0.0)
    final["DiffMask"] = crop_pad_to_bbox(diff_mask, d_bbox, DIFF_XY, pad_value=0.0)

    paths = output_paths(out_base, patient_id)
    paths["T1c"].parent.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        sitk.WriteImage(final[name], str(paths[name]))

    if qa:
        import qa as qa_plots  # matplotlib is only needed when QA figures are requested
        qa_dir = out_base / "QA_Plots"
        qa_dir.mkdir(parents=True, exist_ok=True)
        qa_plots.save_compact_qa(final["T1c"], final["T1"], final["T2"], final["B0"], final["DWI"],
                                 final["ADC"], final["StructMask"], final["DiffMask"], patient_id,
                                 qa_dir / f"{patient_id}_QA_Compact.png")
        qa_plots.save_multimodal_qa(final["T1c"], final["T2"], final["B0"], final["ADC"],
                                    final["StructMask"], final["DiffMask"], patient_id,
                                    qa_dir / f"{patient_id}_QA_MultiSlice.png")

    return {
        "Patient_ID": patient_id, "Status": "Success",
        "Struct_Slices": final["T1c"].GetSize()[2], "Diff_Slices": final["ADC"].GetSize()[2],
        "MI_T2_to_T1c": round(mi_t2_t1c, 4), "MI_B0_to_T2": round(mi_b0_t2, 4),
    }


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
def load_selection(path: Path | None) -> dict:
    """Optional CSV (patient, sequence, file) choosing which series to use when
    ingestion produced several candidates (e.g. T1c_run-2.nii.gz). `file` is
    relative to the patient folder."""
    if path is None:
        return {}
    df = pd.read_csv(path, dtype=str)
    missing = {"patient", "sequence", "file"} - set(df.columns)
    if missing:
        raise ValueError(f"Selection file is missing columns: {sorted(missing)}")
    bad = set(df["sequence"]) - set(SEQUENCES)
    if bad:
        raise ValueError(f"Unknown sequence names in selection file: {sorted(bad)}")
    selection = {}
    for row in df.itertuples(index=False):
        selection.setdefault(row.patient.strip(), {})[row.sequence.strip()] = row.file.strip()
    return selection


def resolve_inputs(patient_dir: Path, chosen: dict) -> dict:
    inputs = {}
    for seq in SEQUENCES:
        path = patient_dir / chosen.get(seq, f"{seq}.nii.gz")
        if not path.exists():
            raise FileNotFoundError(f"Missing {seq}: {path.name}")
        inputs[seq] = path
    return inputs


def run_batch(input_dir: Path, output_dir: Path, seed: int | None, selection_csv: Path | None = None,
              skip_existing: bool = False, qa: bool = True) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = load_selection(selection_csv)
    patient_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    error_log = output_dir / "preprocess_errors.log"
    rows = []

    print(f"Preprocessing {len(patient_dirs)} patients (registration seed: {seed})")
    for i, pdir in enumerate(patient_dirs, 1):
        pid = pdir.name
        print(f"[{i}/{len(patient_dirs)}] {pid}")
        if skip_existing and all(p.exists() for p in output_paths(output_dir, pid).values()):
            print("  -> outputs exist, skipped")
            rows.append({"Patient_ID": pid, "Status": "Skipped"})
            continue
        try:
            res = process_patient(pid, resolve_inputs(pdir, selection.get(pid, {})), output_dir, seed=seed, qa=qa)
            rows.append(res)
            print(f"  -> done: struct Z={res['Struct_Slices']}, diff Z={res['Diff_Slices']}")
        except Exception as e:
            print(f"  -> failed: {e}")
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"=== {pid}\n{traceback.format_exc()}\n")
            rows.append({"Patient_ID": pid, "Status": "Error", "Error_Msg": str(e),
                         "Struct_Slices": None, "Diff_Slices": None,
                         "MI_T2_to_T1c": None, "MI_B0_to_T2": None})

    audit = pd.DataFrame(rows)
    audit_path = output_dir / "Unified_Pipeline_Audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"Audit written to {audit_path}")
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input_dir", type=Path, help="folder with one sub-folder per patient")
    p.add_argument("output_dir", type=Path)
    p.add_argument("--seed", type=int, default=42, help="registration sampling seed (default 42)")
    p.add_argument("--no-seed", action="store_true", help="use a wall-clock seed (non-reproducible)")
    p.add_argument("--selection", type=Path, help="CSV (patient, sequence, file) choosing among duplicate series")
    p.add_argument("--skip-existing", action="store_true", help="skip patients whose outputs already exist")
    p.add_argument("--no-qa", action="store_true", help="do not write QA figures")
    p.add_argument("--threads", type=int,
                   help="ITK threads (default: all); 1 makes runs bit-identical but about 5x slower")
    args = p.parse_args()

    if args.threads:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
    run_batch(args.input_dir, args.output_dir, seed=None if args.no_seed else args.seed,
              selection_csv=args.selection, skip_existing=args.skip_existing, qa=not args.no_qa)


if __name__ == "__main__":
    main()
