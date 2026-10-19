# Preprocessing

From raw DICOM exports to fiducial-centred `.npy` crops. Dependencies: SimpleITK, nibabel,
pydicom, NumPy, pandas, SciPy, matplotlib, and `dcm2niix` for the first step.

## 1. `ingest_dicom.py`: DICOM to NIfTI

```bash
python ingest_dicom.py <dicom_dir> <nifti_dir> --id-map <private>/id_map.csv
```

`<dicom_dir>` holds one `.zip` export or folder per patient. Each DICOM header is read once;
files are grouped by patient, study and series, and every series is converted with dcm2niix.
Series are classified from the series description, image type and b-values:

| Output | Rule (description, case-insensitive) |
|---|---|
| `T2` | contains `t2` but not `dwi` |
| `T1c` | contains `t1` and a contrast marker: `gd`, `gad`, `+c`, `post`, or `ce` as a separate word |
| `T1` | contains `t1` otherwise |
| `ADC` | `adc`, `dadc`, `ddwibs`, or `ADC` in the image type |
| `B0`, `DWI` | diffusion series (b-value file, or `dwi` / `b0` / `trace` / `iso`); b <= 100 -> B0, b >= 700 -> DWI |

Derived maps (eADC, eDWI, exp), phase images, localizers, dynamic series and non-axial
structural series are skipped. A 4D diffusion series is split: the lowest b-value volume
becomes B0, and DWI is the geometric mean of the volumes at the highest b-value (the trace
image). `--dwi-first-direction` keeps the first high-b volume instead.

Patient folders are pseudonymized (`P0001`, and `P0001_2` for a second study of the same
patient). The link to the original PatientID is written only to `--id-map`: keep it private
and out of version control. `--keep-ids` uses the PatientID instead.

When a sequence has several candidates they are kept as `T1c.nii.gz`, `T1c_run-2.nii.gz`, ...
Review them (field of view, artefacts) and record the choice in a selection CSV for step 2:

```
patient,sequence,file
P0003,T1c,T1c_run-2.nii.gz
```

`series_review.csv` lists every converted series, the decision taken and the image geometry;
`ingest_summary.csv` lists, per patient, the candidates per sequence, missing sequences and
repeat studies. Repeat studies of one patient must stay on the same side of any data split;
the modeling code groups `P0001` and `P0001_2` automatically.

## 2. `preprocess.py`: registration and normalization

```bash
python preprocess.py <nifti_dir> <processed_dir> [--selection selection.csv] [--seed 42] [--threads N]
```

Per patient, with T1c as the fixed anchor:

1. reorient all six volumes to RAS;
2. register T2 and T1 to T1c (rigid, Mattes mutual information, 3 resolution levels) and B0 to
   T2 (affine, 2 levels); B0, DWI and ADC are mapped to T1c through the composite transform;
   a registration whose final metric stays above -0.05 is reported as failed;
3. resample everything onto a 1 x 1 x 5 mm grid with the T1c origin and orientation;
4. structural mask: Li threshold on smoothed T1c, largest component, 20 mm in-plane closing,
   hole filling. Diffusion mask: Otsu on log-compressed, clipped B0 with the same clean-up,
   intersected with the structural mask;
5. N4 bias correction of T1c, T1 and T2 (fitted at 1/3 resolution);
6. z-score normalization inside the mask (statistics from the 0.5-99.5 percentile range),
   background set to 0; ADC is masked but keeps its units;
7. crop to the mask bounding box and pad or centre-crop in-plane to 320 x 320.

Outputs: `<id>/<id>_{T1c,T1,T2,StructMask,B0,DWI,ADC,DiffMask}.nii.gz`, two QA figures per
patient in `QA_Plots/` (mid-slice dashboard with overlays, and a multi-slice strip), and
`Unified_Pipeline_Audit.csv` with slice counts and registration metrics. Failures do not stop
the batch; the traceback goes to `preprocess_errors.log`.

Registration samples 15-20% of the voxels at random. `--seed` fixes that sample (default 42).
With several threads ITK sums the metric in a varying order, so repeated runs agree to rounding
error but can occasionally converge to a slightly different transform; `--threads 1` gives
bit-identical runs.

## 3. Fiducial

Place one point at the tumour centre in 3D Slicer on the T1c volume (raw or preprocessed: both
share the same physical coordinates) and save it as `F.mrk.json` (Slicer markups JSON) in the
patient folder. LPS and RAS coordinate systems are both handled.

## 4. `crop.py`: fiducial-centred crops

```bash
python crop.py <processed_dir> <crops_dir> [--fiducial-dir <nifti_dir>] [--box-sizes 48 64 80 90]
```

For each sequence and mask, a square box of fixed physical size (default 48, 64, 80 and 90 mm)
centred on the fiducial is resized to 224 x 224, over 11 slices around the fiducial
(`<box>mm/<id>_<seq>.npy`, shape (11, 224, 224)). `Whole_3Slice/` holds the whole field of view
over 3 slices. Arrays are in radiological orientation (patient right on image left). The
diffusion Z-window is kept within slices where the EPI covers at least 95% of the masked tissue.
The fiducial is mapped into the structural and the diffusion grid separately, and
`crop_audit.csv` reports the offset between the two Z-windows along the slice normal, which
should be 0 for the slice-wise fusion models. QC sheets with the 64 mm box drawn on T1c, DWI and
B0 go to `Expert_QC_Sheets/`.

## `spacing_report.py`

Voxel-spacing statistics per sequence over an ingested dataset (median, IQR, outliers, distinct
spacing combinations), used to choose the common grid.

## Limitations

- Series classification relies on description keywords from our scanners; check
  `series_review.csv` on data from other sites.
- Masks are intensity-based. Check the QA figures, in particular when the tumour is at the edge
  of the field of view or the EPI is strongly distorted.
- ADC is used as exported by the scanner. Exports from different scanners or software versions
  can use different scalings, which a single intensity scaler at training time cannot correct.
