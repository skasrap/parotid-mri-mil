"""Cut fiducial-centred crops from preprocessed volumes and save them as .npy tensors.

A single 3D Slicer fiducial (F.mrk.json or msk.json) marks the tumour centre. For every
patient and sequence this writes:
  <out>/<box>mm/<id>_<seq>.npy   (z_box slices, xy x xy) square boxes of fixed physical size
  <out>/Whole_3Slice/<id>_<seq>.npy  (z_whole slices) whole field of view, aspect-preserving
Arrays are (Z, Y, X) in radiological display orientation (patient right on image left).
The fiducial is mapped separately into the structural and the diffusion grid, so the two
groups stay physically aligned even though preprocess.py crops them to different boxes.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

STRUCT_SEQS = ("T1", "T2", "T1c", "StructMask")
DIFF_SEQS = ("B0", "DWI", "ADC", "DiffMask")
FIDUCIAL_NAMES = ("msk.json", "F.mrk.json")
PAD_VALUE = 0.0


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------
def parse_single_fiducial(json_path: Path, nifti_affine: np.ndarray) -> tuple:
    """First defined control point of a Slicer markup -> voxel indices of the given affine (RAS)."""
    with open(json_path, "r") as f:
        data = json.load(f)

    markups = data.get("markups", [])[0]
    coord_sys = markups.get("coordinateSystem", "LPS")  # Slicer's default when the field is missing
    points = [pt["position"] for pt in markups.get("controlPoints", []) if pt.get("positionStatus") == "defined"]
    if not points:
        raise ValueError(f"No defined fiducial points in {json_path}")

    physical = np.array(points[0])
    if coord_sys == "LPS":
        physical[0] = -physical[0]
        physical[1] = -physical[1]

    voxel = np.linalg.inv(nifti_affine).dot(np.append(physical, 1.0))
    return tuple(np.round(voxel[:3]).astype(int))


def get_true_voxel_bounds(center_idx: int, physical_size_mm: float, spacing: float) -> tuple:
    """Box bounds in voxels; may fall outside the volume (handled by padding)."""
    radius_voxels = int(round((physical_size_mm / 2.0) / spacing))
    return center_idx - radius_voxels, center_idx + radius_voxels


def get_valid_diffusion_z_bounds(b0: np.ndarray, diff_mask: np.ndarray, tolerance: float = 0.05,
                                 pad_val: float = 0.0) -> tuple:
    """First and last slice whose masked tissue is at most `tolerance` padding (EPI coverage
    is often shorter in Z than the structural scans)."""
    valid = []
    for z in range(b0.shape[2]):
        slice_mask = diff_mask[:, :, z]
        expected = np.sum(slice_mask > 0)
        if expected == 0:
            continue
        missing = np.sum((slice_mask > 0) & (np.abs(b0[:, :, z] - pad_val) <= 1e-5))
        if missing / expected <= tolerance:
            valid.append(z)
    if not valid:
        return 0, b0.shape[2] - 1
    return min(valid), max(valid)


def calculate_padded_z_window(center_z: int, size: int, max_z: int, valid_min: int = 0,
                              valid_max: int | None = None) -> tuple:
    """Window of `size` slices around center_z, shifted inside [valid_min, valid_max] and clamped."""
    if valid_max is None:
        valid_max = max_z - 1

    start_z = center_z - (size // 2)
    end_z = start_z + size
    if start_z < valid_min:
        shift = valid_min - start_z
        start_z += shift
        end_z += shift
    if end_z > valid_max + 1:
        shift = end_z - (valid_max + 1)
        start_z -= shift
        end_z -= shift

    return max(valid_min, start_z), min(valid_max + 1, end_z)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def safe_extract_and_pad(volume, si, ei, sj, ej, sk, ek, pad_value):
    max_i, max_j, max_k = volume.shape
    pad = ((max(0, -si), max(0, ei - max_i)),
           (max(0, -sj), max(0, ej - max_j)),
           (max(0, -sk), max(0, ek - max_k)))

    def clip(a, n):
        return max(0, min(a, n))

    extracted = volume[clip(si, max_i):clip(ei, max_i), clip(sj, max_j):clip(ej, max_j),
                       clip(sk, max_k):clip(ek, max_k)]
    return np.pad(extracted, pad, mode="constant", constant_values=pad_value)


def crop_and_resize_volume(volume, si, ei, sj, ej, sk, ek, target_size, pad_value, is_mask=False):
    """Extract (with padding), resize in-plane to target_size, return (Z, Y, X) in radiological view."""
    padded = safe_extract_and_pad(volume, si, ei, sj, ej, sk, ek, pad_value)
    crop_i, crop_j, _ = padded.shape
    resized = zoom(padded, (target_size / crop_i, target_size / crop_j, 1.0),
                   order=0 if is_mask else 1, mode="nearest")
    upright = np.rot90(resized, k=1, axes=(0, 1))
    radiological = np.flip(upright, axis=1)
    return np.transpose(radiological, (2, 0, 1))


def load_canonical_nifti(path: Path) -> nib.Nifti1Image:
    """Reorient to the closest canonical (RAS+) axes so every volume indexes as [X, Y, Z]."""
    return nib.as_closest_canonical(nib.load(str(path)))


def slice_position(img: nib.Nifti1Image, k: int) -> float:
    """Position (mm) of slice k along the slice normal; independent of the in-plane crop."""
    normal = img.affine[:3, 2] / np.linalg.norm(img.affine[:3, 2])
    return float(normal.dot(img.affine.dot([0, 0, k, 1])[:3]))


# ---------------------------------------------------------------------------
# QC figure
# ---------------------------------------------------------------------------
def draw_radiological_rect(ax, si, ei, sj, ej, w_x, w_y, color):
    rect = patches.Rectangle((w_x - ei, w_y - ej), ei - si, ej - sj, linewidth=2,
                             edgecolor=color, facecolor="none")
    ax.add_patch(rect)


def save_qc_sheet(pid, t1c_img, dwi_arr, b0_arr, fid_s, fid_d, sp_s, sp_d, z_s, z_d, xy, z_box, out_path,
                  qc_size=64.0):
    t1c_arr = t1c_img.get_fdata()
    si_t, ei_t = get_true_voxel_bounds(fid_s[0], qc_size, sp_s[0])
    sj_t, ej_t = get_true_voxel_bounds(fid_s[1], qc_size, sp_s[1])
    si_d, ei_d = get_true_voxel_bounds(fid_d[0], qc_size, sp_d[0])
    sj_d, ej_d = get_true_voxel_bounds(fid_d[1], qc_size, sp_d[1])

    fig, axes = plt.subplots(3, 5, figsize=(22, 16))
    fig.suptitle(f"Radiological QC | Patient: {pid} | {xy}x{xy} ({z_box} slices) | Box: {qc_size}mm",
                 fontsize=16, fontweight="bold", color="white")
    fig.patch.set_facecolor("black")

    shows_t = np.linspace(z_s[0], z_s[1] - 1, 5).astype(int)
    shows_d = np.linspace(z_d[0], z_d[1] - 1, 5).astype(int)

    def radio(a):
        return np.flip(np.rot90(a, k=1), axis=1)

    for col in range(5):
        z_t = max(0, min(shows_t[col], t1c_arr.shape[2] - 1))
        z_dd = max(0, min(shows_d[col], b0_arr.shape[2] - 1))
        rows = [
            (radio(t1c_arr[:, :, z_t]), (si_t, ei_t, sj_t, ej_t, t1c_arr.shape[0], t1c_arr.shape[1]), "red",
             f"T1c Z={z_t}", {}),
            (radio(dwi_arr[:, :, z_dd]), (si_d, ei_d, sj_d, ej_d, dwi_arr.shape[0], dwi_arr.shape[1]), "lime",
             f"DWI-b1000 Z={z_dd}", {"vmin": -1.0, "vmax": 3.5}),
            (radio(b0_arr[:, :, z_dd]), (si_d, ei_d, sj_d, ej_d, dwi_arr.shape[0], dwi_arr.shape[1]), "cyan",
             f"DWI-b0 Z={z_dd}", {}),
        ]
        for r, (img, rect, color, title, kw) in enumerate(rows):
            ax = axes[r, col]
            ax.axis("off")
            ax.imshow(np.ma.masked_where(np.abs(img) <= 1e-5, img), cmap="gray", origin="upper", **kw)
            draw_radiological_rect(ax, *rect, color)
            ax.set_title(title, color="white")

    plt.tight_layout()
    plt.savefig(str(out_path), facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
def find_fiducial(pid: str, patient_dir: Path, fiducial_dir: Path | None) -> Path | None:
    base = (fiducial_dir / pid) if fiducial_dir else patient_dir
    for name in FIDUCIAL_NAMES:
        if (base / name).exists():
            return base / name
    return None


def crop_patient(pid, patient_dir, fiducial_path, out_dir, box_sizes_mm, xy, z_box, z_whole, qc_dir):
    files = {seq: patient_dir / f"{pid}_{seq}.nii.gz" for seq in STRUCT_SEQS + DIFF_SEQS}
    missing = [s for s, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing sequences: {missing}")

    diff_mask_img = load_canonical_nifti(files["DiffMask"])
    b0_img = load_canonical_nifti(files["B0"])
    valid_z_min, valid_z_max = get_valid_diffusion_z_bounds(
        b0_img.get_fdata(), diff_mask_img.get_fdata(), tolerance=0.05, pad_val=PAD_VALUE)

    t1c_img = load_canonical_nifti(files["T1c"])
    fid_s = parse_single_fiducial(fiducial_path, t1c_img.affine)
    fid_d = parse_single_fiducial(fiducial_path, b0_img.affine)
    sp_s = t1c_img.header.get_zooms()[:2]
    sp_d = b0_img.header.get_zooms()[:2]

    # Structural windows are not limited to valid diffusion slices; diffusion windows are.
    z_s_box = calculate_padded_z_window(fid_s[2], z_box, t1c_img.shape[2])
    z_d_box = calculate_padded_z_window(fid_d[2], z_box, b0_img.shape[2], valid_z_min, valid_z_max)
    z_s_whole = calculate_padded_z_window(fid_s[2], z_whole, t1c_img.shape[2])
    z_d_whole = calculate_padded_z_window(fid_d[2], z_whole, b0_img.shape[2], valid_z_min, valid_z_max)

    for seq, path in files.items():
        vol = load_canonical_nifti(path).get_fdata()
        is_diff = seq in DIFF_SEQS
        is_mask = "Mask" in seq
        z_b = z_d_box if is_diff else z_s_box
        z_w = z_d_whole if is_diff else z_s_whole
        c_i, c_j = (fid_d[:2] if is_diff else fid_s[:2])
        sx, sy = (sp_d if is_diff else sp_s)

        for size_mm in box_sizes_mm:
            si, ei = get_true_voxel_bounds(c_i, size_mm, sx)
            sj, ej = get_true_voxel_bounds(c_j, size_mm, sy)
            arr = crop_and_resize_volume(vol, si, ei, sj, ej, z_b[0], z_b[1], xy, PAD_VALUE, is_mask=is_mask)
            np.save(str(out_dir / f"{size_mm:g}mm" / f"{pid}_{seq}.npy"), arr.astype(np.float32))

        # Whole slice: pad the shorter in-plane axis so the square resize keeps the aspect ratio.
        max_i, max_j, _ = vol.shape
        max_dim = max(max_i, max_j)
        si_w = -((max_dim - max_i) // 2)
        sj_w = -((max_dim - max_j) // 2)
        arr = crop_and_resize_volume(vol, si_w, si_w + max_dim, sj_w, sj_w + max_dim, z_w[0], z_w[1],
                                     xy, PAD_VALUE, is_mask=is_mask)
        np.save(str(out_dir / "Whole_3Slice" / f"{pid}_{seq}.npy"), arr.astype(np.float32))

    if qc_dir is not None:
        save_qc_sheet(pid, t1c_img, load_canonical_nifti(files["DWI"]).get_fdata(), b0_img.get_fdata(),
                      fid_s, fid_d, sp_s, sp_d, z_s_box, z_d_box, xy, z_box, qc_dir / f"{pid}_QC_Proof.jpg")

    # Early/mid fusion pairs slice k of a structural and a diffusion sequence, so the two
    # windows should start at the same physical level. Report the offset.
    dz_box = slice_position(b0_img, z_d_box[0]) - slice_position(t1c_img, z_s_box[0])
    dz_whole = slice_position(b0_img, z_d_whole[0]) - slice_position(t1c_img, z_s_whole[0])
    return {"Patient_ID": pid, "Status": "Success",
            "Struct_Z_Window": f"{z_s_box[0]}-{z_s_box[1]}", "Diff_Z_Window": f"{z_d_box[0]}-{z_d_box[1]}",
            "Z_Offset_Box_mm": round(dz_box, 2) + 0.0, "Z_Offset_Whole_mm": round(dz_whole, 2) + 0.0}


def run_batch(processed_dir: Path, out_dir: Path, fiducial_dir: Path | None, box_sizes_mm, xy, z_box, z_whole,
              qc: bool = True) -> pd.DataFrame:
    for size_mm in box_sizes_mm:
        (out_dir / f"{size_mm:g}mm").mkdir(parents=True, exist_ok=True)
    (out_dir / "Whole_3Slice").mkdir(parents=True, exist_ok=True)
    qc_dir = out_dir / "Expert_QC_Sheets" if qc else None
    if qc_dir:
        qc_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    patient_dirs = sorted(d for d in processed_dir.iterdir() if d.is_dir() and d.name != "QA_Plots")
    for pdir in patient_dirs:
        pid = pdir.name
        fid = find_fiducial(pid, pdir, fiducial_dir)
        if fid is None:
            print(f"[skip] {pid}: no fiducial file")
            rows.append({"Patient_ID": pid, "Status": "No fiducial"})
            continue
        try:
            row = crop_patient(pid, pdir, fid, out_dir, box_sizes_mm, xy, z_box, z_whole, qc_dir)
            flag = ""
            if abs(row["Z_Offset_Box_mm"]) > 0.5 or abs(row["Z_Offset_Whole_mm"]) > 0.5:
                flag = (f"  WARNING: structural/diffusion windows offset by "
                        f"{row['Z_Offset_Box_mm']} mm (box) / {row['Z_Offset_Whole_mm']} mm (whole)")
            print(f"[ok] {pid}{flag}")
            rows.append(row)
        except Exception as e:
            print(f"[fail] {pid}: {e}")
            rows.append({"Patient_ID": pid, "Status": "Error", "Error_Msg": str(e)})

    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "crop_audit.csv", index=False)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("processed_dir", type=Path, help="output folder of preprocess.py")
    p.add_argument("output_dir", type=Path)
    p.add_argument("--fiducial-dir", type=Path,
                   help="folder with <id>/F.mrk.json; default: inside each processed patient folder")
    p.add_argument("--box-sizes", type=float, nargs="+", default=[48, 64, 80, 90], help="box sizes in mm")
    p.add_argument("--xy", type=int, default=224, help="in-plane output size")
    p.add_argument("--z-box", type=int, default=11, help="slices per box crop")
    p.add_argument("--z-whole", type=int, default=3, help="slices per whole-slice crop")
    p.add_argument("--no-qc", action="store_true", help="do not write QC sheets")
    args = p.parse_args()

    run_batch(args.processed_dir, args.output_dir, args.fiducial_dir, args.box_sizes, args.xy,
              args.z_box, args.z_whole, qc=not args.no_qc)


if __name__ == "__main__":
    main()
