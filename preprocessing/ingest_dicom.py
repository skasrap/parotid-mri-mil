"""Convert raw DICOM exports (zip files or folders) into one NIfTI folder per patient study.

For every source (a .zip file or a sub-folder of the input directory):
  1. read each DICOM header once and group files by patient, study and series;
  2. convert each series with dcm2niix;
  3. classify it as T1, T1c, T2, ADC, B0 or DWI from its description, image type and b-values;
  4. copy it to <out>/<patient>/<SEQ>.nii.gz (further candidates become <SEQ>_run-2.nii.gz, ...).

Outputs next to the patient folders:
  series_review.csv   every converted series and what was done with it
  ingest_summary.csv  per patient: which sequences were found and how many candidates
Choose among duplicate candidates by eye and record the choice in a selection CSV for
preprocess.py (--selection).

Patient folders get pseudonymous names (P0001, P0001_2 for a second study, ...). The
mapping to the original PatientID is written to --id-map, which must stay out of version
control. --keep-ids uses the DICOM PatientID instead.

Requires dcm2niix on PATH.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from nibabel.funcs import squeeze_image

SEQUENCES = ("T1", "T1c", "T2", "ADC", "B0", "DWI")
B0_MAX = 100     # b <= 100 s/mm2 counts as b0
DWI_MIN = 700    # b >= 700 s/mm2 counts as high-b DWI


# ---------------------------------------------------------------------------
# DICOM reading and grouping
# ---------------------------------------------------------------------------
def clean_name(name) -> str:
    name = str(name).strip() if name is not None else ""
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name or "Unknown"


def read_headers(root: Path) -> list:
    """One header read per file. Non-DICOM files and files without a SeriesInstanceUID
    (e.g. DICOMDIR) are skipped."""
    records = []
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        if not series_uid:
            continue
        records.append({
            "path": f,
            "patient_id": str(getattr(ds, "PatientID", "") or ""),
            "study_uid": str(getattr(ds, "StudyInstanceUID", "") or "unknown-study"),
            "study_date": str(getattr(ds, "StudyDate", "") or ""),
            "series_uid": series_uid,
            "series_desc": clean_name(getattr(ds, "SeriesDescription", "") or "UnknownSeries"),
            "series_number": str(getattr(ds, "SeriesNumber", "") or ""),
        })
    return records


class IdMap:
    """Maps (original patient ID, study UID) to an output folder name and persists it."""

    FIELDS = ["original_patient_id", "study_instance_uid", "study_date", "folder"]

    def __init__(self, path: Path | None, keep_ids: bool):
        self.path, self.keep_ids = path, keep_ids
        self.rows = []
        if path is not None and path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))

    def folder_for(self, patient_id: str, study_uid: str, study_date: str) -> str:
        for r in self.rows:
            if r["original_patient_id"] == patient_id and r["study_instance_uid"] == study_uid:
                return r["folder"]
        studies = [r for r in self.rows if r["original_patient_id"] == patient_id]
        if self.keep_ids:
            base = clean_name(patient_id)
        elif studies:
            base = studies[0]["folder"].split("_")[0]
        else:
            n = len({r["original_patient_id"] for r in self.rows}) + 1
            base = f"P{n:04d}"
        folder = base if not studies else f"{base}_{len(studies) + 1}"
        self.rows.append({"original_patient_id": patient_id, "study_instance_uid": study_uid,
                          "study_date": study_date, "folder": folder})
        self.save()
        return folder

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def run_dcm2niix(series_dir: Path, out_dir: Path, log) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["dcm2niix", "-z", "y", "-ba", "y", "-f", "%s_%p", "-o", str(out_dir), str(series_dir)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        # dcm2niix reports most problems on stdout.
        msg = " ".join((res.stderr + " " + res.stdout).split())[-300:]
        log(f"  [!] dcm2niix exit code {res.returncode} for series {series_dir.name}: {msg}")


def has_token(text: str, token: str) -> bool:
    """True if `token` appears as a separate word (delimited by non-letters)."""
    return re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", text) is not None


def classify(desc: str, image_type: list, has_bval: bool) -> tuple:
    """Returns (sequence or None, reason)."""
    d = desc.lower()
    image_type = [str(t).upper() for t in image_type]

    if any(s in d for s in ("eadc", "edwi", "exp", "sdwi")):
        return None, "derived diffusion map (eADC/eDWI/exp/sDWI)"
    if "P" in image_type or "PHASE" in image_type:
        return None, "phase image"
    if any(has_token(d, t) for t in ("localizer", "loc", "scout", "survey")):
        return None, "localizer"

    if "t2" in d and "dwi" not in d:
        if "et2" in d:
            return None, "derived T2 map"
        return "T2", ""
    if "t1" in d:
        # Contrast markers; "ce" only as its own token so e.g. "t1_space" stays T1.
        if any(s in d for s in ("gd", "+c", "post", "gad")) or has_token(d, "ce"):
            return "T1c", ""
        if "et1" in d:
            return None, "derived T1 map"
        return "T1", ""
    if any(s in d for s in ("ddwibs", "dadc", "adc")) or "ADC" in image_type:
        return "ADC", ""
    if has_bval or any(s in d for s in ("dwi", "b0", "trace", "iso")):
        return "DIFFUSION", ""
    return None, "unrecognised description"


def is_axial(img: nib.Nifti1Image, desc: str) -> bool:
    d = desc.lower()
    if "sag" in d or "cor" in d or "3d" in d:
        return False
    return nib.aff2axcodes(img.affine)[2] in ("S", "I")


def next_free_name(dest_dir: Path, seq: str) -> str:
    if not (dest_dir / f"{seq}.nii.gz").exists():
        return f"{seq}.nii.gz"
    n = 2
    while (dest_dir / f"{seq}_run-{n}.nii.gz").exists():
        n += 1
    return f"{seq}_run-{n}.nii.gz"


def shared_run_name(dest_dir: Path, seqs: tuple) -> dict:
    """Same run suffix for B0 and DWI split from one 4D series, so the pair stays linked."""
    run = 1
    for seq in seqs:
        name = next_free_name(dest_dir, seq)
        if name != f"{seq}.nii.gz":
            run = max(run, int(name.split("_run-")[1].split(".")[0]))
    return {seq: (f"{seq}.nii.gz" if run == 1 else f"{seq}_run-{run}.nii.gz") for seq in seqs}


def save_volume(data: np.ndarray, ref: nib.Nifti1Image, path: Path) -> None:
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data.astype(np.float32), ref.affine, header), str(path))


def high_b_trace(data: np.ndarray, bvals: np.ndarray, first_direction: bool) -> tuple:
    """DWI image from a 4D series. Several volumes at the highest b-value are usually
    diffusion directions; their geometric mean is the isotropic (trace) image that
    scanners export. first_direction=True keeps the first one instead."""
    b_max = bvals.max()
    idx = np.where(bvals >= b_max - 50)[0]
    if first_direction or len(idx) == 1:
        return data[..., idx[0]], f"b={bvals[idx[0]]:g}, volume {idx[0]}"
    vols = np.clip(data[..., idx], 1e-6, None)
    trace = np.exp(np.log(vols).mean(axis=-1))
    return trace, f"geometric mean of {len(idx)} volumes at b~{b_max:g}"


def handle_diffusion(nii_path, json_meta, bval_path, dest_dir, first_direction) -> list:
    """Returns a list of (sequence, filename, note)."""
    img = nib.load(str(nii_path))
    bvals = None
    if bval_path.exists():
        bvals = np.atleast_1d(np.loadtxt(bval_path))
    elif json_meta.get("DiffusionBValue") is not None:
        bvals = np.array([json_meta["DiffusionBValue"]], dtype=float)

    is_4d = img.ndim == 4 and img.shape[3] > 1
    if is_4d:
        if bvals is None:
            return [(None, "", "4D diffusion series without b-values")]
        if len(bvals) != img.shape[3]:
            return [(None, "", f"{len(bvals)} b-values for {img.shape[3]} volumes")]
        data = img.get_fdata()
        b0_idx = int(np.argmin(bvals))
        names = shared_run_name(dest_dir, ("B0", "DWI"))
        save_volume(data[..., b0_idx], img, dest_dir / names["B0"])
        out = [("B0", names["B0"], f"b={bvals[b0_idx]:g}, volume {b0_idx} of 4D")]
        if bvals.max() > bvals.min():
            dwi, note = high_b_trace(data, bvals, first_direction)
            save_volume(dwi, img, dest_dir / names["DWI"])
            out.append(("DWI", names["DWI"], note))
        return out

    if bvals is not None and len(bvals) > 0:
        b = float(bvals[0])
        if b <= B0_MAX:
            seq = "B0"
        elif b >= DWI_MIN:
            seq = "DWI"
        else:
            return [(None, "", f"intermediate b-value {b:g}")]
        name = next_free_name(dest_dir, seq)
        nib.save(squeeze_image(img), str(dest_dir / name))
        return [(seq, name, f"b={b:g}")]

    # No b-value anywhere: fall back to the description and flag it for review.
    seq = "B0" if "b0" in json_meta.get("SeriesDescription", "").lower() else "DWI"
    name = next_free_name(dest_dir, seq)
    nib.save(squeeze_image(img), str(dest_dir / name))
    return [(seq, name, "NO B-VALUE: assigned from description, check manually")]


def convert_study(study_files: dict, dest_dir: Path, work_dir: Path, args, review_rows: list, log) -> None:
    """study_files: series_uid -> list of header records. Converts and classifies every series."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for series_uid, recs in sorted(study_files.items(), key=lambda kv: (kv[1][0]["series_number"].zfill(6), kv[0])):
        desc = recs[0]["series_desc"]
        base_row = {"patient": dest_dir.name, "series_number": recs[0]["series_number"],
                    "series_description": desc, "n_files": len(recs)}
        if "dyn" in desc.lower():
            review_rows.append({**base_row, "assigned": "", "file": "", "note": "dynamic series skipped"})
            continue

        series_in = work_dir / "dcm" / series_uid
        nii_out = work_dir / "nii" / series_uid
        for d in (series_in, nii_out):
            shutil.rmtree(d, ignore_errors=True)
        series_in.mkdir(parents=True)
        # Unique names: files from different export folders often share names (IM0001, ...).
        for i, r in enumerate(recs):
            shutil.copy2(r["path"], series_in / f"{i:05d}.dcm")
        run_dcm2niix(series_in, nii_out, log)

        jsons = sorted(nii_out.glob("*.json"))
        if not jsons:
            review_rows.append({**base_row, "assigned": "", "file": "", "note": "dcm2niix produced no output"})
        for json_path in jsons:
            stem = json_path.name[:-len(".json")]
            nii_path = json_path.parent / f"{stem}.nii.gz"
            bval_path = json_path.parent / f"{stem}.bval"
            row = {**base_row, "converted": nii_path.name}
            if not nii_path.exists():
                review_rows.append({**row, "assigned": "", "file": "", "note": "no NIfTI next to sidecar"})
                continue
            try:
                with open(json_path, encoding="utf-8") as f:
                    meta = json.load(f)
                seq, reason = classify(meta.get("SeriesDescription", desc), meta.get("ImageType", []),
                                       bval_path.exists())
                results = (assign_series(seq, nii_path, meta, bval_path, dest_dir, desc, args) if seq
                           else [(None, "", reason)])
            except Exception as e:
                results = [(None, "", f"error: {type(e).__name__}: {e}")]
            for s, fname, note in results:
                img_info = describe_file(dest_dir / fname) if fname else {}
                review_rows.append({**row, "assigned": s or "", "file": fname, "note": note, **img_info})


def assign_series(seq, nii_path, meta, bval_path, dest_dir, desc, args) -> list:
    if seq == "DIFFUSION":
        return handle_diffusion(nii_path, meta, bval_path, dest_dir, args.dwi_first_direction)

    img = nib.load(str(nii_path))
    if img.ndim == 4 and img.shape[3] == 1:
        img = squeeze_image(img)
    if img.ndim != 3:
        return [(None, "", f"{seq}: {img.ndim}D image with shape {img.shape}, skipped")]
    if seq in ("T1", "T1c", "T2") and not is_axial(img, desc):
        return [(None, "", f"{seq}: not axial")]
    name = next_free_name(dest_dir, seq)
    nib.save(img, str(dest_dir / name))
    return [(seq, name, "")]


def describe_file(path: Path) -> dict:
    img = nib.load(str(path))
    zooms = img.header.get_zooms()[:3]
    return {"shape": "x".join(map(str, img.shape)), "spacing_mm": "x".join(f"{z:.3g}" for z in zooms),
            "orientation": "".join(nib.aff2axcodes(img.affine))}


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
def iter_sources(input_dir: Path):
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".zip":
            yield p
        elif p.is_dir():
            yield p


def ingest_source(source: Path, out_dir: Path, id_map: IdMap, args, review_rows: list, seen: set, log) -> None:
    """seen: folders written earlier in this run. A study found again in another source is
    added to its folder; a folder from a previous run is skipped unless --overwrite."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if source.is_file():
            try:
                with zipfile.ZipFile(source) as z:
                    z.extractall(tmp / "src")
            except (zipfile.BadZipFile, OSError) as e:
                log(f"  [!] could not extract {source.name}: {e}")
                return
            root = tmp / "src"
        else:
            root = source

        records = read_headers(root)
        if not records:
            log(f"  [!] no DICOM series found in {source.name}")
            return

        studies = defaultdict(lambda: defaultdict(list))
        study_info = {}
        for r in records:
            pid = r["patient_id"] or source.stem
            key = (pid, r["study_uid"])
            studies[key][r["series_uid"]].append(r)
            study_info[key] = r["study_date"]

        for (pid, study_uid) in sorted(studies, key=lambda k: (k[0], study_info[k], k[1])):
            folder = id_map.folder_for(pid, study_uid, study_info[(pid, study_uid)])
            dest = out_dir / folder
            if folder in seen:
                log(f"  study -> {folder}: also in an earlier source; adding its series to that folder")
            elif dest.exists() and any(dest.glob("*.nii.gz")):
                if not args.overwrite:
                    log(f"  study -> {folder}: converted in a previous run, skipped (--overwrite to redo)")
                    continue
                shutil.rmtree(dest)
            seen.add(folder)
            log(f"  study -> {folder} ({len(studies[(pid, study_uid)])} series)")
            convert_study(studies[(pid, study_uid)], dest, tmp / "work", args, review_rows, log)


def write_csv(path: Path, rows: list, fields: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarize(review_rows: list, id_map: IdMap) -> list:
    counts = defaultdict(lambda: defaultdict(int))
    for r in review_rows:
        if r.get("assigned"):
            counts[r["patient"]][r["assigned"]] += 1
    patients = sorted({r["patient"] for r in review_rows})
    n_studies = defaultdict(int)
    for r in id_map.rows:
        n_studies[r["original_patient_id"]] += 1
    repeat_folders = {r["folder"] for r in id_map.rows if n_studies[r["original_patient_id"]] > 1}
    summary = []
    for p in patients:
        row = {"patient": p}
        row.update({s: counts[p][s] for s in SEQUENCES})
        row["missing"] = " ".join(s for s in SEQUENCES if counts[p][s] == 0)
        row["needs_choice"] = " ".join(s for s in SEQUENCES if counts[p][s] > 1)
        # Several studies of one patient must end up on the same side of any data split.
        row["repeat_study"] = p in repeat_folders
        summary.append(row)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input_dir", type=Path, help="folder of .zip exports and/or DICOM folders")
    p.add_argument("output_dir", type=Path)
    p.add_argument("--id-map", type=Path, help="CSV linking original PatientIDs to folders (keep private)")
    p.add_argument("--keep-ids", action="store_true", help="name folders by DICOM PatientID")
    p.add_argument("--dwi-first-direction", action="store_true",
                   help="for multi-direction 4D DWI keep the first high-b volume instead of the trace")
    p.add_argument("--overwrite", action="store_true", help="re-convert studies that already have output")
    args = p.parse_args()

    if shutil.which("dcm2niix") is None:
        p.error("dcm2niix not found on PATH")
    if not args.keep_ids and args.id_map is None:
        p.error("--id-map is required unless --keep-ids is given")
    if args.id_map is not None and args.output_dir.resolve() in args.id_map.resolve().parents:
        print("WARNING: the ID map is inside the output folder; do not share or commit that folder.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    id_map = IdMap(args.id_map, args.keep_ids)
    review_rows, seen = [], set()
    sources = list(iter_sources(args.input_dir))
    for i, src in enumerate(sources, 1):
        print(f"[{i}/{len(sources)}] {src.name}")
        ingest_source(src, args.output_dir, id_map, args, review_rows, seen, print)

    fields = ["patient", "series_number", "series_description", "n_files", "converted",
              "assigned", "file", "shape", "spacing_mm", "orientation", "note"]
    write_csv(args.output_dir / "series_review.csv", review_rows, fields)
    summary = summarize(review_rows, id_map)
    write_csv(args.output_dir / "ingest_summary.csv", summary,
              ["patient", *SEQUENCES, "missing", "needs_choice", "repeat_study"])

    n_incomplete = sum(1 for s in summary if s["missing"])
    n_choice = sum(1 for s in summary if s["needs_choice"])
    print(f"\n{len(summary)} patient folders | {n_incomplete} missing a sequence | "
          f"{n_choice} with duplicate candidates (see series_review.csv)")


if __name__ == "__main__":
    main()
