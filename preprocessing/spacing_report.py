"""Summarize voxel spacing per sequence across an ingested dataset.

Used to choose the common target grid in preprocess.py. Writes four CSVs to the output
folder: per-sequence summary, every image, distinct spacing combinations, and IQR outliers.
"""

import argparse
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

SEQUENCES = ["T1", "T2", "T1c", "B0", "DWI", "ADC"]


def collect(dataset_dir: Path) -> pd.DataFrame:
    records = []
    for pdir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
        for f in sorted(pdir.glob("*.nii.gz")):
            seq = f.name.split(".")[0].split("_")[0]  # "T1c_run-2.nii.gz" -> "T1c"
            if seq not in SEQUENCES:
                continue
            try:
                zooms = nib.load(str(f)).header.get_zooms()
            except Exception as e:
                print(f"[!] {f}: {e}")
                continue
            if len(zooms) < 3:
                continue
            records.append({"Patient_ID": pdir.name, "Filename": f.name, "Sequence": seq,
                            "X_mm": float(zooms[0]), "Y_mm": float(zooms[1]), "Z_mm": float(zooms[2])})
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame):
    summary, combos = [], []
    df = df.copy()
    df["Outlier"] = False
    for seq, g in df.groupby("Sequence", sort=False):
        row = {"Sequence": seq, "N_images": len(g)}
        for ax in ("X", "Y", "Z"):
            v = g[f"{ax}_mm"]
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df.loc[g.index, "Outlier"] |= (v < lo) | (v > hi)
            row.update({f"Median_{ax}_mm": v.median(), f"Q1_{ax}_mm": q1, f"Q3_{ax}_mm": q3,
                        f"Min_{ax}_mm": v.min(), f"Max_{ax}_mm": v.max(),
                        f"P5_{ax}_mm": v.quantile(0.05), f"P95_{ax}_mm": v.quantile(0.95)})
        row["Potential_Outliers"] = int(df.loc[g.index, "Outlier"].sum())
        summary.append(row)

        counts = Counter(tuple(np.round(r, 4)) for r in g[["X_mm", "Y_mm", "Z_mm"]].to_numpy())
        for (x, y, z), n in counts.most_common():
            combos.append({"Sequence": seq, "X_mm": x, "Y_mm": y, "Z_mm": z, "N_images": n,
                           "Percentage": 100 * n / len(g)})
    return pd.DataFrame(summary), df, pd.DataFrame(combos)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dataset_dir", type=Path, help="folder with one sub-folder per patient (ingest output)")
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()

    df = collect(args.dataset_dir)
    if df.empty:
        raise SystemExit("No NIfTI files found.")
    summary, images, combos = summarize(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "spacing_summary.csv", index=False)
    images.to_csv(args.output_dir / "spacing_all_images.csv", index=False)
    combos.to_csv(args.output_dir / "spacing_combinations.csv", index=False)
    images[images["Outlier"]].to_csv(args.output_dir / "spacing_outliers.csv", index=False)

    print("Median spacing (mm):")
    for _, r in summary.iterrows():
        print(f"  {r['Sequence']:<4} {r['Median_X_mm']:.3f} x {r['Median_Y_mm']:.3f} x {r['Median_Z_mm']:.3f}"
              f"  (n={r['N_images']}, outliers={r['Potential_Outliers']})")


if __name__ == "__main__":
    main()
