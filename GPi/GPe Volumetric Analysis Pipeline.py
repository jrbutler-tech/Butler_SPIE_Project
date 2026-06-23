"""
GPi/GPe Volumetric Analysis Pipeline
=====================================
Three stages, run end to end via main():

  1. EXTRACT  - pull GPi/GPe volumes out of each subject's NextBrain
                stats output
  2. STORE    - write one row per subject to a master CSV
  3. ANALYZE  - generate a histogram + dot plot per structure, and
                report the 10 lowest, 10 highest, and mean volume

NOTE ON CUSTOMIZATION
----------------------
This assumes NextBrain writes a FreeSurfer-style .stats file per subject
(the same column layout as aseg.stats: Index SegId NVoxels Volume_mm3
StructName ...). If your output looks different (e.g. a single combined
CSV report, a JSON, or different label strings), the only piece that
needs to change is `parse_stats_file()` and `STRUCTURE_LABELS` below.
Send me one sample stats file/output and I'll tailor the parser exactly.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# STAGE 1: EXTRACT
# ---------------------------------------------------------------------------

# Edit these to match the EXACT label strings that appear in your stats file
STRUCTURE_LABELS = {
    "GPi_L": "Left-Pallidum-internal",
    "GPi_R": "Right-Pallidum-internal",
    "GPe_L": "Left-Pallidum-external",
    "GPe_R": "Right-Pallidum-external",
}


def parse_stats_file(stats_path, wanted_labels):
    """Parse a FreeSurfer/NextBrain-style .stats file and return
    {struct_name: volume_mm3} for any structure in wanted_labels found."""
    volumes = {}
    with open(stats_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            # Standard column order: Index SegId NVoxels Volume_mm3 StructName
            struct_name = parts[4]
            volume_mm3 = float(parts[3])
            if struct_name in wanted_labels:
                volumes[struct_name] = volume_mm3
    return volumes


def collect_volumes(subjects_dir, stats_filename="aseg.stats", output_csv="gpi_gpe_volumes.csv"):
    """Walk every subject folder in subjects_dir, pull GPi/GPe volumes from
    each one's stats file, and write the combined table to a CSV."""
    rows = []
    subject_dirs = sorted(glob.glob(os.path.join(subjects_dir, "*")))
    wanted = set(STRUCTURE_LABELS.values())

    for subj_path in subject_dirs:
        if not os.path.isdir(subj_path):
            continue
        subject_id = os.path.basename(subj_path)
        stats_path = os.path.join(subj_path, "stats", stats_filename)

        if not os.path.exists(stats_path):
            print(f"  [skip] no stats file for {subject_id}")
            continue

        found = parse_stats_file(stats_path, wanted)

        row = {"subject_id": subject_id}
        for short_name, full_label in STRUCTURE_LABELS.items():
            row[short_name] = found.get(full_label, np.nan)

        row["GPi_total"] = row["GPi_L"] + row["GPi_R"] if not (pd.isna(row["GPi_L"]) or pd.isna(row["GPi_R"])) else np.nan
        row["GPe_total"] = row["GPe_L"] + row["GPe_R"] if not (pd.isna(row["GPe_L"]) or pd.isna(row["GPe_R"])) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} subjects to {output_csv}")
    return df


# ---------------------------------------------------------------------------
# STAGE 2: VISUALIZE
# ---------------------------------------------------------------------------

def plot_structure(df, column, structure_name, output_dir="."):
    """Save a histogram and a dot plot for one structure's volume distribution."""
    data = df[column].dropna()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(data, bins=30, color="#4C72B0", edgecolor="white")
    ax.set_xlabel(f"{structure_name} Volume (mm\u00b3)")
    ax.set_ylabel("Number of Subjects")
    ax.set_title(f"{structure_name} Volume Distribution")
    fig.tight_layout()
    hist_path = os.path.join(output_dir, f"{structure_name}_histogram.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    jitter = np.random.normal(0, 0.02, size=len(data))
    ax.scatter(data, jitter, alpha=0.6, color="#DD8452", s=20)
    ax.set_yticks([])
    ax.set_xlabel(f"{structure_name} Volume (mm\u00b3)")
    ax.set_title(f"{structure_name} Volume \u2014 Dot Plot")
    fig.tight_layout()
    dot_path = os.path.join(output_dir, f"{structure_name}_dotplot.png")
    fig.savefig(dot_path, dpi=150)
    plt.close(fig)

    print(f"Saved {hist_path} and {dot_path}")


# ---------------------------------------------------------------------------
# STAGE 3: SUMMARY STATS (10 lowest / 10 highest / mean)
# ---------------------------------------------------------------------------

def summarize_structure(df, column, structure_name, n=10):
    sub = df[["subject_id", column]].dropna().sort_values(column)
    lowest = sub.head(n)
    highest = sub.tail(n).iloc[::-1]
    mean_val = sub[column].mean()

    print(f"\n=== {structure_name} Summary ===")
    print(f"Mean: {mean_val:.2f} mm\u00b3")
    print(f"\n{n} Lowest:")
    print(lowest.to_string(index=False))
    print(f"\n{n} Highest:")
    print(highest.to_string(index=False))

    return {"mean": mean_val, "lowest": lowest, "highest": highest}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPi/GPe volumetric pipeline")
    parser.add_argument("subjects_dir", help="Folder containing all subject output folders")
    parser.add_argument("--stats_filename", default="aseg.stats", help="Stats filename inside each subject's stats/ folder")
    parser.add_argument("--output_csv", default="gpi_gpe_volumes.csv", help="Where to save the master CSV")
    parser.add_argument("--output_dir", default=".", help="Where to save plots")
    args = parser.parse_args()

    df = collect_volumes(args.subjects_dir, args.stats_filename, args.output_csv)

    for column, name in [("GPi_total", "GPi"), ("GPe_total", "GPe")]:
        plot_structure(df, column, name, args.output_dir)
        summarize_structure(df, column, name)