#!/usr/bin/env python3
"""
aggregate_volumes.py

Search recursively for:
    vols.left.csv
    vols.right.csv

and produce:
    left_volumes.csv
    right_volumes.csv
    combined_volumes.csv
    summary.csv
"""

from pathlib import Path
import pandas as pd
import argparse


def read_volume_csv(csv_path):
    """
    Reads:

    GPi-vol = 629.25214
    GPe-vol = 1202.145

    Returns:
        (gpi, gpe)
    """
    gpi = None
    gpe = None

    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("GPi-vol"):
                gpi = float(line.split("=")[1].strip())

            elif line.startswith("GPe-vol"):
                gpe = float(line.split("=")[1].strip())

    return gpi, gpe


def find_subject_id(csv_path):
    """
    Assumes subject directory is somewhere above
    the csv file.

    Example:
        .../sub-100307/vols.left.csv

    returns:
        sub-100307
    """
    for part in csv_path.parts:
        if part.startswith("sub-"):
            return part

    return csv_path.parent.name


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir",
        help="Directory containing NextBrain outputs"
    )

    parser.add_argument(
        "--outdir",
        default="volume_summary"
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    left_rows = []
    right_rows = []

    #
    # LEFT
    #
    for csv_file in root_dir.rglob("vols.left.csv"):

        subject_id = find_subject_id(csv_file)

        gpi, gpe = read_volume_csv(csv_file)

        left_rows.append({
            "subject_id": subject_id,
            "GPi_left": gpi,
            "GPe_left": gpe
        })

    #
    # RIGHT
    #
    for csv_file in root_dir.rglob("vols.right.csv"):

        subject_id = find_subject_id(csv_file)

        gpi, gpe = read_volume_csv(csv_file)

        right_rows.append({
            "subject_id": subject_id,
            "GPi_right": gpi,
            "GPe_right": gpe
        })

    left_df = pd.DataFrame(left_rows)
    right_df = pd.DataFrame(right_rows)

    left_df.sort_values("subject_id", inplace=True)
    right_df.sort_values("subject_id", inplace=True)

    left_df.to_csv(
        outdir / "left_volumes.csv",
        index=False
    )

    right_df.to_csv(
        outdir / "right_volumes.csv",
        index=False
    )

    #
    # COMBINED
    #
    combined_df = pd.merge(
        left_df,
        right_df,
        on="subject_id",
        how="outer"
    )

    combined_df["GPi_mean"] = (
        combined_df["GPi_left"] +
        combined_df["GPi_right"]
    ) / 2

    combined_df["GPe_mean"] = (
        combined_df["GPe_left"] +
        combined_df["GPe_right"]
    ) / 2

    combined_df.sort_values(
        "subject_id",
        inplace=True
    )

    combined_df.to_csv(
        outdir / "combined_volumes.csv",
        index=False
    )

    #
    # SUMMARY
    #
    summary = pd.DataFrame([
        {
            "metric": "N",
            "value": len(combined_df)
        },
        {
            "metric": "GPi_mean",
            "value": combined_df["GPi_mean"].mean()
        },
        {
            "metric": "GPi_median",
            "value": combined_df["GPi_mean"].median()
        },
        {
            "metric": "GPi_std",
            "value": combined_df["GPi_mean"].std()
        },
        {
            "metric": "GPe_mean",
            "value": combined_df["GPe_mean"].mean()
        },
        {
            "metric": "GPe_median",
            "value": combined_df["GPe_mean"].median()
        },
        {
            "metric": "GPe_std",
            "value": combined_df["GPe_mean"].std()
        }
    ])

    summary.to_csv(
        outdir / "summary.csv",
        index=False
    )

    print(f"Left subjects:  {len(left_df)}")
    print(f"Right subjects: {len(right_df)}")
    print(f"Combined:       {len(combined_df)}")
    print(f"Output:         {outdir}")


if __name__ == "__main__":
    main()