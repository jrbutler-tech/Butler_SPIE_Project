#!/usr/bin/env python3
"""
analyze_volumes.py

Analyze the combined volume table produced by aggregate_volumes.py.

Input:
    combined_volumes.csv

Outputs:
    - GPi histogram
    - GPi dot plot
    - GPe histogram
    - GPe dot plot
    - Printed summary statistics
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_structure(df, column, structure_name, output_dir):
    """Create histogram and dot plot for one structure."""

    data = df[column].dropna()

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(data, bins=30, edgecolor="black")
    ax.set_title(f"{structure_name} Volume Distribution")
    ax.set_xlabel("Volume (mm³)")
    ax.set_ylabel("Subjects")
    fig.tight_layout()
    fig.savefig(output_dir / f"{structure_name}_histogram.png", dpi=300)
    plt.close(fig)

    # Dot plot
    fig, ax = plt.subplots(figsize=(7, 3))
    jitter = np.random.normal(0, 0.02, len(data))
    ax.scatter(data, jitter, s=20, alpha=0.6)
    ax.set_yticks([])
    ax.set_xlabel("Volume (mm³)")
    ax.set_title(f"{structure_name} Volume Dot Plot")
    fig.tight_layout()
    fig.savefig(output_dir / f"{structure_name}_dotplot.png", dpi=300)
    plt.close(fig)


def summarize_structure(df, column, structure_name):

    sub = df[["subject_id", column]].dropna().sort_values(column)

    print(f"\n===== {structure_name} =====")
    print(f"N      : {len(sub)}")
    print(f"Mean   : {sub[column].mean():.2f}")
    print(f"Median : {sub[column].median():.2f}")
    print(f"Std    : {sub[column].std():.2f}")
    print(f"Min    : {sub[column].min():.2f}")
    print(f"Max    : {sub[column].max():.2f}")

    print("\n10 Lowest")
    print(sub.head(10).to_string(index=False))

    print("\n10 Highest")
    print(sub.tail(10).iloc[::-1].to_string(index=False))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "combined_csv",
        help="combined_volumes.csv from aggregate_volumes.py"
    )

    parser.add_argument(
        "--outdir",
        default="analysis"
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.combined_csv)

    plot_structure(df, "GPi_mean", "GPi", outdir)
    plot_structure(df, "GPe_mean", "GPe", outdir)

    summarize_structure(df, "GPi_mean", "GPi")
    summarize_structure(df, "GPe_mean", "GPe")


if __name__ == "__main__":
    main()


