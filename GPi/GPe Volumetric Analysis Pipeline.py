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
    """
    Publication-quality histogram showing:
        - Histogram
        - Mean
        - Median
        - ±1 SD
        - ±2 SD
        - Statistics textbox
    """

    data = df[column].dropna()

    mean = data.mean()
    median = data.median()
    std = data.std()
    var = data.var()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Histogram
    ax.hist(
        data,
        bins=30,
        edgecolor="black",
        alpha=0.75
    )

    # ±2 SD region
    ax.axvspan(
        mean - 2 * std,
        mean + 2 * std,
        alpha=0.10,
        color="gold",
        label="±2 SD"
    )

    # ±1 SD region
    ax.axvspan(
        mean - std,
        mean + std,
        alpha=0.20,
        color="limegreen",
        label="±1 SD"
    )

    # Mean
    ax.axvline(
        mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean ({mean:.2f})"
    )

    # Median
    ax.axvline(
        median,
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"Median ({median:.2f})"
    )

    # Statistics box
    stats = (
        f"N = {len(data)}\n"
        f"Mean = {mean:.2f}\n"
        f"Median = {median:.2f}\n"
        f"Std = {std:.2f}\n"
        f"Variance = {var:.2f}\n"
        f"Min = {data.min():.2f}\n"
        f"Max = {data.max():.2f}"
    )

    ax.text(
        0.98,
        0.98,
        stats,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            alpha=0.90
        )
    )

    ax.set_title(
        f"{structure_name} Volume Distribution",
        fontsize=15,
        weight="bold"
    )

    ax.set_xlabel("Volume (mm³)")
    ax.set_ylabel("Number of Subjects")

    ax.legend(loc="upper left")

    fig.tight_layout()

    fig.savefig(
        output_dir / f"{structure_name}_histogram.png",
        dpi=300
    )

    plt.close(fig)

def plot_structure_qc(df, column, structure_name, output_dir):
    """
    Ranked QC plot.

    Subjects are sorted from smallest to largest volume.
    Highlights:
        - Mean
        - Median
        - ±1 SD region
        - Lowest 10 subjects
        - Highest 10 subjects
    """

    sub = (
        df[["subject_id", column]]
        .dropna()
        .sort_values(column)
        .reset_index(drop=True)
    )

    values = sub[column]

    mean = values.mean()
    median = values.median()
    std = values.std()

    x = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(12, 6))

    # ---------------------------------------------------------
    # Plot all subjects
    # ---------------------------------------------------------

    ax.plot(
        x,
        values,
        color="gray",
        linewidth=1,
        alpha=0.6
    )

    ax.scatter(
        x,
        values,
        s=18,
        color="gray",
        alpha=0.6,
        label="Subjects"
    )

    # ---------------------------------------------------------
    # Lowest 10
    # ---------------------------------------------------------

    lowest = sub.head(10)

    ax.scatter(
        lowest.index,
        lowest[column],
        color="red",
        s=60,
        zorder=5,
        label="10 Lowest"
    )

    # ---------------------------------------------------------
    # Highest 10
    # ---------------------------------------------------------

    highest = sub.tail(10)

    ax.scatter(
        highest.index,
        highest[column],
        color="green",
        s=60,
        zorder=5,
        label="10 Highest"
    )

    # ---------------------------------------------------------
    # Label lowest 10
    # ---------------------------------------------------------

    for _, row in lowest.iterrows():

        ax.annotate(
            row["subject_id"],
            (row.name, row[column]),
            fontsize=8,
            rotation=45,
            xytext=(0, 6),
            textcoords="offset points"
        )

    # ---------------------------------------------------------
    # Label highest 10
    # ---------------------------------------------------------

    for _, row in highest.iterrows():

        ax.annotate(
            row["subject_id"],
            (row.name, row[column]),
            fontsize=8,
            rotation=45,
            xytext=(0, 6),
            textcoords="offset points"
        )

    # ---------------------------------------------------------
    # Mean / Median
    # ---------------------------------------------------------

    ax.axhline(
        mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean ({mean:.2f})"
    )

    ax.axhline(
        median,
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"Median ({median:.2f})"
    )

    # ---------------------------------------------------------
    # ±1 SD
    # ---------------------------------------------------------

    ax.axhspan(
        mean - std,
        mean + std,
        alpha=0.15,
        color="limegreen",
        label="±1 SD"
    )

    # ---------------------------------------------------------
    # Statistics box
    # ---------------------------------------------------------

    text = (
        f"N = {len(values)}\n"
        f"Mean = {mean:.2f}\n"
        f"Median = {median:.2f}\n"
        f"Std = {std:.2f}\n"
        f"Variance = {values.var():.2f}"
    )

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            alpha=0.9
        )
    )

    ax.set_title(
        f"{structure_name} Ranked QC Plot",
        fontsize=15,
        weight="bold"
    )

    ax.set_xlabel("Subjects (Sorted by Volume)")
    ax.set_ylabel("Volume (mm³)")

    ax.grid(alpha=0.3)

    ax.legend(loc="best")

    fig.tight_layout()

    fig.savefig(
        output_dir / f"{structure_name}_QC.png",
        dpi=300
    )

    plt.close(fig)

def plot_structure_box(df, column, structure_name, output_dir):
    """
    Publication-quality boxplot for QA.

    Displays:
        - Median
        - Quartiles (Q1/Q3)
        - Whiskers
        - Outliers
        - Mean marker
    """

    data = df[column].dropna()

    mean = data.mean()
    median = data.median()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    fig, ax = plt.subplots(figsize=(4.5, 7))

    bp = ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        widths=0.45,
        showmeans=True,
        meanprops=dict(
            marker='D',
            markerfacecolor='red',
            markeredgecolor='black',
            markersize=8
        ),
        medianprops=dict(
            color='blue',
            linewidth=2
        ),
        whiskerprops=dict(
            linewidth=1.5
        ),
        capprops=dict(
            linewidth=1.5
        ),
        flierprops=dict(
            marker='o',
            markersize=5,
            markerfacecolor='orange',
            markeredgecolor='black',
            alpha=0.7
        )
    )

    # Color the box
    bp["boxes"][0].set(
        facecolor="lightsteelblue",
        edgecolor="black",
        linewidth=1.5
    )

    # Statistics text
    stats = (
        f"N = {len(data)}\n"
        f"Mean = {mean:.2f}\n"
        f"Median = {median:.2f}\n"
        f"Q1 = {q1:.2f}\n"
        f"Q3 = {q3:.2f}\n"
        f"IQR = {iqr:.2f}"
    )

    ax.text(
        1.28,
        0.98,
        stats,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            alpha=0.9
        )
    )

    ax.set_ylabel("Volume (mm³)")
    ax.set_xticks([1])
    ax.set_xticklabels([structure_name])

    ax.set_title(
        f"{structure_name} Boxplot",
        fontsize=15,
        weight="bold"
    )

    ax.grid(axis="y", alpha=0.30)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"{structure_name}_boxplot.png",
        dpi=300
    )

    plt.close(fig)


def summarize_structure_png(df, column, structure_name, output_dir):
    """
    Create a PNG summary instead of printing to the terminal.
    """

    sub = (
        df[["subject_id", column]]
        .dropna()
        .sort_values(column)
        .reset_index(drop=True)
    )

    values = sub[column]

    summary = [
        ["N", f"{len(values)}"],
        ["Mean", f"{values.mean():.2f}"],
        ["Median", f"{values.median():.2f}"],
        ["Std", f"{values.std():.2f}"],
        ["Variance", f"{values.var():.2f}"],
        ["Minimum", f"{values.min():.2f}"],
        ["Maximum", f"{values.max():.2f}"],
    ]

    lowest = sub.head(10)
    highest = sub.tail(10).iloc[::-1]

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        f"{structure_name} Summary",
        fontsize=18,
        fontweight="bold"
    )

    # -------------------------------
    # Statistics Table
    # -------------------------------

    ax_stats = plt.axes([0.05, 0.60, 0.35, 0.30])
    ax_stats.axis("off")

    stats_table = ax_stats.table(
        cellText=summary,
        colLabels=["Statistic", "Value"],
        cellLoc="center",
        loc="center"
    )

    stats_table.auto_set_font_size(False)
    stats_table.set_fontsize(11)
    stats_table.scale(1.2, 1.7)

    # -------------------------------
    # Lowest 10
    # -------------------------------

    ax_low = plt.axes([0.05, 0.05, 0.42, 0.45])
    ax_low.axis("off")
    ax_low.set_title("10 Lowest Volumes", fontsize=13)

    low_table = ax_low.table(
        cellText=lowest.values,
        colLabels=["Subject", "Volume"],
        cellLoc="center",
        loc="center"
    )

    low_table.auto_set_font_size(False)
    low_table.set_fontsize(9)
    low_table.scale(1.1, 1.5)

    # -------------------------------
    # Highest 10
    # -------------------------------

    ax_high = plt.axes([0.53, 0.05, 0.42, 0.45])
    ax_high.axis("off")
    ax_high.set_title("10 Highest Volumes", fontsize=13)

    high_table = ax_high.table(
        cellText=highest.values,
        colLabels=["Subject", "Volume"],
        cellLoc="center",
        loc="center"
    )

    high_table.auto_set_font_size(False)
    high_table.set_fontsize(9)
    high_table.scale(1.1, 1.5)

    fig.savefig(
        output_dir / f"{structure_name}_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(
        description="Analyze NextBrain volumetric measurements and generate QA figures."
    )

    parser.add_argument(
        "combined_csv",
        help="combined_volumes.csv generated by aggregate_volumes.py"
    )

    parser.add_argument(
        "--outdir",
        default="analysis",
        help="Directory where analysis figures will be saved."
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------

    df = pd.read_csv(args.combined_csv)

    # ----------------------------------------------------------
    # GPi
    # ----------------------------------------------------------

    print("Generating GPi figures...")

    plot_structure(
        df,
        "GPi_mean",
        "GPi",
        outdir
    )

    plot_structure_qc(
        df,
        "GPi_mean",
        "GPi",
        outdir
    )

    plot_structure_box(
        df,
        "GPi_mean",
        "GPi",
        outdir
    )

    summarize_structure_png(
        df,
        "GPi_mean",
        "GPi",
        outdir
    )

    # ----------------------------------------------------------
    # GPe
    # ----------------------------------------------------------

    print("Generating GPe figures...")

    plot_structure(
        df,
        "GPe_mean",
        "GPe",
        outdir
    )

    plot_structure_qc(
        df,
        "GPe_mean",
        "GPe",
        outdir
    )

    plot_structure_box(
        df,
        "GPe_mean",
        "GPe",
        outdir
    )

    summarize_structure_png(
        df,
        "GPe_mean",
        "GPe",
        outdir
    )

    print("\nDone!")
    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()

