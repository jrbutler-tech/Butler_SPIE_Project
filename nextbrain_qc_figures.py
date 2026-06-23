#!/usr/bin/env python3
"""
nextbrain_qc_figures.py

Builds the three QC panels per subject from NextBrain outputs:

  1. <subj>_pallidum_overlay.png   - axial/coronal/sagittal cropped to GPi/GPe;
                                      top row = plain T1, bottom row = T1 + filled label overlay
  2. <subj>_pallidum_contour.png   - same crop/views, label drawn as a contour outline instead
  3. <subj>_whole_brain_labels.png - full-FOV middle axial/coronal/sagittal slices,
                                      ALL NextBrain labels overlaid using the LUT's own colors

--- IMPORTANT: run this once with --list-labels against your real lut.txt before
--- batching across 1000 subjects. The script matches GPi/GPe by searching label
--- NAMES in the LUT for "pallidum"/"gpe"/"gpi" + left/right -- if your LUT spells
--- these differently, the match will silently fail or grab the wrong structure.

Single subject:
    python3 nextbrain_qc_figures.py \
        --t1 /nfs2/harmonization/BIDS/HCP/sub-100307/anat/sub-100307_T1w.nii.gz \
        --seg-left  /path/to/nextbrain_out/sub-100307/seg.left.nii.gz \
        --seg-right /path/to/nextbrain_out/sub-100307/seg.right.nii.gz \
        --lut       /path/to/nextbrain_out/sub-100307/lut.txt \
        --out-dir   /path/to/qc_figures \
        --subject-id sub-100307

List label names found in a LUT (sanity check before batching):
    python3 nextbrain_qc_figures.py --list-labels --lut /path/to/lut.txt

Batch mode (reads the "success" dict written by nextbrain_batch_runner.py):
    python3 nextbrain_qc_figures.py --batch \
        --state-file nextbrain_progress.json \
        --nextbrain-dir /path/to/nextbrain_out \
        --out-dir /path/to/qc_figures
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PAD_VOXELS = 15  # crop margin around the GPi/GPe bounding box


# ---------------------------------------------------------------- LUT handling

def load_lut(lut_path: Path) -> dict:
    """Parse a FreeSurfer-style LUT: id name R G B A (whitespace-separated)."""
    lut = {}
    with open(lut_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            r, g, b = int(parts[2]), int(parts[3]), int(parts[4])
            lut[idx] = (name, (r / 255, g / 255, b / 255))
    return lut


def find_pallidum_labels(lut: dict) -> dict:
    """
    Returns up to 4 entries: Left-GPe, Left-GPi, Right-GPe, Right-GPi -> label id.
    Matches on name text, not on hardcoded numeric IDs, since LUT numbering
    varies by atlas version.
    """
    found = {}
    for idx, (name, _) in lut.items():
        n = name.lower()
        if "pallidu" not in n and "gpe" not in n and "gpi" not in n:
            continue
        is_left = bool(re.search(r"\b(left|lh|l[-_])\b", n))
        is_right = bool(re.search(r"\b(right|rh|r[-_])\b", n))
        is_external = "gpe" in n or "extern" in n
        is_internal = "gpi" in n or "intern" in n
        if not (is_external or is_internal) or not (is_left or is_right):
            continue
        side = "Left" if is_left else "Right"
        struct = "GPe" if is_external else "GPi"
        found[f"{side}-{struct}"] = idx
    return found


def build_label_colormap(lut: dict):
    """ListedColormap covering 0..max_id, background and unknown ids transparent."""
    max_id = max(lut.keys())
    colors = np.zeros((max_id + 1, 4))
    for idx, (_, rgb) in lut.items():
        colors[idx] = (*rgb, 0.55)
    return ListedColormap(colors), max_id


# ---------------------------------------------------------------- volume I/O

def load_canonical(path: Path):
    img = nib.as_closest_canonical(nib.load(str(path)))
    return img.get_fdata(), img


def load_combined_seg(seg_left_path: Path, seg_right_path: Path, reference_img):
    left_img = nib.as_closest_canonical(nib.load(str(seg_left_path)))
    right_img = nib.as_closest_canonical(nib.load(str(seg_right_path)))

    left = left_img.get_fdata()
    right = right_img.get_fdata()

    if left.shape != reference_img.shape:
        log.warning("seg.left grid != T1 grid (%s vs %s); resampling nearest-neighbor", left.shape, reference_img.shape)
        from nibabel.processing import resample_from_to
        left = resample_from_to(left_img, reference_img, order=0).get_fdata()
    if right.shape != reference_img.shape:
        log.warning("seg.right grid != T1 grid (%s vs %s); resampling nearest-neighbor", right.shape, reference_img.shape)
        from nibabel.processing import resample_from_to
        right = resample_from_to(right_img, reference_img, order=0).get_fdata()

    combined = left.copy()
    mask_r = right > 0
    combined[mask_r] = right[mask_r]
    return combined


# ---------------------------------------------------------------- slicing helpers

def get_bbox_and_centroid(mask: np.ndarray, pad: int):
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("No voxels found for the requested labels -- check find_pallidum_labels() matches against your LUT")
    mins = np.maximum(coords.min(0) - pad, 0)
    maxs = np.minimum(coords.max(0) + pad, np.array(mask.shape) - 1)
    centroid = coords.mean(0).astype(int)
    return mins, maxs, centroid


def cropped_views(volume, mins, maxs, centroid):
    """axial (x,y @ z=centroid), coronal (x,z @ y=centroid), sagittal (y,z @ x=centroid)."""
    axial = volume[mins[0]:maxs[0], mins[1]:maxs[1], centroid[2]]
    coronal = volume[mins[0]:maxs[0], centroid[1], mins[2]:maxs[2]]
    sagittal = volume[centroid[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    return [np.rot90(axial), np.rot90(coronal), np.rot90(sagittal)]


def full_fov_middle_views(volume):
    sx, sy, sz = (s // 2 for s in volume.shape)
    axial = volume[:, :, sz]
    coronal = volume[:, sy, :]
    sagittal = volume[sx, :, :]
    return [np.rot90(axial), np.rot90(coronal), np.rot90(sagittal)]


# ---------------------------------------------------------------- figure builders

VIEW_NAMES = ["Axial", "Coronal", "Sagittal"]


def fig_pallidum_overlay(t1_views, label_views, pallidum_ids, out_path, subject_id):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for col, name in enumerate(VIEW_NAMES):
        axes[0, col].imshow(t1_views[col], cmap="gray")
        axes[0, col].set_title(f"{name} (T1)")
        axes[0, col].axis("off")

        axes[1, col].imshow(t1_views[col], cmap="gray")
        mask = np.isin(label_views[col], list(pallidum_ids.values()))
        overlay = np.ma.masked_where(~mask, label_views[col])
        axes[1, col].imshow(overlay, cmap="autumn", alpha=0.5, vmin=min(pallidum_ids.values()), vmax=max(pallidum_ids.values()))
        axes[1, col].set_title(f"{name} (+ GPi/GPe)")
        axes[1, col].axis("off")
    fig.suptitle(f"{subject_id} -- GPi/GPe overlay")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_pallidum_contour(t1_views, label_views, pallidum_ids, out_path, subject_id):
    colors = {"GPe": "cyan", "GPi": "magenta"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for col, name in enumerate(VIEW_NAMES):
        axes[col].imshow(t1_views[col], cmap="gray")
        for struct in ("GPe", "GPi"):
            ids = [v for k, v in pallidum_ids.items() if struct in k]
            if not ids:
                continue
            mask = np.isin(label_views[col], ids).astype(float)
            if mask.max() > 0:
                axes[col].contour(mask, levels=[0.5], colors=colors[struct], linewidths=1.2)
        axes[col].set_title(name)
        axes[col].axis("off")
    fig.suptitle(f"{subject_id} -- GPi (magenta) / GPe (cyan) contours")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_whole_brain_labels(t1_views, label_views, cmap, max_id, out_path, subject_id):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for col, name in enumerate(VIEW_NAMES):
        axes[col].imshow(t1_views[col], cmap="gray")
        overlay = np.ma.masked_equal(label_views[col], 0)
        axes[col].imshow(overlay, cmap=cmap, vmin=0, vmax=max_id)
        axes[col].set_title(f"{name} (middle slice, all labels)")
        axes[col].axis("off")
    fig.suptitle(f"{subject_id} -- whole-brain segmentation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- per-subject driver

def generate_qc_figures(t1_path, seg_left_path, seg_right_path, lut_path, out_dir, subject_id):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lut = load_lut(Path(lut_path))
    pallidum_ids = find_pallidum_labels(lut)
    if len(pallidum_ids) < 2:
        raise ValueError(f"Only matched {pallidum_ids} in LUT -- run --list-labels and fix find_pallidum_labels() matching")

    t1_data, t1_img = load_canonical(Path(t1_path))
    label_data = load_combined_seg(Path(seg_left_path), Path(seg_right_path), t1_img)

    pallidum_mask = np.isin(label_data, list(pallidum_ids.values()))
    mins, maxs, centroid = get_bbox_and_centroid(pallidum_mask, PAD_VOXELS)

    t1_crop = cropped_views(t1_data, mins, maxs, centroid)
    label_crop = cropped_views(label_data, mins, maxs, centroid)
    fig_pallidum_overlay(t1_crop, label_crop, pallidum_ids, out_dir / f"{subject_id}_pallidum_overlay.png", subject_id)
    fig_pallidum_contour(t1_crop, label_crop, pallidum_ids, out_dir / f"{subject_id}_pallidum_contour.png", subject_id)

    t1_full = full_fov_middle_views(t1_data)
    label_full = full_fov_middle_views(label_data)
    cmap, max_id = build_label_colormap(lut)
    fig_whole_brain_labels(t1_full, label_full, cmap, max_id, out_dir / f"{subject_id}_whole_brain_labels.png", subject_id)

    log.info("Wrote 3 QC figures for %s to %s", subject_id, out_dir)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t1", type=Path)
    ap.add_argument("--seg-left", type=Path)
    ap.add_argument("--seg-right", type=Path)
    ap.add_argument("--lut", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--subject-id", type=str)
    ap.add_argument("--list-labels", action="store_true", help="Print LUT entries matching GPi/GPe keywords and exit")
    ap.add_argument("--batch", action="store_true", help="Process all subjects marked 'success' in a batch-runner state file")
    ap.add_argument("--state-file", type=Path)
    ap.add_argument("--nextbrain-dir", type=Path, help="Root dir where nextbrain_batch_runner.py wrote per-subject output folders")
    args = ap.parse_args()

    if args.list_labels:
        lut = load_lut(args.lut)
        matches = find_pallidum_labels(lut)
        print("Matched GPi/GPe labels:")
        for k, v in matches.items():
            print(f"  {k}: id={v} name={lut[v][0]}")
        if len(matches) < 4:
            print("\nWARNING: expected 4 matches (Left/Right x GPe/GPi), found", len(matches))
            print("All LUT entries containing 'pallid'/'gpe'/'gpi' (for manual inspection):")
            for idx, (name, _) in lut.items():
                if "pallid" in name.lower() or "gpe" in name.lower() or "gpi" in name.lower():
                    print(f"  id={idx} name={name}")
        return

    if args.batch:
        state = json.loads(args.state_file.read_text())
        for sid, t1_path in state.get("success", {}).items():
            sub_out = args.nextbrain_dir / sid
            try:
                generate_qc_figures(
                    t1_path=t1_path,
                    seg_left_path=sub_out / "seg.left.nii.gz",
                    seg_right_path=sub_out / "seg.right.nii.gz",
                    lut_path=sub_out / "lut.txt",
                    out_dir=args.out_dir,
                    subject_id=sid,
                )
            except Exception as e:
                log.error("Figure generation failed for %s: %s", sid, e)
        return

    if not all([args.t1, args.seg_left, args.seg_right, args.lut, args.out_dir, args.subject_id]):
        sys.exit("Single-subject mode requires --t1 --seg-left --seg-right --lut --out-dir --subject-id")

    generate_qc_figures(args.t1, args.seg_left, args.seg_right, args.lut, args.out_dir, args.subject_id)


if __name__ == "__main__":
    main()