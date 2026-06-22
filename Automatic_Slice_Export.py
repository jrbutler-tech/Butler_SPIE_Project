#!/usr/bin/env python3

import os
import argparse

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# LUT
# --------------------------------------------------

def load_lut(lut_file):
    lut = {}

    with open(lut_file, "r") as f:
        for line in f:

            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            try:
                label_id = int(parts[0])
            except Exception:
                continue

            label_name = parts[1]
            lut[label_id] = label_name

    return lut


def find_label_id(lut, label):

    if isinstance(label, int):
        return label

    try:
        return int(label)
    except ValueError:
        pass

    for idx, name in lut.items():
        if name.lower() == label.lower():
            return idx

    raise ValueError(f"Label '{label}' not found in LUT")


# --------------------------------------------------
# Volume Loading
# --------------------------------------------------

def load_volume(path):
    print(f"Loading: {path}")
    img = nib.load(path)
    return img.get_fdata()


# --------------------------------------------------
# Slice Extraction
# --------------------------------------------------

def get_slice(volume, axis, index):

    if axis == 0:
        return volume[index, :, :]
    elif axis == 1:
        return volume[:, index, :]
    elif axis == 2:
        return volume[:, :, index]

    raise ValueError("Axis must be 0,1,2")


# --------------------------------------------------
# Image Export (FreeView-style multi-layer)
# --------------------------------------------------

def export_image_mode(base, seg, overlay, start, end, axis, outdir):

    os.makedirs(outdir, exist_ok=True)

    for s in range(start, end + 1):

        base_sl = get_slice(base, axis, s)

        plt.figure(figsize=(8, 8))

        # Base anatomical image
        plt.imshow(np.rot90(base_sl), cmap="gray", origin="lower")

        # Segmentation overlay (optional)
        if seg is not None:
            seg_sl = get_slice(seg, axis, s)
            plt.imshow(
                np.rot90(seg_sl),
                cmap="nipy_spectral",
                alpha=0.4,
                origin="lower"
            )

        # Extra overlay (optional NIfTI)
        if overlay is not None:
            ov_sl = get_slice(overlay, axis, s)
            plt.imshow(
                np.rot90(ov_sl),
                cmap="hot",
                alpha=0.3,
                origin="lower"
            )

        plt.axis("off")

        outfile = os.path.join(outdir, f"slice_{s:04d}.png")

        plt.savefig(outfile, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"Saved {outfile}")


# --------------------------------------------------
# Segmentation Mode (label filtering)
# --------------------------------------------------

def export_segmentation_mode(volume, start, end, axis, outdir,
                             label_id=None, label_name=None, label_all=False):

    os.makedirs(outdir, exist_ok=True)

    for s in range(start, end + 1):

        sl = get_slice(volume, axis, s)

        if label_all:
            display = sl.copy()
            title = "ALL_LABELS"
        else:
            display = np.where(sl == label_id, label_id, 0)
            title = f"{label_name}_{label_id}"

        plt.figure(figsize=(8, 8))

        plt.imshow(np.rot90(display), cmap="nipy_spectral", origin="lower")
        plt.axis("off")

        outfile = os.path.join(outdir, f"{title}_slice_{s:04d}.png")

        plt.savefig(outfile, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"Saved {outfile}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True, help="Base MGZ/NII image")
    parser.add_argument("--seg", help="Segmentation MGZ/NII (optional)")
    parser.add_argument("--overlay", help="Optional overlay NIfTI")

    parser.add_argument("--mode", required=True,
                        choices=["image", "segmentation"])

    parser.add_argument("--orientation", required=True,
                        choices=["axial", "coronal", "sagittal"])

    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)

    parser.add_argument("--outdir", default="output_slices")

    parser.add_argument("--lut", help="LUT file")
    parser.add_argument("--label", help="Label name or ID")
    parser.add_argument("--label-all", action="store_true")

    args = parser.parse_args()

    axis_map = {
        "sagittal": 0,
        "coronal": 1,
        "axial": 2,
    }

    axis = axis_map[args.orientation]

    # Load volumes
    base = load_volume(args.image)
    seg = load_volume(args.seg) if args.seg else None
    overlay = load_volume(args.overlay) if args.overlay else None

    # --------------------------------------------------
    # IMAGE MODE (FreeView-style)
    # --------------------------------------------------
    if args.mode == "image":

        export_image_mode(
            base,
            seg,
            overlay,
            args.start,
            args.end,
            axis,
            args.outdir
        )

    # --------------------------------------------------
    # SEGMENTATION MODE
    # --------------------------------------------------
    elif args.mode == "segmentation":

        if args.label_all:
            label_id = None
            label_name = "ALL"
        else:
            if args.lut is None or args.label is None:
                raise ValueError("--lut and --label required unless --label-all")

            lut = load_lut(args.lut)
            label_id = find_label_id(lut, args.label)
            label_name = lut.get(label_id, str(label_id))

            print(f"Showing label {label_name} ({label_id})")

        export_segmentation_mode(
            seg if seg is not None else base,
            args.start,
            args.end,
            axis,
            args.outdir,
            label_id=label_id,
            label_name=label_name,
            label_all=args.label_all
        )


if __name__ == "__main__":
    main()