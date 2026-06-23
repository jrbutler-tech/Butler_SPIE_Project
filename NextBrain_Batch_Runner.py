#!/usr/bin/env python3
"""
nextbrain_batch_runner.py

Walks a directory of subjects (HCP/BIDS-style: <data_dir>/<subject>/anat/*T1w*.nii.gz),
validates each candidate T1, runs NextBrain on it, and stops once a target number
of SUCCESSFUL segmentations has been reached.

This exists because NextBrain itself only processes one scan at a time -- it has
no built-in directory crawler or "stop after N successes" logic. This script is
the crawler/orchestrator that sits on top of it.

Resumable: progress is checkpointed to a JSON state file after every subject, so
killing the process (Ctrl+C, reboot, walltime limit) and rerunning the same
command picks up where it left off instead of reprocessing or re-counting subjects.

Usage:
    python3 nextbrain_batch_runner.py \
        --data-dir /path/to/HCP \
        --output-dir /path/to/nextbrain_outputs \
        --target 1000 \
        --device cuda \
        --mode invivo

Edit run_nextbrain() below to match your exact NextBrain invocation.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import nibabel as nib
    HAVE_NIB = True
except ImportError:
    HAVE_NIB = False


def setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"success": {}, "failed": {}, "skipped_no_t1": {}}


def save_state(state_path: Path, state: dict) -> None:
    tmp = state_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_path)


def find_t1(subject_dir: Path, glob_pattern: str) -> Optional[Path]:
    """Return the first T1 matching glob_pattern under subject_dir, or None."""
    matches = sorted(subject_dir.glob(glob_pattern))
    return matches[0] if matches else None


def is_valid_t1(t1_path: Path) -> bool:
    """Basic sanity check before spending compute on a scan."""
    if not t1_path.exists() or t1_path.stat().st_size < 1_000_000:
        # smaller than ~1MB is almost certainly not a real 3D T1 volume
        return False
    if HAVE_NIB:
        try:
            img = nib.load(str(t1_path))
            img.header  # forces a header read; raises on corrupt files
            return True
        except Exception:
            return False
    return True  # nibabel not available -- fall back to the size check above


def run_nextbrain(t1_path: Path, out_dir: Path, device: str, mode: str) -> bool:
    """
    Run NextBrain on a single T1, both hemispheres sequentially (so the
    SuperSynth preprocessing step is reused for the second hemisphere).

    *** EDIT THIS to match the exact NextBrain command you already use in
    *** your QC pipeline -- the args below are a placeholder shape, not a
    *** verified invocation for your install.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        cmd = [
            "mri_histo_atlas_segment_fireants",
            str(t1_path),
            str(out_dir),
            device,
            side,
            mode,
        ]
        logging.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error("NextBrain failed on %s (%s): %s", t1_path, side, result.stderr[-2000:])
            return False
    # sanity check expected outputs exist
    if not (out_dir / "seg.left.nii.gz").exists() or not (out_dir / "seg.right.nii.gz").exists():
        logging.error("NextBrain reported success but expected outputs missing for %s", t1_path)
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path, help="Root directory containing subject folders")
    ap.add_argument("--output-dir", required=True, type=Path, help="Root directory for NextBrain outputs")
    ap.add_argument("--target", type=int, default=1000, help="Number of successful segmentations to reach")
    ap.add_argument("--t1-glob", default="anat/*T1w*.nii.gz", help="Glob (relative to each subject dir) to locate the T1")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--mode", default="invivo")
    ap.add_argument("--state-file", type=Path, default=Path("nextbrain_progress.json"))
    ap.add_argument("--log-file", type=Path, default=Path("nextbrain_batch.log"))
    ap.add_argument("--shuffle", action="store_true", help="Randomize subject order (helps avoid site/scanner bias if IDs are sequential per site)")
    ap.add_argument("--dry-run", action="store_true", help="Validate and log what would run, but don't call NextBrain")
    args = ap.parse_args()

    setup_logging(args.log_file)
    state = load_state(args.state_file)

    subject_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir())
    if args.shuffle:
        import random
        random.shuffle(subject_dirs)

    success_count = len(state["success"])
    logging.info("Resuming with %d existing successes (target %d)", success_count, args.target)

    for subject_dir in subject_dirs:
        sid = subject_dir.name

        if sid in state["success"]:
            continue  # already done, don't reprocess and don't recount
        if success_count >= args.target:
            break

        t1_path = find_t1(subject_dir, args.t1_glob)
        if t1_path is None or not is_valid_t1(t1_path):
            logging.warning("Skipping %s: no valid T1 found", sid)
            state["skipped_no_t1"][sid] = True
            save_state(args.state_file, state)
            continue

        if args.dry_run:
            logging.info("[dry-run] would process %s -> %s", sid, t1_path)
            continue

        out_dir = args.output_dir / sid
        ok = run_nextbrain(t1_path, out_dir, args.device, args.mode)

        if ok:
            success_count += 1
            state["success"][sid] = str(t1_path)
            logging.info("SUCCESS %s (%d/%d)", sid, success_count, args.target)
        else:
            state["failed"][sid] = str(t1_path)

        save_state(args.state_file, state)

    logging.info(
        "Done. successes=%d failed=%d skipped_no_t1=%d (target=%d)",
        len(state["success"]), len(state["failed"]), len(state["skipped_no_t1"]), args.target,
    )
    if success_count < args.target:
        logging.warning("Ran out of subjects before reaching target -- only %d/%d successes available in this directory.",
                         success_count, args.target)


if __name__ == "__main__":
    main()