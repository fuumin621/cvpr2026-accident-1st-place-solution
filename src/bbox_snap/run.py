#!/usr/bin/env python3
"""bbox snap post-processing — step 4 in the submission pipeline.

Runs torchvision RetinaNet (BSD-3) detection on uncached videos, then snaps
VLM-predicted accident coordinates onto the nearest vehicle bbox.

Default paths match the Docker layout (/kaggle mount):
    input : /kaggle/working/submission_ensemble.csv
    output: /kaggle/working/submission_final.csv
    cache : /kaggle/results/retinanet_detections/

Usage (inside Docker):
    python3 /kaggle/src/bbox_snap/run.py

    # explicit paths / options
    python3 /kaggle/src/bbox_snap/run.py \\
        --input  /kaggle/working/submission_ensemble.csv \\
        --output /kaggle/working/submission_final.csv \\
        --max-snap-dist 0.2
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# Make src/ importable regardless of working directory
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from bbox_snap.config import DETECTION_CACHE_DIR, SCORE_THRESH  # noqa: E402
from bbox_snap.postprocess.snap import apply_snap  # noqa: E402
from bbox_snap.preprocess.retinanet_detector import (  # noqa: E402
    get_detection_cache_path,
    run_retinanet_detection,
    save_detection_cache,
)

DATASET_ROOT = Path("/kaggle/input/accident")
DEFAULT_INPUT = Path("/kaggle/working/submission_ensemble.csv")
DEFAULT_OUTPUT = Path("/kaggle/working/submission_final.csv")
DEFAULT_DIFF = Path("/kaggle/results/bbox_snap_diff.csv")
SUB_COLS = ["path", "accident_time", "center_x", "center_y", "type"]


def _run_detection_if_needed(
    video_rels: list[str],
    cache_dir: Path,
    score_thresh: float,
    force: bool = False,
) -> None:
    missing = [
        v for v in video_rels
        if force or not get_detection_cache_path(v, cache_dir).exists()
    ]
    if not missing:
        print(f"[bbox_snap] detection cache: all {len(video_rels)} videos cached, skipping")
        return

    print(f"[bbox_snap] running RetinaNet detection on "
          f"{len(missing)}/{len(video_rels)} videos (score_thresh={score_thresh})")
    for i, video_rel in enumerate(missing, start=1):
        video_path = DATASET_ROOT / video_rel
        if not video_path.exists():
            print(f"  [{i}/{len(missing)}] SKIP (not found): {video_rel}")
            continue
        try:
            data = run_retinanet_detection(video_path, score_thresh=score_thresh)
            cache_path = get_detection_cache_path(video_rel, cache_dir)
            save_detection_cache(data, cache_path)
            print(f"  [{i}/{len(missing)}] ok ({data['n_frames']} frames): {video_rel}")
        except Exception as e:
            print(f"  [{i}/{len(missing)}] ERROR {video_rel}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="bbox snap post-processing (step 4)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diff", type=Path, default=DEFAULT_DIFF)
    parser.add_argument("--cache-dir", type=Path, default=DETECTION_CACHE_DIR)
    parser.add_argument("--score-thresh", type=float, default=SCORE_THRESH,
                        help="RetinaNet score threshold")
    parser.add_argument("--window", type=int, default=10, help="Frame search window (±N frames)")
    parser.add_argument("--max-snap-dist", type=float, default=0.2,
                        help="Max snap distance in normalised coords (default: 0.2, matches 1st-place submission)")
    parser.add_argument("--force-detect", action="store_true",
                        help="Re-run detection even if cache exists")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[error] input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    cache_dir: Path = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    rows = df.to_dict("records")
    video_rels: list[str] = [r["path"] for r in rows]

    # Step A: ensure detection cache exists for all videos
    _run_detection_if_needed(
        video_rels, cache_dir,
        score_thresh=args.score_thresh,
        force=args.force_detect,
    )

    # Step B: apply snap
    snapped = apply_snap(rows, cache_dir=cache_dir, window=args.window,
                         max_snap_dist=args.max_snap_dist)

    # Write submission
    out_rows = [{c: r[c] for c in SUB_COLS} for r in snapped]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows, columns=SUB_COLS).to_csv(args.output, index=False)
    applied = sum(1 for r in snapped if r.get("_snap_applied"))
    print(f"[bbox_snap] saved {args.output}  snapped={applied}/{len(rows)}")

    # Write diff CSV
    diff_rows = []
    for r in snapped:
        bef_cx, bef_cy = float(r["_bef_cx"]), float(r["_bef_cy"])
        aft_cx, aft_cy = float(r["center_x"]), float(r["center_y"])
        dist = math.sqrt((aft_cx - bef_cx) ** 2 + (aft_cy - bef_cy) ** 2)
        diff_rows.append({
            "path": r["path"],
            "type": r.get("type", ""),
            "accident_time": r["accident_time"],
            "bef_cx": bef_cx, "bef_cy": bef_cy,
            "aft_cx": aft_cx, "aft_cy": aft_cy,
            "delta_cx": aft_cx - bef_cx,
            "delta_cy": aft_cy - bef_cy,
            "dist_2d": dist,
            "snap_applied": r.get("_snap_applied", False),
        })
    diff_df = (pd.DataFrame(diff_rows)
               .sort_values("dist_2d", ascending=False)
               .reset_index(drop=True))
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    diff_df.to_csv(args.diff, index=False)
    print(f"[bbox_snap] diff saved {args.diff}")


if __name__ == "__main__":
    main()
