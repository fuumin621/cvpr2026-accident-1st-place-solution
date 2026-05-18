#!/usr/bin/env python3
"""Snap VLM-predicted accident coordinates onto the nearest vehicle bbox."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# torchvision COCO (1-indexed, 91 classes) vehicle IDs: car, motorcycle, bus, truck
VEHICLE_CLASSES = {3, 4, 6, 8}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _dist2d(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def snap_with_detection(
    cx: float,
    cy: float,
    accident_time: float,
    cache: dict[str, Any],
    window: int = 10,
    max_snap_dist: float | None = None,
) -> tuple[float, float, bool]:
    """Snap (cx, cy) to the nearest vehicle bbox in the detection cache.

    Returns (new_cx, new_cy, snap_applied).
    All coordinates are normalised to [0, 1].
    """
    fps: float = float(cache.get("fps", 25.0))
    n_frames: int = int(cache.get("n_frames", 0))
    frames_data: dict[str, list[list[float]]] = cache.get("frames", {})

    center_frame = round(accident_time * fps)

    def _collect(w: int) -> list[list[float]]:
        bboxes: list[list[float]] = []
        for fi in range(max(0, center_frame - w), min(n_frames, center_frame + w + 1)):
            for bbox in frames_data.get(str(fi), []):
                if len(bbox) >= 5 and int(bbox[4]) in VEHICLE_CLASSES:
                    bboxes.append(bbox)
        return bboxes

    bboxes = _collect(window)
    if not bboxes:
        bboxes = _collect(window * 2)

    if not bboxes:
        return cx, cy, False

    best_dist = math.inf
    best_cx, best_cy = cx, cy

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        clamped_x = _clamp(cx, x1, x2)
        clamped_y = _clamp(cy, y1, y2)
        d = _dist2d(cx, cy, clamped_x, clamped_y)
        if d < best_dist:
            best_dist = d
            best_cx, best_cy = clamped_x, clamped_y

    if max_snap_dist is not None and best_dist > max_snap_dist:
        return cx, cy, False

    return best_cx, best_cy, True


def apply_snap(
    rows: list[dict[str, Any]],
    cache_dir: str | Path,
    window: int = 10,
    max_snap_dist: float | None = None,
) -> list[dict[str, Any]]:
    """Apply bbox snap to a list of prediction rows.

    Each row must have: path, accident_time, center_x, center_y.
    Returns augmented rows with _bef_cx/_bef_cy/_snap_applied added.
    """
    from bbox_snap.preprocess.retinanet_detector import get_detection_cache_path, load_detection_cache

    cache_dir = Path(cache_dir)
    results: list[dict[str, Any]] = []

    for row in rows:
        video_path = row["path"]
        accident_time = float(row["accident_time"])
        cx = float(row["center_x"])
        cy = float(row["center_y"])

        cache_path = get_detection_cache_path(video_path, cache_dir)
        cache = load_detection_cache(cache_path)

        if cache is None:
            new_row = dict(row)
            new_row["_snap_applied"] = False
            new_row["_bef_cx"] = cx
            new_row["_bef_cy"] = cy
            results.append(new_row)
            continue

        new_cx, new_cy, applied = snap_with_detection(
            cx, cy, accident_time, cache,
            window=window, max_snap_dist=max_snap_dist,
        )

        new_row = dict(row)
        new_row["center_x"] = new_cx
        new_row["center_y"] = new_cy
        new_row["_snap_applied"] = applied
        new_row["_bef_cx"] = cx
        new_row["_bef_cy"] = cy
        results.append(new_row)

    return results
