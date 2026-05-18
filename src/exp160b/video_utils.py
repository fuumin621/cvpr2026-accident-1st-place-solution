#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoInfo:
    fps: float
    n_frames: int
    duration: float
    width: int
    height: int


def read_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = (n_frames / fps) if fps > 0 else 0.0
    return VideoInfo(fps=fps, n_frames=n_frames, duration=duration, width=width, height=height)


def resolve_video_path(dataset_root: Path, video_rel: str) -> Path:
    cand = [dataset_root / video_rel, dataset_root / "sim_dataset" / video_rel]
    out = next((p for p in cand if p.exists()), None)
    if out is None:
        raise FileNotFoundError(f"video not found: {video_rel}")
    return out
