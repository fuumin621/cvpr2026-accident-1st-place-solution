#!/usr/bin/env python3
"""torchvision RetinaNet vehicle detection: cache generation and loading.

Output schema is consumed by the downstream snap logic
(`bbox_snap.postprocess.snap`).

Class IDs in the output use the torchvision COCO convention (1-indexed,
91-class); only vehicles {3,4,6,8} (car, motorcycle, bus, truck) are kept.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# torchvision COCO (1-indexed, 91 classes) vehicle IDs: car, motorcycle, bus, truck
_VEHICLE_CLASSES = {3, 4, 6, 8}


def get_detection_cache_path(video_path: str | Path, cache_dir: str | Path) -> Path:
    key = hashlib.md5(str(video_path).encode()).hexdigest()
    return Path(cache_dir) / f"{key}.json"


def save_detection_cache(data: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)


def load_detection_cache(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    with open(cache_path) as f:
        return json.load(f)


_MODEL_CACHE: dict[str, Any] = {}


def _load_model(device: str, score_thresh: float):
    key = f"{device}:{score_thresh}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from torchvision.models.detection import (
        retinanet_resnet50_fpn_v2,
        RetinaNet_ResNet50_FPN_V2_Weights,
    )
    import torch

    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    model = retinanet_resnet50_fpn_v2(weights=weights, score_thresh=score_thresh)
    model.eval().to(device)
    _MODEL_CACHE[key] = model
    return model


def run_retinanet_detection(
    video_path: str | Path,
    score_thresh: float = 0.25,
    device: str = "cuda",
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run RetinaNet on every frame of *video_path*.

    Returns a dict with fps, dimensions, n_frames, and per-frame bboxes.
    Each bbox is [x1, y1, x2, y2, cls_id] in normalised [0, 1] coordinates,
    with cls_id in the torchvision COCO (1-indexed, 91-class) space
    (only vehicle classes kept).
    """
    import cv2
    import torch
    from torchvision.transforms.functional import to_tensor

    video_path = Path(video_path)
    model = _load_model(device, score_thresh)

    cap = cv2.VideoCapture(str(video_path))
    fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width: float = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
    height: float = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0

    frames: dict[str, list[list[float]]] = {}
    frame_idx = 0
    batch: list[Any] = []
    batch_indices: list[int] = []

    def _flush() -> None:
        if not batch:
            return
        with torch.inference_mode():
            outputs = model(batch)
        for local_i, out in enumerate(outputs):
            boxes = out["boxes"].cpu().tolist()
            scores = out["scores"].cpu().tolist()
            labels = out["labels"].cpu().tolist()
            kept: list[list[float]] = []
            for (x1, y1, x2, y2), _s, lab in zip(boxes, scores, labels):
                cls = int(lab)
                if cls not in _VEHICLE_CLASSES:
                    continue
                kept.append([x1 / width, y1 / height, x2 / width, y2 / height, cls])
            frames[str(batch_indices[local_i])] = kept
        batch.clear()
        batch_indices.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = to_tensor(rgb).to(device)
        batch.append(t)
        batch_indices.append(frame_idx)
        frame_idx += 1
        if len(batch) >= batch_size:
            _flush()

    _flush()
    cap.release()

    return {
        "video_path": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames": frame_idx,
        "frames": frames,
    }
