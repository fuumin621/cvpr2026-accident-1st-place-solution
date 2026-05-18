#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from .video_utils import read_video_info

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

TYPE_CHOICES = ["rear-end", "head-on", "t-bone", "sideswipe", "single"]


def sample_frames(
    video_path: Path,
    out_dir: Path,
    target_fps: float,
    t_start: float | None = None,
    t_end: float | None = None,
    limit: int = 24,
    max_side: int = 640,
) -> list[tuple[float, Path]]:
    info = read_video_info(video_path)
    if info.fps <= 0 or info.n_frames <= 0:
        return []

    t0 = 0.0 if t_start is None else max(0.0, t_start)
    t1 = info.duration if t_end is None else min(info.duration, t_end)
    if t1 <= t0:
        t1 = min(info.duration, t0 + 1.0)

    times = np.arange(t0, t1 + 1e-6, 1.0 / max(target_fps, 0.1))
    if len(times) > limit:
        idx = np.linspace(0, len(times) - 1, limit).round().astype(int)
        times = times[idx]

    cap = cv2.VideoCapture(str(video_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    sampled: list[tuple[float, Path]] = []

    for i, t in enumerate(times):
        frame_idx = int(round(t * info.fps))
        frame_idx = max(0, min(info.n_frames - 1, frame_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        side = max(h, w)
        if side > max_side:
            scale = max_side / float(side)
            frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

        label = f"t={t:.2f}s"
        cv2.rectangle(frame, (8, 8), (250, 46), (0, 0, 0), thickness=-1)
        cv2.putText(frame, label, (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = out_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(out_path), frame)
        sampled.append((float(t), out_path))

    cap.release()
    return sampled


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _prepare_inputs_for_vllm(messages: list[dict[str, Any]], processor: AutoProcessor) -> dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data: dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


class QwenVLPredictor:
    def __init__(
        self,
        checkpoint: str,
        gpu_memory_utilization: float = 0.75,
        max_model_len: int = 65536,
        tensor_parallel_size: int | None = None,
    ):
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        tp_size = tensor_parallel_size
        if tp_size is None or tp_size <= 0:
            tp_size = torch.cuda.device_count()
        self.llm = LLM(
            model=checkpoint,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=False,
            tensor_parallel_size=max(1, tp_size),
            seed=0,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            top_k=-1,
            stop_token_ids=[],
        )

    def generate_json(self, prompt_text: str, image_paths: list[Path]) -> tuple[dict[str, Any] | None, str]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for p in image_paths:
            content.append({"type": "image", "image": str(p)})
        messages = [{"role": "user", "content": content}]
        vllm_input = _prepare_inputs_for_vllm(messages, self.processor)
        output = self.llm.generate([vllm_input], sampling_params=self.sampling_params)[0].outputs[0].text
        return _extract_json(output), output

    def generate_grounding(self, prompt_text: str, image_path: Path) -> tuple[tuple[float, float] | None, str]:
        """Ask model to point to a location in a single image.

        Returns (x_norm, y_norm) in [0,1] range, or None if parsing fails.
        """
        content: list[dict[str, Any]] = [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt_text},
        ]
        messages = [{"role": "user", "content": content}]
        vllm_input = _prepare_inputs_for_vllm(messages, self.processor)
        output = self.llm.generate([vllm_input], sampling_params=self.sampling_params)[0].outputs[0].text
        point = _parse_grounding_point(output)
        return point, output


def _parse_grounding_point(text: str) -> tuple[float, float] | None:
    """Parse point coordinates from Qwen3-VL grounding output.

    Handles multiple formats:
    - (x, y) with values in 0-1000 range (Qwen native grounding)
    - (x, y) with values in 0-1 range
    - JSON {"point": [x, y]} or {"center_x": x, "center_y": y}
    """
    # Try JSON format first
    jdata = _extract_json(text)
    if jdata:
        # {"point": [x, y]}
        if "point" in jdata and isinstance(jdata["point"], (list, tuple)) and len(jdata["point"]) >= 2:
            try:
                x, y = float(jdata["point"][0]), float(jdata["point"][1])
            except (TypeError, ValueError):
                pass
            else:
                if x > 1.0 or y > 1.0:  # 0-1000 scale
                    x, y = x / 1000.0, y / 1000.0
                return (clip(x, 0.0, 1.0), clip(y, 0.0, 1.0))
        # {"center_x": x, "center_y": y}
        if "center_x" in jdata and "center_y" in jdata:
            try:
                x, y = float(jdata["center_x"]), float(jdata["center_y"])
            except (TypeError, ValueError):
                pass
            else:
                if x > 1.0 or y > 1.0:
                    x, y = x / 1000.0, y / 1000.0
                return (clip(x, 0.0, 1.0), clip(y, 0.0, 1.0))

    # Try parenthetical format: (123, 456) or (0.5, 0.6)
    m = re.search(r"\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)", text)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if x > 1.0 or y > 1.0:
            x, y = x / 1000.0, y / 1000.0
        return (clip(x, 0.0, 1.0), clip(y, 0.0, 1.0))

    return None


def extract_single_frame(
    video_path: Path,
    t: float,
    out_path: Path,
    max_side: int = 768,
    burn_timestamp: bool = False,
) -> Path | None:
    """Extract a single frame at time t from video."""
    info = read_video_info(video_path)
    if info.fps <= 0:
        return None
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = int(round(t * info.fps))
    frame_idx = max(0, min(info.n_frames - 1, frame_idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    h, w = frame.shape[:2]
    side = max(h, w)
    if side > max_side:
        scale = max_side / float(side)
        frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    if burn_timestamp:
        label = f"t={t:.2f}s"
        cv2.rectangle(frame, (8, 8), (250, 46), (0, 0, 0), thickness=-1)
        cv2.putText(frame, label, (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return out_path


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_float(v: Any, default: float) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def normalize_prediction(pred: dict[str, Any], duration: float) -> dict[str, Any]:
    out = dict(pred)
    out["accident_time"] = clip(_safe_float(out.get("accident_time"), duration * 0.35), 0.0, duration)
    out["center_x"] = clip(_safe_float(out.get("center_x"), 0.5), 0.0, 1.0)
    out["center_y"] = clip(_safe_float(out.get("center_y"), 0.5), 0.0, 1.0)
    t = str(out.get("type") or "rear-end").strip().lower()
    out["type"] = t if t in TYPE_CHOICES else "rear-end"
    return out
