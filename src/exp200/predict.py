#!/usr/bin/env python3
"""exp200: 235B MoE three-stage VLM pipeline.

Same pipeline as exp160b but uses Qwen3-VL-235B-A22B-Instruct with
tensor_parallel_size=8. Run on 8x RTX PRO 6000 (~10h). Its outputs are
blended 9:1 with exp160b (type taken from exp160b only).
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG
from .video_utils import read_video_info, resolve_video_path
from .vlm_utils import (
    QwenVLPredictor,
    extract_single_frame,
    normalize_prediction,
    sample_frames,
)

SCENE_TYPE_HINTS = {
    "highway": "Common accident types here: rear-end (following too closely), single (loss of control, hitting barrier/guardrail), or sideswipe (lane change).",
    "signalized_intersection": "Common accident types here: t-bone (running red light, perpendicular collision), rear-end (stopping suddenly), or single (running off road).",
    "simple_intersection": "Common accident types here: t-bone (failure to yield at unsignalized intersection), rear-end, or single.",
    "grade_separated_intersection": "Common accident types here: single (loss of control on ramp/merge), rear-end (merging traffic), or t-bone.",
    "city_street": "Common accident types here: t-bone (intersection), rear-end, single (hitting parked car/pole), or sideswipe.",
    "tunnel": "Common accident types here: rear-end (reduced visibility), single (hitting wall), or sideswipe.",
    "parking_lot": "Common accident types here: single (hitting pole/wall), rear-end (backing up), or sideswipe.",
    "roundabout": "Common accident types here: t-bone (failure to yield), sideswipe (lane change in roundabout), or single.",
}

GROUNDING_PROMPT = (
    "This CCTV frame shows a traffic accident. "
    "Point to the EXACT location where the collision or impact occurs. "
    "Output the collision point coordinates as JSON:\n"
    '{"point": [x, y]}\n'
    "where x is horizontal (0=left edge, 1000=right edge) and "
    "y is vertical (0=top edge, 1000=bottom edge)."
)

TIME_REFINE_PROMPT = (
    "These sequential CCTV frames show a traffic accident. "
    "Some frames provide broader context before/after the event, and other frames densely cover a short window around the current predicted collision time. "
    "Each frame has a burned-in timestamp (t=xx.xx s).\n\n"
    "Your task: identify the EXACT timestamp when vehicles first make physical contact, or when a vehicle first hits an object. "
    "Do NOT choose the peak impact, aftermath, debris, spin, or when vehicles are already clearly entangled. "
    "Use the broader context to understand motion, but report ONLY the first contact time.\n\n"
    "Output ONLY this JSON:\n"
    '{"accident_time": <seconds>}'
)

TYPE_CHOICES = ["rear-end", "head-on", "t-bone", "sideswipe", "single"]
T_BONE_IMPOSSIBLE_LAYOUTS = {"highway", "tunnel", "grade_separated_intersection"}


def _load_metadata(metadata_path: Path) -> dict[str, dict[str, str]]:
    if not metadata_path.exists():
        return {}
    result = {}
    with open(metadata_path) as f:
        for row in csv.DictReader(f):
            result[row["path"]] = dict(row)
    return result


def _build_stage1_prompt(duration: float, scene_hint: str) -> str:
    return (
        "These sequential CCTV frames show a traffic accident. "
        "Each frame has a burned-in timestamp (t=xx.xx s). "
        f"The full video is {duration:.1f} seconds long.\n\n"
        "Your task: Find the EXACT MOMENT, LOCATION, and TYPE of the collision.\n\n"
        "1) COLLISION TIME — Carefully examine each frame's timestamp. "
        "Find the frame where vehicles first make contact or a vehicle first hits an object. "
        "Report the timestamp in seconds.\n\n"
        "2) COLLISION POSITION — Where in the frame does the impact happen? "
        "Report as normalized coordinates: "
        "center_x (0=left, 1=right), center_y (0=top, 1=bottom).\n\n"
        "3) ACCIDENT TYPE — Classify the collision type by watching the vehicles' "
        "approach angles and movements across all frames:\n"
        "  • rear-end = a following vehicle strikes the one ahead (both traveling same direction)\n"
        "  • head-on = two vehicles collide front-to-front (traveling opposite directions)\n"
        "  • sideswipe = vehicles traveling parallel make side-to-side contact\n"
        "  • t-bone = perpendicular collision (one vehicle hits the other's side at ~90°)\n"
        "  • single = only one vehicle involved (hits object, barrier, pole, or loses control)\n"
        f"{scene_hint}\n"
        "Look at:\n"
        "- How many vehicles are involved (one = single)\n"
        "- The direction each vehicle is traveling before impact\n"
        "- The angle of collision (same direction = rear-end, opposite = head-on, "
        "perpendicular = t-bone, parallel side = sideswipe)\n\n"
        "Output ONLY this JSON:\n"
        '{"accident_time": <seconds>, "center_x": <0-1>, "center_y": <0-1>, '
        '"type": "rear-end|head-on|t-bone|sideswipe|single"}'
    )


def _apply_scene_type_postfix(accident_type: str, scene_layout: str) -> tuple[str, str]:
    if accident_type == "t-bone" and scene_layout in T_BONE_IMPOSSIBLE_LAYOUTS:
        return "rear-end", f"{scene_layout}:t-bone->rear-end"
    return accident_type, ""


def _merge_predictions(
    pred1: dict[str, Any],
    pred2: dict[str, Any],
    overlap_start: float,
    pass1_end: float,
) -> dict[str, Any]:
    t1 = pred1["accident_time"]
    t2 = pred2["accident_time"]
    if t2 > pass1_end:
        return dict(pred2)
    if t1 < overlap_start:
        return dict(pred1)
    return {
        "accident_time": (t1 + t2) / 2.0,
        "center_x": (pred1["center_x"] + pred2["center_x"]) / 2.0,
        "center_y": (pred1["center_y"] + pred2["center_y"]) / 2.0,
        "type": pred1.get("type", pred2.get("type", "rear-end")),
    }


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _parse_time_from_text(text: str, duration: float) -> float | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "accident_time" in data:
                t = float(data["accident_time"])
                if 0.0 <= t <= duration:
                    return t
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|second)", text)
    if match:
        t = float(match.group(1))
        if 0.0 <= t <= duration:
            return t
    return None


class Exp200Predictor:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.cfg = cfg
        self.vlm = QwenVLPredictor(
            str(cfg["checkpoint"]),
            gpu_memory_utilization=float(cfg["gpu_memory_utilization"]),
            max_model_len=int(cfg["max_model_len"]),
            tensor_parallel_size=int(cfg.get("tensor_parallel_size", 1)),
        )
        meta_path = Path(cfg.get("metadata_path", "/kaggle/input/accident/test_metadata.csv"))
        self.metadata = _load_metadata(meta_path)

    def _get_scene_hint(self, video_path: Path, video_rel: str = "") -> str:
        if video_rel and video_rel in self.metadata:
            layout = self.metadata[video_rel].get("scene_layout", "")
            if layout in SCENE_TYPE_HINTS:
                return f"\nScene context: This is a {layout.replace('_', ' ')} camera. {SCENE_TYPE_HINTS[layout]}\n"
        for suffix in [video_path.name, f"videos/{video_path.name}"]:
            if suffix in self.metadata:
                layout = self.metadata[suffix].get("scene_layout", "")
                if layout in SCENE_TYPE_HINTS:
                    return f"\nScene context: This is a {layout.replace('_', ' ')} camera. {SCENE_TYPE_HINTS[layout]}\n"
        return ""

    def _run_stage1_pass(
        self,
        video_path: Path,
        duration: float,
        t_start: float | None,
        t_end: float | None,
        pass_label: str,
        scene_hint: str,
    ) -> tuple[dict[str, Any], str, list[float]]:
        with tempfile.TemporaryDirectory(prefix=f"exp200_s1_{pass_label}_") as td:
            sampled = sample_frames(
                video_path,
                Path(td) / "frames",
                target_fps=float(self.cfg["whole_fps"]),
                t_start=t_start,
                t_end=t_end,
                limit=int(self.cfg["whole_limit"]),
                max_side=int(self.cfg["image_max_side"]),
            )
            prompt = _build_stage1_prompt(duration, scene_hint)
            pred, raw = self.vlm.generate_json(prompt, [p for _, p in sampled])

        result = normalize_prediction(pred or {}, duration=duration)
        sample_times = [float(t) for t, _ in sampled]
        return result, raw, sample_times

    def _run_stage1_pass_from_sampled(
        self,
        sampled: list[tuple[float, Path]],
        duration: float,
        scene_hint: str,
    ) -> tuple[dict[str, Any], str, list[float]]:
        prompt = _build_stage1_prompt(duration, scene_hint)
        pred, raw = self.vlm.generate_json(prompt, [p for _, p in sampled])
        result = normalize_prediction(pred or {}, duration=duration)
        sample_times = [float(t) for t, _ in sampled]
        return result, raw, sample_times

    def build_stage1_prefetch(self, video_path: Path, video_rel: str = "") -> dict[str, Any]:
        info = read_video_info(video_path)
        scene_hint = self._get_scene_hint(video_path, video_rel)

        split_thresh = float(self.cfg["split_threshold_sec"])
        pass_dur = float(self.cfg["pass_duration_sec"])
        needs_split = info.duration > split_thresh
        pass1_end = min(pass_dur, info.duration)

        cache_dir = Path(tempfile.mkdtemp(prefix="exp200_prefetch_"))
        pass1_sampled = sample_frames(
            video_path,
            cache_dir / "p1",
            target_fps=float(self.cfg["whole_fps"]),
            t_start=0.0,
            t_end=pass1_end,
            limit=int(self.cfg["whole_limit"]),
            max_side=int(self.cfg["image_max_side"]),
        )

        prefetched: dict[str, Any] = {
            "video_path": str(video_path),
            "video_rel": video_rel,
            "info": info,
            "scene_hint": scene_hint,
            "needs_split": needs_split,
            "pass1_end": pass1_end,
            "cache_dir": cache_dir,
            "pass1_sampled": pass1_sampled,
        }

        if needs_split:
            pass2_start = max(0.0, info.duration - pass_dur)
            pass2_sampled = sample_frames(
                video_path,
                cache_dir / "p2",
                target_fps=float(self.cfg["whole_fps"]),
                t_start=pass2_start,
                t_end=info.duration,
                limit=int(self.cfg["whole_limit"]),
                max_side=int(self.cfg["image_max_side"]),
            )
            prefetched["pass2_start"] = pass2_start
            prefetched["pass2_sampled"] = pass2_sampled

        return prefetched

    def cleanup_prefetch(self, prefetched_stage1: dict[str, Any] | None) -> None:
        if not prefetched_stage1:
            return
        cache_dir = prefetched_stage1.get("cache_dir")
        if isinstance(cache_dir, Path):
            shutil.rmtree(cache_dir, ignore_errors=True)

    def _run_time_refinement(
        self,
        video_path: Path,
        predicted_time: float,
        duration: float,
    ) -> tuple[float, dict[str, Any]]:
        if not bool(self.cfg.get("time_refine", True)):
            return predicted_time, {"skipped": True}

        dense_before = float(self.cfg.get("time_refine_window_before", 2.0))
        dense_after = float(self.cfg.get("time_refine_window_after", 2.0))
        dense_fps = float(self.cfg.get("time_refine_fps", 4.0))
        dense_limit = int(self.cfg.get("time_refine_limit", 12))
        context_before = float(self.cfg.get("time_refine_context_before", 8.0))
        context_after = float(self.cfg.get("time_refine_context_after", 4.0))
        context_fps = float(self.cfg.get("time_refine_context_fps", 0.5))
        context_limit = int(self.cfg.get("time_refine_context_limit", 4))
        blend = float(self.cfg.get("time_refine_blend", 0.35))
        max_shift = float(self.cfg.get("time_refine_max_shift_sec", 1.5))

        dense_start = max(0.0, predicted_time - dense_before)
        dense_end = min(duration, predicted_time + dense_after)
        context_start = max(0.0, predicted_time - context_before)
        context_end = min(duration, predicted_time + context_after)

        with tempfile.TemporaryDirectory(prefix="exp200_trefine_") as td:
            context_dir = Path(td) / "context"
            dense_dir = Path(td) / "dense"
            context_frames = sample_frames(
                video_path,
                context_dir,
                target_fps=context_fps,
                t_start=context_start,
                t_end=context_end,
                limit=context_limit,
                max_side=int(self.cfg["image_max_side"]),
            )
            dense_frames = sample_frames(
                video_path,
                dense_dir,
                target_fps=dense_fps,
                t_start=dense_start,
                t_end=dense_end,
                limit=dense_limit,
                max_side=int(self.cfg["image_max_side"]),
            )
            merged = sorted(context_frames + dense_frames, key=lambda x: x[0])
            dedup: list[tuple[float, Path]] = []
            seen: set[float] = set()
            for t, p in merged:
                key = round(float(t), 3)
                if key in seen:
                    continue
                seen.add(key)
                dedup.append((float(t), p))

            if len(dedup) < 4:
                return predicted_time, {
                    "status": "too_few_frames",
                    "predicted_time": predicted_time,
                    "n_frames": len(dedup),
                }

            pred, raw = self.vlm.generate_json(TIME_REFINE_PROMPT, [p for _, p in dedup])

        refined = None
        if pred and "accident_time" in pred:
            try:
                t = float(pred["accident_time"])
                if 0.0 <= t <= duration:
                    refined = t
            except (TypeError, ValueError):
                pass
        if refined is None:
            refined = _parse_time_from_text(raw, duration)

        if refined is None:
            return predicted_time, {
                "status": "fallback",
                "predicted_time": predicted_time,
                "n_frames": len(dedup),
                "sample_times": [float(t) for t, _ in dedup],
                "raw": raw,
            }

        raw_shift = refined - predicted_time
        clipped_shift = _clip(raw_shift, -max_shift, max_shift)
        final_time = _clip(predicted_time + blend * clipped_shift, 0.0, duration)
        return final_time, {
            "status": "refined",
            "predicted_time": predicted_time,
            "raw_refined_time": refined,
            "raw_shift_sec": raw_shift,
            "clipped_shift_sec": clipped_shift,
            "blend": blend,
            "final_time": final_time,
            "dense_window": [dense_start, dense_end],
            "context_window": [context_start, context_end],
            "n_frames": len(dedup),
            "sample_times": [float(t) for t, _ in dedup],
            "raw": raw,
        }

    def _run_stage3_grounding(
        self,
        video_path: Path,
        accident_time: float,
        fallback_cx: float,
        fallback_cy: float,
    ) -> tuple[float, float, str]:
        with tempfile.TemporaryDirectory(prefix="exp200_s3_") as td:
            frame_path = extract_single_frame(
                video_path,
                t=accident_time,
                out_path=Path(td) / "accident_frame.jpg",
                max_side=int(self.cfg["image_max_side"]),
                burn_timestamp=False,
            )
            if frame_path is None:
                return fallback_cx, fallback_cy, "frame_extraction_failed"

            point, raw = self.vlm.generate_grounding(GROUNDING_PROMPT, frame_path)

        if point is not None:
            return point[0], point[1], raw
        return fallback_cx, fallback_cy, f"grounding_failed: {raw}"

    def predict_video(
        self,
        video_path: Path,
        video_rel: str = "",
        prefetched_stage1: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        info = read_video_info(video_path)
        scene_hint = self._get_scene_hint(video_path, video_rel)

        split_thresh = float(self.cfg["split_threshold_sec"])
        pass_dur = float(self.cfg["pass_duration_sec"])
        needs_split = info.duration > split_thresh
        pass1_end = min(pass_dur, info.duration)

        use_prefetch = bool(prefetched_stage1) and prefetched_stage1.get("video_path") == str(video_path)
        if use_prefetch:
            info = prefetched_stage1["info"]
            scene_hint = str(prefetched_stage1["scene_hint"])
            needs_split = bool(prefetched_stage1["needs_split"])
            pass1_end = float(prefetched_stage1["pass1_end"])
            pred1, raw1, times1 = self._run_stage1_pass_from_sampled(
                prefetched_stage1["pass1_sampled"],
                info.duration,
                scene_hint,
            )
        else:
            pred1, raw1, times1 = self._run_stage1_pass(
                video_path, info.duration,
                t_start=0.0, t_end=pass1_end, pass_label="p1",
                scene_hint=scene_hint,
            )

        if not needs_split:
            stage1_pred = pred1
            stage1_debug = {
                "n_passes": 1,
                "raw_s1": raw1,
                "sample_times_s1": times1,
            }
        else:
            if use_prefetch and "pass2_sampled" in prefetched_stage1:
                pass2_start = float(prefetched_stage1["pass2_start"])
                pred2, raw2, times2 = self._run_stage1_pass_from_sampled(
                    prefetched_stage1["pass2_sampled"],
                    info.duration,
                    scene_hint,
                )
            else:
                pass2_start = max(0.0, info.duration - pass_dur)
                pred2, raw2, times2 = self._run_stage1_pass(
                    video_path, info.duration,
                    t_start=pass2_start, t_end=info.duration, pass_label="p2",
                    scene_hint=scene_hint,
                )
            merged = _merge_predictions(pred1, pred2, overlap_start=pass2_start, pass1_end=pass1_end)

            t1 = pred1["accident_time"]
            t2 = pred2["accident_time"]
            if t2 > pass1_end:
                used = "pass2"
            elif t1 < pass2_start:
                used = "pass1"
            else:
                used = "overlap_avg"

            stage1_pred = merged
            stage1_debug = {
                "n_passes": 2,
                "used": used,
                "pass1": {"time": float(t1), "type": pred1.get("type", ""), "raw": raw1},
                "pass2": {"time": float(t2), "type": pred2.get("type", ""), "raw": raw2},
                "sample_times_s1_p1": times1,
                "sample_times_s1_p2": times2,
            }

        base_time = float(stage1_pred["accident_time"])
        s1_cx = float(stage1_pred["center_x"])
        s1_cy = float(stage1_pred["center_y"])

        s1_type = str(stage1_pred.get("type", "rear-end")).strip().lower()
        if s1_type not in TYPE_CHOICES:
            s1_type = "rear-end"

        refined_time, time_refine_debug = self._run_time_refinement(
            video_path, base_time, info.duration
        )

        use_grounding = bool(self.cfg.get("use_grounding", True))
        if use_grounding:
            g_cx, g_cy, g_raw = self._run_stage3_grounding(
                video_path, refined_time, s1_cx, s1_cy,
            )
            grounding_weight = float(self.cfg.get("grounding_weight", 1.0))
            final_cx = s1_cx * (1 - grounding_weight) + g_cx * grounding_weight
            final_cy = s1_cy * (1 - grounding_weight) + g_cy * grounding_weight
            grounding_debug = {
                "s1_cx": s1_cx, "s1_cy": s1_cy,
                "grounding_cx": g_cx, "grounding_cy": g_cy,
                "final_cx": final_cx, "final_cy": final_cy,
                "weight": grounding_weight,
                "raw": g_raw,
            }
        else:
            final_cx, final_cy = s1_cx, s1_cy
            grounding_debug = {"skipped": True}

        scene_layout = ""
        if video_rel and video_rel in self.metadata:
            scene_layout = str(self.metadata[video_rel].get("scene_layout", "")).strip()
        accident_type, postfix_reason = _apply_scene_type_postfix(s1_type, scene_layout)

        return {
            "prediction": {
                "accident_time": refined_time,
                "center_x": final_cx,
                "center_y": final_cy,
                "type": accident_type,
            },
            "video_info": {
                "fps": float(info.fps),
                "n_frames": int(info.n_frames),
                "duration": float(info.duration),
                "width": int(info.width),
                "height": int(info.height),
            },
            "debug": {
                "stage1": stage1_debug,
                "time_refine": time_refine_debug,
                "stage3_grounding": grounding_debug,
                "stage1_type": s1_type,
                "postfix_type": accident_type,
                "postfix_reason": postfix_reason,
                "scene_layout": scene_layout,
                "scene_hint": scene_hint.strip(),
                "stage1_prefetched": use_prefetch,
            },
        }


def resolve_video(dataset_root: Path, video_rel: str) -> Path:
    return resolve_video_path(dataset_root, video_rel)
