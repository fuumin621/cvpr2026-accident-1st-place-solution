#!/usr/bin/env python3
from __future__ import annotations

DEFAULT_CONFIG: dict[str, float | int | str] = {
    "checkpoint": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "gpu_memory_utilization": 0.90,
    "tensor_parallel_size": 8,
    "max_model_len": 98304,
    "whole_fps": 4.0,
    "whole_limit": 128,
    "image_max_side": 960,
    # 2-pass split settings (same as exp046)
    "split_threshold_sec": 32.0,
    "pass_duration_sec": 32.0,
    # Stage 3: Grounding settings
    "use_grounding": True,       # enable grounding stage
    "grounding_weight": 1.0,     # 1.0 = use only grounding, 0.0 = use only Stage1
    # Time refinement (self-contained, no reuse of exp091 outputs)
    "time_refine": True,
    "time_refine_window_before": 2.0,
    "time_refine_window_after": 2.0,
    "time_refine_fps": 4.0,
    "time_refine_limit": 12,
    "time_refine_context_before": 8.0,
    "time_refine_context_after": 4.0,
    "time_refine_context_fps": 0.5,
    "time_refine_context_limit": 4,
    "time_refine_blend": 0.35,
    "time_refine_max_shift_sec": 1.5,
}
