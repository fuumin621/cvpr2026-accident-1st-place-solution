#!/usr/bin/env python3
"""Run exp160b (Qwen3-VL-32B-Instruct-FP8) on the full test set. 1 x RTX PRO 6000."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from exp160b import DEFAULT_CONFIG, Exp160bPredictor, resolve_video  # noqa: E402

DATASET_ROOT = Path("/kaggle/input/accident")
OUT_CSV = Path("/kaggle/working/submission_exp160b.csv")
SUB_COLS = ["path", "accident_time", "center_x", "center_y", "type"]


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    predictor = Exp160bPredictor(config=dict(DEFAULT_CONFIG))
    test_df = pd.read_csv(DATASET_ROOT / "test_metadata.csv")

    done = set(pd.read_csv(OUT_CSV)["path"]) if OUT_CSV.exists() else set()
    rows = pd.read_csv(OUT_CSV)[SUB_COLS].to_dict("records") if OUT_CSV.exists() else []

    total = len(test_df)
    t_start = time.time()
    for i, row in enumerate(test_df.itertuples(index=False), start=1):
        video_rel = str(row.path)
        if video_rel in done:
            continue
        t0 = time.time()
        out = predictor.predict_video(resolve_video(DATASET_ROOT, video_rel), video_rel=video_rel)
        pred = out["prediction"]
        rows.append({
            "path": video_rel,
            "accident_time": float(pred["accident_time"]),
            "center_x": float(pred["center_x"]),
            "center_y": float(pred["center_y"]),
            "type": str(pred["type"]),
        })
        pd.DataFrame(rows, columns=SUB_COLS).to_csv(OUT_CSV, index=False)
        print(f"[{i}/{total}] {video_rel} t={pred['accident_time']:.2f} "
              f"c=({pred['center_x']:.3f},{pred['center_y']:.3f}) type={pred['type']} "
              f"({time.time() - t0:.1f}s)")

    print(f"[done] {OUT_CSV} elapsed={time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
