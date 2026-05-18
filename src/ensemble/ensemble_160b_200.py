#!/usr/bin/env python3
"""Blend: 0.9 * exp160b + 0.1 * exp200. type is taken from exp160b only."""
import pandas as pd

df_32b = pd.read_csv("/kaggle/working/submission_exp160b.csv")
df_235b = pd.read_csv("/kaggle/working/submission_exp200.csv")
m = df_32b.merge(df_235b, on="path", suffixes=("_32b", "_235b"))

out = pd.DataFrame({
    "path": m["path"],
    "accident_time": 0.9 * m["accident_time_32b"] + 0.1 * m["accident_time_235b"],
    "center_x":      0.9 * m["center_x_32b"]      + 0.1 * m["center_x_235b"],
    "center_y":      0.9 * m["center_y_32b"]      + 0.1 * m["center_y_235b"],
    "type": m["type_32b"],
})
out.to_csv("/kaggle/working/submission_ensemble.csv", index=False)
print(f"[saved] /kaggle/working/submission_ensemble.csv rows={len(out)}")
