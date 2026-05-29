# CVPR 2026 AUTOPILOT — Accident (Kaggle) 1st Place Solution

1st-place solution for the [Kaggle CVPR 2026 Accident](https://www.kaggle.com/competitions/accident) competition.

📄 Paper: [arXiv:2605.29325](https://arxiv.org/abs/2605.29325) (Accepted at the AUTOPILOT Workshop, CVPR 2026)

A multi-stage pipeline built on `Qwen3-VL`. Two scales of the same model are run
independently (exp160b: 32B, exp200: 235B), blended 9:1 on time/spatial with
type taken from exp160b, then refined with a bbox-snap post-processing step.

| | Public | Private |
|---|---:|---:|
| exp160b | 0.55215 | 0.56740 |
| exp200 | 0.53118 | 0.55670 |
| 0.9×exp160b + 0.1×exp200 | 0.55415 | 0.56948 |
| + bbox snap (final) | 0.55469 | 0.57080 |

## Hardware
- exp160b: 1 × NVIDIA RTX PRO 6000 (~13h)
- exp200 : 8 × NVIDIA RTX PRO 6000 (~10h)

## Data
Unzip the Kaggle competition data under `./input/accident/` (must contain `test_metadata.csv`).
```bash
kaggle competitions download -c accident -p input && unzip input/accident.zip -d input/accident
```

## Build
```bash
docker build -t accident-1st .
```

## Run
```bash
# 1. exp160b (32B, 1 GPU)
docker run --rm --gpus '"device=0"' --shm-size=32g \
  -v "$PWD":/kaggle -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  accident-1st python3 /kaggle/src/exp160b/submit.py

# 2. exp200 (235B, 8 GPU)
docker run --rm --gpus all --shm-size=64g \
  -v "$PWD":/kaggle -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  accident-1st python3 /kaggle/src/exp200/submit.py

# 3. blend  -> working/submission_ensemble.csv
docker run --rm -v "$PWD":/kaggle accident-1st \
  python3 /kaggle/src/ensemble/ensemble_160b_200.py

# 4. bbox snap post-processing  -> working/submission_final.csv
docker run --rm --gpus all -v "$PWD":/kaggle accident-1st \
  python3 /kaggle/src/bbox_snap/run.py

# 5. submit
kaggle competitions submit -c accident -f working/submission_final.csv -m "final"
```

## Layout
```
├── Dockerfile
├── LICENSE
└── src/
    ├── exp160b/          # 32B FP8 (config/predict/submit/video_utils/vlm_utils)
    ├── exp200/           # 235B MoE (tensor_parallel_size=8)
    ├── ensemble/
    └── bbox_snap/        # RetinaNet-based coordinate snap post-processing
```

## License
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
