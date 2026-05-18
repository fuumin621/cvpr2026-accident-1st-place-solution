FROM gcr.io/kaggle-gpu-images/python:latest

RUN pip uninstall torch torchvision torchaudio torchtext -y \
 && pip install torch torchvision

RUN pip install vllm qwen-vl-utils

WORKDIR /kaggle
