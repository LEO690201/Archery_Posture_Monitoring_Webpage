#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="web"
PYTHON_VERSION="3.9.25"

echo "[1/4] 创建 conda 环境 ${ENV_NAME}"
CONDA_NO_PLUGINS=true conda create --solver classic -y -n "${ENV_NAME}" \
  python="${PYTHON_VERSION}" \
  flask=3.1.2 \
  matplotlib=3.9.2 \
  numpy=2.0.2 \
  pandas=2.3.3 \
  scipy=1.13.1 \
  opencv=4.10.0 \
  pip

echo "[2/4] 安装 CPU 版 PyTorch"
CONDA_NO_PLUGINS=true conda run -n "${ENV_NAME}" python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.8.0+cpu \
  torchvision==0.23.0+cpu \
  torchaudio==2.8.0+cpu

echo "[3/4] 安装网页与姿态分析依赖"
CONDA_NO_PLUGINS=true conda run -n "${ENV_NAME}" python -m pip install \
  pyyaml==6.0.3 \
  requests==2.32.5 \
  psutil==7.2.2 \
  ultralytics-thop==2.0.18

echo "[4/4] 安装 Ultralytics"
CONDA_NO_PLUGINS=true conda run -n "${ENV_NAME}" python -m pip install --no-deps ultralytics==8.4.21

echo
echo "环境配置完成。"
echo "激活环境: conda activate ${ENV_NAME}"
echo "启动项目: python app.py"
