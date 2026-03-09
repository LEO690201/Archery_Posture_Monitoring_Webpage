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

echo "正在下载 YOLO11x 姿态估计模型..."
wget https://cas-bridge.xethub.hf.co/xet-bridge-us/670fb757072e5deeae1fde3f/033d3620a25121a81bd29c35b73b0fe17838a821f26588fa3960ffe4faef4ffb?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20260309%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260309T151905Z&X-Amz-Expires=3600&X-Amz-Signature=b77bcb79726c47eba13fde1405959121ed220d26b81a924db7f11ebb7f9506b9&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=public&response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27yolo11x-pose.pt%3B+filename%3D%22yolo11x-pose.pt%22%3B&x-amz-checksum-mode=ENABLED&x-id=GetObject&Expires=1773073145&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3MzA3MzE0NX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82NzBmYjc1NzA3MmU1ZGVlYWUxZmRlM2YvMDMzZDM2MjBhMjUxMjFhODFiZDI5YzM1YjczYjBmZTE3ODM4YTgyMWYyNjU4OGZhMzk2MGZmZTRmYWVmNGZmYioifV19&Signature=G2LOxUBv1LkvNjgKgU6PM%7EWickXQ-uxavCtgjIKWpooUWynXlPrsvUCvG59y3pijY-lvrG3YKNkvRJP63MqOht1EFinXEpeYtBqLUNZql2Oqhj-9kSj5synfULWaXZqICAqr-i2lAgh7Jt1bMSVUavfn8SywoD8XLjUheElMSEtw38h21OD3a4EFIr821cl-KXc6%7Eg7slYFdaCjKdZ3l0QrEoJDYT7-oLohJDJ207ScRRwquovrvrzbR4OwmA6Lp-7jxxptRdAg3gMQX7wJOFtEME8PWxvdmY7Q8J4fq2SKWb6G5mxYhq4bYOGHOwiUuYyjA8vEfCXtvWumjl53hug__&Key-Pair-Id=K2L8F4GPSG1IFC

echo
echo "环境配置完成。"
echo "激活环境: conda activate ${ENV_NAME}"
echo "启动项目: python app.py"
