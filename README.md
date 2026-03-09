# 射箭姿态评估 Web 版

这是一个可本地运行的射箭姿态分析网页。用户上传视频后，系统会自动识别动作、生成骨架图、输出评分，并可在本机保存本次结果。

## 运行环境

- 操作系统：Windows / Linux
- 包管理：Conda
- Python：3.9

## 一键配置环境

项目目录下已经提供一键脚本，直接执行即可：

```bash
bash setup_conda.sh
```

脚本会自动完成以下内容：

1. 创建 `web` conda 环境
2. 安装 Flask、OpenCV、NumPy、Pandas、SciPy
3. 安装 CPU 版 PyTorch
4. 安装 Ultralytics 及所需依赖

如果本机已经存在同名环境，建议先手动删除旧环境：

```bash
CONDA_NO_PLUGINS=true conda env remove -n web -y
```

## 启动方式

1. 激活环境

```bash
conda activate web
```

2. 启动本地网页

```bash
python app.py
```

3. 浏览器访问

```text
http://127.0.0.1:5000
```

## 关闭方式

在启动服务的终端窗口中按 `Ctrl + C` 即可停止。

## 本地数据说明

程序运行时会自动在项目目录下创建 `web_data/`，用于保存本次分析产生的数据：

- `web_data/uploads/`：上传的视频
- `web_data/runs/`：每次分析生成的结果
- `web_data/profiles/`：用户历史记录
- `web_data/reports/`：趋势图

网页中已经提供：

- `保存本次结果`
- `不保存本次结果`

其中“不保存本次结果”只会撤销当前这一次分析产生的数据，不会影响其他人之前保存的内容。

## Linux 部署方式

本项目已按 Linux 路径方式处理，部署到 Linux 服务器时可直接运行：

```bash
conda activate web
APP_HOST=0.0.0.0 APP_PORT=5000 python app.py
```

如果需要让同一局域网或服务器外部访问，可以把 `APP_HOST` 设为 `0.0.0.0`。

## 主要功能

- 上传射箭视频
- 自动识别起手与射箭关键节点
- 查看原图叠加骨架图
- 查看纯骨架图
- 查看循环动态预览
- 输出评分与建议
- 保存或撤销本次分析结果
