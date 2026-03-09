# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 多文件打包配置脚本 - 射箭姿态分析工具
功能：将基于 YOLO 和 PyQt5 的姿态分析系统打包为可执行文件
适配：Windows 系统，包含 YOLO 模型文件、资源文件和所有依赖库
"""
import sys
import os
from PyInstaller.utils.hooks import collect_all
import torch

# ===================== 路径处理 =====================
# 获取项目根目录（spec 文件所在目录）
# 使用 SPEC 变量获取 spec 文件路径
project_root = os.path.dirname(os.path.abspath(SPEC))

# 增加递归限制（防止打包大模型时递归深度不足）
sys.setrecursionlimit(5000)

# ===================== 打包资源配置 =====================
# 数据文件列表：(源文件路径，目标路径)
# 打包时会将这些文件/文件夹复制到可执行文件目录
datas = [
    # YOLO 姿态模型文件
    (os.path.join(project_root, 'yolo11x-pose.pt'), '.'),
    # 程序图标文件
    (os.path.join(project_root, 'icon.ico'), '.'),
    # 输出帧文件夹
    (os.path.join(project_root, 'output_frames'), '.'),
    # 骨骼图片文件夹
    (os.path.join(project_root, 'black'), '.'),
    # 档案文件文件夹
    (os.path.join(project_root, 'document'), '.'),
    # 原始视频/图片文件夹
    (os.path.join(project_root, 'original'), '.'),
]

# 二进制文件列表（扩展库/动态链接库）
binaries = []

# 收集PyTorch的CUDA相关DLL文件
if hasattr(torch, '_C'):
    torch_dir = os.path.dirname(torch._C.__file__)
    binaries.append((os.path.join(torch_dir, 'lib', 'cudart64_*.dll'), 'torch/lib'))
    binaries.append((os.path.join(torch_dir, 'lib', 'cublas64_*.dll'), 'torch/lib'))
    binaries.append((os.path.join(torch_dir, 'lib', 'cublasLt64_*.dll'), 'torch/lib'))
    binaries.append((os.path.join(torch_dir, 'lib', 'cufft64_*.dll'), 'torch/lib'))
    binaries.append((os.path.join(torch_dir, 'lib', 'curand64_*.dll'), 'torch/lib'))
    binaries.append((os.path.join(torch_dir, 'lib', 'cusolver64_*.dll'), 'torch/lib'))
    binaries.append((os.path.join(torch_dir, 'lib', 'cusparse64_*.dll'), 'torch/lib'))
    binaries.append((os.path.join(torch_dir, 'lib', 'nvrtc-builtins64_*.dll'), 'torch/lib'))
    binaries.append((os.path.join(torch_dir, 'lib', 'nvrtc64_*.dll'), 'torch/lib'))

# 收集torchvision的CUDA相关DLL文件
try:
    import torchvision
    tv_dir = os.path.dirname(torchvision._C.__file__)
    binaries.append((os.path.join(tv_dir, 'lib', '*.dll'), 'torchvision/lib'))
except Exception as e:
    print(f"Warning: Could not collect torchvision CUDA binaries: {e}")

# 二进制文件列表（扩展库/动态链接库）
binaries = []

# 隐式导入列表（解决PyInstaller无法自动检测的依赖）
hiddenimports = [
    # 核心依赖库
    'ultralytics', 'cv2', 'matplotlib', 'torch', 'torchvision', 'torchaudio', 'numpy', 
    'pandas', 'PyQt5', 'math', 're',
    # YOLO子模块
    'ultralytics.models', 'ultralytics.engine', 'ultralytics.utils',
    # PyTorch CUDA和分布式相关模块
    'torch.cuda', 'torch.distributed', 'torch.distributed.rpc', 'torch.distributed.elastic',
    'torch.nn.parallel', 'torch.backends.cudnn', 'torch.optim.lr_scheduler',
    # torchvision子模块
    'torchvision.transforms', 'torchvision.datasets', 'torchvision.models',
    # PyQt5子模块
    'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
    # OpenCV/Matplotlib/Pandas子模块
    'cv2.data', 'matplotlib.backends', 'pandas.core',
    # 测试相关依赖（防止打包遗漏）
    'unittest', 'unittest.mock'
]

# 自动收集ultralytics库的所有依赖（数据文件、二进制文件、隐式导入）
yolo_deps = collect_all('ultralytics')
datas += yolo_deps[0]    # 添加YOLO的数据文件
binaries += yolo_deps[1] # 添加YOLO的二进制文件
hiddenimports += yolo_deps[2] # 添加YOLO的隐式导入

# ===================== PyInstaller核心配置 =====================
# 分析阶段：扫描脚本依赖
a = Analysis(
    # 主程序入口文件
    ['ui.py'],
    # 搜索路径
    pathex=[project_root],
    # 二进制文件
    binaries=binaries,
    # 数据文件
    datas=datas,
    # 隐式导入
    hiddenimports=hiddenimports,
    # 钩子文件路径
    hookspath=[],
    # 钩子配置
    hooksconfig={},
    # 运行时钩子
    runtime_hooks=[],
    # 排除的模块（减小打包体积）
    excludes=['tkinter', 'test'],
    # 是否不压缩为归档文件
    noarchive=False,
    # 优化级别（0=无优化，1=基础优化，2=深度优化）
    optimize=0,
)

# 打包为PYZ归档文件（包含所有Python模块）
pyz = PYZ(
    a.pure,
    a.zipped_data,
    optimize=0  # 保持与Analysis一致的优化级别
)

# 生成可执行文件
exe = EXE(
    pyz,
    a.scripts,
    [],
    # 是否排除二进制文件（True=仅包含脚本）
    exclude_binaries=True,
    # 可执行文件名称
    name='PoseAnalysisTool',
    # 调试模式（False=关闭调试）
    debug=False,
    # 是否忽略引导加载程序信号
    bootloader_ignore_signals=False,
    # 是否剥离符号信息（减小体积）
    strip=False,
    # 是否使用UPX压缩
    upx=True,
    # UPX排除的文件
    upx_exclude=[],
    # 运行时临时目录
    runtime_tmpdir=None,
    # 是否显示控制台窗口（False=无控制台，纯GUI程序）
    console=False,
    # 是否禁用窗口化程序的回溯信息
    disable_windowed_traceback=False,
    # 是否模拟argv（macOS专用）
    argv_emulation=False,
    # 目标架构（None=自动检测）
    target_arch=None,
    # 代码签名标识（macOS专用）
    codesign_identity=None,
    # 权限文件（macOS专用）
    entitlements_file=None,
    # 程序图标文件
    icon=os.path.join(project_root, 'icon.ico'),
)

# 收集所有文件并生成最终的输出目录
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    # 是否剥离符号信息
    strip=False,
    # 是否使用UPX压缩
    upx=True,
    # UPX排除的文件
    upx_exclude=[],
    # 输出目录名称
    name='PoseAnalysisTool'
)