# -*- coding: utf-8 -*-
"""
YOLO11 姿态估计模块 v2.0
功能：加载 YOLO11-pose 模型检测图像中的人体关键点，
      提取左手腕、左手肘、右手肘、右肩、右手腕的坐标信息

特性：自动检测并使用 GPU（CUDA），若无 GPU 则自动回退到 CPU
"""
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import os
import sys

_DEVICE = None
_MODEL = None


def get_device():
    """
    自动检测并返回可用的计算设备
    
    Returns:
        torch.device: 可用的计算设备（GPU 或 CPU）
    
    设备选择优先级：
    1. CUDA GPU（如果有多个，使用第一个）
    2. MPS（Apple Silicon Metal Performance Shaders）
    3. CPU（默认回退）
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    # 详细打印CUDA相关信息用于调试
    print("🔍 正在检测CUDA环境...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"   计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 使用 GPU 加速：{gpu_name}")
        print(f"   GPU 内存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 使用 Apple Metal 加速 (MPS)")
    else:
        device = torch.device('cpu')
        print("💻 使用 CPU 计算（未检测到 GPU）")
    
    _DEVICE = device
    return _DEVICE


def get_model():
    global _MODEL
    if _MODEL is None:
        model = YOLO(get_real_path("yolo11x-pose.pt"))
        model.to(get_device())
        _MODEL = model
    return _MODEL


def get_real_path(relative_path):
    """
    获取文件的真实路径（适配 PyInstaller 打包后的运行环境）
    
    Args:
        relative_path (str): 资源的相对路径
    
    Returns:
        str: 资源的绝对路径
    """
    # 处理 YOLO 模型文件的路径（适配打包后的资源目录）
    if "yolo11x-pose.pt" in relative_path and hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    # 处理打包为可执行文件后的路径
    elif getattr(sys, 'frozen', False):
        base_path = os.path.dirname(os.path.abspath(sys.executable))
        return os.path.join(base_path, relative_path)
    # 开发环境下的相对路径
    return os.path.join(os.path.abspath("."), relative_path)


def yolo(img_path):
    """
    执行 YOLO 人体姿态检测，提取指定关节的坐标信息
    
    Args:
        img_path (str): 输入图像的路径
    
    Returns:
        tuple: 包含以下关节坐标的元组
            - left_wrist (tuple): 左手腕坐标 (x, y)
            - left_elbow (tuple): 左手肘坐标 (x, y)
            - right_elbow (tuple): 右手肘坐标 (x, y)
            - right_shoulder (tuple): 右肩坐标 (x, y)
            - right_wrist (tuple): 右手腕坐标 (x, y)
    """
    # 自动检测并使用可用计算设备
    model = get_model()
    
    # 执行姿态检测推理
    results = model(get_real_path(img_path))
    
    # 提取检测结果：边界框坐标（xyxy格式，转换为CPU张量并转为uint32类型）
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
    # 提取检测结果：关键点数据（转换为CPU张量）
    bboxes_keypoints = results[0].keypoints.data.cpu().numpy()
    # 获取检测到的人体数量
    num_bbox = len(results[0].boxes.cls)
    
    # 读取原始图像（BGR格式）
    img_bgr = cv2.imread(get_real_path(img_path))
    
    # 人体关键点配置映射（ID对应身体部位、显示颜色、绘制半径）
    kpt_color_map = {
        0: {'name': 'Nose', 'color': [0, 0, 255], 'radius': 6},        # 鼻子
        1: {'name': 'Right Eye', 'color': [255, 0, 0], 'radius': 6},   # 右眼
        2: {'name': 'Left Eye', 'color': [255, 0, 0], 'radius': 6},    # 左眼
        3: {'name': 'Right Ear', 'color': [0, 255, 0], 'radius': 6},   # 右耳
        4: {'name': 'Left Ear', 'color': [0, 255, 0], 'radius': 6},    # 左耳
        5: {'name': 'Right Shoulder', 'color': [193, 182, 255], 'radius': 6},  # 右肩
        6: {'name': 'Left Shoulder', 'color': [193, 182, 255], 'radius': 6},   # 左肩
        7: {'name': 'Right Elbow', 'color': [16, 144, 247], 'radius': 6},       # 右肘
        8: {'name': 'Left Elbow', 'color': [16, 144, 247], 'radius': 6},        # 左肘
        9: {'name': 'Right Wrist', 'color': [1, 240, 255], 'radius': 6},        # 右手腕
        10: {'name': 'Left Wrist', 'color': [1, 240, 255], 'radius': 6},        # 左手腕
        11: {'name': 'Right Hip', 'color': [140, 47, 240], 'radius': 6},        # 右髋
        12: {'name': 'Left Hip', 'color': [140, 47, 240], 'radius': 6},         # 左髋
        13: {'name': 'Right Knee', 'color': [223, 155, 60], 'radius': 6},       # 右膝
        14: {'name': 'Left Knee', 'color': [223, 155, 60], 'radius': 6},        # 左膝
        15: {'name': 'Right Ankle', 'color': [139, 0, 0], 'radius': 6},         # 右踝
        16: {'name': 'Left Ankle', 'color': [139, 0, 0], 'radius': 6},          # 左踝
    }
    
    # 关键点标签字体配置（未实际使用，仅保留配置）
    kpt_labelstr = {
        'font_size': 4,
        'font_thickness': 10,
        'offset_x': 0,
        'offset_y': 150,
    }
    
    # 人体骨骼连接配置（起始关键点ID、目标关键点ID、连线颜色、线宽）
    skeleton_map = [
        {'srt_kpt_id': 15, 'dst_kpt_id': 13, 'color': [0, 100, 255], 'thickness': 5},  # 右踝-右膝
        {'srt_kpt_id': 13, 'dst_kpt_id': 11, 'color': [0, 255, 0], 'thickness': 5},    # 右膝-右髋
        {'srt_kpt_id': 16, 'dst_kpt_id': 14, 'color': [255, 0, 0], 'thickness': 5},    # 左踝-左膝
        {'srt_kpt_id': 14, 'dst_kpt_id': 12, 'color': [0, 0, 255], 'thickness': 5},    # 左膝-左髋
        {'srt_kpt_id': 11, 'dst_kpt_id': 12, 'color': [122, 160, 255], 'thickness': 5},# 右髋-左髋
        {'srt_kpt_id': 5, 'dst_kpt_id': 11, 'color': [139, 0, 139], 'thickness': 5},   # 右肩-右髋
        {'srt_kpt_id': 6, 'dst_kpt_id': 12, 'color': [237, 149, 100], 'thickness': 5},  # 左肩-左髋
        {'srt_kpt_id': 5, 'dst_kpt_id': 6, 'color': [152, 251, 152], 'thickness': 5},   # 右肩-左肩
        {'srt_kpt_id': 5, 'dst_kpt_id': 7, 'color': [148, 0, 69], 'thickness': 5},     # 右肩-右肘
        {'srt_kpt_id': 6, 'dst_kpt_id': 8, 'color': [0, 75, 255], 'thickness': 5},     # 左肩-左肘
        {'srt_kpt_id': 7, 'dst_kpt_id': 9, 'color': [56, 230, 25], 'thickness': 5},     # 右肘-右手腕
        {'srt_kpt_id': 8, 'dst_kpt_id': 10, 'color': [0, 240, 240], 'thickness': 5},    # 左肘-左手腕
        {'srt_kpt_id': 1, 'dst_kpt_id': 2, 'color': [224, 255, 255], 'thickness': 5},   # 右眼-左眼
        {'srt_kpt_id': 0, 'dst_kpt_id': 1, 'color': [47, 255, 173], 'thickness': 5},   # 鼻子-右眼
        {'srt_kpt_id': 0, 'dst_kpt_id': 2, 'color': [203, 192, 255], 'thickness': 5},   # 鼻子-左眼
        {'srt_kpt_id': 1, 'dst_kpt_id': 3, 'color': [196, 75, 255], 'thickness': 5},   # 右眼-右耳
        {'srt_kpt_id': 2, 'dst_kpt_id': 4, 'color': [86, 0, 25], 'thickness': 5},      # 左眼-左耳
        {'srt_kpt_id': 3, 'dst_kpt_id': 5, 'color': [255, 255, 0], 'thickness': 5},    # 右耳-右肩
        {'srt_kpt_id': 4, 'dst_kpt_id': 6, 'color': [255, 18, 200], 'thickness': 5},   # 左耳-左肩
    ]
    
    # 关键点置信度阈值
    CONFIDENCE_THRESHOLD = 0.5
    
    # 初始化返回值（默认值表示未检测到）
    left_wrist = None
    left_elbow = None
    right_elbow = None
    right_shoulder = None
    right_wrist = None
    
    # 遍历每个检测到的人体，提取指定关节坐标
    for idx in range(num_bbox):
        # 获取当前人体的关键点数据
        bbox_keypoints = bboxes_keypoints[idx]
        
        # ========== 置信度检查 ==========
        # 检查所有需要的关键点置信度是否都大于阈值
        required_kpt_ids = [5, 7, 8, 9, 10]  # 右肩、右肘、左肘、右手腕、左手腕
        all_confidence_ok = all(
            bbox_keypoints[kpt_id][2] > CONFIDENCE_THRESHOLD 
            for kpt_id in required_kpt_ids
        )
        
        # 只有当所有关键点置信度都达标时才提取坐标
        if all_confidence_ok:
            # 提取左手腕坐标（四舍五入为整数）
            left_wrist = (round(bbox_keypoints[10][0]), round(bbox_keypoints[10][1]))
            # 提取左手肘坐标（四舍五入为整数）
            left_elbow = (round(bbox_keypoints[8][0]), round(bbox_keypoints[8][1]))
            # 提取右手腕坐标（四舍五入为整数）
            right_wrist = (round(bbox_keypoints[9][0]), round(bbox_keypoints[9][1]))
            # 提取右手肘坐标（四舍五入为整数）
            right_elbow = (round(bbox_keypoints[7][0]), round(bbox_keypoints[7][1]))
            # 提取右肩坐标（四舍五入为整数）
            right_shoulder = (round(bbox_keypoints[5][0]), round(bbox_keypoints[5][1]))
            
            # 打印各关键点的置信度信息
            print(f"📍 关键点置信度:")
            print(f"   右肩 (ID=5): {bbox_keypoints[5][2]:.3f}")
            print(f"   右肘 (ID=7): {bbox_keypoints[7][2]:.3f}")
            print(f"   左肘 (ID=8): {bbox_keypoints[8][2]:.3f}")
            print(f"   右手腕 (ID=9): {bbox_keypoints[9][2]:.3f}")
            print(f"   左手腕 (ID=10): {bbox_keypoints[10][2]:.3f}")
        else:
            print(f"⚠️ 警告：关键点置信度不足，跳过该检测结果")
            for kpt_id in required_kpt_ids:
                print(f"   ID={kpt_id}: {bbox_keypoints[kpt_id][2]:.3f}")
    
    # 验证返回值
    if left_wrist is None or left_elbow is None or right_elbow is None or right_shoulder is None or right_wrist is None:
        print("❌ 错误：未能检测到有效的人体关键点！")
        # 返回默认值避免程序崩溃
        return (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)
    
    # 返回提取的关节坐标
    return left_wrist, left_elbow, right_elbow, right_shoulder, right_wrist


# 示例调用（可选，方便测试）
if __name__ == "__main__":
    # 替换为实际的图像路径
    test_img_path = "test_pose.jpg"
    # 调用YOLO姿态检测函数
    lw, le, re, rs, rw = yolo(test_img_path)
    # 打印提取的关节坐标
    print(f"左手腕坐标: {lw}")
    print(f"左手肘坐标: {le}")
    print(f"右手肘坐标: {re}")
    print(f"右肩坐标: {rs}")
    print(f"右手腕坐标: {rw}")
