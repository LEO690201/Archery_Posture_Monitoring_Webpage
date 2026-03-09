# -*- coding: utf-8 -*-
"""
YOLO11 姿态估计（人体关键点检测）模块 v2.0
功能：使用 YOLO11-pose 模型检测图像中的人体关键点，绘制骨骼连线和关键点，
      并提取左手腕和左手肘的坐标信息

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
    获取文件的真实路径（适配打包后的 exe 运行环境）
    
    Args:
        relative_path (str): 文件的相对路径
    
    Returns:
        str: 文件的绝对路径
    """
    base_path = None
    
    # 处理 PyInstaller 打包后的临时路径
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    # 如果是打包后的可执行文件
    elif getattr(sys, 'frozen', False):
        base_path = os.path.dirname(os.path.abspath(sys.executable))
    
    # 默认使用当前目录
    if base_path is None:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


def yolo(img_path):
    """
    执行 YOLO 人体姿态检测，绘制关键点和骨骼，并提取指定关节坐标
    
    Args:
        img_path (str): 输入图像的路径
    
    Returns:
        tuple: 包含以下元素的元组
            - img_bgr (np.ndarray): 绘制了关键点和骨骼的原始图像
            - left_wrist_x (int): 左手腕 x 坐标
            - left_wrist_y (int): 左手腕 y 坐标
            - left_elbow_x (int): 左手肘 x 坐标
            - left_elbow_y (int): 左手肘 y 坐标
            - black_img (np.ndarray): 仅显示骨骼连线的黑色背景图像
    """
    # 自动检测并使用可用计算设备
    model = get_model()

    # 执行姿态检测
    results = model(get_real_path(img_path))

    # 提取检测结果：边界框坐标（xyxy 格式）
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
    # 提取检测结果：关键点数据（每个关键点包含 x,y，置信度）
    bboxes_keypoints = results[0].keypoints.data.cpu().numpy()
    # 获取检测到的人体数量
    num_bbox = len(results[0].boxes.cls)

    # 读取原始图像（BGR 格式）
    img_bgr = cv2.imread(get_real_path(img_path))
    # 创建与原始图像尺寸相同的黑色背景图像（用于仅显示骨骼）
    black_img = np.zeros_like(img_bgr)

    # 边界框绘制配置
    bbox_color = (150, 0, 0)  # 边界框颜色（BGR）
    bbox_thickness = 6        # 边界框线宽
    bbox_labelstr = {         # 边界框标签字体配置
        'font_size': 6,
        'font_thickness': 14,
        'offset_x': 0,
        'offset_y': -80,
    }

    # 人体关键点配置映射（ID 对应身体部位、颜色、绘制半径）
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

    # 关键点标签字体配置
    kpt_labelstr = {
        'font_size': 4,
        'font_thickness': 10,
        'offset_x': 0,
        'offset_y': 150,
    }

    # 人体骨骼连接配置（起始关键点 ID、目标关键点 ID、颜色、线宽）
    skeleton_map = [
        {'srt_kpt_id': 15, 'dst_kpt_id': 13, 'color': [0, 100, 255], 'thickness': 5},  # 右踝 - 右膝
        {'srt_kpt_id': 13, 'dst_kpt_id': 11, 'color': [0, 255, 0], 'thickness': 5},    # 右膝 - 右髋
        {'srt_kpt_id': 16, 'dst_kpt_id': 14, 'color': [255, 0, 0], 'thickness': 5},    # 左踝 - 左膝
        {'srt_kpt_id': 14, 'dst_kpt_id': 12, 'color': [0, 0, 255], 'thickness': 5},    # 左膝 - 左髋
        {'srt_kpt_id': 11, 'dst_kpt_id': 12, 'color': [122, 160, 255], 'thickness': 5},# 右髋 - 左髋
        {'srt_kpt_id': 5, 'dst_kpt_id': 11, 'color': [139, 0, 139], 'thickness': 5},   # 右肩 - 右髋
        {'srt_kpt_id': 6, 'dst_kpt_id': 12, 'color': [237, 149, 100], 'thickness': 5},  # 左肩 - 左髋
        {'srt_kpt_id': 5, 'dst_kpt_id': 6, 'color': [152, 251, 152], 'thickness': 5},   # 右肩 - 左肩
        {'srt_kpt_id': 5, 'dst_kpt_id': 7, 'color': [148, 0, 69], 'thickness': 5},     # 右肩 - 右肘
        {'srt_kpt_id': 6, 'dst_kpt_id': 8, 'color': [0, 75, 255], 'thickness': 5},     # 左肩 - 左肘
        {'srt_kpt_id': 7, 'dst_kpt_id': 9, 'color': [56, 230, 25], 'thickness': 5},     # 右肘 - 右手腕
        {'srt_kpt_id': 8, 'dst_kpt_id': 10, 'color': [0, 240, 240], 'thickness': 5},    # 左肘 - 左手腕
        {'srt_kpt_id': 1, 'dst_kpt_id': 2, 'color': [224, 255, 255], 'thickness': 5},   # 右眼 - 左眼
        {'srt_kpt_id': 0, 'dst_kpt_id': 1, 'color': [47, 255, 173], 'thickness': 5},   # 鼻子 - 右眼
        {'srt_kpt_id': 0, 'dst_kpt_id': 2, 'color': [203, 192, 255], 'thickness': 5},   # 鼻子 - 左眼
        {'srt_kpt_id': 1, 'dst_kpt_id': 3, 'color': [196, 75, 255], 'thickness': 5},   # 右眼 - 右耳
        {'srt_kpt_id': 2, 'dst_kpt_id': 4, 'color': [86, 0, 25], 'thickness': 5},      # 左眼 - 左耳
        {'srt_kpt_id': 3, 'dst_kpt_id': 5, 'color': [255, 255, 0], 'thickness': 5},    # 右耳 - 右肩
        {'srt_kpt_id': 4, 'dst_kpt_id': 6, 'color': [255, 18, 200], 'thickness': 5},   # 左耳 - 左肩
    ]

    # 遍历每个检测到的人体
    for idx in range(num_bbox):
        # 获取当前人体的边界框坐标
        bbox_xyxy = bboxes_xyxy[idx]
        # 获取类别标签（此处固定为第一个类别）
        bbox_label = results[0].names[0]
        # 获取当前人体的关键点数据
        bbox_keypoints = bboxes_keypoints[idx]

        # 绘制骨骼连线
        for skeleton in skeleton_map:
            # 获取起始关键点坐标和置信度
            srt_kpt_id = skeleton['srt_kpt_id']
            srt_kpt_x = round(bbox_keypoints[srt_kpt_id][0])
            srt_kpt_y = round(bbox_keypoints[srt_kpt_id][1])
            srt_kpt_conf = bbox_keypoints[srt_kpt_id][2]
            
            # 获取目标关键点坐标和置信度
            dst_kpt_id = skeleton['dst_kpt_id']
            dst_kpt_x = round(bbox_keypoints[dst_kpt_id][0])
            dst_kpt_y = round(bbox_keypoints[dst_kpt_id][1])
            dst_kpt_conf = bbox_keypoints[dst_kpt_id][2]
            
            # 获取骨骼绘制参数
            skeleton_color = skeleton['color']
            skeleton_thickness = skeleton['thickness']
            
            # 仅当两个关键点置信度都大于 0.5 时才绘制连线
            if srt_kpt_conf > 0.5 and dst_kpt_conf > 0.5:
                img_bgr = cv2.line(
                    img_bgr, 
                    (srt_kpt_x, srt_kpt_y), 
                    (dst_kpt_x, dst_kpt_y), 
                    color=skeleton_color, 
                    thickness=skeleton_thickness
                )
                black_img = cv2.line(
                    black_img, 
                    (srt_kpt_x, srt_kpt_y), 
                    (dst_kpt_x, dst_kpt_y), 
                    color=skeleton_color, 
                    thickness=skeleton_thickness
                )

        # 绘制关键点
        for kpt_id in kpt_color_map:
            # 获取关键点绘制参数
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            
            # 获取关键点坐标和置信度
            kpt_x = round(bbox_keypoints[kpt_id][0])
            kpt_y = round(bbox_keypoints[kpt_id][1])
            kpt_conf = bbox_keypoints[kpt_id][2]
            
            # 仅当关键点置信度大于 0.5 时才绘制
            if kpt_conf > 0.5:
                img_bgr = cv2.circle(
                    img_bgr, 
                    (kpt_x, kpt_y), 
                    kpt_radius, 
                    kpt_color, 
                    -1  # 填充圆
                )

        # 提取左手腕和左手肘坐标
        left_wrist_x = round(bbox_keypoints[10][0])
        left_wrist_y = round(bbox_keypoints[10][1])
        left_elbow_x = round(bbox_keypoints[8][0])
        left_elbow_y = round(bbox_keypoints[8][1])
        
        # 提取右手腕和右手肘坐标（用于支持左手持弓的情况）
        right_wrist_x = round(bbox_keypoints[9][0])
        right_wrist_y = round(bbox_keypoints[9][1])
        right_elbow_x = round(bbox_keypoints[7][0])
        right_elbow_y = round(bbox_keypoints[7][1])

    # 返回处理后的图像和提取的关节坐标（左右手都返回）
    return img_bgr, left_wrist_x, left_wrist_y, left_elbow_x, left_elbow_y, right_wrist_x, right_wrist_y, right_elbow_x, right_elbow_y, black_img


# 示例调用（可选）
if __name__ == "__main__":
    # 替换为你的图像路径
    image_path = "test_pose.jpg"
    # 执行姿态检测
    result_img, lw_x, lw_y, le_x, le_y, skeleton_img = yolo(image_path)
    
    # 保存结果
    cv2.imwrite("pose_result.jpg", result_img)
    cv2.imwrite("skeleton_only.jpg", skeleton_img)
    
    # 打印提取的坐标
    print(f"左手腕坐标：({lw_x}, {lw_y})")
    print(f"左手肘坐标：({le_x}, {le_y})")
