# -*- coding: utf-8 -*-
"""
视频帧提取与YOLO姿态处理模块
功能：
1. 从视频中按间隔提取帧并保存
2. 调用YOLO模型处理帧，提取人体关键点
3. 计算关节位置特征值，找到关键动作帧索引
4. 支持进度回调，适配exe和脚本两种运行环境
"""
import cv2
import os
import sys
import yolo
import numpy as np

def extract_frames(video_path, output_folder, progress_callback=None):
    """
    提取视频帧并进行YOLO姿态处理，支持进度回调
    
    Args:
        video_path (str): 视频文件路径
        output_folder (str): 输出处理后帧的主文件夹
        progress_callback (function, optional): 进度回调函数
            接收参数：当前处理帧数(int)、总帧数(int)
    
    Returns:
        int: 特征值最小值对应的索引（关键动作帧索引）
    
    Raises:
        Exception: 视频文件打开失败时打印错误信息
    """
    # ========== 核心适配：路径处理（兼容exe/脚本运行） ==========
    if getattr(sys, 'frozen', False):
        # 打包为exe运行时：获取可执行文件所在目录
        base_path = os.path.dirname(sys.executable)
    else:
        # 普通脚本运行时：获取当前脚本文件所在目录
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # 拼接各输出文件夹的绝对路径（确保路径正确性）
    # 处理后带关键点的帧输出目录
    output_folder = os.path.join(base_path, output_folder)
    # 仅骨骼线的帧输出目录
    output_folder_2 = os.path.join(base_path, "black")
    # 原始未处理帧的输出目录
    output_folder_3 = os.path.join(base_path, "original")
    # ===========================================================
    
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not video_capture.isOpened():
        print(f"[错误] 无法打开视频文件: {video_path}")
        return None  # 返回None标识失败
    
    # 获取视频总帧数（用于进度计算）
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[信息] 视频总帧数: {total_frames}")
    
    # 确保所有输出文件夹存在，不存在则创建
    for folder in [output_folder, output_folder_2, output_folder_3]:
        os.makedirs(folder, exist_ok=True)
    
    # ========== 初始化变量 ==========
    frame_count = 0          # 全局帧计数器（记录已读取的总帧数）
    process_count = 0        # 间隔计数器（每10帧处理一次）
    # 上一帧左右手腕坐标（用于计算位移）
    left_wrist_y_last = 0
    left_wrist_x_last = 0
    right_wrist_y_last = 0
    right_wrist_x_last = 0
    j = 0                    # 特征值数组索引计数器
    # 初始化特征值数组（长度50，初始值1000）
    array = np.full(50, 1000)
    
    # ========== 逐帧处理视频 ==========
    while True:
        # 读取一帧视频
        ret, frame = video_capture.read()
        
        # 读取失败（已到视频末尾），退出循环
        if not ret:
            break
        
        # 每累计10帧执行一次YOLO处理
        if process_count == 10:
            # 构建各版本帧的保存路径（4位数字格式化，如frame_0010.png）
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04}.png")
            frame_filename_2 = os.path.join(output_folder_2, f"frame_{frame_count:04}.png")
            frame_filename_3 = os.path.join(output_folder_3, f"frame_{frame_count:04}.png")
            
            # 保存原始帧（未处理）
            cv2.imwrite(frame_filename_3, frame)
            # 临时保存原始帧用于YOLO处理
            cv2.imwrite(frame_filename, frame)
            
            # 调用YOLO模型处理帧，提取关键点和绘制骨骼
            # 返回值：处理后帧、左右手腕/肘坐标、仅骨骼帧
            frame_processed, left_wrist_x, left_wrist_y, left_elbow_x, left_elbow_y, right_wrist_x, right_wrist_y, right_elbow_x, right_elbow_y, black_img = yolo.yolo(frame_filename)
            
            # 保存处理后的帧（带关键点）和仅骨骼帧
            cv2.imwrite(frame_filename, frame_processed)
            cv2.imwrite(frame_filename_2, black_img)
            
            # ========== 自动判断拉弦手（后手）==========
            # 计算左右手的位移（与上一帧相比）
            left_displacement = (left_wrist_y - left_wrist_y_last)**2 + (left_wrist_x - left_wrist_x_last)**2
            right_displacement = (right_wrist_y - right_wrist_y_last)**2 + (right_wrist_x - right_wrist_x_last)**2
            
            # 选择位移较大的手作为拉弦手（后手），位移较小的作为持弓手（前手）
            if left_displacement >= right_displacement:
                # 左手是拉弦手（后手），使用左手计算特征值
                wrist_x, wrist_y = left_wrist_x, left_wrist_y
                elbow_x, elbow_y = left_elbow_x, left_elbow_y
                wrist_x_last, wrist_y_last = left_wrist_x_last, left_wrist_y_last
            else:
                # 右手是拉弦手（后手），使用右手计算特征值
                wrist_x, wrist_y = right_wrist_x, right_wrist_y
                elbow_x, elbow_y = right_elbow_x, right_elbow_y
                wrist_x_last, wrist_y_last = right_wrist_x_last, right_wrist_y_last
            
            # ========== 计算特征值 ==========
            # 分子：当前帧肘与手腕的距离平方
            numerator = (elbow_y - wrist_y)**2 + (elbow_x - wrist_x)**2
            # 分母：当前帧与上一帧手腕的位移平方（+0.001避免除零）
            denominator = (wrist_y - wrist_y_last)**2 + (wrist_x - wrist_x_last)**2 + 0.001
            # 计算特征值（相对位移比）
            i_value = numerator / denominator
            
            # 将特征值存入数组（仅前50个有效）
            if j < len(array):
                array[j] = i_value
            
            # ========== 重置状态 ==========
            process_count = 0                          # 重置间隔计数器
            # 更新左右手腕坐标记录
            left_wrist_x_last, left_wrist_y_last = left_wrist_x, left_wrist_y
            right_wrist_x_last, right_wrist_y_last = right_wrist_x, right_wrist_y
            j += 1                                     # 数组索引+1
        
        # ========== 进度更新 ==========
        frame_count += 1       # 全局帧计数器+1
        process_count += 1     # 间隔计数器+1
        
        # 调用进度回调函数更新进度（如果有）
        if progress_callback and total_frames > 0:
            progress_callback(frame_count, total_frames)
        
        # 调试：每处理100帧打印一次进度
        if frame_count % 100 == 0:
            print(f"[进度] 已处理 {frame_count}/{total_frames} 帧")
    
    # ========== 处理完成后清理 ==========
    # 释放视频捕捉对象，释放资源
    video_capture.release()
    
    # ========== 查找特征值最小值对应的索引 ==========
    # 初始化最小值和对应索引（从数组第1位开始，跳过初始值）
    min_value = array[1]
    min_index = 1
    
    # 遍历数组（从第2位到末尾），找到最小值的索引
    for i in range(2, len(array)):
        if array[i] < min_value:
            min_value = array[i]
            min_index = i
    
    print(f"[完成] 帧提取与处理完成，关键帧索引: {min_index}")
    return min_index

# ========== 测试代码（单独运行时执行） ==========
if __name__ == "__main__":
    """测试用例：单独运行该模块时执行"""
    # 自动适配运行环境，获取测试视频路径
    if getattr(sys, 'frozen', False):
        # exe运行环境
        test_video_path = os.path.join(os.path.dirname(sys.executable), "test_video.mp4")
    else:
        # 脚本运行环境
        test_video_path = os.path.join(os.path.dirname(__file__), "test_video.mp4")
    
    # 调用帧提取函数
    result_index = extract_frames(
        video_path=test_video_path,
        output_folder="output_frames"
    )
    print(f"[测试结果] 关键帧索引: {result_index}")