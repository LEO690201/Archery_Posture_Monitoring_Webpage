# -*- coding: utf-8 -*-
"""
角度计算模块 - 射箭姿态分析系统
功能：计算两个向量之间的夹角（基于点积公式）
应用场景：计算射箭动作中手臂关节的弯曲角度
"""
import math

def calculate_angle(A, B, C, D):
    """
    计算向量 AB 和向量 CD 之间的夹角（单位：度）
    
    Args:
        A (tuple/list): 第一个向量起点坐标 (x1, y1)
        B (tuple/list): 第一个向量终点坐标 (x2, y2)
        C (tuple/list): 第二个向量起点坐标 (x3, y3)
        D (tuple/list): 第二个向量终点坐标 (x4, y4)
    
    Returns:
        float: 两个向量之间的夹角（范围：0° ~ 180°）
    
    计算原理：
        1. 向量点积公式：AB · CD = |AB| × |CD| × cosθ
        2. 反余弦求解角度：θ = arccos[(AB · CD) / (|AB| × |CD|)]
        3. 弧度转换为角度：θ(度) = θ(弧度) × (180/π)
    """
    # ========== 步骤1：计算向量 AB 和向量 CD ==========
    # 向量AB = B - A（终点减起点）
    AB = (B[0] - A[0], B[1] - A[1])
    # 向量CD = D - C（终点减起点）
    CD = (D[0] - C[0], D[1] - C[1])

    # ========== 步骤2：计算向量的模长（长度） ==========
    AB_length = math.sqrt(AB[0]**2 + AB[1]**2)  # |AB| = √(x² + y²)
    CD_length = math.sqrt(CD[0]**2 + CD[1]**2)  # |CD| = √(x² + y²)

    # ========== 步骤3：计算两个向量的点积 ==========
    # 点积公式：AB·CD = AB.x × CD.x + AB.y × CD.y
    AB_dot_CD = AB[0] * CD[0] + AB[1] * CD[1]

    # ========== 步骤4：计算夹角的余弦值 ==========
    # cosθ = (AB·CD) / (|AB| × |CD|)
    cos_angle_A = AB_dot_CD / (AB_length * CD_length)

    # ========== 步骤5：计算夹角（弧度转角度） ==========
    # 反余弦求弧度值 → 转换为角度值
    angle_A = math.degrees(math.acos(cos_angle_A))

    return angle_A

# ========== 测试代码（可选） ==========
if __name__ == "__main__":
    """测试用例：验证角度计算正确性"""
    # 测试1：水平同向向量（夹角0°）
    A1, B1 = (0, 0), (1, 0)
    C1, D1 = (0, 0), (2, 0)
    print(f"测试1 - 水平同向向量夹角：{calculate_angle(A1, B1, C1, D1):.1f}°（预期：0.0°）")

    # 测试2：垂直向量（夹角90°）
    A2, B2 = (0, 0), (1, 0)
    C2, D2 = (0, 0), (0, 1)
    print(f"测试2 - 垂直向量夹角：{calculate_angle(A2, B2, C2, D2):.1f}°（预期：90.0°）")

    # 测试3：对角线向量（夹角45°）
    A3, B3 = (0, 0), (1, 0)
    C3, D3 = (0, 0), (1, 1)
    print(f"测试3 - 对角线向量夹角：{calculate_angle(A3, B3, C3, D3):.1f}°（预期：45.0°）")

