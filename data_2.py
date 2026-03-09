# -*- coding: utf-8 -*-
"""
数据可视化模块 - 射箭姿态分析系统
功能：
1. 读取指定txt文件中的数值数据
2. 按规则分组并绘制折线图
3. 适配exe/脚本双运行环境的路径处理
4. 自动创建缺失文件夹，完善异常处理
"""
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# ===================== 路径处理核心函数 =====================
def get_resource_path(relative_path):
    """
    修正版：获取文件的绝对路径（适配PyInstaller打包/未打包环境）
    
    Args:
        relative_path (str): 文件/文件夹的相对路径（如 "document/data.txt"）
    
    Returns:
        str: 适配运行环境的绝对路径
    """
    if getattr(sys, 'frozen', False):
        # 打包成exe运行：基础路径 = 可执行文件所在目录
        base_path = os.path.dirname(os.path.abspath(sys.executable))
    else:
        # 未打包脚本运行：基础路径 = 当前工作目录
        base_path = os.path.abspath(".")
    
    # 拼接并返回规范化的绝对路径
    return os.path.join(base_path, relative_path)

def ensure_folder_exists(folder_path):
    """
    确保目标文件夹存在，不存在则自动创建（容错处理）
    
    Args:
        folder_path (str): 文件夹绝对路径
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 文件夹不存在，已自动创建：{folder_path}")

# ===================== 核心绘图函数 =====================
def plot_data_from_txt(txt_filename):
    """
    读取document文件夹中的txt数据文件，按规则分组并绘制折线图
    
    Args:
        txt_filename (str): txt文件名（仅需文件名，无需包含document路径，如 "data.txt"）
    
    处理逻辑：
        1. 数据分组规则：
           - group1: 第1、4、7...个数据（前手臂弯曲角度）
           - group2: 第2、5、8...个数据取反（后手臂弯曲角度）
           - group3: 第3、6、9...个数据（姿态评分）
        2. 图片保存位置：exe/脚本同级目录（与document文件夹平级）
    """
    document_folder = get_resource_path("document")  # 获取document文件夹绝对路径
    if os.path.isabs(txt_filename):
        abs_file_path = txt_filename
    else:
        abs_file_path = os.path.join(document_folder, txt_filename)
    
    # 2. 确保document文件夹存在（防止用户手动删除导致报错）
    ensure_folder_exists(document_folder)
    
    try:
        # 3. 读取txt文件中的数值数据
        # 兼容空格/制表符/换行分隔的数值，忽略空行
        data = pd.read_csv(
            abs_file_path, 
            header=None,          # 无表头
            sep=r'\s+',           # 匹配任意空白字符作为分隔符
            engine='python'       # 使用Python引擎，兼容更多分隔符格式
        )
        # 将二维数据转换为一维列表（处理多行多列情况）
        data = data.stack().tolist()
        
        # 4. 数据校验：确保数据长度为3的倍数（每组3个数据）
        if len(data) % 3 != 0:
            print(f"⚠️ 警告：数据长度{len(data)}不是3的倍数，可能存在数据缺失！")
        
        # 5. 按规则分组数据
        group1 = data[0::3]                # 第1、4、7...个数据（步长3，从0开始）
        group2 = [-x for x in data[1::3]]  # 第2、5、8...个数据（取反）
        group3 = data[2::3]                # 第3、6、9...个数据（步长3，从2开始）

        # 6. 定义绘图+保存子函数（复用绘图逻辑）
        def plot_and_save(group, title, file_name):
            """
            绘制单组数据的折线图并保存
            
            Args:
                group (list): 待绘图的数据列表
                title (str): 图表标题
                file_name (str): 保存的图片文件名（如 "group1.png"）
            """
            # 解决Matplotlib中文显示乱码问题
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 设置中文字体
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
            
            # 创建绘图画布（尺寸8x5英寸）
            plt.figure(figsize=(8, 5))
            
            # 绘制折线图
            plt.plot(
                group, 
                marker='o',        # 数据点标记为圆形
                markersize=4,      # 标记大小
                linewidth=1.5,     # 线条宽度
                color='#1f77b4'    # 专业蓝色系
            )
            
            # 设置图表样式
            plt.title(title, fontsize=12, pad=10)  # 标题+上边距
            plt.xlabel('序号', fontsize=10)         # X轴标签
            plt.ylabel('数值', fontsize=10)         # Y轴标签
            plt.grid(True, alpha=0.3, linestyle='--')  # 网格线（半透明虚线）
            
            # 核心配置：图片保存到exe/脚本同级目录（与document平级）
            save_path = get_resource_path(file_name)
            # 【注释】原方案：保存到document文件夹内
            # save_path = os.path.join(document_folder, file_name)
            
            # 保存图片（高分辨率，裁剪多余空白）
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()  # 关闭画布释放资源
            print(f"✅ 图片已保存：{save_path}")

        # 7. 绘制并保存三组数据的图表
        plot_and_save(group1, '第一组数据折线图', 'group1.png')
        plot_and_save(group2, '第二组数据（取反）折线图', 'group2.png')
        plot_and_save(group3, '第三组数据折线图', 'group3.png')

        # 8. 输出完成信息
        print("\n🎉 所有图表生成完成！")
        print(f"📄 读取的txt文件：{abs_file_path}")
        print(f"🖼️ 生成的图片位置：{get_resource_path('')}")  # 显示exe/脚本同级目录路径
        
    # 异常处理：文件未找到
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {abs_file_path}")
        print(f"   请确保{txt_filename}放在【exe同级的document文件夹】里！")
    # 异常处理：文件为空
    except pd.errors.EmptyDataError:
        print(f"❌ 错误：文件 {abs_file_path} 为空，无数据可绘制！")
    # 其他未知异常
    except Exception as e:
        print(f"❌ 程序运行出错：{str(e)}")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    """测试用例：单独运行该模块时执行"""
    # 示例：读取document文件夹中的data.txt文件并绘图
    txt_filename = "data.txt"  # 仅需写文件名，无需包含document路径
    plot_data_from_txt(txt_filename)
    
    # 适配exe运行：防止运行完成后直接闪退
    if getattr(sys, 'frozen', False):
        input("\n按回车键退出...")
