# -*- coding: utf-8 -*-
"""
文件写入模块 - 射箭姿态分析系统
功能：
1. 将数值数据写入指定txt文件（适配exe/脚本双运行环境）
2. 自动创建缺失的document文件夹，增强容错性
3. 过滤非数值型数据，避免写入无效内容
4. 完善的异常处理和控制台提示
"""
import os
import sys

def write_to_file(file_name="data", *numbers):
    """
    将数字写入exe/脚本同级document文件夹下的txt文件
    
    Args:
        file_name (str): 文件名（无需加.txt后缀），默认值为"data"
        *numbers (int/float/str): 可变参数，要写入的数值型数据（支持多参数传入）
    
    Returns:
        str/None: 成功写入返回文件完整路径，失败返回None
    
    特性：
        1. 适配PyInstaller打包后的exe运行环境和脚本运行环境
        2. 自动创建document文件夹（防止用户手动删除）
        3. 过滤非数值型数据，仅保留有效数字
        4. 追加模式写入（a），避免覆盖已有数据
        5. 统一使用UTF-8编码，避免中文/特殊字符乱码
    """
    # ========== 核心适配：路径处理（兼容exe/脚本运行） ==========
    if getattr(sys, 'frozen', False):
        # 打包为exe运行时：基础路径 = 可执行文件所在目录
        base_path = os.path.dirname(os.path.abspath(sys.executable))
    else:
        # 普通脚本运行时：基础路径 = 当前脚本文件所在目录
        base_path = os.path.dirname(os.path.abspath(__file__))

    # 拼接document文件夹路径（与exe/脚本文件同级）
    folder_path = os.path.join(base_path, "document")
    
    # ========== 容错处理：确保document文件夹存在 ==========
    # exist_ok=True：文件夹已存在时不报错，简化创建逻辑
    os.makedirs(folder_path, exist_ok=True)

    # ========== 构建目标文件的完整路径 ==========
    # 拼接完整文件名（自动添加.txt后缀）
    file_full_name = f"{file_name}.txt"
    # 拼接文件的绝对路径
    file_path = os.path.join(folder_path, file_full_name)

    # ========== 数据写入（增强健壮性） ==========
    try:
        # 以追加模式打开文件，使用UTF-8编码确保兼容性
        with open(file_path, "a", encoding="utf-8") as file:
            # 第一步：过滤并验证数据，仅保留有效数值
            valid_numbers = []
            for num in numbers:
                try:
                    # 尝试转换为浮点型（兼容int/float/数字字符串）
                    valid_num = float(num)
                    # 转换为字符串，便于后续拼接写入
                    valid_numbers.append(str(valid_num))
                except (ValueError, TypeError):
                    # 捕获非数值型数据异常（如字符串、None、列表等）
                    print(f"⚠️ 警告：'{num}' 不是有效数字，已跳过写入！")
            
            # 第二步：写入有效数据（避免空数据写入）
            if valid_numbers:
                # 多个数字用空格分隔，每行写入一组数据（换行符结尾）
                file.write(" ".join(valid_numbers) + "\n")
                print(f"✅ 数据已成功写入：{file_path}")
            else:
                print(f"⚠️ 警告：无有效数字可写入文件！")
        
        # 返回文件完整路径，方便调用方后续使用
        return file_path
    
    # 异常处理：权限不足（文件被占用/只读等）
    except PermissionError:
        print(f"❌ 错误：没有权限写入文件 {file_path}")
        print(f"   请检查：1. 文件是否被其他程序占用 2. 当前用户是否有写入权限")
    # 其他未知异常（如磁盘满、路径非法等）
    except Exception as e:
        print(f"❌ 写入文件时发生错误：{str(e)}")
        return None

# ========== 测试代码（单独运行时执行） ==========
if __name__ == "__main__":
    """测试用例：验证函数功能"""
    # 测试1：写入正常数字（多参数传入）
    print("=== 测试1：写入有效数字 ===")
    write_to_file("test", 1.2, 3, 4.5, 6)
    
    # 测试2：写入包含无效数据的情况（验证容错能力）
    print("\n=== 测试2：包含无效数据的写入 ===")
    write_to_file("test", 7, "abc", 8.9, None, [1,2])
    
    # 适配exe运行：防止运行完成后直接闪退
    if getattr(sys, 'frozen', False):
        input("\n按回车键退出...")