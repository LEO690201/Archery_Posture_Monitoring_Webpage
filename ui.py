# -*- coding: utf-8 -*-
"""
射箭姿态评估系统 - PyQt5 可视化界面
功能：
1. 选择并播放射箭视频
2. 提取视频帧并进行 YOLO 姿态检测
3. 计算射箭动作的关节角度并评分
4. 展示骨骼检测结果、创建用户档案、数据分析展示

版本：2.0 - 优化评分算法，美化界面设计
"""
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import sys
import os
import re
from PyQt5.QtWidgets import (
    QPushButton, QDialog, QLabel, QVBoxLayout, QMessageBox,
    QWidget, QHBoxLayout, QFileDialog, QLineEdit, QProgressBar,
    QApplication, QGraphicsDropShadowEffect, QFrame, QScrollArea,
    QGridLayout, QTextEdit, QMainWindow, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import torch

# ===================== 路径处理与文件夹初始化 =====================
def get_resource_path(relative_path):
    """
    获取资源文件的真实路径（适配 PyInstaller 打包/脚本双运行环境）
    
    Args:
        relative_path (str): 资源文件的相对路径
    
    Returns:
        str: 资源文件的绝对路径
    """
    # 处理 PyInstaller 打包后的临时路径（用于打包的资源文件，如模型、图标等）
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)


def get_data_path(relative_path):
    """
    获取数据文件夹的真实路径（适配 PyInstaller 打包/脚本双运行环境）
    数据文件夹（output_frames, black, document, original）放在exe同级目录下
    
    Args:
        relative_path (str): 数据文件夹的相对路径
    
    Returns:
        str: 数据文件夹的绝对路径
    """
    # 打包为exe运行时：基础路径 = 可执行文件所在目录
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(os.path.abspath(sys.executable))
    else:
        # 普通脚本运行时：基础路径 = 当前脚本文件所在目录
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)


def init_folders():
    """初始化系统所需的文件夹目录（在exe同级目录下创建）"""
    folders = ["output_frames", "black", "document", "original"]
    for folder in folders:
        folder_path = get_data_path(folder)
        os.makedirs(folder_path, exist_ok=True)


def cleanup_folders():
    """
    清理临时文件夹中的图片文件
    在程序关闭时调用，删除 original、output_frames、black 文件夹中的所有图片
    """
    folders_to_clean = ["original", "output_frames", "black"]
    for folder_name in folders_to_clean:
        folder_path = get_data_path(folder_name)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                # 只删除图片文件，保留其他文件
                if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"删除文件失败 {file_path}: {e}")


# 初始化文件夹
init_folders()

# 导入外部功能模块
import photo
import yolo_2
import angle
import create_document
import data_2

# ===================== 全局数据存储类 =====================
class GlobalData:
    """全局数据存储类，用于跨函数/窗口共享数据"""
    file_path = ""          # 选中的视频文件路径
    min_index = 20          # 关键帧索引
    front_arm_angle = 0     # 前手臂弯曲角度
    behind_arm_angle = 0    # 后手臂弯曲角度
    score = 0               # 姿态评分

# ===================== 进度对话框类 =====================
class ProgressDialog(QDialog):
    """视频处理进度展示对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("处理进度")
        self.setModal(True)  # 模态对话框，阻塞父窗口操作
        self.setFixedSize(400, 100)
        
        # 创建布局和控件
        layout = QVBoxLayout()
        self.label = QLabel("正在处理视频，请稍候...", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)

    def update_progress(self, value, total):
        """
        更新进度条显示
        
        Args:
            value (int): 当前处理进度
            total (int): 总处理数量
        """
        progress = int((value / total) * 100)
        self.progress_bar.setValue(progress)
        self.label.setText(f"正在处理：{value}/{total} 帧")
        QApplication.processEvents()  # 强制刷新 UI

# ===================== 视频处理线程类 =====================
class VideoProcessThread(QThread):
    """视频处理线程（避免 UI 阻塞）"""
    # 自定义信号：进度更新 (当前值，总值)、处理完成 (关键帧索引)
    progress_signal = pyqtSignal(int, int)
    finish_signal = pyqtSignal(int)

    def __init__(self, file_path, output_folder):
        super().__init__()
        self.file_path = file_path    # 视频文件路径
        self.output_folder = output_folder  # 帧输出文件夹

    def run(self):
        """线程执行函数：提取视频帧并发送进度信号"""
        min_index = photo.extract_frames(
            self.file_path, 
            self.output_folder, 
            self.progress_signal.emit
        )
        self.finish_signal.emit(min_index)

# ===================== 主窗口类 =====================
class MainWindow(QMainWindow):
    """射箭姿态评估系统主窗口"""
    
    def __init__(self):
        super().__init__()
        self.ui = Ui_widget()
        self.ui.setupUi(self)
        self.ui.slot_init()
    
    def closeEvent(self, event):
        """
        窗口关闭事件：弹出确认对话框，删除三个文件夹中的所有图片
        """
        try:
            # 弹出确认对话框
            reply = QtWidgets.QMessageBox.question(
                self,
                '确认清理',
                '是否要清理 black、output_frames、original 三个文件夹下的所有图片？',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                # 清理三个文件夹
                cleanup_all_folders()
            
            event.accept()
        except Exception as e:
            print(f"关闭事件处理失败：{e}")
            event.accept()

# ===================== 主界面 UI 类 =====================
class Ui_widget(QWidget):
    """射箭姿态评估系统主界面"""
    
    # 标记是否是第一次选择视频
    is_first_video_select = True
    
    def setupUi(self, widget):
        """初始化 UI 布局和控件"""
        widget.setObjectName("widget")
        widget.setEnabled(True)
        widget.resize(1200, 850)
        widget.setMinimumSize(1200, 850)
        widget.setAcceptDrops(False)

        # ========== 主容器布局 ==========
        widget.setStyleSheet("""
            QWidget#widget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)
        
        # 标题标签
        title_label = QLabel("🏹 射箭姿态评估系统", widget)
        title_label.setGeometry(QtCore.QRect(50, 20, 500, 50))
        title_font = QFont("Microsoft YaHei", 24, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #e94560; background: transparent;")
        title_label.setObjectName("title_label")
        
        # 副标题
        subtitle_label = QLabel("AI 智能姿态分析与评分系统", widget)
        subtitle_label.setGeometry(QtCore.QRect(50, 60, 400, 30))
        subtitle_font = QFont("Microsoft YaHei", 12, QFont.Normal)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setStyleSheet("color: #a0a0a0; background: transparent;")
        
        # 视频显示区域容器（带圆角和阴影）- 增加高度
        video_container = QFrame(widget)
        video_container.setGeometry(QtCore.QRect(30, 100, 800, 550))
        video_container.setStyleSheet("""
            QFrame {
                background-color: #0a0a15;
                border-radius: 15px;
                border: 2px solid #1f4068;
            }
        """)
        video_container.setObjectName("video_container")
        
        # 视频显示标签
        self.video_label = QtWidgets.QLabel(video_container)
        self.video_label.setGeometry(QtCore.QRect(10, 10, 780, 530))
        self.video_label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.video_label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 10px;")
        self.video_label.setObjectName("video_label")
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 100, 255, 100))
        shadow.setOffset(0, 0)
        video_container.setGraphicsEffect(shadow)

        # 侧边栏容器 - 调整位置避免重叠
        sidebar = QFrame(widget)
        sidebar.setGeometry(QtCore.QRect(850, 100, 320, 600))
        sidebar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1f4068, stop:1 #16213e);
                border-radius: 15px;
                border: 2px solid #1f4068;
            }
        """)
        sidebar.setObjectName("sidebar")
        
        # 侧边栏标题
        sidebar_title = QLabel("功能菜单", sidebar)
        sidebar_title.setGeometry(QtCore.QRect(0, 10, 260, 40))
        sidebar_title.setAlignment(Qt.AlignCenter)
        sidebar_title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        sidebar_title.setStyleSheet("color: #ffffff; background: transparent;")
        
        # 功能按钮样式表
        button_style = """
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #{start_color}, stop:1 #{end_color});
                color: white;
                border: none;
                border-radius: 10px;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                font-weight: bold;
                padding: 12px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #{hover_start}, stop:1 #{hover_end});
                transform: scale(1.02);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #{pressed_start}, stop:1 #{pressed_end});
            }}
            QPushButton:disabled {{
                background: #4a4a4a;
                color: #888888;
            }}
        """
        
        # 侧边栏加高以容纳更多按钮
        sidebar.setGeometry(QtCore.QRect(850, 100, 320, 600))
        
        # 功能按钮 - 选择视频（移入侧边栏顶部）- 调整宽度适应新侧边栏
        self.pushButton_7 = QPushButton("📁 选择视频文件", sidebar)
        self.pushButton_7.setGeometry(QtCore.QRect(15, 10, 290, 40))
        self.pushButton_7.setStyleSheet(button_style.format(
            start_color="4ecca3", end_color="#38b37a",
            hover_start="#5fd98d", hover_end="#45c985",
            pressed_start="#3db872", pressed_end="#2da866"
        ))
        self.pushButton_7.setObjectName("pushButton_select")

        # 进度信息标签（移入侧边栏）
        self.progress_info = QLabel("", sidebar)
        self.progress_info.setGeometry(QtCore.QRect(15, 55, 290, 20))
        self.progress_info.setStyleSheet("color: #a0a0a0; background: transparent; font-size: 11px;")
        self.progress_info.setObjectName("progress_info")

        # 功能按钮 - 开始处理（移入侧边栏）
        self.pushButton = QPushButton("▶️ 开始处理视频", sidebar)
        self.pushButton.setGeometry(QtCore.QRect(15, 80, 290, 40))
        self.pushButton.setStyleSheet(button_style.format(
            start_color="4ecca3", end_color="#38b37a",
            hover_start="#5fd98d", hover_end="#45c985",
            pressed_start="#3db872", pressed_end="#2da866"
        ))
        self.pushButton.setObjectName("pushButton_process")

        # 功能按钮 - 起始动作（调整位置）
        self.pushButton_2 = QPushButton("🎯 查看起始动作", sidebar)
        self.pushButton_2.setGeometry(QtCore.QRect(15, 130, 290, 40))
        self.pushButton_2.setStyleSheet(button_style.format(
            start_color="e94560", end_color="#c73e54",
            hover_start="#ff5a7a", hover_end="#e94560",
            pressed_start="#c73e54", pressed_end="#a83347"
        ))
        self.pushButton_2.setObjectName("pushButton_start")

        # 功能按钮 - 结束动作（调整位置）
        self.pushButton_3 = QPushButton("🏹 查看结束动作", sidebar)
        self.pushButton_3.setGeometry(QtCore.QRect(15, 175, 290, 40))
        self.pushButton_3.setStyleSheet(button_style.format(
            start_color="e94560", end_color="#c73e54",
            hover_start="#ff5a7a", hover_end="#e94560",
            pressed_start="#c73e54", pressed_end="#a83347"
        ))
        self.pushButton_3.setObjectName("pushButton_end")

        # 功能按钮 - 姿态评价（调整位置）
        self.pushButton_4 = QPushButton("📊 姿态评价", sidebar)
        self.pushButton_4.setGeometry(QtCore.QRect(15, 220, 290, 40))
        self.pushButton_4.setStyleSheet(button_style.format(
            start_color="f39c12", end_color="#d68910",
            hover_start="#f5b041", hover_end="#e59866",
            pressed_start="#d68910", pressed_end="#b9770e"
        ))
        self.pushButton_4.setObjectName("pushButton_evaluate")

        # 功能按钮 - 创建档案（调整位置）
        self.pushButton_8 = QPushButton("📋 创建用户档案", sidebar)
        self.pushButton_8.setGeometry(QtCore.QRect(15, 265, 290, 40))
        self.pushButton_8.setStyleSheet(button_style.format(
            start_color="3498db", end_color="#2980b9",
            hover_start="#5dade2", hover_end="#3498db",
            pressed_start="#2980b9", pressed_end="#1f618d"
        ))
        self.pushButton_8.setObjectName("pushButton_profile")

        # 功能按钮 - 查看数据分析（调整位置）
        self.pushButton_9 = QPushButton("📈 数据分析报告", sidebar)
        self.pushButton_9.setGeometry(QtCore.QRect(15, 310, 290, 40))
        self.pushButton_9.setStyleSheet(button_style.format(
            start_color="9b59b6", end_color="#8e44ad",
            hover_start="#af6dd6", hover_end="#9b59b6",
            pressed_start="#8e44ad", pressed_end="#732d91"
        ))
        self.pushButton_9.setObjectName("pushButton_analysis")
        
        # 评分显示区域（调整位置）
        score_container = QFrame(sidebar)
        score_container.setGeometry(QtCore.QRect(15, 360, 290, 120))
        score_container.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                border: 1px solid #333;
            }
        """)
        score_container.setObjectName("score_container")
        
        # 评分标题
        score_title = QLabel("姿态评分", score_container)
        score_title.setGeometry(QtCore.QRect(0, 10, 220, 30))
        score_title.setAlignment(Qt.AlignCenter)
        score_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        score_title.setStyleSheet("color: #ffffff; background: transparent;")
        
        # 评分数值显示
        self.score_display = QLabel("--", score_container)
        self.score_display.setGeometry(QtCore.QRect(0, 40, 220, 60))
        self.score_display.setAlignment(Qt.AlignCenter)
        self.score_display.setFont(QFont("Arial", 48, QFont.Bold))
        self.score_display.setStyleSheet("color: #4ecca3; background: transparent;")
        
        # 评分等级显示 - 向下移动避免被遮挡
        self.score_grade = QLabel("", score_container)
        self.score_grade.setGeometry(QtCore.QRect(0, 95, 220, 35))
        self.score_grade.setAlignment(Qt.AlignCenter)
        self.score_grade.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.score_grade.setStyleSheet("color: #a0a0a0; background: transparent;")
        
        # 增加评分容器高度以容纳等级文字
        score_container.setGeometry(QtCore.QRect(15, 360, 290, 140))

        # 详细评价显示区域（缩小宽度避免重叠）- 增加高度和 Y 位置
        evaluation_container = QFrame(widget)
        evaluation_container.setGeometry(QtCore.QRect(30, 670, 800, 150))
        evaluation_container.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                border: 1px solid #1f4068;
            }
        """)
        evaluation_container.setObjectName("evaluation_container")
        
        # 评价标题
        eval_title = QLabel("📝 详细评价", evaluation_container)
        eval_title.setGeometry(QtCore.QRect(15, 10, 180, 30))
        eval_title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        eval_title.setStyleSheet("color: #e94560; background: transparent;")
        
        # 查看完整评价按钮 - 调整位置避免被遮挡
        self.view_full_eval_button = QPushButton("📄 查看评价", evaluation_container)
        self.view_full_eval_button.setGeometry(QtCore.QRect(670, 5, 120, 35))
        self.view_full_eval_button.raise_()  # 确保按钮在最上层
        self.view_full_eval_button.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.view_full_eval_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5dade2, stop:1 #3498db);
            }
        """)
        
        # 评价内容显示（调整尺寸适应容器）
        self.textEdit = QTextEdit(evaluation_container)
        self.textEdit.setGeometry(QtCore.QRect(15, 42, 770, 85))
        self.textEdit.setReadOnly(True)
        self.textEdit.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0.5);
                border: 1px solid #333;
                border-radius: 5px;
                color: #e0e0e0;
                font-family: 'Microsoft YaHei';
                font-size: 13px;
                padding: 8px;
            }
        """)
        self.textEdit.setObjectName("textEdit")
        
        # 状态栏 - 调整位置
        try:
            gpu_status = torch.cuda.is_available()
            device_name = torch.cuda.get_device_name(0) if gpu_status else "CPU"
        except:
            gpu_status = False
            device_name = "CPU"
        
        status_bar = QLabel(f"💻 计算设备：{device_name}", widget)
        status_bar.setGeometry(QtCore.QRect(30, 820, 400, 25))
        status_bar.setStyleSheet("color: #666; background: transparent;")
        status_bar.setObjectName("status_bar")

        # 设置控件文本
        self.retranslateUi(widget)
        QtCore.QMetaObject.connectSlotsByName(widget)

    def retranslateUi(self, widget):
        """设置 UI 控件的显示文本"""
        _translate = QtCore.QCoreApplication.translate
        widget.setWindowTitle(_translate("widget", "射箭姿态评估系统"))
        self.pushButton.setText(_translate("widget", "开始处理"))
        self.pushButton_2.setText(_translate("widget", "起始动作"))
        self.pushButton_3.setText(_translate("widget", "结束动作"))
        self.pushButton_4.setText(_translate("widget", "姿态评价"))
        self.textEdit.setText(_translate("widget", "最终评价:"))
        self.pushButton_7.setText(_translate("widget", "点击选择视频地址"))
        self.pushButton_8.setText(_translate("widget", "创建档案"))
        self.pushButton_9.setText(_translate("widget", "查看数据分析"))

    def slot_init(self):
        """初始化信号槽连接"""
        self.pushButton.clicked.connect(self.Button_clicked)
        self.pushButton_7.clicked.connect(self.open_video)
        self.pushButton_2.clicked.connect(self.Button_clicked_2)
        self.pushButton_3.clicked.connect(self.Button_clicked_3)
        self.pushButton_4.clicked.connect(self.evalute_posture)
        self.pushButton_8.clicked.connect(self.create_document)
        self.pushButton_9.clicked.connect(self.data_analysis)
        self.view_full_eval_button.clicked.connect(self.show_full_evaluation)

    # ===================== 功能按钮槽函数 =====================
    def data_analysis(self):
        """查看数据分析按钮点击事件"""
        try:
            self.image_window = ImageWindow_4()
            self.image_window.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开数据分析窗口失败：{str(e)}")

    def create_document(self):
        """创建档案按钮点击事件"""
        try:
            self.image_window = ImageWindow_3()
            self.image_window.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开创建档案窗口失败：{str(e)}")

    def evalute_posture(self):
        """姿态评价按钮点击事件：计算关节角度并评分（使用新科学评分公式）"""
        try:
            min_index = GlobalData.min_index
            
            # 构造关键帧图片名称
            if min_index < 10:
                image_name = f"frame_00{min_index}0.png"
            else:
                image_name = f"frame_0{min_index}0.png"
            
            # 获取图片路径并检查是否存在（使用数据路径）
            image_path = get_data_path(os.path.join("output_frames", image_name))
            if not os.path.exists(image_path):
                QMessageBox.warning(self, "警告", f"图片文件不存在：{image_path}")
                return

            # ========== 步骤 1：读取图像并进行 YOLO 检测 ==========
            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]
            
            # 进行 YOLO 检测获取关键点坐标
            left_wrist, left_elbow, right_elbow, right_shoulder, right_wrist = yolo_2.yolo(image_path)
            
            # ========== 步骤 2：计算两个手臂的角度 ==========
            # 计算左侧手臂角度（左肩 - 左肘 - 左手腕）
            left_arm_angle = angle.calculate_angle(right_shoulder, left_elbow, left_wrist, (left_elbow[0]+1, left_elbow[1]))
            left_arm_angle = round(left_arm_angle, 1)
            
            # 计算右侧手臂角度（右肩 - 右肘 - 右手腕）
            right_arm_angle = angle.calculate_angle(right_shoulder, right_elbow, right_wrist, (right_elbow[0]+1, right_elbow[1]))
            right_arm_angle = round(right_arm_angle, 1)
            
            # ========== 步骤 3：根据角度大小判断前手和后手 ==========
            # 大于 90 度的视作前手（持弓手，手臂较伸直）
            # 小于 90 度的视作后手（拉弦手，手臂弯曲）
            if left_arm_angle > 90 and right_arm_angle <= 90:
                # 左手角度大，左手是前手；右手角度小，右手是后手
                front_arm_angle = left_arm_angle
                behind_arm_angle = right_arm_angle
                print(f"✓ 左手为前手 ({left_arm_angle}°), 右手为后手 ({right_arm_angle}°)")
            elif right_arm_angle > 90 and left_arm_angle <= 90:
                # 右手角度大，右手是前手；左手角度小，左手是后手
                front_arm_angle = right_arm_angle
                behind_arm_angle = left_arm_angle
                print(f"✓ 右手为前手 ({right_arm_angle}°), 左手为后手 ({left_arm_angle}°)")
            elif left_arm_angle > 90 and right_arm_angle > 90:
                # 两个角度都大于 90 度，取较大的作为前手
                if left_arm_angle >= right_arm_angle:
                    front_arm_angle = left_arm_angle
                    behind_arm_angle = right_arm_angle
                    print(f"✓ 双手均伸直，左手角度更大：前手={left_arm_angle}°, 后手={right_arm_angle}°")
                else:
                    front_arm_angle = right_arm_angle
                    behind_arm_angle = left_arm_angle
                    print(f"✓ 双手均伸直，右手角度更大：前手={right_arm_angle}°, 后手={left_arm_angle}°")
            else:
                # 两个角度都小于等于 90 度，取较大的作为前手
                if left_arm_angle >= right_arm_angle:
                    front_arm_angle = left_arm_angle
                    behind_arm_angle = right_arm_angle
                    print(f"✓ 双手均弯曲，左手角度较大：前手={left_arm_angle}°, 后手={right_arm_angle}°")
                else:
                    front_arm_angle = right_arm_angle
                    behind_arm_angle = left_arm_angle
                    print(f"✓ 双手均弯曲，右手角度较大：前手={right_arm_angle}°, 后手={left_arm_angle}°")
            
            # 保存全局数据
            GlobalData.front_arm_angle = front_arm_angle
            GlobalData.behind_arm_angle = behind_arm_angle
            
            # 计算姿态评分（使用新的科学评分公式）
            score = self.calculate_archery_score(front_arm_angle, behind_arm_angle)
            score = round(score, 1)
            
            # 保存到全局数据
            GlobalData.front_arm_angle = front_arm_angle
            GlobalData.behind_arm_angle = behind_arm_angle
            GlobalData.score = score

            # 生成详细评价信息
            evaluation = self.generate_evaluation(front_arm_angle, behind_arm_angle, score)

            # 更新评分显示
            self.score_display.setText(f"{score}")
            self.score_grade.setText(self.get_score_grade(score))
            
            # 更新评价文本框
            self.textEdit.setHtml(evaluation)
            
            # 保存完整评价内容用于弹窗显示
            self.full_evaluation_html = evaluation
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"姿态评价失败：{str(e)}")

    def show_full_evaluation(self):
        """显示完整评价弹窗"""
        try:
            if hasattr(self, 'full_evaluation_html') and self.full_evaluation_html:
                dialog = EvaluationDetailDialog(self.full_evaluation_html, self)
                dialog.exec_()
            else:
                QMessageBox.information(self, "提示", "请先进行姿态评价以查看详细分析内容")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开评价详情失败：{str(e)}")
    
    def calculate_archery_score(self, front_angle, behind_angle):
        """
        射箭姿态科学评分公式 v2.0
        
        基于射箭运动生物力学原理设计：
        - 前臂拉弓角度：理想 175-180°，越接近 180°越好
        - 后臂拉弦角度：理想 0-15°，越小表示后臂越直
        
        评分权重分配：
        - 前臂角度评分：40%（拉弓稳定性）
        - 后臂角度评分：35%（拉弦直线度）
        - 综合协调性：25%（两臂配合）
        
        Args:
            front_angle (float): 前臂弯曲角度（°）
            behind_angle (float): 后臂与箭矢夹角（°）
        
        Returns:
            float: 综合评分（0-100 分）
        """
        # ========== 前臂角度评分（40% 权重） ==========
        # 理想范围：175-180°，使用分段函数评分
        if front_angle >= 175:
            front_score = 100
        elif front_angle >= 160:
            # 160-175°之间线性递减
            front_score = 70 + (front_angle - 160) / 15 * 30
        elif front_angle >= 140:
            # 140-160°之间线性递减
            front_score = 40 + (front_angle - 140) / 20 * 30
        else:
            # 低于 140°给予基础分
            front_score = max(0, front_angle / 140 * 40)
        
        # ========== 后臂角度评分（35% 权重） ==========
        # 理想范围：0-15°，越小越好
        if behind_angle <= 15:
            behind_score = 100
        elif behind_angle <= 30:
            # 15-30°之间线性递减
            behind_score = 70 + (30 - behind_angle) / 15 * 30
        elif behind_angle <= 60:
            # 30-60°之间线性递减
            behind_score = 40 + (60 - behind_angle) / 30 * 30
        else:
            # 超过 60°给予基础分
            behind_score = max(0, (90 - behind_angle) / 90 * 40)
        
        # ========== 综合协调性评分（25% 权重） ==========
        # 评估两臂配合程度：前后臂角度差应在合理范围内
        angle_diff = abs(180 - front_angle - behind_angle)
        if angle_diff <= 20:
            coordination_score = 100
        elif angle_diff <= 40:
            coordination_score = 70 + (40 - angle_diff) / 20 * 30
        elif angle_diff <= 70:
            coordination_score = 40 + (70 - angle_diff) / 30 * 30
        else:
            coordination_score = max(0, (100 - angle_diff) / 100 * 40)
        
        # ========== 综合评分 ==========
        total_score = front_score * 0.40 + behind_score * 0.35 + coordination_score * 0.25
        
        return total_score
    
    def get_score_grade(self, score):
        """
        根据评分返回等级评价
        
        Args:
            score (float): 姿态评分
        
        Returns:
            str: 等级评价文本
        """
        if score >= 90:
            return "🏆 大师级"
        elif score >= 80:
            return "⭐ 优秀"
        elif score >= 70:
            return "✓ 良好"
        elif score >= 60:
            return "△ 合格"
        else:
            return "⚠ 需改进"
    
    def generate_evaluation(self, front_angle, behind_angle, score):
        """
        生成详细评价 HTML 文本
        
        Args:
            front_angle (float): 前臂角度
            behind_angle (float): 后臂角度
            score (float): 综合评分
        
        Returns:
            str: HTML 格式的评价文本
        """
        # 收集各项建议
        suggestions = []
        issues = []
        
        # 前臂角度评价
        if front_angle >= 175:
            suggestions.append("<span style='color:#4ecca3'>✓ 前臂拉弓角度优秀</span>")
        elif front_angle >= 160:
            issues.append("<span style='color:#f39c12'>⚠ 前臂未完全伸直，建议加强伸展</span>")
        else:
            issues.append("<span style='color:#e74c3c'>✗ 前臂弯曲过大，严重影响发力</span>")
        
        # 后臂角度评价
        if behind_angle <= 15:
            suggestions.append("<span style='color:#4ecca3'>✓ 后臂拉弦直线度优秀</span>")
        elif behind_angle <= 30:
            issues.append("<span style='color:#f39c12'>⚠ 后臂略有弯曲，注意保持直线</span>")
        else:
            issues.append("<span style='color:#e74c3c'>✗ 后臂弯曲过大，影响箭矢飞行</span>")
        
        # 生成 HTML 评价
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Microsoft YaHei'; font-size: 13px; }}
                .title {{ color: #4ecca3; font-weight: bold; font-size: 15px; margin-bottom: 8px; }}
                .item {{ margin: 5px 0; padding-left: 10px; }}
                .data {{ color: #a0a0a0; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="title">📊 姿态分析结果</div>
            <div class="item data">前臂拉弓角度：<strong>{front_angle}°</strong> | 后臂拉弦角度：<strong>{behind_angle}°</strong></div>
            <div class="item data">综合评分：<strong style='color:#4ecca3'>{score}分</strong> | 等级：<strong>{self.get_score_grade(score)}</strong></div>
            <br>
            <div class="title">💡 改进建议</div>
            {''.join([f'<div class="item">{s}</div>' for s in issues])}
            {''.join([f'<div class="item">{s}</div>' for s in suggestions])}
            {f'<div class="item" style="color:#4ecca3">🎉 姿态完美，继续保持！</div>' if len(issues) == 0 and len(suggestions) > 0 else ''}
        </body>
        </html>
        """
        return html

    def Button_clicked_2(self):
        """起始动作按钮点击事件"""
        try:
            self.image_window = ImageWindow()
            self.image_window.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开起始动作窗口失败：{str(e)}")

    def Button_clicked_3(self):
        """结束动作按钮点击事件"""
        try:
            self.image_window = ImageWindow_2()
            self.image_window.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开结束动作窗口失败：{str(e)}")

    def open_video(self):
        """选择并打开视频文件按钮点击事件 - 显示选择对话框"""
        try:
            # 只在非第一次选择视频时询问是否清理旧文件
            if not Ui_widget.is_first_video_select:
                reply = QMessageBox.question(
                    self,
                    '清理旧文件',
                    '是否要清理 black、output_frames、original 三个文件夹下的所有图片？',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    cleanup_all_folders()
            else:
                # 第一次选择视频，标记为非第一次
                Ui_widget.is_first_video_select = False
            
            # 打开选择对话框，让用户选择上传文件或使用摄像头
            dialog = VideoSourceDialog(self)
            result = dialog.exec_()
            
            if result == QtWidgets.QDialog.Accepted:
                # 用户选择上传文件
                file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    "选择视频文件",
                    "",
                    "Video Files (*.mp4 *.avi *.mkv *.mov)"
                )
                
                if file_path:
                    # 保存视频路径到全局数据
                    GlobalData.file_path = file_path
                    GlobalData.source_type = "file"
                    
                    # 停止之前的视频播放
                    if hasattr(self, 'timer') and self.timer.isActive():
                        self.timer.stop()
                    if hasattr(self, 'cap'):
                        self.cap.release()
                    
                    # 打开视频并播放
                    self.cap = cv2.VideoCapture(file_path)
                    if not self.cap.isOpened():
                        QtWidgets.QMessageBox.warning(self, "错误", "无法打开视频文件！")
                        return
                    
                    # 创建定时器播放视频帧
                    self.timer = QtCore.QTimer()
                    self.timer.timeout.connect(self.update_frame)
                    self.timer.start(30)
                    
                    # 重置处理按钮状态
                    self.pushButton.setText("▶️ 开始处理视频")
                    self.pushButton.setEnabled(True)
                    # 重新连接按钮点击事件
                    try:
                        self.pushButton.clicked.disconnect(self.Button_clicked)
                    except:
                        pass
                    try:
                        self.pushButton.clicked.disconnect(self.show_dialog)
                    except:
                        pass
                    self.pushButton.clicked.connect(self.Button_clicked)
                    
                    # 重置进度信息
                    self.progress_info.setText("")
                    
            elif result == QtWidgets.QDialog.Rejected:
                # 用户选择摄像头录制
                self.open_camera_record()
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开视频失败：{str(e)}")

    def open_camera_record(self):
        """打开摄像头录制视频"""
        try:
            # 打开摄像头选择对话框
            camera_dialog = CameraSelectDialog(self)
            camera_result = camera_dialog.exec_()
            
            if camera_result != QtWidgets.QDialog.Accepted:
                return
            
            camera_index = camera_dialog.camera_index
            
            # 打开摄像头
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                QtWidgets.QMessageBox.warning(self, "错误", "无法打开摄像头！")
                return
            
            # 创建并显示录制控制窗口
            self.record_window = RecordWindow(cap, camera_index, self)
            self.record_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开摄像头失败：{str(e)}")

    def update_frame(self):
        """更新视频播放帧"""
        try:
            ret, frame = self.cap.read()
            if ret:
                # 转换颜色空间并创建 QImage
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = channel * width
                q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                
                # 显示到标签控件
                pixmap = QtGui.QPixmap.fromImage(q_image)
                self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio))
            else:
                self.timer.stop()  # 视频播放完毕停止定时器
        except Exception as e:
            self.timer.stop()
            QMessageBox.warning(self, "警告", f"视频播放出错：{str(e)}")

    def replace_path(self, file_path):
        """规范化文件路径"""
        return os.path.normpath(file_path)

    def Button_clicked(self):
        """开始处理按钮点击事件：启动视频处理线程"""
        try:
            file_path = GlobalData.file_path
            if not file_path:
                QMessageBox.warning(self, "警告", "请先选择视频文件！")
                return

            # 设置输出文件夹（使用数据路径，确保在exe同级目录下）
            output_folder = get_data_path("output_frames")
            os.makedirs(output_folder, exist_ok=True)

            # 更新按钮状态
            self.pushButton.setText("正在处理...")
            self.pushButton.setEnabled(False)

            # 显示进度对话框
            self.progress_dialog = ProgressDialog(self)
            self.progress_dialog.show()

            # 规范化路径并启动处理线程
            file_path = self.replace_path(file_path)
            self.process_thread = VideoProcessThread(file_path, output_folder)
            self.process_thread.progress_signal.connect(self.progress_dialog.update_progress)
            self.process_thread.finish_signal.connect(self.on_process_finished)
            self.process_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理视频失败：{str(e)}")
            self.pushButton.setText("开始处理")
            self.pushButton.setEnabled(True)

    def on_process_finished(self, result_min_index):
        """视频处理完成回调函数"""
        try:
            # 保存关键帧索引到全局数据
            GlobalData.min_index = result_min_index
            
            # 关闭进度对话框，更新按钮状态
            self.progress_dialog.close()
            self.pushButton.setText("展示处理结果")
            self.pushButton.setEnabled(True)

            # 重新连接按钮点击事件
            if self.pushButton.clicked:
                try:
                    self.pushButton.clicked.disconnect(self.Button_clicked)
                except:
                    pass
            self.pushButton.clicked.connect(self.show_dialog)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理完成回调失败：{str(e)}")

    def show_dialog(self):
        """展示处理结果：弹出骨骼显示选择对话框"""
        try:
            dialog = SkeletonDialog()
            result = dialog.exec_()
            
            # 根据用户选择播放不同的图片序列
            if result == QtWidgets.QDialog.Accepted:
                self.play_image_sequence()  # 仅显示骨骼
            else:
                self.play_image_sequence_2()  # 显示原始帧
        except Exception as e:
            QMessageBox.critical(self, "错误", f"展示结果失败：{str(e)}")

    def extract_number(self, filename):
        """从文件名中提取数字（用于排序）"""
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group(0))
        return 0

    def play_image_sequence(self):
        """播放仅显示骨骼的图片序列"""
        try:
            folder_path = get_data_path("black")
            os.makedirs(folder_path, exist_ok=True)

            # 获取文件夹中的图片文件并按数字排序
            self.image_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            self.image_files.sort(key=self.extract_number)

            # 初始化播放索引和定时器
            self.current_index = 0
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_frame_2)
            self.timer.start(300)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"播放骨骼图片序列失败：{str(e)}")

    def update_frame_2(self):
        """更新骨骼图片序列播放帧"""
        try:
            if self.current_index < len(self.image_files):
                image_path = self.image_files[self.current_index]
                if not os.path.exists(image_path):
                    self.current_index += 1
                    return

                # 读取并显示图片
                image_matrix = cv2.imread(image_path)
                frame = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = channel * width
                q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio))

                self.current_index += 1
            else:
                self.timer.stop()  # 播放完毕停止定时器
        except Exception as e:
            self.timer.stop()
            QMessageBox.warning(self, "警告", f"播放图片序列出错：{str(e)}")

    def play_image_sequence_2(self):
        """播放原始视频帧图片序列"""
        try:
            folder_path = get_data_path("output_frames")
            
            # 获取文件夹中的图片文件并按数字排序
            self.image_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            self.image_files.sort(key=self.extract_number)

            # 初始化播放索引和定时器
            self.current_index = 0
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_frame_3)
            self.timer.start(300)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"播放输出图片序列失败：{str(e)}")

    def update_frame_3(self):
        """更新原始帧图片序列播放帧"""
        try:
            if self.current_index < len(self.image_files):
                image_path = self.image_files[self.current_index]
                if not os.path.exists(image_path):
                    self.current_index += 1
                    return

                # 读取并显示图片
                image_matrix = cv2.imread(image_path)
                frame = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = channel * width
                q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio))

                self.current_index += 1
            else:
                self.timer.stop()  # 播放完毕停止定时器
        except Exception as e:
            self.timer.stop()
            QMessageBox.warning(self, "警告", f"播放图片序列出错：{str(e)}")

# ===================== 全局清理函数 =====================
def cleanup_all_folders():
    """清理三个文件夹中的所有图片"""
    folders_to_clean = ["original", "output_frames", "black"]
    for folder_name in folders_to_clean:
        folder_path = get_data_path(folder_name)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"删除文件失败 {file_path}: {e}")

# ===================== 起始动作窗口类 =====================
class ImageWindow(QDialog):
    """射箭起始动作展示窗口（显示姿态评价时的关键帧图片）"""
    def __init__(self):
        super().__init__()
        self.min_index = GlobalData.min_index
        self.setWindowTitle("🎯 射箭起始动作")
        self.setGeometry(300, 50, 1000, 800)
        
        # 应用深色主题样式
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题标签
        title_label = QLabel("🏹 射箭起始动作分析（关键帧）")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setStyleSheet("color: #e94560; background: transparent; padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 创建图片显示容器（带圆角和边框）
        image_container = QFrame(self)
        image_container.setStyleSheet("""
            QFrame {
                background-color: #0a0a15;
                border-radius: 15px;
                border: 2px solid #1f4068;
            }
        """)
        image_container_layout = QVBoxLayout(image_container)
        image_container_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建图片显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("background-color: #000000; border-radius: 10px;")
        image_container_layout.addWidget(self.image_label)
        
        main_layout.addWidget(image_container)

        # 创建切换骨骼显示按钮
        button_style = """
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e94560, stop:1 #c73e54);
                color: white;
                border: none;
                border-radius: 10px;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff5a7a, stop:1 #e94560);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #c73e54, stop:1 #a83347);
            }}
        """
        self.button = QPushButton("💀 切换骨骼显示", self)
        self.button.clicked.connect(self.on_button_clicked)
        self.button.setStyleSheet(button_style)
        self.button.setMaximumWidth(200)
        main_layout.addWidget(self.button, alignment=Qt.AlignCenter)
        
        self.setLayout(main_layout)
        
        # 加载关键帧图片（与姿态评价相同）
        self.load_key_frame_image("output_frames", self.min_index)

    def load_key_frame_image(self, folder, index):
        """
        加载并显示关键帧图片（与姿态评价相同）
        
        Args:
            folder (str): 图片文件夹
            index (int): 关键帧索引
        """
        try:
            # 构造关键帧图片名称（直接使用索引，不加 1）
            if index < 10:
                image_name = f"frame_00{index}0.png"
            else:
                image_name = f"frame_0{index}0.png"
            
            # 获取图片路径并加载（使用数据路径）
            image_path = get_data_path(os.path.join(folder, image_name))
            if not os.path.exists(image_path):
                self.image_label.setText(f"图片不存在：{image_path}\n（尝试加载：{image_name}）")
                return
            
            pixmap = QPixmap(image_path)
            # 缩放图片以适应窗口
            if pixmap.height() > self.height() or pixmap.width() > self.width():
                pixmap = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            self.image_label.setText(f"加载图片失败：{str(e)}")

    def on_button_clicked(self):
        """切换到仅显示骨骼的图片"""
        self.load_key_frame_image("black", self.min_index)

# ===================== 结束动作窗口类 =====================
class ImageWindow_2(QDialog):
    """射箭结束动作展示窗口"""
    def __init__(self):
        super().__init__()
        self.min_index = GlobalData.min_index
        self.setWindowTitle("🏹 射箭结束动作")
        self.setGeometry(300, 50, 1000, 800)
        
        # 应用深色主题样式
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题标签
        title_label = QLabel("🎯 射箭结束动作分析")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setStyleSheet("color: #e94560; background: transparent; padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 创建图片显示容器（带圆角和边框）
        image_container = QFrame(self)
        image_container.setStyleSheet("""
            QFrame {
                background-color: #0a0a15;
                border-radius: 15px;
                border: 2px solid #1f4068;
            }
        """)
        image_container_layout = QVBoxLayout(image_container)
        image_container_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建图片显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("background-color: #000000; border-radius: 10px;")
        image_container_layout.addWidget(self.image_label)
        
        main_layout.addWidget(image_container)

        # 创建切换骨骼显示按钮
        button_style = """
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e94560, stop:1 #c73e54);
                color: white;
                border: none;
                border-radius: 10px;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                font-weight: bold;
                padding: 12px 24px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff5a7a, stop:1 #e94560);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #c73e54, stop:1 #a83347);
            }}
        """
        self.button = QPushButton("💀 切换骨骼显示", self)
        self.button.clicked.connect(self.on_button_clicked)
        self.button.setStyleSheet(button_style)
        self.button.setMaximumWidth(200)
        main_layout.addWidget(self.button, alignment=Qt.AlignCenter)
        
        self.setLayout(main_layout)
        
        # 加载图片
        self.load_image("output_frames", self.min_index)

    def load_image(self, folder, index):
        """
        加载并显示指定索引的结束动作图片
        
        Args:
            folder (str): 图片文件夹
            index (int): 图片索引
        """
        try:
            # 构造结束动作图片名称（索引 +1）
            target_index = index + 1
            # 根据目标索引值选择正确的文件名格式
            if target_index < 10:
                image_name = f"frame_00{target_index}0.png"
            elif target_index < 100:
                image_name = f"frame_0{target_index}0.png"
            else:
                image_name = f"frame_{target_index}0.png"
            
            # 获取图片路径并加载（使用数据路径）
            image_path = get_data_path(os.path.join(folder, image_name))
            if not os.path.exists(image_path):
                self.image_label.setText(f"图片不存在：{image_path}\n（尝试加载：{image_name}）")
                return
            
            pixmap = QPixmap(image_path)
            # 缩放图片以适应窗口
            if pixmap.height() > self.height() or pixmap.width() > self.width():
                pixmap = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            self.image_label.setText(f"加载图片失败：{str(e)}")

    def on_button_clicked(self):
        """切换到仅显示骨骼的图片"""
        self.load_image("black", self.min_index)

# ===================== 骨骼显示选择对话框 =====================
class SkeletonDialog(QDialog):
    """骨骼显示选择对话框"""
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """初始化对话框 UI"""
        self.setWindowTitle("📷 显示选项")
        self.setModal(True)
        self.setGeometry(700, 400, 350, 180)
        
        # 应用深色主题
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建提示标签
        self.label = QtWidgets.QLabel("🤔 选择显示模式", self)
        self.label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("color: #ffffff; background: transparent; padding: 10px;")
        main_layout.addWidget(self.label)
        
        # 说明文本
        desc_label = QLabel("请选择要播放的图片类型", self)
        desc_label.setAlignment(QtCore.Qt.AlignCenter)
        desc_label.setStyleSheet("color: #a0a0a0; background: transparent;")
        main_layout.addWidget(desc_label)

        # 按钮样式
        button_style = """
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #{start_color}, stop:1 #{end_color});
                color: white;
                border: none;
                border-radius: 10px;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                font-weight: bold;
                padding: 12px 30px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #{hover_start}, stop:1 #{hover_end});
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #{pressed_start}, stop:1 #{pressed_end});
            }}
        """
        
        # 创建按钮容器
        button_layout = QHBoxLayout()
        
        # 创建确认按钮
        self.yes_button = QtWidgets.QPushButton("💀 仅显示骨骼", self)
        self.yes_button.clicked.connect(self.accept)
        self.yes_button.setStyleSheet(button_style.format(
            start_color="e94560", end_color="#c73e54",
            hover_start="#ff5a7a", hover_end="#e94560",
            pressed_start="#c73e54", pressed_end="#a83347"
        ))
        button_layout.addWidget(self.yes_button)

        # 创建取消按钮
        self.no_button = QtWidgets.QPushButton("📷 显示完整帧", self)
        self.no_button.clicked.connect(self.reject)
        self.no_button.setStyleSheet(button_style.format(
            start_color="4ecca3", end_color="#38b37a",
            hover_start="#5fd98d", hover_end="#45c985",
            pressed_start="#3db872", pressed_end="#2da866"
        ))
        button_layout.addWidget(self.no_button)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

# ===================== 详细评价弹窗类 =====================
class EvaluationDetailDialog(QDialog):
    """详细评价弹窗 - 显示完整的姿态分析评价内容"""
    def __init__(self, html_content, parent=None):
        super().__init__(parent)
        self.html_content = html_content
        self.init_ui()

    def init_ui(self):
        """初始化弹窗 UI"""
        self.setWindowTitle("📊 姿态分析详细评价")
        self.setModal(False)
        self.setGeometry(400, 200, 600, 500)
        
        # 应用深色主题
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题标签
        title_label = QLabel("🏹 姿态分析详细评价")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setStyleSheet("color: #e94560; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 评价内容显示区域（使用滚动区域）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: rgba(0, 0, 0, 0.3);
                border: 1px solid #1f4068;
                border-radius: 10px;
            }
            QScrollBar:vertical {
                background-color: #1f4068;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #4ecca3;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # 内容容器
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: transparent;")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)
        
        # 使用 QTextEdit 显示 HTML 内容
        text_browser = QTextEdit()
        text_browser.setHtml(self.html_content)
        text_browser.setReadOnly(True)
        text_browser.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0.5);
                border: none;
                border-radius: 5px;
                color: #e0e0e0;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                padding: 10px;
            }
        """)
        text_browser.setMinimumWidth(500)
        text_browser.setMinimumHeight(300)
        
        content_layout.addWidget(text_browser)
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # 关闭按钮
        button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4ecca3, stop:1 #38b37a);
                color: white;
                border: none;
                border-radius: 10px;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                font-weight: bold;
                padding: 12px 30px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5fd98d, stop:1 #45c985);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3db872, stop:1 #2da866);
            }
        """
        
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        close_button.setStyleSheet(button_style)
        close_button.setMaximumWidth(150)
        main_layout.addWidget(close_button, alignment=Qt.AlignCenter)
        
        self.setLayout(main_layout)


# ===================== 创建档案窗口类 =====================
class ImageWindow_3(QDialog):
    """创建用户档案窗口"""
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """初始化档案创建窗口 UI"""
        # 应用深色主题
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)
        
        self.setWindowTitle("📋 创建用户档案")
        self.setModal(True)
        self.setGeometry(600, 300, 400, 250)
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题标签
        title_label = QLabel("🏹 射箭姿态档案")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setStyleSheet("color: #e94560; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 姓名输入区域容器
        input_container = QFrame(self)
        input_container.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                border: 1px solid #1f4068;
                padding: 15px;
            }
        """)
        input_layout = QVBoxLayout(input_container)
        
        # 姓名输入标签
        label = QLabel("请输入您的姓名：", input_container)
        label.setFont(QFont("Microsoft YaHei", 12))
        label.setStyleSheet("color: #ffffff; background: transparent;")
        input_layout.addWidget(label)
        
        # 姓名输入框
        self.name_input = QLineEdit(input_container)
        self.name_input.setFont(QFont("Microsoft YaHei", 12))
        self.name_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.5);
                border: 1px solid #333;
                border-radius: 5px;
                color: #e0e0e0;
                padding: 8px;
            }
            QLineEdit:focus {
                border: 1px solid #4ecca3;
            }
        """)
        input_layout.addWidget(self.name_input)
        
        main_layout.addWidget(input_container)
        
        # 按钮样式
        button_style = """
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4ecca3, stop:1 #38b37a);
                color: white;
                border: none;
                border-radius: 10px;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                font-weight: bold;
                padding: 12px 30px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5fd98d, stop:1 #45c985);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3db872, stop:1 #2da866);
            }}
        """
        
        # 确认按钮
        self.ok_button = QPushButton("✅ 创建档案", self)
        self.ok_button.clicked.connect(self.close_window)
        self.ok_button.setStyleSheet(button_style)
        self.ok_button.setMaximumWidth(150)
        main_layout.addWidget(self.ok_button, alignment=Qt.AlignCenter)
        
        self.setLayout(main_layout)

    def close_window(self):
        """确认创建档案并保存到文件"""
        try:
            name = self.name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "警告", "请输入姓名！")
                return

            # 获取全局数据中的姿态参数
            front_arm_angle = GlobalData.front_arm_angle
            behind_arm_angle = GlobalData.behind_arm_angle
            score = GlobalData.score

            # 写入档案文件
            create_document.write_to_file(name, front_arm_angle, behind_arm_angle, score)
            QMessageBox.information(self, "成功", f"档案创建成功：{name}.txt")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建档案失败：{str(e)}")

# ===================== 数据分析窗口类 =====================
class ImageWindow_4(QDialog):
    """数据分析展示窗口"""
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """初始化数据分析窗口 UI"""
        # 应用深色主题
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)
        
        self.setWindowTitle("📈 数据分析报告")
        self.setModal(True)
        self.setGeometry(200, 50, 1300, 750)
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题标签
        title_label = QLabel("🏹 射箭姿态数据分析")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setStyleSheet("color: #e94560; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 输入区域容器
        input_container = QFrame(self)
        input_container.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                border: 1px solid #1f4068;
                padding: 15px;
            }
        """)
        input_layout = QHBoxLayout(input_container)
        
        # 姓名输入标签
        input_label = QLabel("请输入姓名：", input_container)
        input_label.setFont(QFont("Microsoft YaHei", 12))
        input_label.setStyleSheet("color: #ffffff; background: transparent;")
        input_layout.addWidget(input_label)
        
        # 姓名输入框
        self.name_input = QLineEdit(input_container)
        self.name_input.setFont(QFont("Microsoft YaHei", 12))
        self.name_input.setMinimumWidth(200)
        self.name_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.5);
                border: 1px solid #333;
                border-radius: 5px;
                color: #e0e0e0;
                padding: 8px;
            }
            QLineEdit:focus {
                border: 1px solid #4ecca3;
            }
        """)
        input_layout.addWidget(self.name_input)
        
        # 按钮样式
        button_style = """
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #9b59b6, stop:1 #8e44ad);
                color: white;
                border: none;
                border-radius: 8px;
                font-family: 'Microsoft YaHei';
                font-size: 12px;
                font-weight: bold;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #af6dd6, stop:1 #9b59b6);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #8e44ad, stop:1 #732d91);
            }}
        """
        
        # 提交按钮
        self.submit_button = QPushButton("🔍 加载数据", input_container)
        self.submit_button.clicked.connect(self.on_button_clicked)
        self.submit_button.setStyleSheet(button_style)
        input_layout.addWidget(self.submit_button)
        
        main_layout.addWidget(input_container)
        
        # 图表容器
        charts_container = QFrame(self)
        charts_container.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.2);
                border-radius: 10px;
                border: 1px solid #1f4068;
                padding: 15px;
            }
        """)
        charts_layout = QVBoxLayout(charts_container)
        
        # 图片展示行布局
        self.image_labels = []
        self.text_labels = []
        row_layout = QHBoxLayout()

        # 图表标题
        chart_titles = ["前手臂弯曲角度", "后手臂弯曲角度", "综合评分"]
        chart_colors = ["#4ecca3", "#e94560", "#f39c12"]
        
        # 创建 3 列图片 + 文本展示区域
        for i in range(3):
            column_layout = QVBoxLayout()
            
            # 图表标题
            chart_title = QLabel(chart_titles[i])
            chart_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            chart_title.setStyleSheet(f"color: {chart_colors[i]}; background: transparent;")
            chart_title.setAlignment(Qt.AlignCenter)
            column_layout.addWidget(chart_title)
            
            # 图片显示标签容器
            img_container = QFrame()
            img_container.setStyleSheet("""
                QFrame {
                    background-color: rgba(0, 0, 0, 0.5);
                    border-radius: 8px;
                    border: 1px solid #333;
                }
            """)
            img_layout = QVBoxLayout(img_container)
            
            # 图片显示标签
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setFixedSize(380, 350)
            image_label.setText("📊 点击加载数据")
            image_label.setStyleSheet("color: #666; background: transparent;")
            img_layout.addWidget(image_label)
            self.image_labels.append(image_label)
            column_layout.addWidget(img_container)
            
            # 文本说明标签
            text_label = QLabel()
            text_label.setAlignment(Qt.AlignCenter)
            text_label.setFixedHeight(60)
            text_label.setStyleSheet("color: #a0a0a0; background: transparent;")
            self.text_labels.append(text_label)
            column_layout.addWidget(text_label)
            
            row_layout.addLayout(column_layout)

        charts_layout.addLayout(row_layout)
        main_layout.addWidget(charts_container)
        
        self.setLayout(main_layout)

    def on_button_clicked(self):
        """加载并处理指定用户的数据分析"""
        try:
            name = self.name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "警告", "请输入姓名！")
                return

            # 检查档案文件是否存在（使用数据路径）
            txt_path = get_data_path(os.path.join("document", f"{name}.txt"))
            if not os.path.exists(txt_path):
                QMessageBox.warning(self, "警告", f"档案文件不存在：{txt_path}")
                return

            # 处理数据并生成图表
            data_2.plot_data_from_txt(txt_path)
            self.show_images_and_text()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理数据时发生错误：{e}")

    def show_images_and_text(self):
        """显示数据分析图表和说明文本"""
        try:
            # 图表图片路径和对应的说明文本（使用数据路径，适配打包后的exe环境）
            image_paths = [
                get_data_path("group1.png"),
                get_data_path("group2.png"),
                get_data_path("group3.png")
            ]
            text_descriptions = [
                "前手臂弯曲角度。",
                "后手臂弯曲角度。",
                "前后手臂评分。"
            ]

            # 加载并显示每张图片和对应文本
            for label, pixmap_path, text in zip(self.image_labels, image_paths, text_descriptions):
                if os.path.exists(pixmap_path):
                    pixmap = QPixmap(pixmap_path)
                    if not pixmap.isNull():
                        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))
                    else:
                        label.setText(f"图片损坏：{os.path.basename(pixmap_path)}")
                else:
                    label.setText(f"图片不存在：{os.path.basename(pixmap_path)}")
                self.text_labels[self.image_labels.index(label)].setText(text)
            
            self.update()  # 刷新 UI
        except Exception as e:
            QMessageBox.warning(self, "警告", f"显示图片失败：{str(e)}")

# ===================== 视频来源选择对话框 =====================
class VideoSourceDialog(QDialog):
    """视频来源选择对话框 - 让用户选择上传文件或摄像头录制"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """初始化对话框 UI"""
        self.setWindowTitle("📹 选择视频来源")
        self.setModal(True)
        self.setFixedSize(450, 280)
        
        # 应用深色主题
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题标签
        title_label = QLabel("🤔 选择视频来源", self)
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #ffffff; background: transparent; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # 说明文本
        desc_label = QLabel("请选择您要使用的视频来源方式", self)
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("color: #a0a0a0; background: transparent; font-size: 13px;")
        main_layout.addWidget(desc_label)
        
        # 选项容器
        options_container = QFrame(self)
        options_container.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                border: 1px solid #1f4068;
                padding: 15px;
            }
        """)
        options_layout = QVBoxLayout(options_container)
        options_layout.setSpacing(15)
        
        # 选项 1：上传文件
        file_option = QFrame(options_container)
        file_option.setStyleSheet("""
            QFrame {
                background-color: rgba(78, 204, 163, 0.1);
                border-radius: 8px;
                border: 1px solid #4ecca3;
                padding: 10px;
            }
        """)
        file_layout = QHBoxLayout(file_option)
        
        file_icon = QLabel("📁", file_option)
        file_icon.setFont(QFont("Segoe UI Emoji", 24))
        file_layout.addWidget(file_icon)
        
        file_text = QVBoxLayout()
        file_title = QLabel("上传视频文件", file_option)
        file_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        file_title.setStyleSheet("color: #4ecca3;")
        file_desc = QLabel("支持 MP4、AVI、MKV、MOV 等格式", file_option)
        file_desc.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        file_text.addWidget(file_title)
        file_text.addWidget(file_desc)
        file_layout.addLayout(file_text)
        
        options_layout.addWidget(file_option)
        
        # 选项 2：摄像头录制
        camera_option = QFrame(options_container)
        camera_option.setStyleSheet("""
            QFrame {
                background-color: rgba(233, 69, 96, 0.1);
                border-radius: 8px;
                border: 1px solid #e94560;
                padding: 10px;
            }
        """)
        camera_layout = QHBoxLayout(camera_option)
        
        camera_icon = QLabel("📷", camera_option)
        camera_icon.setFont(QFont("Segoe UI Emoji", 24))
        camera_layout.addWidget(camera_icon)
        
        camera_text = QVBoxLayout()
        camera_title = QLabel("摄像头录制", camera_option)
        camera_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        camera_title.setStyleSheet("color: #e94560;")
        camera_desc = QLabel("使用摄像头实时录制射箭动作", camera_option)
        camera_desc.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        camera_text.addWidget(camera_title)
        camera_text.addWidget(camera_desc)
        camera_layout.addLayout(camera_text)
        
        options_layout.addWidget(camera_option)
        
        main_layout.addWidget(options_container)
        
        # 按钮样式
        button_style = """
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #{start_color}, stop:1 #{end_color});
                color: white;
                border: none;
                border-radius: 10px;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                font-weight: bold;
                padding: 12px 30px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #{hover_start}, stop:1 #{hover_end});
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #{pressed_start}, stop:1 #{pressed_end});
            }}
        """
        
        # 按钮容器
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        
        # 上传文件按钮
        self.file_button = QPushButton("📁 上传文件", self)
        self.file_button.clicked.connect(self.accept_file)
        self.file_button.setStyleSheet(button_style.format(
            start_color="4ecca3", end_color="#38b37a",
            hover_start="#5fd98d", hover_end="#45c985",
            pressed_start="#3db872", pressed_end="#2da866"
        ))
        button_layout.addWidget(self.file_button)
        
        # 摄像头录制按钮
        self.camera_button = QPushButton("📷 摄像头录制", self)
        self.camera_button.clicked.connect(self.accept_camera)
        self.camera_button.setStyleSheet(button_style.format(
            start_color="e94560", end_color="#c73e54",
            hover_start="#ff5a7a", hover_end="#e94560",
            pressed_start="#c73e54", pressed_end="#a83347"
        ))
        button_layout.addWidget(self.camera_button)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def accept_file(self):
        """用户选择上传文件"""
        self.done(QtWidgets.QDialog.Accepted)
    
    def accept_camera(self):
        """用户选择摄像头录制"""
        self.done(QtWidgets.QDialog.Rejected)


# ===================== 摄像头选择对话框 =====================
class CameraSelectDialog(QDialog):
    """摄像头选择对话框 - 让用户选择要使用的摄像头"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_index = 0
        self.init_ui()

    def init_ui(self):
        """初始化对话框 UI"""
        self.setWindowTitle("📷 选择摄像头")
        self.setModal(True)
        self.setFixedSize(400, 250)
        
        # 应用深色主题
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题标签
        title_label = QLabel("📷 选择摄像头设备", self)
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #ffffff; background: transparent;")
        main_layout.addWidget(title_label)
        
        # 说明文本
        desc_label = QLabel("请选择要使用的摄像头设备", self)
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("color: #a0a0a0; background: transparent;")
        main_layout.addWidget(desc_label)
        
        # 摄像头选择下拉框
        self.camera_combo = QtWidgets.QComboBox(self)
        self.camera_combo.setFont(QFont("Microsoft YaHei", 12))
        self.camera_combo.setMinimumHeight(40)
        self.camera_combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                border: 2px solid #666;
                border-radius: 8px;
                color: #000000;
                padding: 10px 15px;
                min-height: 30px;
            }
            QComboBox::drop-down {
                border: none;
                width: 40px;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
                width: 0;
                height: 0;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 8px solid #666;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                border: 2px solid #666;
                border-radius: 8px;
                color: #000000;
                selection-background-color: #4ecca3;
                selection-color: #ffffff;
                padding: 5px;
                min-height: 30px;
            }
            QComboBox QAbstractItemView::item {
                min-height: 30px;
                padding: 8px;
            }
        """)
        
        # 检测可用的摄像头
        self.detect_cameras()
        
        main_layout.addWidget(self.camera_combo)
        
        # 按钮样式
        button_style = """
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4ecca3, stop:1 #38b37a);
                color: white;
                border: none;
                border-radius: 10px;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                font-weight: bold;
                padding: 12px 30px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5fd98d, stop:1 #45c985);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3db872, stop:1 #2da866);
            }}
        """
        
        # 确认按钮
        self.ok_button = QPushButton("✅ 确认选择", self)
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setStyleSheet(button_style)
        self.ok_button.setMaximumWidth(150)
        main_layout.addWidget(self.ok_button, alignment=Qt.AlignCenter)
        
        self.setLayout(main_layout)
    
    def detect_cameras(self):
        """检测系统中可用的摄像头"""
        cameras_found = False
        
        # 尝试检测前 5 个摄像头索引
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.camera_combo.addItem(f"摄像头 {i}", i)
                    cameras_found = True
                cap.release()
            else:
                cap.release()
        
        if not cameras_found:
            # 如果没有检测到摄像头，添加默认选项
            self.camera_combo.addItem("默认摄像头 (索引 0)", 0)
            self.camera_combo.addItem("外部摄像头 (索引 1)", 1)
    
    def accept(self):
        """用户确认选择"""
        self.camera_index = self.camera_combo.currentData()
        super().accept()


# ===================== 录制控制窗口 =====================
class RecordWindow(QDialog):
    """摄像头录制控制窗口"""
    recording_finished = pyqtSignal(str)  # 录制完成信号，返回视频路径
    
    def __init__(self, cap, camera_index, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.camera_index = camera_index
        self.is_recording = False
        self.is_paused = False
        self.record_timer = QTimer()
        self.recording_time = 0
        self.video_writer = None
        self.output_path = ""
        self.frames = []
        self.init_ui()

    def init_ui(self):
        """初始化窗口 UI"""
        self.setWindowTitle("📷 摄像头录制")
        self.setModal(False)
        self.setGeometry(100, 100, 1000, 700)
        
        # 应用深色主题
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题标签
        title_label = QLabel("📷 摄像头录制 - 请保持射箭动作在画面中")
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #e94560; background: transparent;")
        main_layout.addWidget(title_label)
        
        # 视频显示区域
        self.video_display = QLabel(self)
        self.video_display.setMinimumSize(800, 600)
        self.video_display.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border-radius: 10px;
                border: 2px solid #1f4068;
            }
        """)
        self.video_display.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_display)
        
        # 控制按钮容器
        control_container = QFrame(self)
        control_container.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                border: 1px solid #1f4068;
                padding: 15px;
            }
        """)
        control_layout = QHBoxLayout(control_container)
        control_layout.setSpacing(20)
        control_layout.setAlignment(Qt.AlignCenter)
        
        # 录制按钮样式
        record_btn_style = """
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #{start_color}, stop:1 #{end_color});
                color: white;
                border: none;
                border-radius: 50px;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                font-weight: bold;
                padding: 15px 30px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #{hover_start}, stop:1 #{hover_end});
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #{pressed_start}, stop:1 #{pressed_end});
            }}
        """
        
        # 开始/停止录制按钮
        self.record_button = QPushButton("🔴 开始录制", control_container)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setStyleSheet(record_btn_style.format(
            start_color="e94560", end_color="#c73e54",
            hover_start="#ff5a7a", hover_end="#e94560",
            pressed_start="#c73e54", pressed_end="#a83347"
        ))
        control_layout.addWidget(self.record_button)
        
        # 时间显示
        self.time_label = QLabel("00:00", control_container)
        self.time_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.time_label.setStyleSheet("color: #4ecca3; background: transparent; min-width: 100px;")
        self.time_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.time_label)
        
        main_layout.addWidget(control_container)
        
        # 提示信息
        self.hint_label = QLabel("💡 提示：点击开始录制，再次点击停止录制并自动处理视频", self)
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setStyleSheet("color: #a0a0a0; background: transparent; font-size: 12px;")
        main_layout.addWidget(self.hint_label)
        
        self.setLayout(main_layout)
        
        # 启动摄像头预览
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(30)
        
        # 录制定时器
        self.record_timer.timeout.connect(self.update_recording_time)
    
    def update_preview(self):
        """更新摄像头预览画面"""
        try:
            ret, frame = self.cap.read()
            if ret:
                # 如果是录制中，保存帧
                if self.is_recording and not self.is_paused:
                    self.frames.append(frame.copy())
                
                # 转换颜色空间并显示
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = channel * width
                q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                self.video_display.setPixmap(pixmap.scaled(self.video_display.size(), Qt.KeepAspectRatio))
        except Exception as e:
            print(f"预览更新失败：{e}")
    
    def toggle_recording(self):
        """切换录制状态"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """开始录制"""
        self.is_recording = True
        self.frames = []
        self.recording_time = 0
        self.record_button.setText("⏹️ 停止录制")
        self.record_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4ecca3, stop:1 #38b37a);
                color: white;
                border: none;
                border-radius: 50px;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                font-weight: bold;
                padding: 15px 30px;
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5fd98d, stop:1 #45c985);
            }
        """)
        self.hint_label.setText("💡 录制中... 请完成射箭动作后点击停止录制")
        self.hint_label.setStyleSheet("color: #e94560; background: transparent; font-size: 12px;")
        self.record_timer.start(1000)
    
    def stop_recording(self):
        """停止录制并保存视频"""
        self.is_recording = False
        self.record_timer.stop()
        self.record_button.setText("🔴 开始录制")
        self.record_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e94560, stop:1 #c73e54);
                color: white;
                border: none;
                border-radius: 50px;
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                font-weight: bold;
                padding: 15px 30px;
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff5a7a, stop:1 #e94560);
            }
        """)
        self.hint_label.setText("💡 正在保存视频，请稍候...")
        self.hint_label.setStyleSheet("color: #4ecca3; background: transparent; font-size: 12px;")
        
        # 保存录制的视频
        if len(self.frames) > 0:
            self.save_video()
        else:
            QMessageBox.warning(self, "警告", "未录制到任何内容！")
            self.hint_label.setText("💡 提示：点击开始录制，再次点击停止录制并自动处理视频")
            self.hint_label.setStyleSheet("color: #a0a0a0; background: transparent; font-size: 12px;")
    
    def update_recording_time(self):
        """更新录制时间显示"""
        self.recording_time += 1
        minutes = self.recording_time // 60
        seconds = self.recording_time % 60
        self.time_label.setText(f"{minutes:02d}:{seconds:02d}")
    
    def save_video(self):
        """保存录制的视频到文件"""
        try:
            if len(self.frames) == 0:
                QMessageBox.warning(self, "警告", "未录制到任何内容！")
                return
            
            # 获取视频帧参数
            height, width, _ = self.frames[0].shape
            fps = 30  # 假设 30fps
            
            # 生成输出文件名（使用数据路径，确保在exe同级目录下）
            timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
            output_folder = get_data_path("original")
            os.makedirs(output_folder, exist_ok=True)
            self.output_path = os.path.join(output_folder, f"recorded_{timestamp}.mp4")
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            # 写入所有帧
            for frame in self.frames:
                self.video_writer.write(frame)
            
            self.video_writer.release()
            
            # 设置全局数据
            GlobalData.file_path = self.output_path
            GlobalData.source_type = "camera"
            
            # 显示成功消息
            QMessageBox.information(self, "成功", f"视频已保存：{self.output_path}\n\n即将开始处理视频...")
            
            # 关闭录制窗口
            self.close()
            
            # 通知主窗口开始处理
            if self.parent():
                self.parent().pushButton.setText("正在处理...")
                self.parent().pushButton.setEnabled(False)
                
                # 显示进度对话框
                self.parent().progress_dialog = ProgressDialog(self.parent())
                self.parent().progress_dialog.show()
                
                # 启动处理线程（使用数据路径，确保在exe同级目录下）
                output_folder = get_data_path("output_frames")
                os.makedirs(output_folder, exist_ok=True)
                
                self.parent().process_thread = VideoProcessThread(self.output_path, output_folder)
                self.parent().process_thread.progress_signal.connect(self.parent().progress_dialog.update_progress)
                self.parent().process_thread.finish_signal.connect(self.parent().on_process_finished)
                self.parent().process_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存视频失败：{str(e)}")
            self.hint_label.setText("💡 提示：点击开始录制，再次点击停止录制并自动处理视频")
            self.hint_label.setStyleSheet("color: #a0a0a0; background: transparent; font-size: 12px;")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.preview_timer.stop()
        self.record_timer.stop()
        if self.video_writer:
            self.video_writer.release()
        self.cap.release()
        event.accept()


# ===================== 程序入口 =====================
if __name__ == "__main__":
    """程序主入口"""
    # 启用高 DPI 缩放支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    # 创建应用程序实例
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("射箭姿态评估系统")
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序事件循环
    sys.exit(app.exec_())