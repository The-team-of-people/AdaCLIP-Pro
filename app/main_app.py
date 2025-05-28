import sys
import os
import json
import traceback
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QLineEdit, QFileDialog, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QSplitter, QStackedWidget, QCheckBox, QMessageBox, QHeaderView, QStatusBar,
    QFrame, QDialog, QGroupBox, QGridLayout, QProgressBar, QToolButton, QAction, QMenu
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve, QTimer, QPoint
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap, QCursor
from icon_and_font_updates import IconManager
# 引入新样式表
from resources.stylesheet import get_modern_style, get_nav_button_style
# 引入接口
from api.adaclip_api import (
    uploadDirectory, videoQuery, scanAndCheckDictory,
    addVideoToDictory, deleteVideo
)

BASE_DIR = os.path.dirname(__file__)  # 获取当前脚本所在目录
SETTINGS_FILE = os.path.join(BASE_DIR, "user_data", "settings.json")  # 相对路径
HISTORY_FILE = os.path.join(BASE_DIR, "user_data", "history.json")  # 相对路径


def get_modern_style():
    return """
    /* 全局字体设置 */
    * {
        font-family: "Microsoft YaHei", "Segoe UI", "Noto Sans CJK SC", sans-serif;
        font-size: 14px;
    }

    QMainWindow {
        background-color: #f8f9fa;
    }

    /* 按钮样式 */
    QPushButton {
        background-color: #0d6efd;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    QPushButton:hover {
        background-color: #0b5ed7;
        transition: background-color 0.3s;
    }

    QPushButton:pressed {
        background-color: #0a58ca;
        transform: translateY(1px);
    }

    QPushButton:disabled {
        background-color: #6c757d;
        opacity: 0.65;
    }

    /* 输入框样式 */
    QLineEdit {
        padding: 10px 14px;
        border: 2px solid #dee2e6;
        border-radius: 6px;
        background-color: white;
        selection-background-color: #0d6efd;
        selection-color: white;
        font-size: 14px;
    }

    QLineEdit:focus {
        border-color: #86b7fe;
        outline: 0;
        box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25);
    }

    /* 表格样式 */
    QTableWidget {
        background-color: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        gridline-color: #f1f3f5;
    }

    QTableWidget::item {
        padding: 10px;
        border-bottom: 1px solid #f1f3f5;
        font-size: 14px;
    }

    QTableWidget::item:selected {
        background-color: #e7f1ff;
        color: #000;
    }

    QHeaderView::section {
        background-color: #f8f9fa;
        padding: 12px;
        border: none;
        border-bottom: 2px solid #dee2e6;
        font-weight: bold;
        color: #495057;
        font-size: 15px;
    }

    /* 列表样式 */
    QListWidget {
        background-color: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        outline: none;
    }

    QListWidget::item {
        padding: 10px 14px;
        border-bottom: 1px solid #f1f3f5;
        font-size: 14px;
    }

    QListWidget::item:selected {
        background-color: #e7f1ff;
        color: #000;
    }

    QListWidget::item:hover {
        background-color: #f8f9fa;
    }

    /* 状态栏样式 */
    QStatusBar {
        background-color: #f8f9fa;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        padding: 6px;
        font-size: 13px;
    }

    /* 进度条样式 */
    QProgressBar {
        border: none;
        border-radius: 4px;
        background-color: #e9ecef;
        text-align: center;
        color: white;
        font-size: 12px;
        font-weight: 500;
    }

    QProgressBar::chunk {
        background-color: #0d6efd;
        border-radius: 4px;
    }

    /* 滚动条样式 */
    QScrollBar:vertical {
        border: none;
        background-color: #f8f9fa;
        width: 10px;
        margin: 0px;
    }

    QScrollBar::handle:vertical {
        background-color: #dee2e6;
        border-radius: 5px;
        min-height: 30px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #adb5bd;
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }

    /* 左侧面板样式 */
    #leftPanel {
        background-color: #0d6efd;
        border-right: none;
    }

    #leftPanel QPushButton {
        background-color: transparent;
        color: white;
        text-align: left;
        padding: 14px 24px;
        border-radius: 0;
        font-size: 15px;
        font-weight: 500;
    }

    #leftPanel QPushButton:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }

    #leftPanel QLabel {
        color: white;
        font-size: 24px;
        font-weight: bold;
        padding: 24px;
    }
    """

def normalize(path: str) -> str:
    # 统一大小写、去掉末尾分隔符、正斜杠
    p = os.path.normcase(os.path.normpath(path))
    return p

def open_path(path):
    """统一的文件/文件夹打开函数"""
    try:
        if os.path.exists(path):
            os.startfile(path)
        else:
            QMessageBox.warning(None, "提示", f"路径不存在: {path}")
    except Exception as e:
        QMessageBox.warning(None, "错误", f"打开失败: {str(e)}")

class SearchThread(QThread):
    search_complete = pyqtSignal(list)  # 发送搜索结果
    status_update = pyqtSignal(str)     # 发送状态更新

    def __init__(self, search_dirs, search_text):
        super().__init__()
        self.search_dirs = search_dirs
        self.search_text = search_text

    def run(self):
        try:
            self.status_update.emit("正在搜索...")
            # 调用API进行视频搜索
            api_results = videoQuery(self.search_dirs, self.search_text)
            self.search_complete.emit(api_results)
        except Exception as e:
            self.status_update.emit(f"搜索失败: {str(e)}")
            self.search_complete.emit([])  # 发送空结果

class SearchingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("搜索中")
        self.setFixedSize(200, 100)
        self.setStyleSheet(get_modern_style())

        layout = QVBoxLayout(self)
        label = QLabel("正在进行搜索，请稍候...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

class UploadThread(QThread):
    progress_updated = pyqtSignal(str, int, str)  # 文件夹路径, 进度百分比, 状态消息
    task_completed = pyqtSignal(str, str)  # 文件夹路径, 最终结果

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        # 启动时主动发送0%进度，让前端进度条立即刷新
        self.progress_updated.emit(self.folder, 0, "开始处理...")

        def progress_callback(folder, percent, message):
            # 进度回调由后端调用，线程安全，直接emit
            self.progress_updated.emit(folder, percent, message)

        try:
            result = uploadDirectory(self.folder, progress_callback=progress_callback)

            if result.get("success"):
                # 处理结束时发100%
                self.progress_updated.emit(self.folder, 100, "处理完成")
                self.task_completed.emit(self.folder, result.get('msg', '处理完成'))
            else:
                self.progress_updated.emit(self.folder, 100, f"处理失败: {result.get('msg', '未知错误')}")
                self.task_completed.emit(self.folder, result.get('msg', '未知错误'))

        except Exception as e:
            print(f"UploadThread 异常: {type(e).__name__}, 信息: {str(e)}")
            traceback.print_exc()
            self.progress_updated.emit(self.folder, 100, f"处理异常: {str(e)}")
            self.task_completed.emit(self.folder, f"处理异常: {str(e)}")


class ProgressMonitor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("上传进度监控")
        self.setMinimumWidth(600)
        self.setStyleSheet(get_modern_style())
        self.setWindowIcon(parent.icon_manager.get_icon())  # 设置对话框图标

        self.layout = QVBoxLayout(self)
        self.task_widgets = {}  # 存储每个任务的UI组件

        # 添加关闭按钮
        btn_layout = QHBoxLayout()
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.hide)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)

        self.layout.addLayout(btn_layout)

    def add_task(self, folder):
        # 为每个任务创建一个分组框
        group_box = QGroupBox(f"处理: {os.path.basename(folder)}")
        group_box.setStyleSheet("QGroupBox { border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }")

        task_layout = QGridLayout()

        # 显示文件夹路径
        path_label = QLabel(f"路径: {folder}")
        path_label.setWordWrap(True)

        # 进度条
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)

        # 状态标签
        status_label = QLabel("等待中...")

        # 添加到布局
        task_layout.addWidget(QLabel("状态:"), 0, 0)
        task_layout.addWidget(status_label, 0, 1)
        task_layout.addWidget(QLabel("进度:"), 1, 0)
        task_layout.addWidget(progress_bar, 1, 1)
        task_layout.addWidget(path_label, 2, 0, 1, 2)

        group_box.setLayout(task_layout)
        self.layout.insertWidget(self.layout.count() - 1, group_box)  # 插入到关闭按钮之前

        # 保存UI组件引用
        self.task_widgets[folder] = {
            "group_box": group_box,
            "progress_bar": progress_bar,
            "status_label": status_label
        }

        self.show()

    def update_progress(self, folder, progress, status):
        folder = normalize(folder)
        print(f"[DEBUG] update_progress 收到: {folder=} {progress=} {status=}")
        if folder in self.task_widgets:
            self.task_widgets[folder]["progress_bar"].setValue(progress)
            self.task_widgets[folder]["status_label"].setText(status)
            QApplication.processEvents()  # 关键：强制刷新UI

    def complete_task(self, folder, result):
        folder = normalize(folder)
        if folder in self.task_widgets:
            self.task_widgets[folder]["status_label"].setText(f"✓ {result}")
            self.task_widgets[folder]["progress_bar"].setValue(100)
            # 设置为绿色完成状态
            self.task_widgets[folder]["progress_bar"].setStyleSheet("""
                QProgressBar {
                    border: 1px solid grey;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                }
            """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.icon_manager = IconManager()

        self.setWindowTitle("文影灵搜")
        self.setWindowIcon(self.icon_manager.get_icon())
        self.resize(1200, 800)  # 适当增大默认窗口尺寸
        self.selected_search_dirs = []
        self.processed_dirs = []
        self.upload_threads = {}  # 存储所有上传线程
        self.progress_monitor = None  # 进度监控窗口
        self.search_thread = None  # 搜索线程
        self.searching_dialog = None  # 搜索提示弹窗

        # 确保图标存在
        self._ensure_icons_exist()

        # 设置全局样式
        from resources.stylesheet import get_modern_style
        self.setStyleSheet(get_modern_style())

        # 主分割器
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧导航栏
        self.left_widget = QFrame()
        self.left_widget.setObjectName("leftPanel")

        left_layout = QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(0, 20, 0, 20)
        left_layout.setSpacing(15)

        # 应用名称和logo
        logo_layout = QHBoxLayout()
        app_icon = QLabel()
        app_icon.setPixmap(self.icon_manager.get_icon().pixmap(32, 32))
        app_icon.setAlignment(Qt.AlignCenter)
        app_icon.setStyleSheet("background-color: transparent;")

        self.app_name = QLabel("文影灵搜")
        self.app_name.setAlignment(Qt.AlignCenter)
        self.app_name.setStyleSheet("font-size: 22px; font-weight: bold; padding: 10px; background-color: transparent;")

        logo_layout.addStretch(1)
        logo_layout.addWidget(app_icon)
        logo_layout.addWidget(self.app_name)
        logo_layout.addStretch(1)

        # 导航分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: rgba(255, 255, 255, 0.2); margin: 5px 15px;")

        # 导航按钮
        nav_layout = QVBoxLayout()
        self.btn_main = QPushButton("  主页")
        self.btn_main.setCheckable(True)
        self.btn_main.setChecked(True)
        self.btn_main.setIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "home.png")))
        self.btn_main.setIconSize(QSize(20, 20))

        self.btn_settings = QPushButton("  设置")
        self.btn_settings.setCheckable(True)
        self.btn_settings.setIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "settings.png")))
        self.btn_settings.setIconSize(QSize(20, 20))

        self.btn_history = QPushButton("  历史记录")
        self.btn_history.setCheckable(True)
        self.btn_history.setIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "history.png")))
        self.btn_history.setIconSize(QSize(20, 20))

        for btn in [self.btn_main, self.btn_settings, self.btn_history]:
            btn.setStyleSheet(get_nav_button_style())
            btn.setMinimumHeight(50)

        self.btn_main.clicked.connect(self.show_main)
        self.btn_settings.clicked.connect(self.show_settings)
        self.btn_history.clicked.connect(self.show_history)

        nav_layout.addWidget(self.btn_main)
        nav_layout.addWidget(self.btn_settings)
        nav_layout.addWidget(self.btn_history)
        nav_layout.setSpacing(8)
        nav_layout.addStretch()

        left_layout.addLayout(logo_layout)
        left_layout.addWidget(separator)
        left_layout.addLayout(nav_layout)
        left_layout.addStretch()

        # 版本信息
        version_label = QLabel("V 1.0.0")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: rgba(255,255,255,0.4); font-size: 12px;")
        left_layout.addWidget(version_label)

        # 右侧内容区
        self.stacked_widget = QStackedWidget()
        self.main_page = self.build_main_page()
        self.settings_page = self.build_settings_page()
        self.history_page = self.build_history_page()

        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.settings_page)
        self.stacked_widget.addWidget(self.history_page)

        main_layout.addWidget(self.left_widget, 1)
        main_layout.addWidget(self.stacked_widget, 5)

        self.setCentralWidget(main_widget)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("欢迎使用文影灵搜")

    def _ensure_icons_exist(self):
        """确保必要的图标文件存在"""
        icons_dir = os.path.join(BASE_DIR, "resources", "icons")
        os.makedirs(icons_dir, exist_ok=True)

        # 创建基本图标 - 如果不存在的话
        icons = {
            "home.png": "#0d6efd",
            "settings.png": "#0d6efd",
            "history.png": "#0d6efd",
            "search.png": "#0d6efd",
            "upload.png": "#198754",
            "select.png": "#0d6efd",
            "folder.png": "#6c757d",
            "delete.png": "#dc3545",
            "next.png": "#0d6efd",
            "prev.png": "#0d6efd"
        }

        # 检查图标是否存在，不存在则跳过(避免重复创建和依赖PIL)
        for icon_name in icons:
            icon_path = os.path.join(icons_dir, icon_name)
            if not os.path.exists(icon_path):
                print(f"警告: 缺少图标文件 {icon_name}，请确保resources/icons目录中包含所需的图标文件。")

    def build_main_page(self):
        widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # 顶部标题区域
        title_label = QLabel("视频内容搜索")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #212529; margin-bottom: 10px;")

        # 创建搜索卡片
        search_card = QWidget()
        search_card.setObjectName("searchCard")
        search_card.setStyleSheet("""
            #searchCard {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #dee2e6;
            }
        """)

        search_layout = QVBoxLayout(search_card)
        search_layout.setContentsMargins(25, 25, 25, 25)

        # 上传和选择按钮区域
        action_layout = QHBoxLayout()

        # 使用带有图标的按钮
        self.btn_upload = QPushButton(" 上传文件夹")
        self.btn_upload.setIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "upload.png")))
        self.btn_upload.setIconSize(QSize(18, 18))
        self.btn_upload.setStyleSheet("background-color: #198754;")  # 绿色按钮
        self.btn_upload.setMinimumHeight(40)

        self.btn_select = QPushButton(" 选择搜索范围")
        self.btn_select.setIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "select.png")))
        self.btn_select.setIconSize(QSize(18, 18))
        self.btn_select.setMinimumHeight(40)

        action_layout.addWidget(self.btn_upload)
        action_layout.addWidget(self.btn_select)
        action_layout.addStretch()

        self.btn_upload.clicked.connect(self.upload_folder)
        self.btn_select.clicked.connect(self.select_search_dirs)

        # 创建搜索输入区域
        search_frame = QFrame()
        search_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #dee2e6;
                border-radius: 24px;
                background-color: #f8f9fa;
                padding: 5px;
                margin-top: 15px;
            }
        """)

        search_input_layout = QHBoxLayout(search_frame)
        search_input_layout.setContentsMargins(15, 5, 5, 5)
        search_input_layout.setSpacing(10)

        # 搜索图标
        search_icon = QLabel()
        search_pixmap = QPixmap(os.path.join(BASE_DIR, "resources", "icons", "search.png"))
        search_icon.setPixmap(search_pixmap.scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 搜索输入框
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("输入内容进行视频搜索...")
        self.input_text.setStyleSheet("""
            QLineEdit {
                border: none;
                background-color: transparent;
                font-size: 16px;
                padding: 8px;
            }
        """)
        self.input_text.setMinimumHeight(40)

        # 搜索按钮
        self.btn_search = QPushButton("搜索")
        self.btn_search.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                border-radius: 20px;
                padding: 10px 25px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
        """)
        self.btn_search.setMinimumHeight(40)
        self.btn_search.clicked.connect(self.do_search)

        # 把组件添加到搜索栏布局
        search_input_layout.addWidget(search_icon)
        search_input_layout.addWidget(self.input_text, 1)
        search_input_layout.addWidget(self.btn_search)

        # 把所有组件添加到搜索卡片
        search_layout.addLayout(action_layout)
        search_layout.addWidget(search_frame)

        # 添加搜索卡片到主布局
        main_layout.addWidget(title_label)
        main_layout.addWidget(search_card)

        # 结果区域标题
        results_label = QLabel("搜索结果")
        results_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #495057; margin-top: 10px;")

        # 文件展示表格 - 美化表格样式
        self.table_files = QTableWidget(0, 4)  # 增加为4列，添加序号列
        self.table_files.setHorizontalHeaderLabels(["序号", "文件名", "大小", "修改时间"])
        self.table_files.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_files.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_files.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # 仅文件名列自动伸缩
        self.table_files.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 序号列自适应内容宽度
        self.table_files.setStyleSheet("""
            QTableWidget {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 5px;
                background-color: white;
                gridline-color: #f1f3f5;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 15px 10px;
                border: none;
                border-bottom: 2px solid #dee2e6;
                font-weight: bold;
                color: #495057;
            }
            QTableWidget::item {
                padding: 12px 6px;
                border-bottom: 1px solid #f1f3f5;
            }
            QTableWidget::item:selected {
                background-color: #e7f1ff;
                color: #000;
            }
        """)
        self.table_files.doubleClicked.connect(self.open_selected_file)
        self.table_files.setAlternatingRowColors(True)  # 交替行颜色

        # 分页控制区
        pagination_layout = QHBoxLayout()
        pagination_layout.setContentsMargins(0, 10, 0, 0)

        self.page_info_label = QLabel("第 0/0 页，共 0 条记录")

        self.btn_prev_page = QPushButton("上一页")
        self.btn_prev_page.setIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "prev.png")))
        self.btn_prev_page.setIconSize(QSize(16, 16))
        self.btn_prev_page.clicked.connect(self.prev_page)
        self.btn_prev_page.setEnabled(False)

        self.btn_next_page = QPushButton("下一页")
        self.btn_next_page.setIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "next.png")))
        self.btn_next_page.setIconSize(QSize(16, 16))
        self.btn_next_page.clicked.connect(self.next_page)
        self.btn_next_page.setEnabled(False)

        pagination_layout.addWidget(self.page_info_label)
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.btn_prev_page)
        pagination_layout.addWidget(self.btn_next_page)

        # 添加结果标题、表格和分页控制到主布局
        main_layout.addWidget(results_label)
        main_layout.addWidget(self.table_files, 1)  # 表格可伸缩
        main_layout.addLayout(pagination_layout)

        # 分页相关变量
        self.page_size = 10  # 每页显示10条记录
        self.current_page = 1
        self.total_pages = 1
        self.all_results = []  # 存储所有搜索结果

        widget.setLayout(main_layout)
        return widget

    def build_settings_page(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # 顶部标题区域
        title_label = QLabel("应用设置")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #212529; margin-bottom: 10px;")

        # 创建设置卡片
        settings_card = QWidget()
        settings_card.setObjectName("settingsCard")
        settings_card.setStyleSheet("""
            #settingsCard {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #dee2e6;
            }
        """)

        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(25, 25, 25, 25)
        settings_layout.setSpacing(20)

        # 帧保存路径设置区
        path_group = QGroupBox("视频帧保存路径")
        path_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                color: #495057;
                font-size: 16px;
            }
        """)

        path_layout = QVBoxLayout(path_group)
        path_layout.setContentsMargins(15, 20, 15, 15)
        path_layout.setSpacing(15)

        # 帧保存路径说明
        path_desc = QLabel("设置视频帧图像的默认保存路径，用于存储视频处理过程中生成的帧图像文件。")
        path_desc.setWordWrap(True)
        path_desc.setStyleSheet("color: #6c757d; font-size: 14px;")

        # 帧路径选择区
        folder_layout = QHBoxLayout()
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("请选择文件夹路径...")
        self.folder_path_edit.setStyleSheet("""
            QLineEdit {
                border: 2px solid #dee2e6;
                border-radius: 6px;
                padding: 10px;
                background-color: #f8f9fa;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #86b7fe;
            }
        """)
        self.folder_path_edit.setMinimumHeight(40)

        btn_choose = QPushButton("选择路径")
        btn_choose.setIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "folder.png")))
        btn_choose.setIconSize(QSize(16, 16))
        btn_choose.setMinimumHeight(40)
        btn_choose.clicked.connect(self.choose_settings_folder)

        folder_layout.addWidget(self.folder_path_edit, 3)
        folder_layout.addWidget(btn_choose, 1)

        path_layout.addWidget(path_desc)
        path_layout.addLayout(folder_layout)

        # 其他设置项 (示例)
        app_settings = QGroupBox("应用设置")
        app_settings.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                color: #495057;
                font-size: 16px;
            }
        """)

        app_layout = QVBoxLayout(app_settings)
        app_layout.setContentsMargins(15, 20, 15, 15)

        # 添加一些示例设置选项
        auto_start = QCheckBox("启动时自动加载上次搜索")
        auto_start.setStyleSheet("font-size: 14px;")

        save_history = QCheckBox("保存搜索历史记录")
        save_history.setChecked(True)
        save_history.setStyleSheet("font-size: 14px;")

        app_layout.addWidget(auto_start)
        app_layout.addWidget(save_history)
        app_layout.addStretch()

        # 添加所有设置组到设置卡片
        settings_layout.addWidget(path_group)
        settings_layout.addWidget(app_settings)
        settings_layout.addStretch()

        # 底部按钮区
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("保存设置")
        btn_save.setStyleSheet("background-color: #198754;")  # 绿色按钮
        btn_save.setMinimumHeight(40)
        btn_save.setMinimumWidth(120)
        btn_save.clicked.connect(self.save_settings)

        btn_back = QPushButton("返回主页")
        btn_back.setMinimumHeight(40)
        btn_back.setMinimumWidth(120)
        btn_back.clicked.connect(self.show_main)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_back)

        # 添加到主布局
        layout.addWidget(title_label)
        layout.addWidget(settings_card, 1)
        layout.addLayout(btn_layout)

        widget.setLayout(layout)
        # 加载保存的设置
        self.load_settings_folder()
        return widget

    def save_settings(self):
        """保存所有设置"""
        try:
            folder_path = self.folder_path_edit.text().strip()
            if folder_path and not os.path.isdir(folder_path):
                QMessageBox.warning(self, "提示", "设置的路径不是有效文件夹")
                return

            settings = {}
            if os.path.exists(SETTINGS_FILE):
                try:
                    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                        settings = json.load(f)
                except:
                    settings = {}

            # 更新设置
            settings["frames_addr"] = folder_path

            # 保存设置
            os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)

            QMessageBox.information(self, "成功", "设置已成功保存")
            self.statusBar().showMessage("设置已保存", 3000)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存设置失败: {str(e)}")
            print(f"保存设置失败: {e}")

    def load_settings_folder(self):
        """加载已保存的设置路径"""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                    frames_addr = settings.get("frames_addr", "")
                    if frames_addr:
                        self.folder_path_edit.setText(frames_addr)
            except Exception as e:
                print(f"加载设置文件失败: {e}")

    def build_history_page(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # 顶部标题区域
        title_label = QLabel("历史记录")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #212529; margin-bottom: 10px;")

        # 搜索栏
        search_layout = QHBoxLayout()
        search_frame = QFrame()
        search_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #dee2e6;
                border-radius: 6px;
                background-color: #f8f9fa;
                padding: 2px;
            }
        """)

        search_input_layout = QHBoxLayout(search_frame)
        search_input_layout.setContentsMargins(10, 2, 10, 2)
        search_input_layout.setSpacing(5)  # 减小间距

        # 搜索图标
        search_icon = QLabel()
        search_pixmap = QPixmap(os.path.join(BASE_DIR, "resources", "icons", "search.png"))
        search_icon.setPixmap(search_pixmap.scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        search_icon.setFixedWidth(20)  # 固定宽度，防止空白

        # 搜索输入框
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("搜索历史记录...")
        self.history_search.setStyleSheet("""
            QLineEdit {
                border: none;
                background-color: transparent;
                font-size: 14px;
                padding: 5px;
            }
        """)
        self.history_search.textChanged.connect(self.filter_history)

        search_input_layout.addWidget(search_icon)
        search_input_layout.addWidget(self.history_search)

        search_layout.addWidget(search_frame)

        # 历史记录列表
        self.history_list = QListWidget()
        self.history_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 5px;
                background-color: white;
            }
            QListWidget::item {
                padding: 10px 14px;
                border-bottom: 1px solid #f1f3f5;
                font-size: 14px;
            }
            QListWidget::item:selected {
                background-color: #e7f1ff;
                color: #000;
            }
        """)
        self.history_list.itemDoubleClicked.connect(self.open_history_item)

        layout.addWidget(title_label)
        layout.addLayout(search_layout)
        layout.addWidget(self.history_list)

        widget.setLayout(layout)
        self.load_history()
        return widget

    def load_history(self):
        """加载历史记录"""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    for item in history:
                        list_item = QListWidgetItem(item)
                        self.history_list.addItem(list_item)
            except Exception as e:
                print(f"加载历史记录失败: {e}")

    def filter_history(self, text):
        """过滤历史记录"""
        for i in range(self.history_list.count()):
            item = self.history_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def open_history_item(self, item):
        """打开历史记录项"""
        QMessageBox.information(self, "历史记录", f"你双击了: {item.text()}")

    def show_main(self):
        self.stacked_widget.setCurrentWidget(self.main_page)
        self.btn_main.setChecked(True)
        self.btn_settings.setChecked(False)
        self.btn_history.setChecked(False)

    def show_settings(self):
        self.stacked_widget.setCurrentWidget(self.settings_page)
        self.btn_main.setChecked(False)
        self.btn_settings.setChecked(True)
        self.btn_history.setChecked(False)

    def show_history(self):
        self.stacked_widget.setCurrentWidget(self.history_page)
        self.btn_main.setChecked(False)
        self.btn_settings.setChecked(False)
        self.btn_history.setChecked(True)

    def choose_settings_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            self.folder_path_edit.setText(folder)

    def upload_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            self.processed_dirs.append(folder)
            self.start_upload_task(folder)

    def select_search_dirs(self):
        folders = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folders:
            self.selected_search_dirs.append(folders)
            QMessageBox.information(self, "选择成功", f"已选择文件夹: {folders}")

    def start_upload_task(self, folder):
        if folder in self.upload_threads:
            QMessageBox.warning(self, "提示", "该文件夹正在处理，请勿重复添加")
            return

        # 创建上传线程
        upload_thread = UploadThread(folder)
        upload_thread.progress_updated.connect(self.update_progress)
        upload_thread.task_completed.connect(self.complete_task)
        self.upload_threads[folder] = upload_thread

        # 创建进度监控窗口
        if not self.progress_monitor:
            self.progress_monitor = ProgressMonitor(self)

        self.progress_monitor.add_task(folder)
        upload_thread.start()

    def update_progress(self, folder, progress, status):
        if self.progress_monitor:
            self.progress_monitor.update_progress(folder, progress, status)

    def complete_task(self, folder, result):
        if self.progress_monitor:
            self.progress_monitor.complete_task(folder, result)

    def do_search(self):
        search_text = self.input_text.text().strip()
        if not search_text:
            QMessageBox.warning(self, "提示", "请输入搜索内容")
            return

        if not self.selected_search_dirs:
            QMessageBox.warning(self, "提示", "请选择搜索范围")
            return

        # 创建搜索线程
        self.search_thread = SearchThread(self.selected_search_dirs, search_text)
        self.search_thread.search_complete.connect(self.display_search_results)
        self.search_thread.status_update.connect(self.update_status)

        # 显示搜索提示弹窗
        self.searching_dialog = SearchingDialog(self)
        self.searching_dialog.show()

        self.search_thread.start()

    def display_search_results(self, results):
        self.searching_dialog.close()
        self.all_results = results
        self.current_page = 1
        self.total_pages = (len(self.all_results) + self.page_size - 1) // self.page_size
        self.update_pagination()

    def update_pagination(self):
        self.table_files.setRowCount(0)
        start_index = (self.current_page - 1) * self.page_size
        end_index = min(start_index + self.page_size, len(self.all_results))

        for i in range(start_index, end_index):
            result = self.all_results[i]
            row_position = self.table_files.rowCount()
            self.table_files.insertRow(row_position)
            self.table_files.setItem(row_position, 0, QTableWidgetItem(str(i + 1)))
            self.table_files.setItem(row_position, 1, QTableWidgetItem(result["name"]))
            self.table_files.setItem(row_position, 2, QTableWidgetItem(result["size"]))
            self.table_files.setItem(row_position, 3, QTableWidgetItem(result["modified"]))

        self.page_info_label.setText(f"第 {self.current_page}/{self.total_pages} 页，共 {len(self.all_results)} 条记录")
        self.btn_prev_page.setEnabled(self.current_page > 1)
        self.btn_next_page.setEnabled(self.current_page < self.total_pages)

    def prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.update_pagination()

    def next_page(self):
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.update_pagination()

    def update_status(self, status):
        self.status_bar.showMessage(status)

    def open_selected_file(self):
        selected_row = self.table_files.currentRow()
        if selected_row >= 0:
            file_name = self.table_files.item(selected_row, 1).text()
            file_path = os.path.join(self.selected_search_dirs[0], file_name)
            open_path(file_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
