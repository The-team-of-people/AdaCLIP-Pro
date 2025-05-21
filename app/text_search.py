import sys
import os
import platform
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QFileDialog, QTreeWidget, QTreeWidgetItem,
                               QHeaderView, QStatusBar, QProgressBar, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QIcon, QFont, QColor, QPalette, QBrush, QPixmap


class SearchThread(QThread):
    """搜索文件的后台线程"""
    update_results = Signal(list)
    search_complete = Signal(int)
    status_update = Signal(str)

    def __init__(self, folder_path, search_text):
        super().__init__()
        self.folder_path = folder_path
        self.search_text = search_text

    def run(self):
        try:
            results = []
            count = 0

            self.status_update.emit(f"正在搜索 '{self.search_text}' 在 '{self.folder_path}'")

            for root, dirs, files in os.walk(self.folder_path):
                # 搜索文件夹
                for dir_name in dirs:
                    if self.search_text.lower() in dir_name.lower():
                        dir_path = os.path.join(root, dir_name)
                        try:
                            stats = os.stat(dir_path)
                            modified_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                            results.append((dir_name, "文件夹", "", modified_time, dir_path))
                            count += 1

                            # 每100个结果发送一次更新
                            if count % 100 == 0:
                                self.update_results.emit(results)
                                results = []
                        except Exception as e:
                            pass

                # 搜索文件
                for file_name in files:
                    if self.search_text.lower() in file_name.lower():
                        file_path = os.path.join(root, file_name)
                        try:
                            stats = os.stat(file_path)
                            modified_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                            size = self.format_size(stats.st_size)
                            ext = os.path.splitext(file_name)[1].upper().replace('.', '') or '文件'
                            results.append((file_name, ext, size, modified_time, file_path))
                            count += 1

                            # 每100个结果发送一次更新
                            if count % 100 == 0:
                                self.update_results.emit(results)
                                results = []
                        except Exception as e:
                            pass

            # 发送剩余结果
            if results:
                self.update_results.emit(results)

            self.search_complete.emit(count)
            self.status_update.emit(f"搜索完成，找到 {count} 个文件")

        except Exception as e:
            self.status_update.emit(f"搜索出错: {str(e)}")

    def format_size(self, size_bytes):
        """格式化文件大小显示"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


class FileSearchTool(QMainWindow):
    """文件搜索工具主窗口"""

    def __init__(self):
        super().__init__()

        self.setWindowIcon(QIcon("prime_search.png"))  # 替换为你的PNG路径

        # 设置窗口标题和大小
        self.setWindowTitle("文影灵搜")
        self.resize(900, 600)

        # 设置中文字体
        if platform.system() == "Windows":
            self.default_font = QFont("Microsoft YaHei UI", 10)
            self.title_font = QFont("Microsoft YaHei UI", 14, QFont.Bold)
        elif platform.system() == "Darwin":  # macOS
            self.default_font = QFont("Heiti TC", 10)
            self.title_font = QFont("Heiti TC", 14, QFont.Bold)
        else:  # Linux
            self.default_font = QFont("WenQuanYi Micro Hei", 10)
            self.title_font = QFont("WenQuanYi Micro Hei", 14, QFont.Bold)

        self.setFont(self.default_font)

        # 自定义颜色
        self.colors = {
            "primary": "#3B82F6",
            "secondary": "#6B7280",
            "success": "#10B981",
            "warning": "#F59E0B",
            "danger": "#EF4444",
            "light": "#F3F4F6",
            "dark": "#1F2937",
            "card_bg": "#FFFFFF",
            "card_shadow": "#F0F0F0"
        }

        # 设置样式表
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #F9FAFB;
            }}
            QLabel {{
                color: {self.colors["dark"]};
            }}
            QLineEdit {{
                background-color: white;
                color: #000000;  /* 文本框文字为黑色 */
                border: 1px solid #E5E7EB;
                border-radius: 4px;
                padding: 5px;
                selection-background-color: {self.colors["primary"]};
                selection-color: white;
            }}
            QPushButton {{
                background-color: #E5E7EB;  /* 按钮背景为浅灰色 */
                color: #000000;  /* 按钮文字为黑色 */
                border: 1px solid #D1D5DB;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #D1D5DB;  /* 悬停时颜色加深 */
            }}
            QPushButton:pressed {{
                background-color: #9CA3AF;  /* 按下时颜色更深 */
            }}
            /* 特殊样式：搜索按钮使用主色调 */
            QPushButton#searchButton {{
                background-color: {self.colors["primary"]};  /* 主色调背景 */
                color: white;  /* 搜索按钮文字为白色 */
                border: none;
                font-weight: bold;
            }}
            QPushButton#searchButton:hover {{
                background-color: #2563EB;
            }}
            QPushButton#searchButton:pressed {{
                background-color: #1D4ED8;
            }}
            QTreeWidget {{
                alternate-background-color: #F9FAFB;
                background-color: white;
                border: 1px solid #E5E7EB;
                border-radius: 4px;
                color: #000000;  /* 文件列表文字为黑色 */
            }}
            QTreeWidget::item {{
                height: 28px;
                border-bottom: 1px solid #F0F0F0;
                color: #000000;
            }}
            QTreeWidget::item:selected {{
                background-color: {self.colors["primary"]};
                color: white;
            }}
            QHeaderView::section {{
                background-color: {self.colors["light"]};
                color: {self.colors["dark"]};
                border: none;
                border-bottom: 1px solid #E5E7EB;
                padding: 6px;
                font-weight: bold;
            }}
            QStatusBar {{
                background-color: #E5E7EB;
                color: {self.colors["dark"]};
                border-top: 1px solid #D1D5DB;
            }}
        """)

        # 创建中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 创建标题区域
        title_frame = QWidget()
        title_frame.setStyleSheet(f"background-color: {self.colors['primary']}; border-radius: 6px;")
        title_frame.setFixedHeight(60)

        title_layout = QHBoxLayout(title_frame)
        title_layout.setContentsMargins(20, 0, 0, 0)

        title_label = QLabel("文影灵搜")
        title_label.setFont(self.title_font)
        title_label.setStyleSheet("color: white;")
        title_layout.addWidget(title_label)

        main_layout.addWidget(title_frame)

        # 创建搜索区域（修改后的布局）
        search_frame = QWidget()
        search_frame.setStyleSheet(
            f"background-color: {self.colors['card_bg']}; border-radius: 6px; padding: 15px; border: 1px solid {self.colors['card_shadow']};")

        # 使用垂直布局容器
        search_container = QVBoxLayout(search_frame)
        search_container.setSpacing(12)

        # 第一行：文件夹路径选择
        folder_layout = QHBoxLayout()
        folder_layout.setSpacing(10)

        folder_label = QLabel("目标文件夹:")
        folder_layout.addWidget(folder_label)

        self.folder_path = QLineEdit()
        self.folder_path.setPlaceholderText("点击右侧按钮选择搜索目录...")
        folder_layout.addWidget(self.folder_path, 3)  # 3倍宽度占比

        browse_button = QPushButton("浏览")
        browse_button.setFixedWidth(80)
        browse_button.clicked.connect(self.browse_folder)
        folder_layout.addWidget(browse_button)

        search_container.addLayout(folder_layout)

        # 第二行：搜索输入和按钮
        search_input_layout = QHBoxLayout()
        search_input_layout.setSpacing(10)

        search_text_label = QLabel("搜索内容:")
        search_input_layout.addWidget(search_text_label)

        self.search_text = QLineEdit()
        self.search_text.setPlaceholderText("输入文件名关键词...")
        search_input_layout.addWidget(self.search_text, 2)  # 2倍宽度占比

        # 搜索按钮
        search_button = QPushButton("开始搜索")
        search_button.setFixedWidth(100)
        search_button.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {self.colors["primary"]};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        padding: 8px 16px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{ background-color: #2563EB; }}
                    QPushButton:pressed {{ background-color: #1D4ED8; }}
                """)
        search_button.clicked.connect(self.search_files)
        search_input_layout.addWidget(search_button)

        search_container.addLayout(search_input_layout)

        main_layout.addWidget(search_frame)

        # 创建结果统计区域
        stats_frame = QWidget()
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(0, 0, 0, 0)

        self.result_count = QLabel("找到 0 个文件")
        stats_layout.addWidget(self.result_count, 0, Qt.AlignLeft)

        main_layout.addWidget(stats_frame)

        # 创建文件列表区域
        self.file_tree = QTreeWidget()
        self.file_tree.setColumnCount(4)
        self.file_tree.setHeaderLabels(["名称", "类型", "大小", "修改日期"])
        self.file_tree.setAlternatingRowColors(True)
        self.file_tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.file_tree.setUniformRowHeights(True)
        self.file_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.file_tree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.file_tree.header().setSectionResizeMode(2, QHeaderView.Fixed)
        self.file_tree.header().setSectionResizeMode(3, QHeaderView.Fixed)
        self.file_tree.setColumnWidth(1, 100)  # 类型列
        self.file_tree.setColumnWidth(2, 100)  # 大小列
        self.file_tree.setColumnWidth(3, 180)  # 修改日期列

        # 双击事件
        self.file_tree.itemDoubleClicked.connect(self.on_double_click)

        main_layout.addWidget(self.file_tree)

        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        # 初始化搜索线程
        self.search_thread = None

    def browse_folder(self):
        """浏览并选择文件夹"""
        folder_selected = QFileDialog.getExistingDirectory(self, "选择文件夹", "")
        if folder_selected:
            self.folder_path.setText(folder_selected)
            self.status_bar.showMessage(f"已选择文件夹: {folder_selected}")

    def search_files(self):
        """搜索文件"""
        search_text = self.search_text.text().strip()
        folder_path = self.folder_path.text().strip()

        if not folder_path:
            QMessageBox.warning(self, "警告", "请选择文件夹路径")
            return

        if not os.path.isdir(folder_path):
            QMessageBox.critical(self, "错误", "选择的路径不是有效文件夹")
            return

        # 清空表格
        self.file_tree.clear()
        self.result_count.setText("搜索中...")

        # 创建并启动搜索线程
        self.search_thread = SearchThread(folder_path, search_text)
        self.search_thread.update_results.connect(self.update_results)
        self.search_thread.search_complete.connect(self.search_complete)
        self.search_thread.status_update.connect(self.status_bar.showMessage)
        self.search_thread.start()

    def update_results(self, results):
        """更新搜索结果"""
        for item in results:
            name, file_type, size, modified, path = item
            tree_item = QTreeWidgetItem(self.file_tree)
            tree_item.setText(0, name)
            tree_item.setText(1, file_type)
            tree_item.setText(2, size)
            tree_item.setText(3, modified)
            tree_item.setData(0, Qt.UserRole, path)  # 存储路径信息

    def search_complete(self, count):
        """搜索完成处理"""
        self.result_count.setText(f"找到 {count} 个文件")

    def on_double_click(self, item, column):
        """双击事件处理 - 打开文件或文件夹"""
        path = item.data(0, Qt.UserRole)
        if path and os.path.exists(path):
            try:
                if platform.system() == "Windows":
                    os.startfile(path)
                elif platform.system() == "Darwin":
                    os.system(f"open '{path}'")
                else:
                    os.system(f"xdg-open '{path}'")
                self.status_bar.showMessage(f"已打开: {path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法打开: {str(e)}")
        else:
            QMessageBox.critical(self, "错误", "文件或文件夹不存在")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 在 QApplication 实例化后立即添加
    # app.setAttribute(Qt.AA_EnableHighDpiScaling)  # 启用自动缩放
    # app.setAttribute(Qt.AA_UseHighDpiPixmaps)  # 高清图标支持
    # 设置应用样式
    app.setStyle("Fusion")

    window = FileSearchTool()
    window.show()

    sys.exit(app.exec())