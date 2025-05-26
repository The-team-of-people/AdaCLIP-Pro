import sys
import os
import json
import traceback
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QLineEdit, QFileDialog, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QSplitter, QStackedWidget, QCheckBox, QMessageBox, QHeaderView, QStatusBar,
    QFrame, QDialog, QGroupBox, QGridLayout, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
from icon_and_font_updates import IconManager
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



def open_path(path):
    """统一的文件/文件夹打开函数"""
    try:
        if os.path.exists(path):
            os.startfile(path)
        else:
            QMessageBox.warning(None, "提示", f"路径不存在: {path}")
    except Exception as e:
        QMessageBox.warning(None, "错误", f"打开失败: {str(e)}")


class UploadThread(QThread):
    progress_updated = pyqtSignal(str, int, str)  # 文件夹路径, 进度百分比, 状态消息
    task_completed = pyqtSignal(str, str)  # 文件夹路径, 最终结果

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        try:
            # 调用后端上传目录接口

            result = uploadDirectory(self.folder)

            if result.get("success"):
                self.progress_updated.emit(self.folder, 100, "处理完成")
                self.task_completed.emit(self.folder, result.get('msg', '处理完成'))
            else:
                self.progress_updated.emit(self.folder, 100, f"处理失败: {result.get('msg', '未知错误')}")
                self.task_completed.emit(self.folder, result)  # 直接传递字典

        except Exception as e:
            print(f"UploadThread 异常: {type(e).__name__}, 信息: {str(e)}")
            traceback.print_exc()  # 需要导入 traceback
            self.progress_updated.emit(self.folder, 100, f"处理异常: {str(e)}")
            self.task_completed.emit(self.folder, {"success": False, "msg": str(e)})


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
        if folder in self.task_widgets:
            self.task_widgets[folder]["progress_bar"].setValue(progress)
            self.task_widgets[folder]["status_label"].setText(status)

    def complete_task(self, folder, result):
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
        self.resize(1024, 700)
        self.selected_search_dirs = []
        self.processed_dirs = []
        self.upload_threads = {}  # 存储所有上传线程
        self.progress_monitor = None  # 进度监控窗口

        # 设置全局样式
        self.setStyleSheet(get_modern_style())

        # 主分割器
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧导航栏（15%宽度）
        self.left_widget = QFrame()
        self.left_widget.setObjectName("leftPanel")
        self.left_widget.setStyleSheet("""
            #leftPanel {
                background-color: #1E88E5;
                border-right: 1px solid #1976D2;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: transparent;
                color: white;
                text-align: left;
                padding: 10px;
                border: none;
                border-radius: 0;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)

        left_layout = QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(0, 20, 0, 20)
        left_layout.setSpacing(10)

        # 应用名称
        self.app_name = QLabel("文影灵搜")
        self.app_name.setAlignment(Qt.AlignCenter)
        self.app_name.setStyleSheet("font-size: 20px; font-weight: bold; padding: 20px;")

        # 导航按钮
        self.btn_main = QPushButton("主页")
        self.btn_settings = QPushButton("设置")
        self.btn_history = QPushButton("历史记录")

        for btn in [self.btn_main, self.btn_settings, self.btn_history]:
            btn.setFont(QFont("Arial", 10))

        self.btn_main.clicked.connect(self.show_main)
        self.btn_settings.clicked.connect(self.show_settings)
        self.btn_history.clicked.connect(self.show_history)

        left_layout.addWidget(self.app_name)
        left_layout.addWidget(self.btn_main)
        left_layout.addWidget(self.btn_settings)
        left_layout.addWidget(self.btn_history)
        left_layout.addStretch()

        # 右侧内容区
        self.stacked_widget = QStackedWidget()
        self.main_page = self.build_main_page()
        self.settings_page = self.build_settings_page()
        self.history_page = self.build_history_page()

        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.settings_page)
        self.stacked_widget.addWidget(self.history_page)

        main_layout.addWidget(self.left_widget, 15)
        main_layout.addWidget(self.stacked_widget, 85)

        self.setCentralWidget(main_widget)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def build_main_page(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # 上传和选择按钮区
        top_buttons = QHBoxLayout()
        self.btn_upload = QPushButton("上传")
        self.btn_select = QPushButton("选择")
        top_buttons.addWidget(self.btn_upload)
        top_buttons.addWidget(self.btn_select)
        top_buttons.addStretch()

        self.btn_upload.clicked.connect(self.upload_folder)
        self.btn_select.clicked.connect(self.select_search_dirs)

        # 搜索区
        search_layout = QHBoxLayout()
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("请输入搜索内容...")
        self.btn_search = QPushButton("搜索")
        self.btn_search.clicked.connect(self.do_search)

        search_layout.addWidget(self.input_text)
        search_layout.addWidget(self.btn_search)

        # 文件展示表格
        self.table_files = QTableWidget(0, 3)
        self.table_files.setHorizontalHeaderLabels(["文件名", "大小", "修改时间"])
        self.table_files.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_files.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_files.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_files.doubleClicked.connect(self.open_selected_file)

        layout.addLayout(top_buttons)
        layout.addSpacing(10)
        layout.addLayout(search_layout)
        layout.addSpacing(10)
        layout.addWidget(self.table_files)

        widget.setLayout(layout)
        return widget

    def build_settings_page(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # 返回按钮
        btn_back = QPushButton("返回主页")
        btn_back.clicked.connect(self.show_main)

        # 文件夹选择区
        folder_layout = QHBoxLayout()
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("请选择文件夹路径...")
        btn_choose = QPushButton("选择文件夹")
        btn_choose.clicked.connect(self.choose_settings_folder)

        folder_layout.addWidget(self.folder_path_edit)
        folder_layout.addWidget(btn_choose)

        layout.addWidget(btn_back)
        layout.addSpacing(20)
        layout.addLayout(folder_layout)
        layout.addStretch()

        widget.setLayout(layout)
        self.load_settings_folder()
        return widget

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
                self.statusBar().showMessage(f"加载设置失败: {str(e)}", 3000)
        else:
            # 文件不存在时不做处理，保持占位符显示
            pass

    def build_history_page(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # 返回按钮
        btn_back = QPushButton("返回主页")
        btn_back.clicked.connect(self.show_main)

        # 历史记录列表
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self.open_history_item)

        layout.addWidget(btn_back)
        layout.addSpacing(20)
        layout.addWidget(self.history_list)

        widget.setLayout(layout)
        return widget

    # -- 事件处理函数 --
    def upload_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择要上传的文件夹")
        if folder in self.upload_threads:
            QMessageBox.warning(self, "提示", "该文件夹正在处理中，请等待完成！")
            return
        if folder in self.processed_dirs:
            QMessageBox.warning(self, "提示", "该文件夹已处理完成，无需重复上传！")
            return
        if folder:
            self.statusBar().showMessage("正在上传处理...")

            # 创建进度监控窗口(如果不存在)
            if not self.progress_monitor:
                self.progress_monitor = ProgressMonitor(self)

            # 添加新任务到监控窗口
            self.progress_monitor.add_task(folder)

            # 创建并启动上传线程
            thread = UploadThread(folder)
            self.upload_threads[folder] = thread

            # 连接信号
            thread.progress_updated.connect(self.update_progress)
            thread.task_completed.connect(self.on_task_completed)

            thread.start()

    def update_progress(self, folder, progress, status):
        if self.progress_monitor:
            self.progress_monitor.update_progress(folder, progress, status)

    def on_task_completed(self, folder, msg):
        if folder in self.upload_threads:
            # 从线程列表中移除已完成的线程
            self.upload_threads[folder].deleteLater()
            del self.upload_threads[folder]

        # 更新进度窗口

        if self.progress_monitor:
            self.progress_monitor.complete_task(folder, msg)

        # 添加到已处理目录列表
        self.processed_dirs.append(folder)
        self.statusBar().showMessage(f"{folder} 处理完成", 3000)

    # 其他方法保持不变...

    def on_upload_done(self, msg):
        print(msg)
        self.processed_dirs.append(self.upload_thread.folder)
        self.statusBar().showMessage(msg, 3000)

    def select_search_dirs(self):
        if not self.processed_dirs:
            QMessageBox.warning(self, "提示", "请先上传并处理文件夹！")
            return
        dlg = SelectDirDialog(self.processed_dirs, self)
        if dlg.exec_():
            self.selected_search_dirs = dlg.get_selected_dirs()
            print("选择的搜索路径：", self.selected_search_dirs)

    def do_search(self):
        text = self.input_text.text().strip()
        if not text or not self.selected_search_dirs:
            QMessageBox.warning(self, "提示", "请输入搜索内容并选择文件夹！")
            return

        self.statusBar().showMessage("正在搜索...")
        try:
            # 调用API进行视频搜索
            api_results = videoQuery(self.selected_search_dirs, text)
            result_count = len(api_results)
            if result_count > 30:
                api_results = api_results[:30]
            else:
                pass
            print(len(api_results))
            # 处理搜索结果
            display_results = []
            for path in api_results:
                if os.path.exists(path):
                    file_stat = os.stat(path)
                    display_results.append({
                        "name": os.path.basename(path),
                        "size": f"{file_stat.st_size / (1024 * 1024):.2f}MB",
                        "mtime": datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        "path": path
                    })

            # 显示搜索结果
            self.show_search_results(display_results)

            # 保存到历史记录
            if display_results:
                self.save_to_history(text, [r["path"] for r in display_results[:5]])

            # 更新状态栏
            self.statusBar().showMessage(f"搜索完成，找到 {len(display_results)} 个结果")
        except Exception as e:
            print(f"搜索错误: {str(e)}")
            self.statusBar().showMessage(f"搜索失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"搜索失败: {str(e)}")

    def save_to_history(self, search_text, file_paths):
        """保存搜索历史"""
        history = {"records": []}
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except:
                pass

        # 添加新记录
        new_record = {
            "text": search_text,
            "files": file_paths,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 确保records字段存在
        if "records" not in history:
            history["records"] = []

        # 添加新记录到开头
        history["records"].insert(0, new_record)

        # 只保留最近50条记录
        history["records"] = history["records"][:50]

        # 保存历史记录
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存历史记录失败: {e}")
    def show_search_results(self, files):
        self.table_files.setRowCount(len(files))
        self.table_files.files_data = files  # 保存完整数据用于双击打开

        for row, file in enumerate(files):
            self.table_files.setItem(row, 0, QTableWidgetItem(file["name"]))
            self.table_files.setItem(row, 1, QTableWidgetItem(file["size"]))
            self.table_files.setItem(row, 2, QTableWidgetItem(file["mtime"]))

        self.statusBar().showMessage(f"共搜索到 {len(files)} 个文件")

    def open_selected_file(self, idx):
        file_info = self.table_files.files_data[idx.row()]
        open_path(file_info["path"])

    def choose_settings_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            if not os.path.isdir(folder):
                QMessageBox.warning(self, "提示", "选择的路径不是有效文件夹")
                return

            self.folder_path_edit.setText(folder)
            try:
                with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                    json.dump({"frames_addr": folder}, f, ensure_ascii=False, indent=2)
                self.statusBar().showMessage(f"设置已保存: {folder}", 3000)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存设置失败: {str(e)}")


    def load_history(self):
        self.history_list.clear()
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
                for record in history.get("records", []):
                    txt = f"{record['text']} -> {', '.join(record['files'])}"
                    item = QListWidgetItem(txt)
                    item.file_paths = record['files']
                    self.history_list.addItem(item)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载历史记录失败: {str(e)}")

    def open_history_item(self, item):
        if hasattr(item, 'file_paths') and item.file_paths:
            open_path(item.file_paths[0])

    # -- 界面切换 --
    def show_main(self):
        self.stacked_widget.setCurrentWidget(self.main_page)
        self.statusBar().showMessage("主页")

    def show_settings(self):
        self.stacked_widget.setCurrentWidget(self.settings_page)
        self.statusBar().showMessage("设置")

    def show_history(self):
        self.stacked_widget.setCurrentWidget(self.history_page)
        self.load_history()
        self.statusBar().showMessage("历史记录")


class SelectDirDialog(QDialog):
    def __init__(self, dirs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择搜索文件夹")
        self.setMinimumWidth(500)
        self.setStyleSheet(get_modern_style())
        self.setWindowIcon(parent.icon_manager.get_icon())  # 设置对话框图标

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # 文件夹列表
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for d in dirs:
            item = QListWidgetItem(d)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)
        self.list_widget.itemDoubleClicked.connect(lambda x: open_path(x.text()))

        # 按钮区
        btn_layout = QHBoxLayout()
        self.btn_all = QPushButton("全选")
        self.btn_none = QPushButton("全不选")
        self.btn_refresh = QPushButton("刷新")
        self.btn_ok = QPushButton("确定")
        self.btn_cancel = QPushButton("取消")

        self.btn_all.clicked.connect(self.select_all)
        self.btn_none.clicked.connect(self.select_none)
        self.btn_refresh.clicked.connect(self.refresh_dirs)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_all)
        btn_layout.addWidget(self.btn_none)
        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)

        layout.addWidget(self.list_widget)
        layout.addLayout(btn_layout)

    def select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)

    def select_none(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

    def refresh_dirs(self):
        """刷新目录列表，重新检查处理状态"""
        try:
            # 获取当前选中状态
            current_states = {}
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                current_states[item.text()] = item.checkState()

            # 检查目录状态
            result = scanAndCheckDictory()
            if not result:
                QMessageBox.warning(self, "错误", "目录扫描失败")
                return

            # 清空列表
            self.list_widget.clear()

            # 从 uploaded_directory.json 重新加载目录列表
            upload_info_file = os.path.join(BASE_DIR, "user_data", "uploaded_directory.json")
            if os.path.exists(upload_info_file):
                with open(upload_info_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 添加目录到列表，保持原有的选中状态
                    for dir_path in data.keys():
                        item = QListWidgetItem(dir_path)
                        item.setCheckState(current_states.get(dir_path, Qt.Unchecked))
                        self.list_widget.addItem(item)

            QMessageBox.information(self, "完成", "目录已刷新完成")

        except Exception as e:
            print(f"刷新目录失败: {str(e)}")
            traceback.print_exc()
            QMessageBox.warning(self, "错误", f"刷新目录失败: {str(e)}")

    def get_selected_dirs(self):
        return [
            self.list_widget.item(i).text()
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.Checked
        ]


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # ---------------------------------------------------------------
    # 添加这些行来设置默认字体
    default_font = QFont("Microsoft YaHei")  # 设置默认字体为微软雅黑
    default_font.setPixelSize(16)  # 设置默认字体大小
    # -----------------------------------------------------------------
    # --------------------------------------------------------------
    # 设置应用程序图标
    icon_manager = IconManager()
    app.setWindowIcon(icon_manager.get_icon())
    # ---------------------------------------------------------------

    app.setStyle('Fusion')  # 使用 Fusion 风格获得更现代的外观
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())