import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QLineEdit, QFileDialog, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QSplitter, QStackedWidget, QCheckBox, QMessageBox, QHeaderView, QStatusBar,
    QFrame, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt6.QtGui import QIcon

SETTINGS_FILE = "settings.json"
HISTORY_FILE = "history.json"


def get_style():
    return """
    QMainWindow {
        background-color: #f5f5f5;
    }
    QPushButton {
        background-color: #2196F3;
        color: white;
        border: none;
        padding: 5px 15px;
        border-radius: 4px;
        min-height: 25px;
    }
    QPushButton:hover {
        background-color: #1976D2;
    }
    QPushButton:pressed {
        background-color: #0D47A1;
    }
    QLineEdit {
        padding: 5px;
        border: 1px solid #BBBBBB;
        border-radius: 4px;
        background-color: white;
    }
    QTableWidget {
        background-color: white;
        border: 1px solid #DDDDDD;
        border-radius: 4px;
    }
    QListWidget {
        background-color: white;
        border: 1px solid #DDDDDD;
        border-radius: 4px;
    }
    QStatusBar {
        background-color: #E3F2FD;
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
    done = pyqtSignal(str)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        print(f"后台上传处理中: {self.folder}")
        self.sleep(2)  # 模拟耗时
        self.done.emit(self.folder + " 上传处理完成")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("文影灵搜")
        # self.setWindowIcon(QIcon("prime_search.png"))  # 替换为你的PNG路径
        self.resize(1024, 700)
        self.selected_search_dirs = []
        self.processed_dirs = []

        # 设置全局样式
        self.setStyleSheet(get_style())

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
        return widget

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
        if folder:
            self.statusBar().showMessage("正在上传处理...")
            self.upload_thread = UploadThread(folder)
            self.upload_thread.done.connect(self.on_upload_done)
            self.upload_thread.start()

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

        print(f"搜索文本: {text}")
        print(f"搜索路径: {self.selected_search_dirs}")

        # 模拟搜索结果
        results = [
            {"name": "视频1.mp4", "size": "120MB", "mtime": "2024-06-01 20:13", "path": r"C:\Videos\视频1.mp4"},
            {"name": "视频2.mp4", "size": "150MB", "mtime": "2024-06-01 20:15", "path": r"C:\Videos\视频2.mp4"},
        ]
        self.show_search_results(results)

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
            self.folder_path_edit.setText(folder)
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({"folder": folder}, f, ensure_ascii=False, indent=2)
            self.statusBar().showMessage(f"设置已保存: {folder}", 3000)

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
        self.setStyleSheet(get_style())

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
        print("刷新目录，检查需要重新处理的路径...")
        QMessageBox.information(self, "刷新", "目录已刷新完成")

    def get_selected_dirs(self):
        return [
            self.list_widget.item(i).text()
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.Checked
        ]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用 Fusion 风格获得更现代的外观
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())