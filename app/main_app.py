import sys
import os
import json
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QLineEdit, QFileDialog, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QSplitter, QStackedWidget, QCheckBox, QMessageBox, QHeaderView, QStatusBar,
    QFrame, QDialog, QGroupBox, QGridLayout, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt6.QtGui import QIcon

# 确保必要的目录和文件存在
os.makedirs("user_data", exist_ok=True)
SETTINGS_FILE = os.path.join("user_data", "settings.json")
HISTORY_FILE = os.path.join("user_data", "history.json")

# 如果设置文件不存在，创建默认设置
if not os.path.exists(SETTINGS_FILE):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed_dirs": []}, f, ensure_ascii=False, indent=2)

# 如果历史文件不存在，创建默认历史记录
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"records": []}, f, ensure_ascii=False, indent=2)

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
    QProgressBar {
        border: 1px solid grey;
        border-radius: 3px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #2196F3;
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

#这里FileProcessor类负责处理文件夹中的文件，提取文件信息并保存到JSON文件中。
class FileProcessor:
    def __init__(self):
        self.processed_files = {}
        self.user_data_dir = Path("user_data")
        self.user_data_dir.mkdir(exist_ok=True)
        self.upload_info_file = self.user_data_dir / "uploaded_directory.json"
        self.load_processed_files()

    def load_processed_files(self):
        if self.upload_info_file.exists():
            try:
                with open(self.upload_info_file, "r", encoding="utf-8") as f:
                    self.processed_files = json.load(f)
            except:
                self.processed_files = {}

    def save_processed_files(self):
        with open(self.upload_info_file, "w", encoding="utf-8") as f:
            json.dump(self.processed_files, f, ensure_ascii=False, indent=2)

    def process_folder(self, folder_path):
        """处理文件夹中的所有文件"""
        folder_path = Path(folder_path)
        files_info = {}
        
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files_info[str(file_path)] = {
                    "name": file_path.name,
                    "size": self.format_size(stat.st_size),
                    "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "path": str(file_path),
                    "content_type": self.get_content_type(file_path)
                }
        
        self.processed_files[str(folder_path)] = {
            "files": files_info,
            "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_processed_files()
        return len(files_info)

    def search_files(self, search_text, search_dirs):
        """搜索文件"""
        results = []
        search_text = search_text.lower()
        
        for dir_path in search_dirs:
            if dir_path not in self.processed_files:
                continue
                
            for file_info in self.processed_files[dir_path]["files"].values():
                if (search_text in file_info["name"].lower() or
                    search_text in str(file_info["path"]).lower()):
                    results.append(file_info)
                    
        return results

    @staticmethod
    def format_size(size):
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f}{unit}"
            size /= 1024
        return f"{size:.2f}TB"

    @staticmethod
    def get_content_type(file_path):
        """获取文件类型"""
        ext = file_path.suffix.lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return "视频"
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return "图片"
        elif ext in ['.doc', '.docx', '.pdf', '.txt']:
            return "文档"
        else:
            return "其他"

class UploadThread(QThread):
    done = pyqtSignal(str, bool, str)  # 文件夹路径, 是否成功, 消息

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        try:
            # 调用API处理视频
            result = uploadDirectory(self.folder)
            if result["success"]:
                self.done.emit(self.folder, True, result["msg"])
            else:
                self.done.emit(self.folder, False, result["msg"])
        except Exception as e:
            self.done.emit(self.folder, False, str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("文影灵搜")
        self.resize(1024, 700)
        self.selected_search_dirs = []
        self.processed_dirs = []
        # 添加文件处理器实例
        self.file_processor = FileProcessor()  # 添加这一行

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

    def show_main(self):
        """显示主页面"""
        self.stacked_widget.setCurrentIndex(0)
        self.statusBar().showMessage("主页")

    def show_settings(self):
        """显示设置页面"""
        self.stacked_widget.setCurrentIndex(1)
        self.statusBar().showMessage("设置")

    def show_history(self):
        """显示历史记录页面"""
        self.stacked_widget.setCurrentIndex(2)
        self.load_history()  # 加载历史记录
        self.statusBar().showMessage("历史记录")

    def load_history(self):
        """加载历史记录"""
        self.history_list.clear()
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    for record in history.get("records", []):
                        text = record["text"]
                        time = record["time"]
                        item = QListWidgetItem(f"{time} - {text}")
                        item.setData(Qt.UserRole, record)  # 存储完整记录数据
                        self.history_list.addItem(item)
            except Exception as e:
                print(f"加载历史记录失败: {e}")

    def open_history_item(self, item):
        """打开历史记录项"""
        record = item.data(Qt.UserRole)
        if record and "files" in record:
            if not record["files"]:
                QMessageBox.information(self, "提示", "该记录没有关联文件")
                return
            
            # 显示文件列表对话框
            dlg = QDialog(self)
            dlg.setWindowTitle("历史记录文件")
            layout = QVBoxLayout(dlg)
            
            list_widget = QListWidget()
            for file_path in record["files"]:
                list_widget.addItem(QListWidgetItem(file_path))
            list_widget.itemDoubleClicked.connect(lambda item: open_path(item.text()))
            
            layout.addWidget(list_widget)
            dlg.exec_()

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

    def on_upload_done(self, folder, success, msg):
        if success:
            self.processed_dirs.append(folder)
            self.statusBar().showMessage(msg, 3000)
        else:
            self.statusBar().showMessage(f"上传处理失败: {msg}", 5000)

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
        if not text:
            QMessageBox.warning(self, "提示", "请输入搜索内容！")
            return
            
        if not self.selected_search_dirs:
            QMessageBox.warning(self, "提示", "请先选择要搜索的文件夹！")
            return

        self.statusBar().showMessage("正在搜索...")
        try:
            # 先通过 FileProcessor 尝试搜索本地文件
            local_results = self.file_processor.search_files(text, self.selected_search_dirs)
            
            # 再通过 API 搜索视频
            api_results = videoQuery(self.selected_search_dirs, text)
            
            # 合并结果
            all_paths = set(r["path"] for r in local_results) | set(api_results)
            display_results = []
            #这里是你要的list返回值
            for path in all_paths:
                if os.path.exists(path):
                    file_stat = os.stat(path)
                    display_results.append({
                        "name": os.path.basename(path),
                        "size": self.format_size(file_stat.st_size),
                        "mtime": datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        "path": path
                    })

            self.show_search_results(display_results)
            
            # 保存到历史记录
            if display_results:
                self.save_to_history(text, [r["path"] for r in display_results[:5]])
                
        except Exception as e:
            self.statusBar().showMessage(f"搜索失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"搜索失败: {str(e)}")

    def show_search_results(self, files):
        """显示搜索结果"""
        self.table_files.setRowCount(len(files))
        self.table_files.files_data = files  # 保存完整数据用于双击打开
        
        for row, file in enumerate(files):
            self.table_files.setItem(row, 0, QTableWidgetItem(file["name"]))
            self.table_files.setItem(row, 1, QTableWidgetItem(file["size"]))
            self.table_files.setItem(row, 2, QTableWidgetItem(file["mtime"]))
        
        self.statusBar().showMessage(f"共搜索到 {len(files)} 个文件")
        
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

    def select_search_dirs(self):
        if not self.processed_dirs:
            QMessageBox.warning(self, "提示", "请先上传并处理文件夹！")
            return
        dlg = SelectDirDialog(self.processed_dirs, self)
        if dlg.exec_():
            self.selected_search_dirs = dlg.get_selected_dirs()
            print("选择的搜索路径：", self.selected_search_dirs)

    def open_selected_file(self, idx):
        file_info = self.table_files.files_data[idx.row()]
        open_path(file_info["path"])

    def choose_settings_folder(self):
        """在设置页面选择文件夹"""
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
                self.statusBar().showMessage(f"保存设置失败: {str(e)}")
                QMessageBox.warning(self, "错误", f"保存设置失败: {str(e)}")

# API接口模拟（这里是简化模拟，到时候你替换成api调用）
def uploadDirectory(directory):
    """模拟上传目录API"""
    return {"success": True, "msg": f"目录 {directory} 处理完成"}

def videoQuery(directories, query):
    """模拟视频查询API"""
    # 这里模拟返回一些测试数据，你到时候换一下在测试
    return [
        os.path.join(directories[0], "test1.mp4"),
        os.path.join(directories[0], "test2.mp4"),
    ]

def scanAndCheckDictory():
    """模拟目录扫描API"""
    return True

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
    try:
        app = QApplication(sys.argv)
        # 设置应用程序图标
        if os.path.exists("app/prime_search.png"):
            app.setWindowIcon(QIcon("app/prime_search.png"))
            
        # 创建并显示主窗口
        window = MainWindow()
        window.show()
        
        # 运行应用程序
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序启动失败: {str(e)}")
        # 显示错误对话框
        if 'app' in locals():
            QMessageBox.critical(None, "错误", f"程序启动失败: {str(e)}")
        sys.exit(1)

