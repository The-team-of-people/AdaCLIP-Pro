"""
样式表文件 - 集中管理应用的所有样式
"""

def get_modern_style():
    return """
    /* 全局字体设置 */
    * {
        font-family: "Microsoft YaHei", "Segoe UI", "Noto Sans CJK SC", sans-serif;
        font-size: 14px;
    }

    QMainWindow, QDialog {
        background-color: #f8f9fa;
    }

    /* 按钮样式 - 主要按钮 */
    QPushButton {
        background-color: #0d6efd;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0.3px;
        margin: 2px;
        min-height: 38px;
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

    /* 次要按钮样式 */
    QPushButton.secondary {
        background-color: #6c757d;
        color: white;
    }

    QPushButton.secondary:hover {
        background-color: #5c636a;
    }

    QPushButton.secondary:pressed {
        background-color: #565e64;
    }

    /* 成功按钮样式 */
    QPushButton.success {
        background-color: #198754;
        color: white;
    }

    QPushButton.success:hover {
        background-color: #157347;
    }

    QPushButton.success:pressed {
        background-color: #146c43;
    }

    /* 危险按钮样式 */
    QPushButton.danger {
        background-color: #dc3545;
        color: white;
    }

    QPushButton.danger:hover {
        background-color: #bb2d3b;
    }

    QPushButton.danger:pressed {
        background-color: #b02a37;
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
        alternate-background-color: #f8f9fa;
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
        padding: 4px;
    }

    QListWidget::item {
        padding: 10px 14px;
        border-bottom: 1px solid #f1f3f5;
        font-size: 14px;
        margin: 2px 0;
        border-radius: 4px;
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
        min-height: 10px;
        max-height: 10px;
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

    /* 标签样式 */
    QLabel {
        color: #212529;
    }

    QLabel.title {
        font-size: 20px;
        font-weight: bold;
        color: #212529;
        padding: 10px 0;
    }

    QLabel.subtitle {
        font-size: 16px;
        font-weight: bold;
        color: #495057;
    }

    /* 分组框样式 */
    QGroupBox {
        font-weight: bold;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        margin-top: 20px;
        padding-top: 20px;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 10px;
        color: #495057;
    }

    /* 左侧面板样式 */
    #leftPanel {
        background-color: #212529;
        border-right: none;
    }

    #leftPanel QPushButton {
        background-color: transparent;
        color: #adb5bd;
        text-align: left;
        padding: 14px 24px;
        border-radius: 0;
        font-size: 15px;
        font-weight: normal;
        border-left: 4px solid transparent;
    }

    #leftPanel QPushButton:hover {
        background-color: #2c3034;
        color: #fff;
    }

    #leftPanel QPushButton:checked {
        background-color: #2c3034;
        color: #fff;
        border-left: 4px solid #0d6efd;
        font-weight: bold;
    }

    #leftPanel QLabel {
        color: white;
        font-size: 22px;
        font-weight: bold;
        padding: 24px;
    }

    /* 搜索框样式 */
    QLineEdit#searchBox {
        background-color: #ffffff;
        border: 2px solid #dee2e6;
        border-radius: 20px;
        padding: 10px 14px 10px 40px;
        font-size: 14px;
    }

    QLineEdit#searchBox:focus {
        border-color: #86b7fe;
    }

    /* 卡片样式 */
    .card {
        background-color: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
    }
    
    /* 弹窗样式 */
    QDialog {
        border-radius: 8px;
    }
    """

def get_nav_button_style(is_active=False):
    """获取导航按钮的样式，可以根据是否激活返回不同样式"""
    base_style = """
        QPushButton {
            background-color: transparent;
            color: #adb5bd;
            text-align: left;
            padding: 14px 24px 14px 54px;
            border-radius: 0;
            font-size: 15px;
            font-weight: normal;
            border-left: 4px solid transparent;
            background-repeat: no-repeat;
            background-position: 20px center;
        }
        QPushButton:hover {
            background-color: #2c3034;
            color: #fff;
        }
    """

    active_style = base_style + """
        QPushButton {
            background-color: #2c3034;
            color: #fff;
            border-left: 4px solid #0d6efd;
            font-weight: bold;
        }
    """

    return active_style if is_active else base_style
