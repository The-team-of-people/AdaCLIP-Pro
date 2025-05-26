from PyQt5.QtGui import QIcon, QFont
import os



# 添加图标管理类
class IconManager:
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)  # 获取当前文件所在目录
        self.icon_path = os.path.join(self.base_dir, "resources", "icons", "app_icon.png")
        self._ensure_icon_exists()

    def _ensure_icon_exists(self):
        """确保图标文件存在，如果不存在则创建"""
        if not os.path.exists(self.icon_path):
            os.makedirs(os.path.dirname(self.icon_path), exist_ok=True)
            self._create_default_icon()

    def _create_default_icon(self):
        """创建默认图标的代码，使用PIL创建一个简单的图标"""
        try:
            from PIL import Image, ImageDraw

            # 创建一个 128x128 的图像
            img = Image.new('RGBA', (128, 128), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # 绘制一个简单的图标（蓝色圆形背景）
            draw.ellipse([10, 10, 118, 118], fill='#0d6efd')

            # 绘制一个白色的"文"字样
            if hasattr(draw, 'textbbox'):  # PIL 8.0.0 及以上版本
                font_size = 60
                from PIL import ImageFont
                font = ImageFont.truetype("simhei.ttf", font_size)  # 使用黑体
                bbox = draw.textbbox((64, 64), "文", font=font, anchor="mm")
                draw.text((64, 64), "文", fill='white', font=font, anchor="mm")
            else:
                # 如果没有高级文本功能，就画一个简单的白色方块
                draw.rectangle([44, 44, 84, 84], fill='white')

            # 保存图标
            img.save(self.icon_path, 'PNG')
        except Exception as e:
            print(f"创建默认图标失败: {e}")
            # 如果创建失败，生成一个1x1的透明图片作为替代
            img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            img.save(self.icon_path, 'PNG')

    def get_icon(self):
        """获取应用图标"""
        return QIcon(self.icon_path)