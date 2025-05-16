import os


def scan_directory(root_path='D:\\GitCode\\AdaCLIP-Pro'):
    """
    递归扫描目录结构，对data目录只扫描一级子目录，完全跳过.git目录

    Args:
        root_path: 要扫描的根目录，默认为 D:\\GitCode\\AdaCLIP-Pro
    """
    result = []

    # 遍历根目录
    for root, dirs, files in os.walk(root_path, topdown=True):
        # 先检查是否是.git目录，如果是则跳过
        if '.git' in os.path.normpath(root).split(os.sep):
            # 不输出任何.git目录的信息，并跳过对.git子目录的访问
            continue
        if '.idea' in os.path.normpath(root).split(os.sep):
            # 不输出任何.git目录的信息，并跳过对.git子目录的访问
            continue

        # 检查当前目录路径
        path_parts = os.path.normpath(root).split(os.sep)

        # 检查是否在data目录的深层子目录中
        if 'data' in path_parts:
            data_index = path_parts.index('data')
            # 如果当前目录深度超过了data的直接子目录，跳过更深层的遍历
            if len(path_parts) - data_index > 2:
                # 清空dirs列表，防止os.walk继续遍历
                dirs[:] = []
                continue

        # 在dirs列表中移除'.git'，防止遍历.git目录
        if '.git' in dirs:
            dirs.remove('.git')

        # 打印当前目录
        print(f"目录: {root}")
        result.append(f"目录: {root}")

        # 打印当前目录下的文件
        for file in files:
            file_path = os.path.join(root, file)
            print(f"  文件: {file}")
            result.append(f"  文件: {file}")

        # 打印空行，增加可读性
        if files:
            print()
            result.append("")

    return result


if __name__ == "__main__":
    print("开始扫描目录结构...")
    root_dir = 'D:\\GitCode\\AdaCLIP-Pro'
    print(f"根目录: {root_dir}")
    scan_directory(root_dir)
    print("扫描完成!")