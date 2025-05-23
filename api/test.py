import os

# 假设 adaclip_api.py 已在同目录或已加入PYTHONPATH
from adaclip_api import (
    uploadDirectory,
    videoQuery,
    scanAndCheckDictory,
    addVideoToDictory,
    deleteVideo
)

# 测试目录（你的视频目录）
test_video_dir = r"D:\GitCode\AdaCLIP-Pro\data\test_video"

def print_divider(title):
    print("=" * 30)
    print(title)
    print("=" * 30)

def test_upload():
    print_divider("测试上传目录")
    result = uploadDirectory(test_video_dir, device="cuda")
    print(result)
    return result

def test_query():
    print_divider("测试视频语义检索")
    # 假设只查一个目录，也可以多个
    dirs = [test_video_dir]
    # 查询文本
    message = "一个人在打篮球"
    result = videoQuery(dirs, message, device="cuda")
    print("按相似度排序的绝对路径:")
    for p in result:
        print(p)
    return result

def test_scan_check():
    print_divider("测试目录一致性检查/刷新")
    # 为空则自动查所有上传过的目录
    result = scanAndCheckDictory()
    print(result)
    return result

def test_add_video():
    print_divider("测试手动添加视频到目录")
    # 先随便拿test_video_dir下的一个视频复制一份，模拟“新视频”
    video_files = [os.path.join(test_video_dir, f) for f in os.listdir(test_video_dir) if f.endswith((".mp4", ".mkv", ".webm", ".avi", ".mov"))]
    if not video_files:
        print("无视频文件，跳过新增测试")
        return
    # 假定你想再加回自己（模拟多次添加/去重）
    result = addVideoToDictory(video_files, test_video_dir)
    print(result)
    return result

def test_delete_video():
    print_divider("测试删除视频")
    # 随便删一个
    video_files = [os.path.join(test_video_dir, f) for f in os.listdir(test_video_dir) if f.endswith((".mp4", ".mkv", ".webm", ".avi", ".mov"))]
    if not video_files:
        print("无视频文件，跳过删除测试")
        return
    # 只删第一个
    to_delete = [video_files[0]]
    result = deleteVideo(to_delete)
    print(result)
    return result

if __name__ == "__main__":
    # 按需调用各测试函数
    test_upload()
    test_query()
    test_scan_check()
    test_add_video()
    test_delete_video()