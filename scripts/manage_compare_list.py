"""
用于管理 data/application/server_compare_list.json 文件，
可将自定义视频id列表写入该文件，供 server.py 检索时限定比较范围。
"""

import json
import os

def main():
    compare_json = "D:\GitCode\AdaCLIP\data/application/server_compare_list.json"
    video_ids_json = "D:\GitCode\AdaCLIP\data/application/video_ids.json"

    if not os.path.isfile(video_ids_json):
        print(f"{video_ids_json} 不存在！请先运行 process.py 生成。")
        return

    with open(video_ids_json, "r") as f:
        all_video_ids = json.load(f)

    print(f"共有 {len(all_video_ids)} 个已提特征视频可选。")
    print("请输入要加入 compare_list 的视频名，用英文逗号分隔（如 video123,video456,video789）：")
    raw = input().strip()
    if not raw:
        print("未输入，退出。")
        return

    input_ids = [x.strip() for x in raw.split(",") if x.strip() in all_video_ids]
    if not input_ids:
        print("没有输入有效视频名，退出。")
        return

    with open(compare_json, "w") as f:
        json.dump(input_ids, f)
    print(f"已将 {len(input_ids)} 个视频id写入 {compare_json}")

if __name__ == "__main__":
    main()