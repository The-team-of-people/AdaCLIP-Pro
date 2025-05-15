import json
import numpy as np

# 路径根据你的实际情况修改
video_ids_path = "data/application/video_ids.json"
test_ids_path = "data/MSRVTT/test_video_ids.txt"  # 一行一个video id，比如 video0
video_embs_path = "data/application/video_embs.npy"

# 读取 video_ids.json
with open(video_ids_path, "r") as f:
    video_ids = json.load(f)

# 读取 test set id（如MSRVTT测试集1000条）
with open(test_ids_path, "r") as f:
    test_ids = [line.strip() for line in f if line.strip()]

# 读取 video_embs 的 shape
video_embs = np.load(video_embs_path)
assert len(video_embs) == len(video_ids), f"video_embs和video_ids长度不一致！{len(video_embs)} vs {len(video_ids)}"

# 检查 test_ids 是否都在 video_ids 中
missing = [vid for vid in test_ids if vid not in video_ids]
if missing:
    print(f"以下test set id在video_ids.json中找不到：{missing}")
else:
    print("所有test set id都包含在video_ids.json中。")

# 检查顺序
all_match = True
for idx, vid in enumerate(test_ids):
    if video_ids[idx] != vid:
        print(f"顺序不一致: test_ids[{idx}]={vid}, video_ids[{idx}]={video_ids[idx]}")
        all_match = False

if all_match:
    print("✅ test set id顺序和video_ids.json完全一致！")
else:
    print("❌ test set id顺序和video_ids.json有不一致，请检查！")