import os
import json
import numpy as np
import torch
from modeling.model import AdaCLIP
from modeling.clip import tokenize
from modeling.clip_model import CLIP
from configs.config import parser, parse_with_config

def pad_tokens(tokens, max_txt_len):
    out = torch.zeros((len(tokens), max_txt_len), dtype=torch.long)
    for i, tks in enumerate(tokens):
        if len(tks) > max_txt_len:
            tks = tks[:max_txt_len-1] + [tks[-1]]
        out[i, :len(tks)] = torch.tensor(tks)
    return out

def check_order_and_shape(test_ids, video_ids, video_embs):
    """
    检查test_ids（如MSRVTT测试集id顺序）与video_ids.json是否一一对应并顺序一致，同时检查embs shape
    """
    if not test_ids or len(test_ids) == 0:
        print("⚠️ 未指定test_ids，跳过顺序检查。")
        return

    print(f"检查 test_ids({len(test_ids)})，video_ids({len(video_ids)})，video_embs({len(video_embs)})...")

    # 检查 test_ids 是否都在 video_ids
    missing = [vid for vid in test_ids if vid not in video_ids]
    if missing:
        print(f"❌ 以下test set id在video_ids.json中找不到: {missing}")
    else:
        print("✅ 所有test set id都包含在video_ids.json中。")

    # 检查数量
    if len(test_ids) > len(video_ids):
        print(f"❌ test_ids数量({len(test_ids)})大于video_ids({len(video_ids)})，请检查！")

    if len(video_embs) != len(video_ids):
        print(f"❌ video_embs数量({len(video_embs)})与video_ids数量({len(video_ids)})不一致，请检查！")
    else:
        print("✅ video_embs数量与video_ids数量一致。")

    # 检查顺序
    all_match = True
    min_len = min(len(test_ids), len(video_ids))
    for idx in range(min_len):
        if video_ids[idx] != test_ids[idx]:
            print(f"❌ 顺序不一致: test_ids[{idx}]={test_ids[idx]}, video_ids[{idx}]={video_ids[idx]}")
            all_match = False
    if all_match and len(test_ids) == len(video_ids):
        print("✅ test set id顺序和video_ids.json完全一致！")
    elif all_match:
        print("⚠️ test_ids和video_ids顺序一致，但长度不同。")
    else:
        print("❌ test set id顺序和video_ids.json有不一致，请检查！")
    return all_match

def search(model, cfg, video_embs, video_ids, text, device, subset_ids=None):
    tokens = tokenize([text], context_length=cfg.max_txt_len)
    tokens = pad_tokens(tokens, cfg.max_txt_len)
    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(1)  # [batch, 1, seq_len]
    tokens = tokens.to(device)
    with torch.no_grad():
        text_embd, _ = model.get_text_output(tokens, return_hidden=False)  # (1, 1, D)
        text_embd = text_embd.squeeze(0).squeeze(0)
        text_embd = text_embd / text_embd.norm(dim=-1, keepdim=True)
    # 视频编码
    if subset_ids is not None:
        indices = [video_ids.index(vid) for vid in subset_ids if vid in video_ids]
        sub_video_embs = video_embs[indices]
        sub_video_ids = [video_ids[i] for i in indices]
    else:
        sub_video_embs = video_embs
        sub_video_ids = video_ids
    # 转为 torch，搬到 device
    video_embs_tensor = torch.tensor(sub_video_embs, dtype=torch.float32, device=device)
    video_embs_tensor = video_embs_tensor / video_embs_tensor.norm(dim=-1, keepdim=True)
    # 批量相似度
    sims = torch.matmul(video_embs_tensor, text_embd)
    sims = sims.cpu().numpy()
    sorted_idx = np.argsort(-sims)
    results = [(sub_video_ids[i], float(sims[i])) for i in sorted_idx]
    return results, sims, sub_video_ids

if __name__ == "__main__":
    import argparse
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("--config", type=str, default="configs/msrvtt-jsfusion.json")
    parser_arg.add_argument("--video_embs", type=str, default="data/application/video_embs.npy")
    parser_arg.add_argument("--video_ids", type=str, default="data/application/video_ids.json")
    parser_arg.add_argument("--device", type=str, default="cuda")
    parser_arg.add_argument("--frames_dir", type=str, default="data/MSRVTT/frames")
    parser_arg.add_argument("--compare_json", type=str, default="data/application/server_compare_list.json")
    parser_arg.add_argument("--processed_json", type=str, default="data/application/processed_videos.json")
    parser_arg.add_argument("--test_ids", type=str, default="data/MSRVTT/test_video_ids.txt", help="测试集id顺序文件（每行一个video id，可为空）")
    args = parser_arg.parse_args()

    parsed_args = parser.parse_args(args=[])
    parsed_args.config = args.config
    cfg = parse_with_config(parsed_args)
    cfg.frames_dir = args.frames_dir

    # 自动切换CPU/GPU
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # 加载全部视频特征和ID
    video_embs = np.load(args.video_embs)
    with open(args.video_ids, "r") as f:
        video_ids = json.load(f)

    # 加载test_ids
    test_ids = []
    if args.test_ids and os.path.isfile(args.test_ids):
        with open(args.test_ids, "r") as f:
            test_ids = [line.strip() for line in f if line.strip()]

    # 检查顺序和shape
    check_order_and_shape(test_ids, video_ids, video_embs)

    # 优先用 compare_json，如果没有则用 processed_json（全量视频）
    if os.path.isfile(args.compare_json):
        with open(args.compare_json, "r") as f:
            compare_ids = json.load(f)
            print(f"Loaded {len(compare_ids)} video ids for comparison from {args.compare_json}")
    elif os.path.isfile(args.processed_json):
        with open(args.processed_json, "r") as f:
            compare_ids = json.load(f)
            print(f"Loaded {len(compare_ids)} video ids for comparison from {args.processed_json}")
    else:
        compare_ids = None
        print("No compare_json or processed_json found, will use all video embeddings.")

    # 加载模型
    clip_state_dict = CLIP.get_config(pretrained_clip_name=cfg.clip_backbone)
    model = AdaCLIP(cfg, clip_state_dict)
    ckpt_path = os.path.join("checkpoints", cfg.dataset, "trained_model.pth")
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'].state_dict(), strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    if device.type == "cpu":
        model = model.float()
    model.eval()

    print(f"Ready for text-to-video retrieval! (device: {device})")
    print("如需顺序debug，请确保--test_ids为测试集id顺序文件，每行一个video id。")
    # 当前用于自动索引的query序号
    query_idx = 0
    while True:
        try:
            query = input("请输入文本查询（回车退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出检索。")
            break
        if not query:
            break

        subset_ids = compare_ids  # compare_ids 可能为 None，表示全量对比

        results, sims, sub_video_ids = search(model, cfg, video_embs, video_ids, query, device, subset_ids)

        print("全部排序后的视频ID及相似度：")
        for rank, (vid, score) in enumerate(results[:50], 1):
            print(f"{rank}. {vid}\tScore: {score:.4f}")

        # 如果test_ids存在且不为空，输出ground-truth排名
        if test_ids and len(test_ids) > query_idx:
            gt_vid = test_ids[query_idx]
            gt_rank = None
            for i, (vid, score) in enumerate(results):
                if vid == gt_vid:
                    gt_rank = i+1
                    print(f"👉 Ground-truth视频id: {gt_vid} 在当前query下排名: {gt_rank}, 相似度: {score:.4f}")
                    break
            if gt_rank is None:
                print(f"⚠️ Ground-truth视频id: {gt_vid} 未在compare_ids或检索结果中！")
            query_idx += 1
        else:
            print("未指定ground-truth或查询序号超出test_ids范围。")