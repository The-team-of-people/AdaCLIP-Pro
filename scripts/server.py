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
    return results

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
    while True:
        try:
            query = input("请输入文本查询（回车退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出检索。")
            break
        if not query:
            break

        subset_ids = compare_ids  # compare_ids 可能为 None，表示全量对比

        results = search(model, cfg, video_embs, video_ids, query, device, subset_ids)
        print("全部排序后的视频ID及相似度：")
        # for vid, score in results:
        #     print(f"{vid}\tScore: {score:.4f}")
        # print("\nTop-10:")
        for rank, (vid, score) in enumerate(results[:50], 1):
            print(f"{rank}. {vid}\tScore: {score:.4f}")