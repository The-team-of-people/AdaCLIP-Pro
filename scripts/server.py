# scripts/server.py

import os
import json
import numpy as np
import torch
from modeling.model import AdaCLIP
from modeling.clip import tokenize      # 依赖项目内的 tokenize 实现
from modeling.clip_model import CLIP
from configs.config import parser, parse_with_config

def pad_tokens(tokens, max_len, device):
    """
    把 tokenize 得到的 ID 列表 pad 或截断到 max_len，并返回 GPU 张量
    """
    batch_size = len(tokens)
    out = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, tks in enumerate(tokens):
        if len(tks) > max_len:
            tks = tks[: max_len - 1] + [tks[-1]]
        out[i, : len(tks)] = torch.tensor(tks, device=device)
    return out

def load_model(cfg, device):
    """
    初始化 AdaCLIP 并加载训练好的 checkpoint
    """
    # 1) 加载 CLIP 预训练权重配置
    clip_state = CLIP.get_config(pretrained_clip_name=cfg.clip_backbone)
    model = AdaCLIP(cfg, clip_state)

    # 2) 加载 AdaCLIP 训练好的 checkpoint
    ckpt_path = os.path.join("checkpoints", cfg.dataset, "trained_model.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    # 3) 放到 GPU/CPU 并切换到 eval 模式
    model.to(device)
    if device.type == "cpu":
        model = model.float()
    model.eval()
    return model

def search(model, cfg, video_embs, video_ids, query, device, subset_ids=None, topk=10):
    """
    对单条 query 文本进行检索，返回前 topk 个 (video_id, score)
    """
    # 1) tokenize + pad → GPU
    token_lists = tokenize([query], context_length=cfg.max_txt_len)  # list of lists
    tokens = pad_tokens(token_lists, cfg.max_txt_len, device)        # (1, L)
    tokens = tokens.unsqueeze(1)                                     # (1,1,L)

    # 2) 文本编码 + 归一化  （在 GPU 上）
    with torch.no_grad():
        text_embd, _ = model.get_text_output(tokens, return_hidden=False)  # (1,1,D)
        text_embd = text_embd.squeeze(0).squeeze(0)                        # (D,)
        text_embd = text_embd / text_embd.norm()

    # 3) 子集筛选（可选）
    if subset_ids is not None:
        idx_map = {vid: idx for idx, vid in enumerate(video_ids)}
        idxs   = [idx_map[v] for v in subset_ids if v in idx_map]
        ids    = [video_ids[i] for i in idxs]
        embs   = video_embs[idxs]
    else:
        ids  = video_ids
        embs = video_embs

    # 4) 视频特征张量化 + 归一化（GPU）
    v = torch.from_numpy(embs).to(device)               # (N, D)
    v = v / v.norm(dim=1, keepdim=True)

    # 5) 相似度计算（GPU）并乘 logit_scale
    with torch.no_grad():
        scale = model.clip.logit_scale.exp().to(device)
        sims  = scale * (v @ text_embd)                  # (N,)

    # 6) topk 排序（GPU），再转 CPU 提取结果
    topk_idx = torch.topk(sims, k=topk, largest=True).indices  # (topk,)
    topk_scores = sims[topk_idx].cpu().tolist()
    topk_ids    = [ids[i] for i in topk_idx.cpu().tolist()]

    return list(zip(topk_ids, topk_scores))

if __name__ == "__main__":
    import argparse

    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("--config",        type=str, default="configs/msrvtt-jsfusion.json")
    parser_arg.add_argument("--video_embs",    type=str, default="data/application/video_embs.npy")
    parser_arg.add_argument("--video_ids",     type=str, default="data/application/video_ids.json")
    parser_arg.add_argument("--subset_json",   type=str, default=None,
                             help="可选，仅检索 subset_json 中的 IDs")
    parser_arg.add_argument("--device",        type=str, default="cuda")
    parser_arg.add_argument("--topk",          type=int, default=10)
    args = parser_arg.parse_args()

    # 1) 加载配置
    base_args = parser.parse_args([])
    base_args.config = args.config
    cfg = parse_with_config(base_args)

    # 2) 设备选择
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda")
                          else "cpu")

    # 3) 加载预计算的视频 embeddings & IDs （顺序对齐）
    video_embs = np.load(args.video_embs)     # shape: (N, D)
    with open(args.video_ids, "r") as f:
        video_ids = json.load(f)

    # 4) 如果提供 subset_json，则只检索该子集
    subset_ids = None
    if args.subset_json and os.path.exists(args.subset_json):
        subset_ids = json.load(open(args.subset_json))
        print(f"Using subset: {len(subset_ids)} videos")

    # 5) 加载模型（仅需要 text encoder）
    model = load_model(cfg, device)
    print(f"Server ready! Device: {device}. Enter empty line to exit.\n")

    # 6) 交互式检索
    while True:
        try:
            q = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            break

        results = search(model, cfg, video_embs, video_ids, q, device, subset_ids, topk=args.topk)
        print(f"\nTop-{args.topk} results:")
        for rank, (vid, score) in enumerate(results, 1):
            print(f"{rank:02d}. {vid}\t{score:.4f}")
        print()
