# scripts/preprocess.py

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed

from modeling.model import AdaCLIP
from modeling.clip_model import CLIP
from configs.config import parser, parse_with_config
from datasets.dataset import BaseDataset

# ======= 配置区 =======
FRAMES_DIR      = "data/MSRVTT/frames"
IMG_TMPL        = "image_{:05d}.jpg"
CKPT_PATH       = "checkpoints/msrvtt/trained_model.pth"
OUTPUT_EMBS     = "data/application/video_embs.npy"
OUTPUT_IDS      = "data/application/video_ids.json"
PROCESSED_JSON  = "data/application/processed_videos.json"
CONFIG_PATH     = "configs/msrvtt-jsfusion.json"
# ======================

def extract_frames_tensor(frames_dir, vid, num_frm, img_tmpl, preprocess):
    """
    返回 (frames_tensor, vid, None) 或 (None, vid, err_msg)
    frames_tensor: Tensor of shape (num_frm, C, H, W) on CPU
    """
    frame_dir = os.path.join(frames_dir, vid)
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not files:
        return None, vid, f"No frames for {vid}"
    idxs = np.linspace(0, len(files)-1, num=num_frm, dtype=int)
    imgs = []
    for idx in idxs:
        path = os.path.join(frame_dir, img_tmpl.format(idx+1))
        if not os.path.exists(path):
            return None, vid, f"Missing frame {path}"
        try:
            img = preprocess(Image.open(path).convert("RGB"))
        except Exception as e:
            return None, vid, f"Error loading {path}: {e}"
        imgs.append(img)
    return torch.stack(imgs), vid, None

def clear_outputs():
    for f in (OUTPUT_EMBS, OUTPUT_IDS, PROCESSED_JSON):
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device",        default="cuda", help="cuda or cpu")
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--num_workers",   type=int, default=8)
    p.add_argument("--clear",         action="store_true")
    args = p.parse_args()

    if args.clear:
        clear_outputs()

    # ——— 加载配置 ———
    base_args = parser.parse_args([])
    base_args.config = CONFIG_PATH
    cfg = parse_with_config(base_args)
    cfg.frames_dir      = FRAMES_DIR
    cfg.img_tmpl        = IMG_TMPL
    # 强制使用 MLP 聚合
    cfg.frame_agg       = "mlp"
    cfg.frame_agg_temp  = getattr(cfg, "frame_agg_temp", 1.0)

    # ——— 设备判断 ———
    use_cuda = torch.cuda.is_available() and args.device.lower().startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # ——— 视频列表 & 已处理集 ———
    all_vids   = sorted(d for d in os.listdir(FRAMES_DIR)
                       if os.path.isdir(os.path.join(FRAMES_DIR, d)))
    processed  = set(json.load(open(PROCESSED_JSON)) if os.path.exists(PROCESSED_JSON) else [])
    to_process = [v for v in all_vids if v not in processed]
    print(f"Total videos: {len(all_vids)}, to process: {len(to_process)}")
    if not to_process:
        print("Nothing to process."); exit(0)

    # ——— 构建并加载模型 ———
    clip_sd = CLIP.get_config(pretrained_clip_name=cfg.clip_backbone)
    model   = AdaCLIP(cfg, clip_sd)
    ckpt    = torch.load(CKPT_PATH, map_location="cpu")
    sd      = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=False)

    # 迁移到 GPU/CPU
    model.to(device)
    # ========= 新增：如果在 CPU 上，强制转为 float32 =========
    if device.type == "cpu":
        model.float()
    # ===========================================
    model.eval()

    # ——— 图像预处理（保持在 CPU） ———
    preprocess = BaseDataset(cfg, "annots/empty.json", is_train=False).clip_preprocess

    # ——— 如果已有存量，则加载 ———
    all_pairs = []
    if os.path.exists(OUTPUT_EMBS) and os.path.exists(OUTPUT_IDS):
        embs = np.load(OUTPUT_EMBS)
        ids  = json.load(open(OUTPUT_IDS))
        all_pairs = list(zip(ids, embs))

    save_cnt = 0

    with torch.no_grad():
        pbar = tqdm(total=len(to_process), desc="Extracting")
        for i in range(0, len(to_process), args.batch_size):
            batch = to_process[i:i+args.batch_size]
            # 并行读取 & CPU 预处理
            results = Parallel(n_jobs=args.num_workers)(
                delayed(extract_frames_tensor)(
                    FRAMES_DIR, vid, cfg.num_frm, IMG_TMPL, preprocess
                )
                for vid in batch
            )
            vids_batch, frames_batch = [], []
            for frames, vid, err in results:
                if frames is None:
                    tqdm.write(f"[SKIP] {vid}: {err}")
                else:
                    vids_batch.append(vid)
                    frames_batch.append(frames)

            if not frames_batch:
                pbar.update(len(batch))
                continue

            # CPU→GPU
            batch_tensor = torch.stack(frames_batch).to(device)  # (B,N,C,H,W)
            feats        = model.get_visual_output(batch_tensor)  # (B,N,D)
            feats        = model.frame_transformation(feats)     # (B,N,D)
            logits       = model.frame_agg_mlp(feats)            # (B,N,1)
            weights      = torch.softmax(logits / cfg.frame_agg_temp, dim=1)
            emb_batch    = (feats * weights).sum(dim=1)          # (B,D)
            emb_batch    = emb_batch.cpu().numpy()               # →CPU numpy

            # 收集
            for vid, emb in zip(vids_batch, emb_batch):
                all_pairs.append((vid, emb))
                processed.add(vid)
                save_cnt += 1

            pbar.update(len(batch))

            # 中间保存
            if save_cnt >= args.save_interval:
                ids, embs = zip(*all_pairs)
                np.save(OUTPUT_EMBS, np.stack(embs, axis=0))
                json.dump(list(ids), open(OUTPUT_IDS, 'w'))
                json.dump(list(processed), open(PROCESSED_JSON, 'w'))
                save_cnt = 0

        # 最终保存（若有剩余）
        if save_cnt > 0:
            ids, embs = zip(*all_pairs)
            np.save(OUTPUT_EMBS, np.stack(embs, axis=0))
            json.dump(list(ids), open(OUTPUT_IDS, 'w'))
            json.dump(list(processed), open(PROCESSED_JSON, 'w'))

        pbar.close()

    print(f"Finished. Total embeddings saved: {len(all_pairs)} → {OUTPUT_EMBS}")
