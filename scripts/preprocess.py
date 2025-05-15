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
FRAMES_DIR = "data/MSRVTT/frames"
IMG_TMPL = "image_{:05d}.jpg"
CKPT_PATH = "checkpoints/msrvtt/trained_model.pth"
OUTPUT_EMBS = "data/application/video_embs.npy"
OUTPUT_IDS = "data/application/video_ids.json"
PROCESSED_JSON = "data/application/processed_videos.json"
CONFIG_PATH = "configs/msrvtt-jsfusion.json"
# ======================

def extract_frames_tensor(frames_dir, vid, num_frm, img_tmpl, preprocess):
    frame_dir = os.path.join(frames_dir, vid)
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    if len(frame_files) == 0:
        return None, vid, f"No frames found for video {vid}"
    if len(frame_files) < num_frm:
        indices = np.linspace(0, len(frame_files)-1, num=len(frame_files), dtype=int)
    else:
        indices = np.linspace(0, len(frame_files)-1, num=num_frm, dtype=int)
    imgs = []
    for idx in indices:
        frame_path = os.path.join(frame_dir, img_tmpl.format(idx+1))
        if not os.path.exists(frame_path):
            return None, vid, f"Frame file does not exist: {frame_path}"
        try:
            img = preprocess(Image.open(frame_path).convert("RGB"))
        except Exception as e:
            return None, vid, f"Error loading {frame_path}: {e}"
        imgs.append(img)
    return torch.stack(imgs, dim=0), vid, None

def clear_files():
    for f in [OUTPUT_EMBS, OUTPUT_IDS, PROCESSED_JSON]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    import argparse

    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("--device", type=str, default="cuda")
    parser_arg.add_argument("--batch_size", type=int, default=8)
    parser_arg.add_argument("--save_interval", type=int, default=100)
    parser_arg.add_argument("--num_workers", type=int, default=8)
    parser_arg.add_argument("--clear", action="store_true", help="Clear output files before run")
    args = parser_arg.parse_args()

    if args.clear:
        clear_files()

    # 读取配置
    parsed_args = parser.parse_args(args=[])
    parsed_args.config = CONFIG_PATH
    cfg = parse_with_config(parsed_args)
    cfg.frames_dir = FRAMES_DIR
    cfg.img_tmpl = IMG_TMPL

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # 获取所有帧目录下的视频ID
    all_video_list = [name for name in os.listdir(FRAMES_DIR) if os.path.isdir(os.path.join(FRAMES_DIR, name))]
    all_video_list = sorted(all_video_list)
    print(f"Found {len(all_video_list)} videos with frames.")

    # 初始化已处理视频
    processed_videos = set()
    if os.path.isfile(PROCESSED_JSON):
        with open(PROCESSED_JSON, "r") as f:
            processed_videos = set(json.load(f))

    # 本次需要处理的视频
    video_list = [vid for vid in all_video_list if vid not in processed_videos]
    print(f"{len(video_list)} videos to be processed this run.")

    if not video_list:
        print("No new videos to process. Exiting.")
        exit(0)

    # 加载模型
    clip_state_dict = CLIP.get_config(pretrained_clip_name=cfg.clip_backbone)
    model = AdaCLIP(cfg, clip_state_dict)
    if os.path.isfile(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH, map_location="cpu")
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
            model.load_state_dict(checkpoint['model'].state_dict(), strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"❌ Model weights not found at: {CKPT_PATH}")
        exit(1)
    model = model.to(device)
    model.eval()

    # 获取clip预处理
    clip_preprocess = BaseDataset(cfg, "annots/empty.json", is_train=False).clip_preprocess

    # 如果已存在旧的embs和ids，先载入，否则初始化
    if os.path.isfile(OUTPUT_EMBS) and os.path.isfile(OUTPUT_IDS):
        video_embs = list(np.load(OUTPUT_EMBS))
        with open(OUTPUT_IDS, "r") as f:
            video_ids = json.load(f)
    else:
        video_embs = []
        video_ids = []

    new_video_embs = []
    new_video_ids = []
    save_count = 0
    BATCH_SIZE = args.batch_size
    SAVE_INTERVAL = args.save_interval
    NUM_WORKERS = args.num_workers

    with torch.no_grad():
        pbar = tqdm(total=len(video_list), desc="Extracting video features")
        for start_idx in range(0, len(video_list), BATCH_SIZE):
            batch_vids = video_list[start_idx:start_idx+BATCH_SIZE]
            # 多线程并行帧图片读取+预处理（仅CPU/IO密集，安全）
            results = Parallel(n_jobs=NUM_WORKERS)(
                delayed(extract_frames_tensor)(
                    FRAMES_DIR, vid, cfg.num_frm, IMG_TMPL, clip_preprocess
                )
                for vid in batch_vids
            )
            batch_tensors = []
            batch_valid_vids = []
            for frames_tensor, vid, err in results:
                if frames_tensor is None:
                    tqdm.write(f"[SKIP] {vid} due to error: {err}")
                    continue
                batch_tensors.append(frames_tensor)
                batch_valid_vids.append(vid)
            if not batch_tensors:
                pbar.update(len(batch_vids))
                continue
            frames_tensor = torch.stack(batch_tensors).to(device)
            try:
                if hasattr(cfg, 'use_policy') and cfg.use_policy:
                    if getattr(cfg, 'policy_backbone', None) == "clip":
                        frame_embd = model.get_visual_output(frames_tensor)
                        actions, logits = model.sampler(frame_embd)
                    else:
                        actions, logits = model.sampler(frames_tensor)
                        frame_embd = model.get_visual_output(frames_tensor)
                    frame_embd = torch.matmul(actions, frame_embd)
                    frame_embd = frame_embd.sum(dim=1, keepdim=True)
                else:
                    frame_embd = model.get_visual_output(frames_tensor)
                    frame_embd = frame_embd.mean(dim=1, keepdim=True)
                frame_embd = model.frame_transformation(frame_embd)
                embs = frame_embd.squeeze(1).cpu().numpy()
            except Exception as e:
                tqdm.write(f"[ERROR] Feature extraction failed for batch: {batch_valid_vids}, error: {e}")
                continue
            for emb, vid in zip(embs, batch_valid_vids):
                new_video_embs.append(emb)
                new_video_ids.append(vid)
                processed_videos.add(vid)
                save_count += 1
            pbar.update(len(batch_vids))
            if save_count >= SAVE_INTERVAL:
                all_video_embs = list(video_embs) + new_video_embs
                all_video_ids = list(video_ids) + new_video_ids
                np.save(OUTPUT_EMBS, np.stack(all_video_embs))
                with open(OUTPUT_IDS, "w") as f:
                    json.dump(all_video_ids, f)
                video_embs = all_video_embs.copy()
                video_ids = all_video_ids.copy()
                with open(PROCESSED_JSON, "w") as f:
                    json.dump(list(processed_videos), f)
                tqdm.write(f"[INFO] Auto-saved progress at {len(all_video_ids)} videos.")
                new_video_embs = []
                new_video_ids = []
                save_count = 0
        if new_video_embs:
            all_video_embs = list(video_embs) + new_video_embs
            all_video_ids = list(video_ids) + new_video_ids
            np.save(OUTPUT_EMBS, np.stack(all_video_embs))
            with open(OUTPUT_IDS, "w") as f:
                json.dump(all_video_ids, f)
            video_embs = all_video_embs.copy()
            video_ids = all_video_ids.copy()
            with open(PROCESSED_JSON, "w") as f:
                json.dump(list(processed_videos), f)
            tqdm.write(f"[INFO] Final save at {len(all_video_ids)} videos.")
    print(f"Finished. Saved {len(video_ids)} video embeddings to {OUTPUT_EMBS}, ids to {OUTPUT_IDS}")
    print(f"Final processed video record at {PROCESSED_JSON}")