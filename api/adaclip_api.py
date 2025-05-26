import os
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import shutil

from modeling.model import AdaCLIP
from modeling.clip_model import CLIP
from configs.config import parser, parse_with_config
from datasets.dataset import BaseDataset
from modeling.simple_tokenizer import SimpleTokenizer

# ------------- 配置读取 -------------
def get_frames_root():
    settings_path = os.path.join(os.path.dirname(__file__), "..", "app", "user_data", "settings.json")
    settings_path = os.path.abspath(settings_path)
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)
    return settings["frames_addr"]

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
FRAMES_ROOT = get_frames_root()
IMG_TMPL = "image_%05d.jpg"
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "msrvtt", "trained_model.pth")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "msrvtt-jsfusion.json")
EMBS_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "application")

def normalize(path: str) -> str:
    # 统一大小写、去掉末尾分隔符、正斜杠
    p = os.path.normcase(os.path.normpath(path))
    return p
def extract_frames_for_dir(video_dir, out_dir, prefix=IMG_TMPL, frame_rate=-1, frame_size=-1, progress_callback=None):
    video_dir = Path(video_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_formats = [".mp4", ".mkv", ".webm", ".avi", ".mov"]
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in accepted_formats]
    total = len(videos)
    folder = normalize(str(video_dir))
    if progress_callback:
        progress_callback(folder, 0, f"开始切帧，共{total}个视频...")

    for idx, video_path in enumerate(videos):
        video_name = video_path.stem
        dst_directory_path = out_dir / video_name
        dst_directory_path.mkdir(parents=True, exist_ok=True)
        if any(dst_directory_path.iterdir()):
            if progress_callback:
                percent = int(((idx + 1) / total) * 50)
                progress_callback(folder, percent, f"[{idx+1}/{total}] 已存在帧目录: {video_name}，跳过。")
            continue
        frame_rate_str = f"-r {frame_rate}" if frame_rate > 0 else ""
        frame_size_str = ""
        if frame_size > 0:
            try:
                import ffmpeg
                probe = ffmpeg.probe(str(video_path))
                video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
                if not video_streams:
                    continue
                w = int(video_streams[0]['width'])
                h = int(video_streams[0]['height'])
            except Exception:
                w, h = 0, 0
            if min(w, h) <= frame_size:
                frame_size_str = ""
            elif w > h:
                frame_size_str = f"-vf scale=-1:{frame_size}"
            else:
                frame_size_str = f"-vf scale={frame_size}:-1"
        cmd = f'ffmpeg -nostats -loglevel 0 -i "{video_path}" -q:v 2 {frame_size_str} {frame_rate_str} "{dst_directory_path}/{prefix}"'
        # print(f"[DEBUG] Running ffmpeg: {cmd}")
        ret = subprocess.call(cmd, shell=True)
        # if ret != 0:
        #     print(f"[ERROR] ffmpeg failed for {video_path}, return code: {ret}")
        # 每处理一个视频都推送一次进度
        if progress_callback:
            percent = int(((idx + 1) / total) * 50)
            progress_callback(folder, percent, f"[{idx+1}/{total}] 切帧中: {video_name}")

    if progress_callback:
        progress_callback(folder, 50, f"切帧完成，共{total}个视频。")

def save_embeddings_per_video(embs_dir, vids, embs):
    for vid, emb in zip(vids, embs):
        vid_dir = os.path.join(embs_dir, vid)
        os.makedirs(vid_dir, exist_ok=True)
        emb_path = os.path.join(vid_dir, "embs.npy")
        np.save(emb_path, emb)

def preprocess_frames_folder(frames_dir, output_ids, processed_json, embs_dir,
                            config_path=CONFIG_PATH, ckpt_path=CKPT_PATH, device_str="cuda",
                            batch_size=8, save_interval=100, num_workers=8, progress_callback=None):
    base_args = parser.parse_args([])
    base_args.config = config_path
    cfg = parse_with_config(base_args)
    cfg.frames_dir = str(frames_dir)
    cfg.img_tmpl = IMG_TMPL
    cfg.frame_agg = "mlp"
    cfg.frame_agg_temp = getattr(cfg, "frame_agg_temp", 1.0)
    device_str = device_str.lower()
    use_cuda = torch.cuda.is_available() and device_str.startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")
    all_vids = sorted([d.name for d in Path(frames_dir).iterdir() if d.is_dir()])
    processed = set()
    if os.path.exists(processed_json):
        processed = set(json.load(open(processed_json)))
    to_process = [v for v in all_vids if v not in processed]
    total = len(to_process)
    frames_dir_norm = normalize(str(frames_dir))
    if not to_process:
        if progress_callback:
            progress_callback(frames_dir_norm, 100, "所有视频已处理，无需重复处理。")
        return
    if progress_callback:
        progress_callback(frames_dir_norm, 50, f"开始特征提取，共{total}个视频...")

    clip_sd = CLIP.get_config(pretrained_clip_name=cfg.clip_backbone)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = AdaCLIP(cfg, clip_sd)
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    if device.type == "cpu":
        model.float()
    model.eval()
    annots_path = os.path.join(PROJECT_ROOT, "annots", "empty.json")
    preprocess = BaseDataset(cfg, annots_path, is_train=False).clip_preprocess
    all_pairs = []
    save_cnt = 0
    from joblib import Parallel, delayed

    def extract_frames_tensor(frames_dir, vid, num_frm, img_tmpl, preprocess):
        frame_dir = os.path.join(frames_dir, vid)
        files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
        if not files:
            return None, vid, f"No frames for {vid}"
        idxs = np.linspace(0, len(files) - 1, num=num_frm, dtype=int)
        imgs = []
        for idx in idxs:
            path = os.path.join(frame_dir, img_tmpl % (idx + 1))
            if not os.path.exists(path):
                return None, vid, f"Missing frame {path}"
            try:
                img = preprocess(Image.open(path).convert("RGB"))
            except Exception as e:
                return None, vid, f"Error loading {path}: {e}"
            imgs.append(img)
        return torch.stack(imgs), vid, None

    with torch.no_grad():
        processed_count = 0
        for i in range(0, total, batch_size):
            batch = to_process[i:i + batch_size]
            results = Parallel(n_jobs=num_workers)(
                delayed(extract_frames_tensor)(
                    str(frames_dir), vid, cfg.num_frm, IMG_TMPL, preprocess
                )
                for vid in batch
            )
            vids_batch, frames_batch = [], []
            for frames, vid, err in results:
                if frames is None:
                    # 跳过无帧的视频，但也算进进度
                    processed_count += 1
                    if progress_callback:
                        percent = 50 + int((processed_count / total) * 50)
                        progress_callback(frames_dir_norm, percent, f"[{processed_count}/{total}] 特征提取中（无帧跳过: {vid}）")
                else:
                    vids_batch.append(vid)
                    frames_batch.append(frames)

            # 处理本批中实际有帧的视频
            if frames_batch:
                batch_tensor = torch.stack(frames_batch).to(device)
                feats = model.get_visual_output(batch_tensor)
                feats = model.frame_transformation(feats)
                logits = model.frame_agg_mlp(feats)
                weights = torch.softmax(logits / cfg.frame_agg_temp, dim=1)
                emb_batch = (feats * weights).sum(dim=1)
                emb_batch = emb_batch.cpu().numpy()

                for idx_in_batch, (vid, emb) in enumerate(zip(vids_batch, emb_batch)):
                    all_pairs.append((vid, emb))
                    processed.add(vid)
                    save_cnt += 1
                    processed_count += 1
                    if progress_callback:
                        percent = 50 + int((processed_count / total) * 50)
                        progress_callback(frames_dir_norm, percent, f"[{processed_count}/{total}] 特征提取中: {vid}")

            if save_cnt >= save_interval:
                ids, embs = zip(*all_pairs)
                json.dump(list(ids), open(output_ids, 'w'))
                save_embeddings_per_video(embs_dir, ids, embs)
                json.dump(list(processed), open(processed_json, 'w'))
                save_cnt = 0

        if all_pairs:
            ids, embs = zip(*all_pairs)
            json.dump(list(ids), open(output_ids, 'w'))
            save_embeddings_per_video(embs_dir, ids, embs)
            json.dump(list(processed), open(processed_json, 'w'))
    if progress_callback:
        progress_callback(frames_dir_norm, 100, f"特征提取完成，共{len(all_pairs)}个视频。")
def uploadDirectory(dictory_addr, frame_root=None, embs_root=EMBS_OUTPUT_ROOT, device="cuda", progress_callback=None):
    dictory_addr = os.path.abspath(dictory_addr)
    dir_name = os.path.basename(dictory_addr.rstrip("/\\"))
    frame_root = frame_root or FRAMES_ROOT
    frame_dir = os.path.join(frame_root, dir_name)
    embs_dir = os.path.join(embs_root, dir_name)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(embs_dir, exist_ok=True)

    if progress_callback:
        progress_callback(dictory_addr, 0, "上传初始化...")

    try:
        extract_frames_for_dir(dictory_addr, frame_dir, progress_callback=progress_callback)
    except Exception as e:
        if progress_callback:
            progress_callback(dictory_addr, 100, f"上传失败：视频切帧出错 - {str(e)}")
        return {"success": False, "msg": f"上传失败：视频切帧出错 - {str(e)}"}

    output_ids = os.path.join(embs_dir, "video_ids.json")
    processed_json = os.path.join(embs_dir, "processed_videos.json")
    try:
        preprocess_result = preprocess_frames_folder(
            frame_dir,
            output_ids,
            processed_json,
            embs_dir,
            device_str=device,
            progress_callback=progress_callback,
        )
    except Exception as e:
        if progress_callback:
            progress_callback(dictory_addr, 100, f"上传失败：特征提取出错 - {str(e)}")
        return {"success": False, "msg": f"上传失败：特征提取出错 - {str(e)}"}

    user_data_dir = os.path.join(os.path.dirname(__file__), "..", "app", "user_data")
    history_path = os.path.join(user_data_dir, "uploaded_dictory.json")
    os.makedirs(user_data_dir, exist_ok=True)
    history = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except Exception:
                history = []
    if dictory_addr not in history:
        history.append(dictory_addr)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    emb_files = {}
    if os.path.exists(output_ids):
        with open(output_ids, "r", encoding="utf-8") as f:
            vids = json.load(f)
        for vid in vids:
            emb_path = os.path.join(embs_dir, vid, "embs.npy")
            emb_files[vid] = emb_path

    if progress_callback:
        progress_callback(dictory_addr, 100, f"目录 {dictory_addr} 中的视频已全部处理完毕。")

    return {
        "frame_dir": frame_dir,
        "embs_dir": embs_dir,
        "video_ids": output_ids,
        "processed_json": processed_json,
        "video_emb_files": emb_files,
        "success": True,
        "msg": f"目录 {dictory_addr} 中的视频已经预处理完毕。"
    }
def load_per_video_embs(embs_dir):
    """
    加载 embs_dir 下所有视频的 embedding。返回 (video_id_list, embs_array, video_path_dict)
    embs_dir: 形如 data/application/某个上传目录/
    返回：
      - video_ids: [str, ...]
      - embs: [N, D] np.ndarray
      - video_path_dict: {video_id: 视频绝对路径}
    """
    ids_path = os.path.join(embs_dir, "video_ids.json")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"找不到 video_ids.json: {ids_path}")
    with open(ids_path, "r", encoding="utf-8") as f:
        video_ids = json.load(f)
    embs = []
    video_path_dict = {}
    for vid in video_ids:
        emb_path = os.path.join(embs_dir, vid, "embs.npy")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"embedding文件不存在: {emb_path}")
        emb = np.load(emb_path)
        embs.append(emb)
        # 真实视频（原文件）路径——假设原始目录保存在 user_data/uploaded_dictory.json
        # 这里假定视频名和原始视频名一致，否则可传递绝对路径映射
        # 用帧推断原始目录
        # 建议：你可以在 embedding 预处理时，把原始绝对路径存到一个 json 里做映射
        # 这里假定帧目录为 FRAMES_ROOT/目录名/vid，原视频为上传目录/vid + .mp4等
        # 实际更好做法：上传时存下完整原始视频绝对路径列表到 embs_dir/video_abs_paths.json
        # 这里用一种通用的推测方式（如需更稳妥请用映射表）
        video_path_dict[vid] = None  # 这里留空，由外部补全
    embs = np.stack(embs, axis=0)
    return video_ids, embs, video_path_dict

def videoQuery(dictory_addrs, message, device="cuda"):
    """
    dictory_addrs: 已上传目录（绝对路径组成的list），如 [".../my_videos", ".../sports"]
    message: 用户输入文本
    返回：全部视频真实绝对路径合集（按与文本相似度降序排序，list）
    """
    # 读取 user_data/uploaded_dictory.json 建立 {上传目录: [原视频绝对路径]} 映射
    # 若每个上传目录下有 embs_dir/video_abs_paths.json 用它更好
    user_data_dir = os.path.join(os.path.dirname(__file__), "..", "app", "user_data")
    history_path = os.path.join(user_data_dir, "uploaded_dictory.json")
    video_abs_paths_map = {}  # {目录名: {vid: abs_path}}
    # 建议：上传时顺带存一份 embs_dir/video_abs_paths.json
    for dir_addr in dictory_addrs:
        dir_name = os.path.basename(dir_addr.rstrip("/\\"))
        abs_map_path = os.path.join(EMBS_OUTPUT_ROOT, dir_name, "video_abs_paths.json")
        if os.path.exists(abs_map_path):
            with open(abs_map_path, "r", encoding="utf-8") as f:
                video_abs_paths_map[dir_name] = json.load(f)
        else:
            # fallback: 尝试用上传目录下查找
            video_abs_paths_map[dir_name] = {}
            for ext in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
                for f in os.listdir(dir_addr):
                    if f.endswith(ext):
                        name = os.path.splitext(f)[0]
                        video_abs_paths_map[dir_name][name] = os.path.abspath(os.path.join(dir_addr, f))

    all_video_ids = []
    all_embs = []
    all_id_to_path = {}

    for dir_addr in dictory_addrs:
        dir_name = os.path.basename(dir_addr.rstrip("/\\"))
        embs_dir = os.path.join(EMBS_OUTPUT_ROOT, dir_name)
        if not os.path.exists(embs_dir):
            continue
        video_ids, embs, _ = load_per_video_embs(embs_dir)
        all_video_ids.extend([f"{dir_name}:{vid}" for vid in video_ids])
        all_embs.append(embs)
        for vid in video_ids:
            abs_path = video_abs_paths_map.get(dir_name, {}).get(vid, None)
            if abs_path is not None:
                all_id_to_path[f"{dir_name}:{vid}"] = abs_path
            else:
                # fallback：帧目录/上传目录猜测
                # 不推荐，仅做兜底
                guessed_path = None
                for ext in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
                    p = os.path.join(dir_addr, vid + ext)
                    if os.path.exists(p):
                        guessed_path = os.path.abspath(p)
                        break
                if guessed_path:
                    all_id_to_path[f"{dir_name}:{vid}"] = guessed_path

    if not all_embs:
        return []
    all_embs = np.concatenate(all_embs, axis=0)

    # 2. 文本转embedding
    base_dir = dictory_addrs[0]
    config_path = CONFIG_PATH
    ckpt_path = CKPT_PATH
    base_args = parser.parse_args([])
    base_args.config = config_path
    cfg = parse_with_config(base_args)
    device_str = device.lower()
    use_cuda = torch.cuda.is_available() and device_str.startswith("cuda")
    device_obj = torch.device("cuda" if use_cuda else "cpu")
    clip_sd = CLIP.get_config(pretrained_clip_name=cfg.clip_backbone)
    model = AdaCLIP(cfg, clip_sd)
    ckpt = torch.load(ckpt_path, map_location=device_obj)
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.to(device_obj)
    if device_obj.type == "cpu":
        model.float()
    model.eval()
    tokenizer = SimpleTokenizer()
    # 编码文本
    tokens = tokenizer.encode(message)
    tokens = tokens[:cfg.max_txt_len]
    tokens_tensor = torch.zeros((1, cfg.max_txt_len), dtype=torch.int64)
    tokens_tensor[0, :len(tokens)] = torch.tensor(tokens)
    tokens_tensor = tokens_tensor.to(device_obj)
    if tokens_tensor.ndim == 2:
        tokens_tensor = tokens_tensor.unsqueeze(1)  # [1, 1, max_txt_len]
    with torch.no_grad():
        text_feat = model.get_text_output(tokens_tensor)
        if isinstance(text_feat, tuple):
            text_feat = text_feat[0]
        text_feat = text_feat.cpu().numpy().reshape(-1)

    # 3. 检索
    sims = np.dot(all_embs, text_feat) / (np.linalg.norm(all_embs, axis=1) * np.linalg.norm(text_feat) + 1e-8)
    idx_sorted = np.argsort(-sims)
    # 返还全部视频真实绝对路径（按相似度排序）
    video_addrs = []
    for idx in idx_sorted:
        abs_path = all_id_to_path.get(all_video_ids[idx], None)
        if abs_path is not None:
            video_addrs.append(abs_path)
    return video_addrs

def scanAndCheckDictory(dictory_addrs=None):
    """
    扫描并检查上传目录列表的状态（预处理文件与视频实际存在性）。
    - dictory_addrs: 可为None或目录绝对路径列表，如果为None则扫描全部 data/application 下的目录。
    功能：
        - 检查每个目录下视频的 video_ids.json、embs.npy、帧目录等是否与实际一致。
        - 如有视频被删除，则移除对应embedding和帧目录、更新 video_ids.json/processed_videos.json。
        - 如有新视频加入，则触发帧抽取/embedding生成。
    返回：
        - 每个目录的最新视频列表（含已处理与未处理）、状态、错误信息等。
    """
    results = {}
    # 若未指定则扫描全部 application 下的目录
    root_dir = EMBS_OUTPUT_ROOT
    if dictory_addrs is None or len(dictory_addrs) == 0:
        # 默认扫描全部已上传的目录
        dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    else:
        # 用户指定目录
        dirs = []
        for d in dictory_addrs:
            dir_name = os.path.basename(d.rstrip("/\\"))
            dirs.append(os.path.join(root_dir, dir_name))

    for embs_dir in dirs:
        dir_name = os.path.basename(embs_dir)
        result = {
            "embs_dir": embs_dir,
            "frame_dir": os.path.join(FRAMES_ROOT, dir_name),
            "video_ids": [],
            "removed": [],
            "added": [],
            "errors": [],
            "success": True
        }
        # 1. 获取物理帧目录下的视频名
        frame_dir = result["frame_dir"]
        if not os.path.exists(frame_dir):
            result["errors"].append(f"帧目录不存在 {frame_dir}")
            result["success"] = False
            results[dir_name] = result
            continue
        actual_videos = sorted([d for d in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, d))])

        # 2. 获取 video_ids.json 中记录
        ids_path = os.path.join(embs_dir, "video_ids.json")
        if os.path.exists(ids_path):
            with open(ids_path, "r", encoding="utf-8") as f:
                ids_videos = json.load(f)
        else:
            ids_videos = []
        result["video_ids"] = ids_videos

        # 3. 检查已删除/新增
        removed = [vid for vid in ids_videos if vid not in actual_videos]
        added = [vid for vid in actual_videos if vid not in ids_videos]
        result["removed"] = removed
        result["added"] = added

        # 4. 对于已删除视频，移除embedding、更新json
        for vid in removed:
            emb_path = os.path.join(embs_dir, vid, "embs.npy")
            emb_dir = os.path.dirname(emb_path)
            if os.path.exists(emb_path):
                try:
                    os.remove(emb_path)
                except Exception as e:
                    result["errors"].append(f"移除embedding失败 {emb_path}: {e}")
            if os.path.exists(emb_dir):
                try:
                    os.rmdir(emb_dir)
                except Exception as e:
                    pass  # 目录可能非空，不强制删除
        # 更新 video_ids.json
        new_ids_videos = [vid for vid in ids_videos if vid not in removed]
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(new_ids_videos, f, ensure_ascii=False, indent=2)
        # processed_videos.json 也同步更新
        processed_json = os.path.join(embs_dir, "processed_videos.json")
        if os.path.exists(processed_json):
            with open(processed_json, "r", encoding="utf-8") as f:
                processed = set(json.load(f))
            processed = [vid for vid in processed if vid not in removed]
            with open(processed_json, "w", encoding="utf-8") as f:
                json.dump(list(processed), f, ensure_ascii=False, indent=2)

        # 5. 对于新加视频，自动触发帧抽取和embedding生成
        if added:
            added_dirs = [os.path.join(frame_dir, vid) for vid in added]
            # 这里只处理已切帧的（即帧目录存在且有图片）
            # 若你希望自动切帧，可加入视频文件扫描和extract_frames_for_dir调用
            # 自动补embedding
            preprocess_frames_folder(
                frame_dir,
                ids_path,
                processed_json,
                embs_dir,
                device_str="cuda" if torch.cuda.is_available() else "cpu"
            )

        # 最终刷新一次video_ids
        if os.path.exists(ids_path):
            with open(ids_path, "r", encoding="utf-8") as f:
                result["video_ids"] = json.load(f)

        results[dir_name] = result

    return results

import os
import shutil
import json
from pathlib import Path

# ...其余依赖已在你前面的代码中导入...

def addVideoToDictory(video_addrs, dictory_addr, frame_root=None, embs_root=EMBS_OUTPUT_ROOT, device="cuda"):
    """
    将video_addrs中的全部视频（绝对路径）拷贝到dictory_addr（绝对路径）目录下，自动去重（不覆盖同名），
    并对新增视频做切帧和embedding预处理。返回新增视频名列表和处理结果。
    """
    dictory_addr = os.path.abspath(dictory_addr)
    os.makedirs(dictory_addr, exist_ok=True)
    added_videos = []
    skipped_videos = []
    # 1. 拷贝视频
    for vpath in video_addrs:
        vpath = os.path.abspath(vpath)
        vname = os.path.basename(vpath)
        dst_path = os.path.join(dictory_addr, vname)
        if os.path.exists(dst_path):
            skipped_videos.append(vname)
            continue  # 不覆盖同名
        try:
            shutil.copy2(vpath, dst_path)
            added_videos.append(vname)
        except Exception as e:
            skipped_videos.append(vname)
    # 2. 预处理（只处理新增）
    if added_videos:
        dir_name = os.path.basename(dictory_addr.rstrip("/\\"))
        frame_root = frame_root or FRAMES_ROOT
        frame_dir = os.path.join(frame_root, dir_name)
        os.makedirs(frame_dir, exist_ok=True)
        # 只对新加视频切帧
        extract_frames_for_dir(dictory_addr, frame_dir)
        # 只对新增帧做embedding（只补充未处理帧）
        embs_dir = os.path.join(embs_root, dir_name)
        os.makedirs(embs_dir, exist_ok=True)
        output_ids = os.path.join(embs_dir, "video_ids.json")
        processed_json = os.path.join(embs_dir, "processed_videos.json")
        preprocess_frames_folder(
            frame_dir,
            output_ids,
            processed_json,
            embs_dir,
            device_str=device,
        )
    return {
        "added": added_videos,
        "skipped": skipped_videos,
        "success": True,
        "msg": f"新增视频:{added_videos} 已处理；跳过:{skipped_videos}"
    }

def deleteVideo(video_addrs, frame_root=None, embs_root=EMBS_OUTPUT_ROOT):
    """
    删除已上传目录中的视频（绝对路径列表），并清理帧目录和embedding目录下相关预处理文件。
    """
    deleted = []
    not_found = []
    for vpath in video_addrs:
        vpath = os.path.abspath(vpath)
        vname = os.path.basename(vpath)
        dir_addr = os.path.dirname(vpath)
        dir_name = os.path.basename(dir_addr.rstrip("/\\"))
        # 1. 删除原视频文件
        if os.path.exists(vpath):
            try:
                os.remove(vpath)
                deleted.append(vpath)
            except Exception:
                not_found.append(vpath)
                continue
        else:
            not_found.append(vpath)
            continue
        # 2. 删除帧目录
        frame_root_ = frame_root or FRAMES_ROOT
        frame_dir = os.path.join(frame_root_, dir_name, os.path.splitext(vname)[0])
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir, ignore_errors=True)
        # 3. 删除embedding
        embs_dir = os.path.join(embs_root, dir_name, os.path.splitext(vname)[0])
        if os.path.exists(embs_dir):
            shutil.rmtree(embs_dir, ignore_errors=True)
        # 4. 更新video_ids.json和processed_videos.json
        embs_dir_root = os.path.join(embs_root, dir_name)
        ids_path = os.path.join(embs_dir_root, "video_ids.json")
        processed_json = os.path.join(embs_dir_root, "processed_videos.json")
        vid_noext = os.path.splitext(vname)[0]
        # 更新 ids
        if os.path.exists(ids_path):
            with open(ids_path, "r", encoding="utf-8") as f:
                ids = json.load(f)
            ids = [i for i in ids if i != vid_noext]
            with open(ids_path, "w", encoding="utf-8") as f:
                json.dump(ids, f, ensure_ascii=False, indent=2)
        # 更新 processed
        if os.path.exists(processed_json):
            with open(processed_json, "r", encoding="utf-8") as f:
                processed = json.load(f)
            processed = [i for i in processed if i != vid_noext]
            with open(processed_json, "w", encoding="utf-8") as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)
    return {
        "deleted": deleted,
        "not_found": not_found,
        "success": True,
        "msg": f"成功删除:{deleted}；未找到或无法删除:{not_found}"
    }