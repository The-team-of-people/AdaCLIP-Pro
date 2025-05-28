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
def extract_frames_for_dir(video_dir, out_dir,
                           prefix=IMG_TMPL,
                           frame_rate=-1,
                           frame_size=-1,
                           progress_callback=None,
                           origin_folder=None):
    """
    切帧阶段占前 50% 进度，每处理完一个视频就回调一次。
    已存在帧目录时跳过，但仍会回调进度。
    """
    video_dir = Path(video_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 支持的视频格式
    accepted_formats = [".mp4", ".mkv", ".webm", ".avi", ".mov"]
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in accepted_formats]
    total = len(videos)
    folder = normalize(str(video_dir))

    # 0% 初始回调
    if progress_callback:
        progress_callback(folder, 0, f"开始切帧，共{total}个视频...")

    for idx, video_path in enumerate(videos, start=1):
        video_name = video_path.stem
        dst = out_dir / video_name
        dst.mkdir(parents=True, exist_ok=True)

        # 如果已存在帧目录，则跳过并回调
        if any(dst.iterdir()):
            if progress_callback:
                pct = int(idx / total * 50)
                progress_callback(folder, pct,
                                  f"[{idx}/{total}] 已存在帧目录，跳过：{video_name}")
            continue

        # 构建 ffmpeg 参数
        frame_rate_str = f"-r {frame_rate}" if frame_rate > 0 else ""
        frame_size_str = ""
        if frame_size > 0:
            try:
                import ffmpeg
                probe = ffmpeg.probe(str(video_path))
                vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")
                w, h = int(vstream["width"]), int(vstream["height"])
                if min(w, h) > frame_size:
                    if w > h:
                        frame_size_str = f"-vf scale=-1:{frame_size}"
                    else:
                        frame_size_str = f"-vf scale={frame_size}:-1"
            except Exception:
                pass

        # 执行切帧
        cmd = (
            f'ffmpeg -nostats -loglevel 0 -i "{video_path}" -q:v 2 '
            f'{frame_size_str} {frame_rate_str} '
            f'"{dst}/{prefix}"'
        )
        subprocess.call(cmd, shell=True)

        # 每个视频后立即回调进度
        if progress_callback:
            pct = int(idx / total * 50)
            progress_callback(folder, pct,
                              f"[{idx}/{total}] 已切帧：{video_name}")

    # 切帧完成后发 50%
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
                            batch_size=8, save_interval=100, num_workers=8, progress_callback=None, origin_folder=None):
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
    # 关键：始终用原始目录做 progress_callback 的 key
    folder_for_callback = normalize(origin_folder if origin_folder is not None else str(frames_dir))
    if not to_process:
        if progress_callback:
            progress_callback(folder_for_callback, 100, "所有视频已处理，无需重复处理。")
        return
    if progress_callback:
        progress_callback(folder_for_callback, 50, f"开始特征提取，共{total}个视频...")

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
                        progress_callback(folder_for_callback, percent, f"[{processed_count}/{total}] 特征提取中（无帧跳过: {vid}）")
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
                        progress_callback(folder_for_callback, percent, f"[{processed_count}/{total}] 特征提取中: {vid}")

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
        progress_callback(folder_for_callback, 100, f"特征提取完成，共{len(all_pairs)}个视频。")

def uploadDirectory(dictory_addr, frame_root=None, embs_root=EMBS_OUTPUT_ROOT, device="cuda", progress_callback=None):
    dictory_addr = os.path.abspath(dictory_addr)
    dir_name = os.path.basename(dictory_addr.rstrip("/\\"))
    frame_root = frame_root or FRAMES_ROOT
    frame_dir = os.path.join(frame_root, dir_name)
    embs_dir = os.path.join(embs_root, dir_name)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(embs_dir, exist_ok=True)

    folder_for_callback = normalize(dictory_addr)
    if progress_callback:
        progress_callback(folder_for_callback, 0, "上传初始化...")

    try:
        extract_frames_for_dir(dictory_addr, frame_dir, progress_callback=progress_callback, origin_folder=dictory_addr)
    except Exception as e:
        if progress_callback:
            progress_callback(folder_for_callback, 100, f"上传失败：视频切帧出错 - {str(e)}")
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
            origin_folder=dictory_addr,  # 新增参数
        )
    except Exception as e:
        if progress_callback:
            progress_callback(folder_for_callback, 100, f"上传失败：特征提取出错 - {str(e)}")
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
        progress_callback(folder_for_callback, 100, f"目录 {dictory_addr} 中的视频已全部处理完毕。")

    return {
        "frame_dir": frame_dir,
        "embs_dir": embs_dir,
        "video_ids": output_ids,
        "processed_json": processed_json,
        "video_emb_files": emb_files,
        "success": True,
        "msg": f"目录 {dictory_addr} 中的视频已经预处理完毕。"
    }
import os
import json
import numpy as np
import torch
from modeling.model import AdaCLIP
from modeling.clip_model import CLIP
from configs.config import parser, parse_with_config
from modeling.simple_tokenizer import SimpleTokenizer

# 全局缓存对象
class VideoSearchCache:
    def __init__(self):
        self.model = None
        self.cfg = None
        self.device = None
        self.all_video_ids = []
        self.all_embs = None
        self.all_id_to_path = {}

    def is_ready(self):
        return (self.model is not None and
                self.cfg is not None and
                self.device is not None and
                self.all_embs is not None and
                len(self.all_video_ids) > 0)

VIDEO_SEARCH_CACHE = VideoSearchCache()

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
        video_path_dict[vid] = None  # 这里留空，由外部补全
    embs = np.stack(embs, axis=0)
    return video_ids, embs, video_path_dict

def load_video_search_assets(dictory_addrs, device="cuda"):
    """
    加载模型和全部embedding到内存，device可以是"cuda"或"cpu"
    """
    user_data_dir = os.path.join(os.path.dirname(__file__), "..", "app", "user_data")
    video_abs_paths_map = {}
    for dir_addr in dictory_addrs:
        dir_name = os.path.basename(dir_addr.rstrip("/\\"))
        abs_map_path = os.path.join(EMBS_OUTPUT_ROOT, dir_name, "video_abs_paths.json")
        if os.path.exists(abs_map_path):
            with open(abs_map_path, "r", encoding="utf-8") as f:
                video_abs_paths_map[dir_name] = json.load(f)
        else:
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
                guessed_path = None
                for ext in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
                    p = os.path.join(dir_addr, vid + ext)
                    if os.path.exists(p):
                        guessed_path = os.path.abspath(p)
                        break
                if guessed_path:
                    all_id_to_path[f"{dir_name}:{vid}"] = guessed_path

    if not all_embs:
        raise RuntimeError("未找到任何embedding数据。")
    all_embs = np.concatenate(all_embs, axis=0)

    # ---- 加载模型 ----
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

    # ---- 写入全局缓存 ----
    VIDEO_SEARCH_CACHE.model = model
    VIDEO_SEARCH_CACHE.cfg = cfg
    VIDEO_SEARCH_CACHE.device = device_obj
    VIDEO_SEARCH_CACHE.all_video_ids = all_video_ids
    VIDEO_SEARCH_CACHE.all_embs = all_embs
    VIDEO_SEARCH_CACHE.all_id_to_path = all_id_to_path

def cached_video_query(message, topk=None):
    """
    用内存中的热数据做检索，返回按相似度排序的绝对路径list
    """
    cache = VIDEO_SEARCH_CACHE
    if not cache.is_ready():
        raise RuntimeError("模型和embedding尚未加载，请先调用load_video_search_assets()")

    model = cache.model
    cfg = cache.cfg
    device_obj = cache.device
    all_video_ids = cache.all_video_ids
    all_embs = cache.all_embs
    all_id_to_path = cache.all_id_to_path

    # 文本转embedding
    tokenizer = SimpleTokenizer()
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

    # 检索
    sims = np.dot(all_embs, text_feat) / (np.linalg.norm(all_embs, axis=1) * np.linalg.norm(text_feat) + 1e-8)
    idx_sorted = np.argsort(-sims)
    if topk:
        idx_sorted = idx_sorted[:topk]
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