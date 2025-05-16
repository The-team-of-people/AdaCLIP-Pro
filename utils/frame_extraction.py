from __future__ import print_function, division
import os
import time
import subprocess
from tqdm import tqdm
import argparse
from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset processor: Video->Frames")
    parser.add_argument("dir_path", type=str, help="original dataset path")
    parser.add_argument("dst_dir_path", type=str, help="dest path to save the frames")
    parser.add_argument("--prefix", type=str, default="image_%05d.jpg", help="output image type")
    parser.add_argument("--accepted_formats", type=str, nargs="+",
                        help="list of input video formats (e.g. .mp4 .mkv .webm .avi)")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=666666666)
    parser.add_argument("--file_list", type=str, default="")
    parser.add_argument("--frame_rate", type=int, default=-1, help="frame sampling rate. -1 is native frame rate")
    parser.add_argument("--frame_size", type=int, default=-1, help="shortest frame size. -1 for native size")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()
    if args.accepted_formats is None:
        args.accepted_formats = [".mp4", ".mkv", ".webm", ".avi"]
    args.accepted_formats = [f.lower() for f in args.accepted_formats]
    return args

def par_job(args_tuple):
    command, dry_run = args_tuple
    if dry_run:
        print(command)
    else:
        subprocess.call(command, shell=True)

if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    dir_path = args.dir_path
    dst_dir_path = args.dst_dir_path

    # 文件名获取
    if args.file_list == "":
        file_names = sorted(os.listdir(dir_path))
    else:
        with open(args.file_list, "r", encoding="utf-8") as f:
            file_names = [x.strip() for x in f if x.strip()]
    filtered_file_names = []
    for file_name in file_names:
        _, ext = os.path.splitext(file_name)
        if ext.lower() in args.accepted_formats:
            filtered_file_names.append(file_name)
    file_names = filtered_file_names
    file_names = file_names[args.begin:args.end + 1]
    print("%d videos to handle" % (len(file_names)))
    cmd_list = []
    for file_name in tqdm(file_names, desc="Generating commands"):
        name, ext = os.path.splitext(file_name)
        dst_directory_path = os.path.join(dst_dir_path, name)
        video_file_path = os.path.join(dir_path, file_name)
        if not os.path.exists(dst_directory_path):
            os.makedirs(dst_directory_path, exist_ok=True)
        if os.listdir(dst_directory_path):
            continue
        frame_rate_str = "-r %d" % args.frame_rate if args.frame_rate > 0 else ""
        frame_size_str = ""
        if args.frame_size > 0:
            try:
                result = os.popen(
                    f"ffprobe -hide_banner -loglevel error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 \"{video_file_path}\""
                )
                wh_str = result.readline().rstrip()
                if not wh_str or "," not in wh_str:
                    raise ValueError("ffprobe result empty or format error")
                w, h = [int(d) for d in wh_str.split(",")]
            except Exception as e:
                print(f"Error with video {video_file_path}! {e}")
                continue
            if min(w, h) <= args.frame_size:
                frame_size_str = ""
            elif w > h:
                frame_size_str = "-vf scale=-1:%d" % args.frame_size
            else:
                frame_size_str = "-vf scale=%d:-1" % args.frame_size
        cmd = 'ffmpeg -nostats -loglevel 0 -i "{}" -q:v 2 {} {} "{}/{}"'.format(
            video_file_path, frame_size_str, frame_rate_str, dst_directory_path, args.prefix)
        if not args.parallel:
            if args.dry_run:
                print(cmd)
            else:
                subprocess.call(cmd, shell=True)
        cmd_list.append(cmd)

    if args.parallel:
        task_list = [(cmd, args.dry_run) for cmd in cmd_list]
        with Pool(processes=args.num_workers) as pool:
            with tqdm(total=len(task_list), desc="Extracting frames") as pbar:
                for _ in pool.imap_unordered(par_job, task_list):
                    pbar.update()
    t1 = time.time()
    print("Finished in %.4f seconds" % (t1 - t0))
    if os.name == "posix":  # 只在 Linux/macOS 下执行
        os.system("stty sane")