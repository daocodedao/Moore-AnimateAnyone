import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
import sys
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# from moviepy.editor import VideoFileClip
from moviepy.editor import *
from utils.logger_settings import api_logger
import subprocess
import time

def log_subprocess_output(inStr):
    if len(inStr) > 0:
        inStr = inStr.decode(sys.stdout.encoding)
        logStrList = inStr.split('\n')
        for line in logStrList:
            api_logger.info(line)


def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)
    
    print(f"path={path}")
    dirPath = os.path.dirname(path)
    print(f"dirPath={dirPath}")
    os.makedirs(dirPath, exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps


def split_video(filename, segment_length, output_dir):
    clip = VideoFileClip(filename)
    duration = clip.duration

    start_time = 0
    end_time = segment_length
    i = 1

    # Extract the filename without extension
    basename = os.path.basename(filename).split('.')[0]

    # Extract directory path
    # dir_path = os.path.dirname(filename)

    # output_path = os.path.join(dir_path, output_dir)

    # Create output directory if it doesn't exist
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    while start_time < duration:
        output = os.path.join(output_dir, f"{basename}_part{i}.mp4")
        ffmpeg_extract_subclip(filename, start_time, min(end_time, duration), targetname=output)
        start_time = end_time
        end_time += segment_length
        i += 1
    print(f'Video split into {i-1} parts.')

def video_duration(filename):
    clip = VideoFileClip(filename)
    return clip.duration

def changeVideoFps(filePath, fps=30, outFilePath=None):
    print(f"now fps = {int(get_fps(filePath))}")
    clip = VideoFileClip(filePath)
    if outFilePath is None:
        outFilePath = filePath

    clip.write_videofile(outFilePath, fps=fps)
    print(f"now fps = {int(get_fps(outFilePath))}")


def concatenate(video_clip_paths, output_path, method="compose"):
    """Concatenates several video files into one video file
    and save it to `output_path`. Note that extension (mp4, etc.) must be added to `output_path`
    `method` can be either 'compose' or 'reduce':
        `reduce`: Reduce the quality of the video to the lowest quality on the list of `video_clip_paths`.
        `compose`: type help(concatenate_videoclips) for the info"""
    # create VideoFileClip object for each video file
    clips = [VideoFileClip(c) for c in video_clip_paths]
    if method == "reduce":
        # calculate minimum width & height across all clips
        min_height = min([c.h for c in clips])
        min_width = min([c.w for c in clips])
        # resize the videos to the minimum
        clips = [c.resize(newsize=(min_width, min_height)) for c in clips]
        # concatenate the final video
        final_clip = concatenate_videoclips(clips)
    elif method == "compose":
        # concatenate the final video with the compose method provided by moviepy
        final_clip = concatenate_videoclips(clips, method="compose")
    # write the output video file
    final_clip.write_videofile(output_path)


def extractAudioFromVideo(srcVideoPath, outAudioPath):
    api_logger.info(f"从视频剥离音频文件 {srcVideoPath}")
    command = f"ffmpeg -y -i {srcVideoPath} -vn -acodec pcm_f32le -ar 44100 -ac 2 {outAudioPath}"
    api_logger.info(command)
    result = subprocess.check_output(command, shell=True)
    log_subprocess_output(result)


def extractBgMusic(srcAudioPath, processId, audioInsPath):
    try:
        for tryIndex in range(0,5):
            try:
                api_logger.info(f"第{tryIndex}获取背景音乐")
                command = f"/data/work/GPT-SoVITS/start-urv.sh -s {srcAudioPath} -i {processId} -n {audioInsPath}"
                api_logger.info(f"命令：")
                api_logger.info(command)
                result = subprocess.check_output(command, shell=True)
                log_subprocess_output(result)
                if os.path.exists(audioInsPath):
                    api_logger.info(f'完成音频urv任务: {audioInsPath}')
                    break
            except Exception as e:
                api_logger.error(f"第{tryIndex}次，获取背景音乐失败：{e} 休息2秒后重试")
                time.sleep(2)

        if os.path.exists(audioInsPath):
            api_logger.info(f"背景音乐 {audioInsPath} 获取成功")
        else:
            api_logger.error(f"背景音乐 {audioInsPath} 获取失败")
    except Exception as e:
        api_logger.error(f"视频加上背景音乐失败：{e}")
