from src.dwpose import DWposeDetector
import os
from pathlib import Path

from src.utils.util import get_fps, read_frames, save_videos_from_pil
import numpy as np
from utils.logger_settings import api_logger
from utils.Tos import TosService

# python vid2pose.py --video_path ./youtube/6TvTJIxZca4/6TvTJIxZca4.mp4


def export_pose_video(srcVideoPath, poseVideoPath):


    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--video_path", type=str)
    # args = parser.parse_args()

    if not os.path.exists(srcVideoPath):
        raise ValueError(f"Path: {srcVideoPath} not exists")

    # dir_path, video_name = (
    #     os.path.dirname(srcVideoPath),
    #     os.path.splitext(os.path.basename(srcVideoPath))[0],
    # )
    # out_path = os.path.join(dir_path, video_name + "_kps.mp4")

    detector = DWposeDetector()
    detector = detector.to(f"cuda")

    fps = get_fps(srcVideoPath)
    frames = read_frames(srcVideoPath)
    kps_results = []
    for i, frame_pil in enumerate(frames):
        result, score = detector(frame_pil)
        score = np.mean(score, axis=-1)

        kps_results.append(result)

    api_logger.info(f"out_path={poseVideoPath}")
    save_videos_from_pil(kps_results, poseVideoPath, fps=fps)

    # curVideoPath = out_path
    # bucketName = "magicphoto-1315251136"
    # resultUrlPre = f"animate/video/{video_name}/"
    # reusultUrl = f"{resultUrlPre}{curVideoPath}"
    # api_logger.info(f"上传视频 {curVideoPath}")
    # if os.path.exists(curVideoPath):
    #     api_logger.info(f"上传视频到OSS，curVideoPath:{curVideoPath}, task.key:{reusultUrl}, task.bucketName:{bucketName}")
    #     TosService.upload_file(curVideoPath, reusultUrl, bucketName)
    #     KCDNPlayUrl="http://magicphoto.cdn.yuebanjyapp.com/"
    #     playUrl = f"{KCDNPlayUrl}{reusultUrl}"
    #     api_logger.info(f"播放地址= {playUrl}")

