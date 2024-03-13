from src.utils.util import *
import os
import shutil

pose_video_path = "./youtube/6TvTJIxZca4/6TvTJIxZca4_kps.mp4"
outFpsDIr = "./youtube/6TvTJIxZca4/split"
out_video_path = "./youtube/6TvTJIxZca4/6TvTJIxZca4_kps_fps.mp4"



# shutil.rmtree(outFpsDIr, ignore_errors=True)
videoDuraion = video_duration(pose_video_path)
fps = get_fps(pose_video_path)

changeVideoFps(filePath=pose_video_path, outFilePath=out_video_path, fps=24)

# if not os.path.exists(outFpsDIr):
    # os.makedirs(outFpsDIr, exist_ok=True)

# pose_images = read_frames(pose_video_path)
# split_video(filename=pose_video_path, segment_length = 10, output_dir=outFpsDIr)
print("done")
