from src.utils.util import *
import os
import shutil

# pose_video_path = "./youtube/6TvTJIxZca4/6TvTJIxZca4_kps.mp4"
# outFpsDIr = "./youtube/6TvTJIxZca4/split"
# out_video_path = "./youtube/6TvTJIxZca4/6TvTJIxZca4_kps_fps.mp4"


# video_out_dir = "./output/6TvTJIxZca4_kps/gen/"
# video_compose_path = "./output/6TvTJIxZca4_kps/compose.mp4"
# model_pths = [os.path.join(video_out_dir, i) for i in os.listdir(video_out_dir) if i.endswith('mp4')]
# model_pths.sort()
# concatenate(model_pths, video_compose_path)

# # shutil.rmtree(outFpsDIr, ignore_errors=True)
# videoDuraion = video_duration(pose_video_path)
# fps = get_fps(pose_video_path)

# changeVideoFps(filePath=pose_video_path, outFilePath=out_video_path, fps=24)

# # if not os.path.exists(outFpsDIr):
#     # os.makedirs(outFpsDIr, exist_ok=True)

# # pose_images = read_frames(pose_video_path)
# # split_video(filename=pose_video_path, segment_length = 10, output_dir=outFpsDIr)
# print("done")


processId = "6TvTJIxZca4"
outDir = "./output/"

outVideoDir = os.path.join(outDir, processId)
os.makedirs(outVideoDir, exist_ok=True)

src_video_path = "./youtube/6TvTJIxZca4/6TvTJIxZca4.mp4"
out_audio_path = os.path.join(outVideoDir, f"{processId}.wav")
audioInsPath = os.path.join(outVideoDir, f"{processId}-ins.wav")


extractAudioFromVideo(src_video_path, out_audio_path)
extractBgMusic(out_audio_path, "6TvTJIxZca4", audioInsPath)