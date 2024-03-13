from src.utils.util import *
import os
import shutil

video_out_dir = "/Users/linzhiji/Downloads/6TvTJIxZca4_kps/gen"
video_compose_path = "/Users/linzhiji/Downloads/6TvTJIxZca4_kps/compose.mp4"
model_pths = [os.path.join(video_out_dir, i) for i in os.listdir(video_out_dir) if i.endswith('mp4')]
model_pths.sort()
concatenate(model_pths, video_compose_path)