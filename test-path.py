import os



videoPath = "output/20240312/2030--seed_42-512x784/anyone-2_anyone-video-2_784x512_3_2030.mp4"


dirPath = os.path.dirname(videoPath)
print(f"dirPath={dirPath}")
os.makedirs(dirPath, exist_ok=True)