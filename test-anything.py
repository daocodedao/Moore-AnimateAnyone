import os
from pathlib import Path
from utils.util import Util



reImageDir = "/Users/linzhiji/Documents/code/Moore-AnimateAnyone/configs/inference/ref_images/girl"

refImagePaths = Util.get_image_paths_from_folder(reImageDir)

for refImagePath in refImagePaths:
    refImageName = Path(refImagePath).stem
    print(refImageName)