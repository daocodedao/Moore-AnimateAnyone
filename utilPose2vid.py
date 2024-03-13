import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

import av
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from configs.prompts.test_cases import TestCasesDict
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid


from utils.logger_settings import api_logger
from utils.Tos import TosService

#  python -m utilPose2vid --config ./configs/prompts/animation.yaml -W 512 -H 784 --posVideoPath './youtube/6TvTJIxZca4/6TvTJIxZca4_kps.mp4' --refImagePath './configs/inference/ref_images/anyone-2.png'


dtype = torch.bfloat16
cuda0 = "cuda:0"
cuda1 = "cuda:1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=784)
    parser.add_argument("-L", type=int, default=-1)
    
    parser.add_argument("--posVideoPath", type=str)
    parser.add_argument("--refImagePath", type=str)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

    ref_image_path = args.refImagePath
    pose_video_path = args.posVideoPath

    if config.weight_dtype == "fp16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    api_logger.info(f"weight_dtype={weight_dtype}")

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to(cuda1, dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=cuda1)

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=cuda1)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=cuda1
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device=cuda1)

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    pipe:Pose2VideoPipeline = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(cuda1, dtype=weight_dtype)
    pipe.enable_vae_slicing()
    pipe.enable_sequential_cpu_offload()

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{width}x{height}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    ref_name = Path(ref_image_path).stem
    pose_name = Path(pose_video_path).stem.replace("_kps", "")

    ref_image_pil = Image.open(ref_image_path).convert("RGB")

    pose_list = []
    pose_tensor_list = []
    pose_images = read_frames(pose_video_path)
    src_fps = get_fps(pose_video_path)

    frameCount = args.L
    if frameCount == -1:
        frameCount = len(pose_images)

    api_logger.info(f"frameCount={frameCount}")
    api_logger.info(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )
    for pose_image_pil in pose_images[: frameCount]:
        pose_tensor_list.append(pose_transform(pose_image_pil))
        pose_list.append(pose_image_pil)

    ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
    ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(
        0
    )  # (1, c, 1, h, w)
    ref_image_tensor = repeat(
        ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=frameCount
    )

    pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
    pose_tensor = pose_tensor.transpose(0, 1)
    pose_tensor = pose_tensor.unsqueeze(0)

    api_logger.info(f"pipe video")

    video = pipe(
        ref_image_pil,
        pose_list,
        width,
        height,
        frameCount,
        args.steps,
        args.cfg,
        generator=generator,
    ).videos

    video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
    videoName = f"{ref_name}_{pose_name}_{height}x{width}_{int(args.cfg)}_{time_str}"
    curVideoPath = f"{save_dir}/{videoName}.mp4"
    api_logger.info(f"saving video to {curVideoPath}")
    save_videos_grid(
        video,
        curVideoPath,
        n_rows=3,
        fps=src_fps if args.fps is None else args.fps,
    )

    bucketName = "magicphoto-1315251136"
    resultUrlPre = f"animate/video/{videoName}/"
    reusultUrl = f"{resultUrlPre}{curVideoPath}"
    api_logger.info(f"上传视频 {curVideoPath}")
    if os.path.exists(curVideoPath):
        api_logger.info(f"上传视频到OSS，curVideoPath:{curVideoPath}, task.key:{reusultUrl}, task.bucketName:{bucketName}")
        TosService.upload_file(curVideoPath, reusultUrl, bucketName)
        KCDNPlayUrl="http://magicphoto.cdn.yuebanjyapp.com/"
        playUrl = f"{KCDNPlayUrl}{reusultUrl}"
        api_logger.info(f"播放地址= {playUrl}")




# scp -r  -P 10065 fxbox@frp.fxait.com:/data/work/Moore-AnimateAnyone/output/20240312  /Users/linzhiji/Downloads/ 

if __name__ == "__main__":
    main()