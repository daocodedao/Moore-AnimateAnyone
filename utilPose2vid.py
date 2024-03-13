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
from src.utils.util import *


from utils.logger_settings import api_logger
from utils.Tos import TosService

from utilVid2pose import *
#  /data/work/Moore-AnimateAnyone/venv/bin/python -m utilPose2vid --config ./configs/prompts/animation.yaml -W 512 -H 784 --srcVideoPath './youtube/6TvTJIxZca4/6TvTJIxZca4.mp4' --refImagePath './configs/inference/ref_images/anyone-3.png' --processId '6TvTJIxZca4'


# scp -r  -P 10080 fxbox@frp.fxait.com:/data/work/Moore-AnimateAnyone/output/6TvTJIxZca4_kps  /Users/linzhiji/Downloads/ 


dtype = torch.bfloat16
cuda0 = "cuda:0"
cuda1 = "cuda:1"
kMaxPoseVideoDuration = 6
kFixedFps = 24


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=784)
    parser.add_argument("-L", type=int, default=-1)
    
    parser.add_argument("--srcVideoPath", type=str)
    # parser.add_argument("--posVideoPath", type=str)
    parser.add_argument("--refImagePath", type=str)
    parser.add_argument("--processId", type=str)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    return args

def initResource(args, config):
    api_logger.info("初始化各种model")
    if config.weight_dtype == "fp16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    api_logger.info(f"weight_dtype={weight_dtype}")

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to(dtype=weight_dtype, device=cuda1)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet"
    ).to(dtype=weight_dtype, device=cuda1)

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs
    ).to(dtype=weight_dtype, device=cuda1)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=cuda1 
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path, 
    ).to(dtype=weight_dtype, device=cuda0 )

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    # width, height = args.W, args.H

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
    pipe = pipe.to(dtype=weight_dtype, device=cuda0)
    pipe.enable_vae_slicing()
    # pipe.enable_sequential_cpu_offload()
    return pipe, generator

def generateVideo(args, pipe, generator, videoPosePath, ref_image_path, outVideoPath):
    api_logger.info(f"准备生成视频， poseVideo={videoPosePath}, refImage={ref_image_path}, outVideo={outVideoPath}")
    width, height = args.W, args.H
    save_individual_videos = True
    ref_image_pil = Image.open(ref_image_path).convert("RGB")

    pose_list = []
    pose_tensor_list = []
    pose_images = read_frames(videoPosePath)
    src_fps = get_fps(videoPosePath)


    frameCount = args.L
    if frameCount == -1:
        frameCount = len(pose_images)

    api_logger.info(f"frameCount={frameCount}")
    api_logger.info(f"{videoPosePath} 视频有 {len(pose_images)} 帧, 帧率 {int(src_fps)} fps")
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

    if not save_individual_videos:
        video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
        # save_videos_from_pil
        save_videos_grid(
            video,
            outVideoPath,
            n_rows=3,
            fps=src_fps,
        )
    else:
        save_videos_grid(
            video,
            outVideoPath,
            n_rows=3,
            fps=src_fps,
        )


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)
    
    api_logger.info("1---------准备各种文件")
    ref_image_path = args.refImagePath
    # videoPosePath = args.posVideoPath
    src_video_path = args.srcVideoPath
    processId = args.processId

    outDir = f"/data/work/dance/{processId}"
    os.makedirs(outDir, exist_ok=True)

    videoSrcPath = os.path.join(outDir, f"{processId}.mp4")
    curVideoPath = videoSrcPath
    videoPosePath = os.path.join(outDir, f"{processId}-pose.mp4")

    videoSrcFixFpsPath = os.path.join(outDir, f"{processId}-fps-fixed.mp4")

    videoAudioPath = os.path.join(outDir, f"{processId}.wav")
    videoAudioInsPath = os.path.join(outDir, f"{processId}-ins.wav")
    
    videoComposePath = os.path.join(outDir, f"{processId}-composed.mp4")
    videoComposeBGMusicPath = os.path.join(outDir, f"{processId}-composed-bg.mp4")

    # pose 切割视频输出文件夹
    outSplitDir = os.path.join(outDir, "split")
    # 最终合成视频输出文件夹
    outGenDir = os.path.join(outDir, "gen")

    api_logger.info("---------检查视频文件和POSE文件")
    if not os.path.exists(videoSrcPath):
        api_logger.info(f"拷贝文件从 {src_video_path} 到 {videoSrcPath}")
        shutil.copy(src_video_path, videoSrcPath)

    if not os.path.exists(videoPosePath):
        api_logger.info(f"生成POSE文件")
        export_pose_video(videoSrcPath, videoPosePath)


    api_logger.info("---------调整POSE视频FPS")
    src_fps = get_fps(videoPosePath)
    if not os.path.exists(videoSrcFixFpsPath) and  int(src_fps) > kFixedFps:
        api_logger.info(f"原视频FPS需要调整为{kFixedFps}")
        changeVideoFps(videoPosePath, kFixedFps, videoSrcFixFpsPath)
        videoPosePath = videoSrcFixFpsPath
        api_logger.info(f"fps调整完成，现在的videoPosePath={videoPosePath}")

    api_logger.info("---------是否要提取视频里的音频")
    if not os.path.exists(videoAudioPath):
        extractAudioFromVideo(videoSrcPath, videoAudioPath)

    api_logger.info("---------是否要提取背景音乐")
    if not os.path.exists(videoAudioInsPath):
        extractBgMusic(videoAudioPath, processId, videoAudioInsPath)

    api_logger.info("---------确保切割后的文件夹存在并清空")
    shutil.rmtree(outSplitDir, ignore_errors=True)
    os.makedirs(outSplitDir, exist_ok=True)

    # shutil.rmtree(outGenDir, ignore_errors=True)
    os.makedirs(outGenDir, exist_ok=True)

    videoDuraion = video_duration(videoPosePath)


    api_logger.info("2---------初始化models")
    pipe, generator = initResource(args, config)


    api_logger.info("3---------检查切割POSE视频")
    poseVideoList = []
    if videoDuraion > kMaxPoseVideoDuration:
        api_logger.info(f"pose视频时长{videoDuraion}, 需要切割视频，{kMaxPoseVideoDuration}秒一切割")
        split_video(videoPosePath, kMaxPoseVideoDuration, outSplitDir)
        poseVideoList = [os.path.join(outSplitDir, i)  for i in os.listdir(outSplitDir) if i.endswith('mp4')]
        api_logger.info(f"切割视频完成，共有{len(poseVideoList)}个视频")
    else:
        api_logger.info(f"pose视频时长{videoDuraion}, 无需要切割视频")
        poseVideoList.append(videoPosePath)

    api_logger.info("4---------开始-合成视频-耗时比较长-耐心等待")

    model_pths = [os.path.join(outGenDir, i) for i in os.listdir(outGenDir) if i.endswith('mp4')]
    if len(model_pths) == 0:
        outVideoPathList = []
        poseVideoList.sort()
        for idx, video_path in enumerate(poseVideoList):
            outVideoPath = os.path.join(outGenDir, f"{idx}.mp4")
            try:
                generateVideo(args, pipe, generator, video_path, ref_image_path, outVideoPath)
                if os.path.exists(outVideoPath):
                    api_logger.info(f"生成视频成功，路径:{outVideoPath}")
                    outVideoPathList.append(outVideoPath)
                else:
                    api_logger.info(f"生成视频失败，路径:{outVideoPath}不存在")
            except Exception as e:
                api_logger.error(f"生成视频失败，路径:{outVideoPath}")
                api_logger.error(e)
    else:
        api_logger.info("临时处理，无需合成")

    api_logger.info("4---------结束-合成视频")

    api_logger.info("5---------合并视频,")
    model_pths = [os.path.join(outGenDir, i) for i in os.listdir(outGenDir) if i.endswith('mp4')]
    model_pths.sort()
    api_logger.info(f"共有 {len(model_pths)}")
    concatenate(model_pths, videoComposePath)
    curVideoPath = videoComposePath


    api_logger.info("6---------添加背景音乐")
    # command = f"ffmpeg -y -i {curVideoPath}  -i {videoAudioInsPath} -c:v copy -filter_complex '[0:a]aformat=fltp:44100:stereo,apad[0a];[1]aformat=fltp:44100:stereo,volume=0.6[1a];[0a][1a]amerge[a]' -map 0:v -map '[a]' -ac 2 {videoComposeBGMusicPath}"
    command = f"ffmpeg -i {curVideoPath}  -i {videoAudioInsPath} -c copy {videoComposeBGMusicPath}"



    api_logger.info(f"命令：")
    api_logger.info(command)
    result = subprocess.check_output(command, shell=True)
    log_subprocess_output(result)
    api_logger.info(f'完成背景音乐合并任务: {videoComposeBGMusicPath}')
    curVideoPath = videoComposeBGMusicPath


    api_logger.info("6---------上传到腾讯")
    bucketName = "magicphoto-1315251136"
    resultUrlPre = f"dance/video/{processId}/"
    reusultUrl = f"{resultUrlPre}{curVideoPath}"
    api_logger.info(f"上传视频 {curVideoPath}")
    if os.path.exists(curVideoPath):
        api_logger.info(f"上传视频到OSS，curVideoPath:{curVideoPath}, task.key:{reusultUrl}, task.bucketName:{bucketName}")
        TosService.upload_file(curVideoPath, reusultUrl, bucketName)
        KCDNPlayUrl="http://magicphoto.cdn.yuebanjyapp.com/"
        playUrl = f"{KCDNPlayUrl}{reusultUrl}"
        api_logger.info(f"播放地址= {playUrl}")




if __name__ == "__main__":
    main()
