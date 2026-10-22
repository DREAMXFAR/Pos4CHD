#!/usr/bin/env python
# coding=utf-8
"""
LoRA + ControlNet批量推理脚本 - 根据metadata文件中的提示词生成对应的图像
"""

import argparse
import json
from PIL import Image
from tqdm import tqdm
import os
import torch
from transformers import AutoTokenizer, CLIPTextModel
import ipdb 

from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline, 
    ControlNetModel,
    AutoencoderKL,
    UniPCMultistepScheduler,
    UNet2DConditionModel,
    DDPMScheduler
)

import numpy as np 
from src import SpaTextControlNetModel, StableDiffusionSpatextControlNetPipeline


def load_models(
    args,
    pretrained_model_path,
    controlnet_model_path,
    unet_weights_type,
    unet_weights_path,
    vae_model_path=None,
    device="cuda",
    torch_dtype=torch.float16
):
    print(f"Loading models from base: {pretrained_model_path}")

    # 1. 加载 Tokenizer 和 Text Encoder
    # main.py 中使用了 import_model_class... 这里简化为直接加载 CLIP，适用于 SD 1.5
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer", use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder", torch_dtype=torch_dtype
    )

    # 2. 加载 VAE (完全复刻 main.py 的逻辑)
    if vae_model_path is not None:
        print(f"Loading custom VAE from: {vae_model_path}")
        # 尝试直接加载或从 subfolder 加载，兼容 main.py 的保存结构
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=torch_dtype)
        except OSError:
            vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder="vae", torch_dtype=torch_dtype)
    else:
        print("Loading VAE from base model...")
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_path, subfolder="vae", torch_dtype=torch_dtype
        )

    # 3. 加载 UNet
    print("Loading UNet...")
    if unet_weights_type == "lora":
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_path, subfolder="unet", torch_dtype=torch_dtype
        )

        # 4. 加载 LoRA 权重到 UNet (复刻 main.py 的加载方式)
        # main.py 使用 unet.load_attn_procs 而不是 pipeline.load_lora_weights
        if unet_weights_path and os.path.exists(unet_weights_path):
            print(f"Loading LoRA weights into UNet from: {unet_weights_path}")
            try:
                unet.load_attn_procs(unet_weights_path)
            except Exception as e:
                print(f"Error loading LoRA: {e}")
                raise e
        else:
            print("No LoRA weights provided or path does not exist, using base UNet only")
    elif unet_weights_type == "full":
        unet = UNet2DConditionModel.from_pretrained(
            unet_weights_path, subfolder="unet", torch_dtype=torch_dtype
        )
        print(f"Loading Full weights into UNet from: {unet_weights_path}")
    else:
        raise Exception("Weights must be lora or full!")

    # 5. 加载或初始化 ControlNet
    if args.use_spatext_ctrlnet:
        controlnet_cls = SpaTextControlNetModel
    else:
        controlnet_cls = ControlNetModel
    
    if controlnet_model_path and os.path.exists(controlnet_model_path):
        print(f"Loading ControlNet from: {controlnet_model_path}")
        controlnet = controlnet_cls.from_pretrained(
            controlnet_model_path,
            torch_dtype=torch_dtype
        )
    else:
        print("Initializing ControlNet from UNet weights (Random/Copy)...")
        controlnet = controlnet_cls.from_unet(
            unet,
            # conditioning_channels=3
        )

    # 6. 加载 Scheduler (为了推理，我们使用 UniPC，但从底模配置初始化)
    scheduler = UniPCMultistepScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

    # 7. 组装 Pipeline
    print("Assembling Pipeline...")
    if args.use_spatext_ctrlnet:
        pipeline_cls = StableDiffusionSpatextControlNetPipeline
    else:
        pipeline_cls = StableDiffusionControlNetPipeline
    
    pipeline = pipeline_cls(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )

    pipeline = pipeline.to(device)

    # 启用内存高效注意力
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention")
    except Exception as e:
        print(f"xformers not available: {e}, using default attention")

    pipeline.set_progress_bar_config(disable=True)

    return pipeline


def prepare_control_image(control_image_path, resolution=512, spatext_mode=False):
    """准备控制图像"""
    if spatext_mode:
        control_image = Image.open(control_image_path).convert("L")
        if resolution is not None:
            control_image = control_image.resize((resolution, resolution), resample=Image.NEAREST)
        
        control_image = (np.array(control_image))[np.newaxis, :, :]
        control_image = torch.tensor(control_image)
    else:
        control_image = Image.open(control_image_path).convert("RGB")
        # 调整图像大小
        if resolution is not None:
            control_image = control_image.resize((resolution, resolution), Image.LANCZOS)
        
    return control_image


def load_metadata(metadata_path):
    """加载metadata文件"""
    print(f"Loading metadata from: {metadata_path}")

    if metadata_path.endswith('.jsonl'):
        # 读取JSONL文件
        metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))
    elif metadata_path.endswith('.json'):
        # 读取JSON文件
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path}")

    print(f"Loaded {len(metadata)} entries from metadata")
    return metadata


def batch_generate_from_metadata(
        args,
        pipeline,
        metadata,
        test_data_dir,
        output_base_dir="batch_inference_results",
        num_images_per_prompt=4,
        num_inference_steps=20,
        guidance_scale=7.5,
        base_seed=42,
        controlnet_conditioning_scale=1.0,
        resolution=512
):
    """
    根据metadata批量生成图像
    """

    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)

    # 保存生成参数
    params_file = os.path.join(output_base_dir, "generation_parameters.txt")
    with open(params_file, "w", encoding="utf-8") as f:
        f.write(f"Number of images per mask: {num_images_per_prompt}\n")
        f.write(f"Inference steps: {num_inference_steps}\n")
        f.write(f"Guidance scale: {guidance_scale}\n")
        f.write(f"Base seed: {base_seed}\n")
        f.write(f"ControlNet conditioning scale: {controlnet_conditioning_scale}\n")
        f.write(f"Resolution: {resolution}\n")
        f.write(f"Total samples: {len(metadata)}\n")

    # 处理每个metadata条目
    for idx, item in enumerate(tqdm(metadata, desc="Processing metadata")):
        try:
            # 提取信息
            if args.enhanced_prompt: text = item.get('enhanced_prompt', '').strip()
            else: text = item.get('default_prompt', '').strip()
            if not text:
                print(f"[{idx + 1}/{len(metadata)}] Warning: No text found in metadata, using default prompt")
                text = "An image of sks medical scan"

            if args.use_spatext_ctrlnet:
                assert (not args.colorful_segmentation), "use conditioning image only for Spatext ControlNet"

            if args.colorful_segmentation: 
                conditioning_image_path = item.get('colorful_conditioning_image', '')
            else: 
                conditioning_image_path = item.get('conditioning_image', '')
            
            if not conditioning_image_path:
                print(f"[{idx + 1}/{len(metadata)}] Error: No conditioning_image path in metadata")
                continue

            # 构建完整的路径
            if not os.path.isabs(conditioning_image_path):
                conditioning_image_path = os.path.join(test_data_dir, conditioning_image_path)
                # conditioning_image_path = conditioning_image_path.replace("colorful_conditioning_image", "")
                # here to change the conditioning image path

            # ipdb.set_trace()
            
            if not os.path.exists(conditioning_image_path):
                print(f"[{idx + 1}/{len(metadata)}] Error: Conditioning image not found: {conditioning_image_path}")
                continue

            # 获取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(conditioning_image_path))[0]

            # 根据生成数量决定输出目录结构
            if num_images_per_prompt > 1:
                # 为每个样本创建独立文件夹
                sample_output_dir = os.path.join(output_base_dir, filename)
                os.makedirs(sample_output_dir, exist_ok=True)
            else:
                # 所有图像放在同一文件夹
                sample_output_dir = output_base_dir

            print(f"\n[{idx + 1}/{len(metadata)}] Processing: {filename}")
            print(f"  Prompt: {text[:50]}..." if len(text) > 50 else f"  Prompt: {text}")

            # 加载控制图像
            try:
                control_image = prepare_control_image(conditioning_image_path, resolution, spatext_mode=args.use_spatext_ctrlnet)
            except Exception as e:
                print(f"  Error loading control image: {e}")
                continue

            # 生成图像
            for j in range(num_images_per_prompt):
                print(f"  Generating image {j + 1}/{num_images_per_prompt}...")

                # 设置随机种子
                if base_seed is not None:
                    current_seed = base_seed + idx * 1000 + j  # 确保每个图像有不同的种子
                    generator = torch.Generator(device=pipeline.device).manual_seed(current_seed)
                else:
                    generator = None

                try:
                    # 生成图像
                    image = pipeline(
                        text,
                        control_image,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        guidance_scale=guidance_scale,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                    ).images[0]

                    # 保存图像
                    if num_images_per_prompt > 1:
                        image_filename = f"generated_{j + 1}.png"
                    else:
                        image_filename = f"{filename}{args.save_suffix}.png"

                    image_path = os.path.join(sample_output_dir, image_filename)
                    image.save(image_path)

                    print(f"  Saved: {image_filename}")

                except Exception as e:
                    print(f"  Error generating image {j + 1}: {e}")
                    continue

        except Exception as e:
            print(f"[{idx + 1}/{len(metadata)}] Error processing metadata item: {e}")
            continue

    print(f"\n=== Finish inference ===")
    print(f"所有输出已保存到: {output_base_dir}")
    print(f"处理了 {len(metadata)} 个样本")


def create_image_grid(imgs, rows, cols):
    """创建图像网格"""
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def main():
    parser = argparse.ArgumentParser(description="LoRA + ControlNet批量推理脚本 - 根据metadata生成图像")

    parser.add_argument("--colorful_segmentation", action="store_true", help="Whether to use colorful segmentation.")
    parser.add_argument("--enhanced_prompt", action="store_true", help="Whether to use enhanced prompt.")

    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default=None,
        help="Path to a pretrained VAE model (optional).",
    )
    # 模型参数
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="预训练基础模型路径（如runwayml/stable-diffusion-v1-5）"
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        default=None,
        help="ControlNet模型路径（可选，如果没有会随机初始化）"
    )
    parser.add_argument(
        "--unet_weights_type",
        type=str,
        choices=["lora", "full"], 
        default="lora",
        required=False,
        help="use full or lora weights"
    )
    parser.add_argument(
        "--unet_weights_path",
        type=str,
        default=None,
        required=False,
        help="e.g., path/to/pytorch_lora_weights.safetensors or folder"
    )

    # 数据参数
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
        help="测试数据目录"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="metadata.jsonl",
        help="metadata文件路径（相对于test_data_dir或绝对路径）"
    )

    # 生成参数
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=4,
        help="每个掩码生成的图像数量"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="推理步数"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="指导尺度（CFG scale）"
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="ControlNet条件尺度"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备：cuda 或 cpu"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="数据类型"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="图像分辨率"
    )
    parser.add_argument(
        "--save_suffix",
        type=str,
        default="_generated",
        help="保存图像文件名称后缀",
    )
    
    # use spatext controlnet
    parser.add_argument(
        "--use_spatext_ctrlnet",
        action="store_true",
        help="Use Spatext ControlNet",
    )

    args = parser.parse_args()

    # 验证参数
    if not os.path.exists(args.test_data_dir):
        raise ValueError(f"测试数据目录不存在: {args.test_data_dir}")

    # 确定metadata文件路径
    if os.path.isabs(args.metadata_file):
        metadata_path = args.metadata_file
    else:
        metadata_path = os.path.join(args.test_data_dir, args.metadata_file)

    if not os.path.exists(metadata_path):
        raise ValueError(f"metadata文件不存在: {metadata_path}")

    # 设置数据类型
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print("=== LoRA + ControlNet批量推理 ===")
    print(f"Pretrained model: {args.pretrained_model_path}")
    print(f"Pretrained VAE: {args.pretrained_vae_path}")
    print(f"Use Spatext ControlNet: {args.use_spatext_ctrlnet}")
    print(f"ControlNet model: {args.controlnet_model_path}")
    print(f"UNet weights type: {args.unet_weights_type}")
    print(f"UNet weights path: {args.unet_weights_path}")
    # print(f"LoRA weights: {args.lora_weights_path}")
    print(f"Test data directory: {args.test_data_dir}")
    print(f"Metadata file: {metadata_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of images per mask: {args.num_images_per_prompt}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")
    print(f"Resolution: {args.resolution}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"ControlNet conditioning scale: {args.controlnet_conditioning_scale}")
    print(f"Seed: {args.seed}")
    print("=" * 50)

    try:
        # 1. 加载metadata
        metadata = load_metadata(metadata_path)

        if len(metadata) == 0:
            print("Error: No data found in metadata file!")
            return

        # 2. 加载模型
        pipeline = load_models(
            args, 
            args.pretrained_model_path,
            args.controlnet_model_path,
            unet_weights_type=args.unet_weights_type, 
            unet_weights_path=args.unet_weights_path,
            vae_model_path=args.pretrained_vae_path,
            device=args.device,
            torch_dtype=torch_dtype
        )

        # 3. 批量生成
        batch_generate_from_metadata(
            args=args,
            pipeline=pipeline,
            metadata=metadata,
            test_data_dir=args.test_data_dir,
            output_base_dir=args.output_dir,
            num_images_per_prompt=args.num_images_per_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            base_seed=args.seed,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            resolution=args.resolution
        )

    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()