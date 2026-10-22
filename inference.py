# -*- coding: utf-8 -*- 
import os 
from PIL import Image
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from transformers import AutoTokenizer, CLIPTextModel

from diffusers import (
    AutoencoderKL,
    UniPCMultistepScheduler,
    UNet2DConditionModel,
)

import numpy as np 
from src import SpaTextControlNetModel, StableDiffusionSpatextControlNetPipeline


def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def inference_pipeline(
    torch_dtype=torch.float16, 
    device="cuda", 
):
    # 0. Config
    pretrained_model_path = r"./pretrained_models/stable-diffusion-v1-5"
    print(f"Loading models from base: {pretrained_model_path}")
    
    vae_model_path = r"./pretrained_models/vae_4CH_unlabeled_bz2_res512_e20_ckpt4"
    
    unet_weights_type="lora" 
    unet_weights_path = "./pretrained_models/FEGen_c3_hist01_ftsdlora_4CH_target14_processed_v2_bz16_ckpt55000"

    controlnet_model_path = r"./pretrained_models/FEGen_c3_hist01_ftsdlora_4CH_target14_processed_v2_bz16_ckpt55000/controlnet"

    # 1. Tokenizer & Text encoder
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer", use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder", torch_dtype=torch_dtype
    )

    # 2. VAE
    if vae_model_path is not None:
        print(f"Loading custom VAE from: {vae_model_path}")
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=torch_dtype)
        except OSError:
            vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder="vae", torch_dtype=torch_dtype)
    else:
        print("Loading VAE from base model...")
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_path, subfolder="vae", torch_dtype=torch_dtype
        )

    # 3. UNet
    print("Loading UNet...")
    if unet_weights_type == "lora":
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_path, subfolder="unet", torch_dtype=torch_dtype
        )

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

    # 5. ControlNet
    assert os.path.exists(controlnet_model_path)
    print(f"Loading ControlNet from: {controlnet_model_path}")
    controlnet = SpaTextControlNetModel.from_pretrained(
        controlnet_model_path,
        torch_dtype=torch_dtype
    )

    # 6. Scheduler
    scheduler = UniPCMultistepScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

    # 7. Pipeline
    print("Loading Pipeline...")
    pipeline = StableDiffusionSpatextControlNetPipeline(
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

    return pipeline


def prepare_control_image(control_image_path, resolution=512):
    control_image = Image.open(control_image_path).convert("L")
    if resolution is not None:
        control_image = control_image.resize((resolution, resolution), resample=Image.NEAREST)
    
    control_image = (np.array(control_image))[np.newaxis, :, :]
    control_image = torch.tensor(control_image)
    
    return control_image


def inference():
    # load pipeline
    pipeline = inference_pipeline()

    pipeline.enable_model_cpu_offload()
    pipeline.safety_checker = None

    # prompt
    prompt = "A realistic ultrasound image, fetal cardiac, standard four-chamber view."
    
    # condition
    condition_image_path = r"./dataset/demo_masks/conditioning_images/normal.png"
    condition_image = prepare_control_image(condition_image_path)
    
    # generator 
    generator = torch.manual_seed(666)
    
    output_image = pipeline(
        prompt=prompt,
        image=condition_image,  
        num_inference_steps=30, 
        generator=generator, 
        num_images_per_prompt=4,
        guidance_scale=2.0, 
        controlnet_conditioning_scale=1.0, 
    ).images
    
    mix_image = image_grid(output_image, rows=1, cols=4)
    save_path = "./output_dir/infer_demo.png"
    mix_image.save(save_path) 
    
    print("==> Synthesized images saved in:{}".format(save_path))
    
    
def main():   
    # demo
    inference()


if __name__ == "__main__":
    main()
    
