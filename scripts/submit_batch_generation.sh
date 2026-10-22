#!/bin/bash

# 4CH_target14_controlnet_format ----------------------------------------------------------------------------
BASE_MODEL_PATH=/path/to/stable-diffusion-v1-5/
VAE_PATH=/path/to/vae
CTRLNET_PATH=/path/to/fegen

TEST_DATA_DIR=./dataset/demo_dataset/train
META_DATA=./dataset/demo_dataset/train/metadata.jsonl

OUTPUT_ROOT=./output_dir/generation_results

SEED=42

for CKPT_ID in "${CKPT_LIST[@]}"
do
    OUTPUT_DIR=${OUTPUT_ROOT}/scale_disc_ftvae_4CH_target14_${CTRLNET_PATH}_seed42_cfg2p0
    
    UNET_LORA=./pretrained_models/${CTRLNET_PATH}

    CUDA_VISIBLE_DEVICES=0 python generate.py \
        --use_spatext_ctrlnet \
        --pretrained_vae_path ${VAE_PATH} \
        --pretrained_model_path ${BASE_MODEL_PATH} \
        --controlnet_model_path ${CTRLNET_PATH} \
        --unet_weights_type "lora" \
        --unet_weights_path ${UNET_LORA} \
        --test_data_dir ${TEST_DATA_DIR} \
        --metadata_file ${META_DATA} \
        --output_dir ${OUTPUT_DIR} \
        --num_images_per_prompt 1 \
        --num_inference_steps 30 \
        --guidance_scale 2.0 \
        --controlnet_conditioning_scale 1.0 \
        --seed ${SEED} \
        --resolution 512 \
        --save_suffix ""
done