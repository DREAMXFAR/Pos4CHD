#!/bin/bash
nnUNet_raw="./nnUNet_dataset/nnUNet_raw"
nnUNet_preprocessed="./nnUNet_dataset/nnUNet_preprocessed"
nnUNet_results="./output_dir/checkpoints/nnUNet"

### VSD data copy
INPUT_ROOT=./dataset/demo_dataset/train
DIR_NAME=images
OUTPUT_ROOT=./downstream_tasks/nnUNet/output_dir/nnUNet_pred

GT_MASK=./dataset/demo_dataset/train/conditioning_images


CUDA_VISIBLE_DEVICES=2 python run_prediction_nnUNet_dice_for_dir.py \
    --dir_name ${DIR_NAME} \
    --input_root ${INPUT_ROOT} \
    --output_root ${OUTPUT_ROOT} \
    --gt_mask_dir ${GT_MASK} \
    --dataset_id 451 \
    --fold 0 \
    --target_classes 1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
    --nnUNet_results ${nnUNet_results} \
    --nnUNet_preprocessed ${nnUNet_preprocessed} \
    --nnUNet_raw ${nnUNet_raw}