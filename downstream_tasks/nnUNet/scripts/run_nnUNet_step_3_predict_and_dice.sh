#!/bin/bash
export nnUNet_raw="./nnUNet_dataset/nnUNet_raw"
export nnUNet_preprocessed="./nnUNet_dataset/nnUNet_preprocessed"
export nnUNet_results="./output_dir/checkpoints/nnUNet"

### for 4CH_target14_processed_other_val100_v2
# Predictions
IMAGE_DIR=./nnUNet_dataset/nnUNet_raw/Dataset401_4CH_target14_processed_other_val100_v2_controlnet_format/imagesTs

EXP_NAME=Dataset430_demo_dataset
FOLD=4

OUTPUT_DIR=./output_dir/nnUNet_inference_results/${EXP_NAME}_fold${FOLD}_test_best

DATASET_ID=430

CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict \
    -i ${IMAGE_DIR} \
    -o ${OUTPUT_DIR} \
    -d ${DATASET_ID} \
    -c 2d \
    -f ${FOLD} \
    -tr nnUNetTrainer_200epoches \
    -chk checkpoint_best.pth
    # --disable_tta

echo "DATASET ID = "${DATASET_ID}

# # Compute Dice
# GT_DIR=./nnUNet_dataset/nnUNet_raw/Dataset401_4CH_target14_processed_other_val100_v2_controlnet_format/labelsTs
# PRED_DIR=${OUTPUT_DIR}
# CLS_NUM=14

# CUDA_VISIBLE_DEVICES=0 python metrics/Dice_with_acc.py \
#     --gt_dir ${GT_DIR} \
#     --pred_dir ${PRED_DIR} \
#     --num_class ${CLS_NUM}
