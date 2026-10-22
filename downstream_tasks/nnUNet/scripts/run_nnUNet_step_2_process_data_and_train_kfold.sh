#!/bin/bash

export nnUNet_raw="./nnUNet_dataset/nnUNet_raw"
export nnUNet_preprocessed="./nnUNet_dataset/nnUNet_preprocessed"
export nnUNet_results="./output_dir/checkpoints/nnUNet"

DATASET_ID=430

### prepare dateset
nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity

# for fold in 0 1 2 3 4
for fold in 0 
do 
    echo "=============================="
    echo "Training fold ${DATASET_ID} -- ${fold}"
    echo "=============================="
    
    CUDA_VISIBLE_DEVICES=2 nnUNetv2_train ${DATASET_ID} 2d ${fold} -tr nnUNetTrainer_200epoches

done
