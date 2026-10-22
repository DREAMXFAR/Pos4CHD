#!/bin/bash

DATASET_NAME=demo_dataset

python DatasetConvert.py \
    --source_dir "./dataset/${DATASET_NAME}" \
    --output_dir "./output_dir/nnUNet_raw" \
    --dataset_id 429 \
    --mode "train" \
    --dataset_name ${DATASET_NAME} \
    --modality_name "Ultrasound"