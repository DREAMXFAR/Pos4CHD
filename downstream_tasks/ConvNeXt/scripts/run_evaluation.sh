#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

### EA
# DATA_PATH=./data/PosDet_EA_v0726/train
# EVAL_DATA_PATH=./data/PosDet_EA_v0726/test
# EXP_NAME=EbA_genneg_ft_warmup5_seed9172_convnext_tiny_e200_lr3e-4_+minlr1e-4_res512

### APVC
DATA_PATH=./data/PosDet_APVC_v0726/train
EVAL_DATA_PATH=./data/PosDet_APVC_v0726/test
EXP_NAME=TAPVC_genneg_ft_warmup5_seed30469_convnext_tiny_e200_lr5e-4_+minlr1e-4_res512

CKPT_PATH=./Pos4CHD_updating/downstream_tasks/ConvNeXt/pretrain_models/${EXP_NAME}/checkpoint-best.pth

### no-ema
echo "------------------------------------ no-ema -------------------------------------"

python main.py \
    --model convnext_tiny --eval true \
    --dist_eval False \
    --data_set image_folder \
    --nb_classes 2 \
    --resume ${CKPT_PATH} \
    --input_size 512 --drop_path 0.1 \
    --data_path ${DATA_PATH} \
    --eval_data_path ${EVAL_DATA_PATH} \
    --use_custom_evaluator
