#!/bin/bash
export CUDA_VISIBLE_DEVICES=1


# config
MODEL_TYPE=convnext_tiny
EPOCH=200
LR=3e-4
MIN_LR=1e-4
INPUT_SIZE=512
EMA=True
EMA_DECAY=0.999

CUTMIX=0
MIXUP=0
COLOR_JITTER=0.2
AA=rand-m8-mstd0.5-inc1
REPROB=0

HEAD_INIT_SCALE=1.0

SEED=9172


EXP_NAME=Final_EA_spatextctrlnet_genneg_ft_warmup5_seed${SEED}_${MODEL_TYPE}_e${EPOCH}_lr${LR}_+minlr${MIN_LR}_res${INPUT_SIZE}_ema${EMA}_emadecay${EMA_DECAY}_ft_headinitscale${HEAD_INIT_SCALE}_aug_cutmix${CUTMIX}_mixup${MIXUP}_colorjit${COLOR_JITTER}_aa${AA}_reprob${REPROB}

OUTPUT_DIR=./output_dir/${EXP_NAME}

DATA_PATH=./dataset/PosDet_EA_v0812_gennegtrain/train
EVAL_DATA_PATH=./dataset/PosDet_EA_v0812_gennegtrain/test

python main.py \
    --model ${MODEL_TYPE} --drop_path 0.1 --input_size ${INPUT_SIZE} \
    --nb_classes 2 --data_set image_folder \
    --epochs ${EPOCH} \
    --seed  ${SEED} \
    --smoothing 0 \
    --batch_size 8 --lr ${LR} --update_freq 4 --min_lr ${MIN_LR}\
    --save_ckpt_freq 5 \
    --warmup_epochs 5 \
    --model_ema ${EMA} --model_ema_eval ${EMA} \
    --model_ema_decay ${EMA_DECAY} \
    --head_init_scale ${HEAD_INIT_SCALE} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --cutmix ${CUTMIX} --mixup ${MIXUP} --color_jitter ${COLOR_JITTER} --aa ${AA} --reprob ${REPROB}\
    --eval_data_path ${EVAL_DATA_PATH} \
    --log_dir ./output_dir/tensorboard_logs/${EXP_NAME} \
    --finetune ./pretrained_models/convnext_tiny_1k_224_ema.pth