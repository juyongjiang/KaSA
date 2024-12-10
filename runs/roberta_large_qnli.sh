#!/bin/bash
cd ../
mkdir -p logs/roberta-large


# variables
CUDA_DEVICE=1

MODEL_NAME_OR_PATH="roberta-large"

DATASET="qnli"
TASK="qnli"

BATCH_SIZE=8 # 32
MAX_LENGTH=512
NUM_EPOCH=20

HEAD_LR=4e-4
MODULE_LR=4e-4 

LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.0

BETA=0.01
GEMMA=0.001

SEED=0
WEIGHT_DECAY=0.0


# run
LOG_FILE="logs/${MODEL_NAME_OR_PATH}/${MODEL_NAME_OR_PATH}_${TASK}_bs_${BATCH_SIZE}_maxlen_${MAX_LENGTH}_lora_r_${LORA_R}_lora_alpha_${LORA_ALPHA}_lora_dropout_${LORA_DROPOUT}_modulelr_${MODULE_LR}_headlr_${HEAD_LR}_beta_${BETA}_gemma_${GEMMA}_weight_decay_${WEIGHT_DECAY}_seed_${SEED}.log"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python main.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset $DATASET \
    --task $TASK \
    --max_length $MAX_LENGTH \
    --bs $BATCH_SIZE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_epoch $NUM_EPOCH \
    --head_lr $HEAD_LR \
    --module_lr $MODULE_LR \
    --beta $BETA \
    --gemma $GEMMA \
    --weight_decay $WEIGHT_DECAY \
    --seed $SEED 2>&1 | tee $LOG_FILE