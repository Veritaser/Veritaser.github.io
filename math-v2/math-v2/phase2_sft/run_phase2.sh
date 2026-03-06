#!/bin/bash
# Phase 2: SFT Training with DeepSpeed
# 使用DeepSpeed进行SFT训练

set -e

echo "========================================"
echo "Phase 2: Supervised Fine-Tuning (SFT)"
echo "========================================"

# 配置路径
MODEL_NAME=${MODEL_NAME:-"../Qwen2.5-1.5B"}
TRAIN_FILE=${TRAIN_FILE:-"../data/sft/train.jsonl"}
VAL_FILE=${VAL_FILE:-"../data/sft/val.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"../outputs/sft"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"../config/deepspeed_zero2.json"}

# 训练参数
NUM_EPOCHS=${NUM_EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-2e-5}
MAX_LENGTH=${MAX_LENGTH:-2048}

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Train file: $TRAIN_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Learning rate: $LR"
echo ""

# 使用DeepSpeed启动训练
deepspeed --num_gpus=1 train_sft.py \
    --model_name_or_path $MODEL_NAME \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --max_length $MAX_LENGTH \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --eval_steps 500 \
    --eval_strategy steps \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --bf16 True \
    --gradient_checkpointing True \
    --deepspeed $DEEPSPEED_CONFIG \
    --report_to tensorboard

echo ""
echo "========================================"
echo "SFT Training Complete!"
echo "========================================"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Running sanity check..."
python sanity_check.py --model_path $OUTPUT_DIR
