#!/bin/bash
# Phase 3: GRPO Training with DeepSpeed
# 使用DeepSpeed进行GRPO强化学习训练

set -e

echo "========================================"
echo "Phase 3: GRPO Reinforcement Learning"
echo "========================================"

# 配置路径
MODEL_PATH=${MODEL_PATH:-"../outputs/sft"}
OUTPUT_DIR=${OUTPUT_DIR:-"../outputs/grpo-lora-18-512-accm6"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"../config/deepspeed_zero3.json"}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LR=${LR:-1e-6}
# 创建输出目录
mkdir -p $OUTPUT_DIR


# 使用原生Python启动GRPO训练 (不使用DeepSpeed)
# 1.5B模型+LoRA显存占用很小，不需要ZeRO3，直接运行更稳定
deepspeed --include localhost:0 train_grpo_final.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --num_generations 18 \
    --max_new_tokens 512 \
    --lr $LR

# 如果显存不足(OOM)，可以尝试减小 num_generations 或 batch_size
# deepspeed --include localhost:0 train_grpo_final.py \
#     --model_path $MODEL_PATH \
#     --output_dir $OUTPUT_DIR \
#     --num_epochs $NUM_EPOCHS \
#     --num_generations 32 \
#     --max_new_tokens 1024 \
#     --lr $LR \
#     --deepspeed $DEEPSPEED_CONFIG

echo ""
echo "========================================"
echo "GRPO Training Complete!"
echo "========================================"
echo "Model saved to: $OUTPUT_DIR"
