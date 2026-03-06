#!/bin/bash
# Phase 1: Data Construction Pipeline
# 数据构建流程脚本

set -e

echo "========================================"
echo "Phase 1: Data Construction"
echo "========================================"

# 配置
PROVIDER=${PROVIDER:-"openai"}
MODEL=${MODEL:-"gpt-4o"}
NUM_SAMPLES=${NUM_SAMPLES:-""}  # 空表示全部
MAX_WORKERS=${MAX_WORKERS:-4}
OUTPUT_DIR="./data"

# Step 1: 生成Teacher数据
echo ""
echo "[Step 1/3] Generating teacher responses..."
python teacher_generate.py \
    --provider $PROVIDER \
    --model $MODEL \
    ${NUM_SAMPLES:+--num_samples $NUM_SAMPLES} \
    --max_workers $MAX_WORKERS \
    --output_dir "$OUTPUT_DIR/raw"

# Step 2: 数据清洗和验证
echo ""
echo "[Step 2/3] Cleaning and verifying data..."
python data_cleaning.py \
    --raw_data "$OUTPUT_DIR/raw/teacher_responses.jsonl" \
    --gsm8k_answers "$OUTPUT_DIR/raw/gsm8k_answers.jsonl" \
    --output "$OUTPUT_DIR/cleaned/verified_data.jsonl"

# Step 3: 格式化为SFT训练格式
echo ""
echo "[Step 3/3] Formatting dataset for SFT..."
python format_dataset.py \
    --input "$OUTPUT_DIR/cleaned/verified_data.jsonl" \
    --output_dir "$OUTPUT_DIR/sft" \
    --format messages \
    --val_ratio 0.05 \
    --only_correct

echo ""
echo "========================================"
echo "Phase 1 Complete!"
echo "========================================"
echo "SFT training data saved to: $OUTPUT_DIR/sft/"
echo "  - train.jsonl"
echo "  - val.jsonl"
