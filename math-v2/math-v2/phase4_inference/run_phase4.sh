#!/bin/bash
# Phase 4: Evaluation
# 在GSM8K测试集上评估模型

set -e

echo "========================================"
echo "Phase 4: Evaluation on GSM8K"
echo "========================================"

# 配置
MODEL_PATH=${MODEL_PATH:-"../outputs/grpo-lora/checkpoint-1000"}
NUM_SAMPLES=${NUM_SAMPLES:-8}
STRATEGY=${STRATEGY:-"score_weighted_vote"}
OUTPUT_FILE=${OUTPUT_FILE:-"../outputs/eval_results_lora_1000.json"}
MAX_EXAMPLES=${MAX_EXAMPLES:-""}  # 空表示全部

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Samples per question: $NUM_SAMPLES"
echo "  Strategy: $STRATEGY"
echo "  Output: $OUTPUT_FILE"
echo ""

# 运行评估
python evaluate.py \
    --model_path $MODEL_PATH \
    --num_samples $NUM_SAMPLES \
    --strategy $STRATEGY \
    --output_file $OUTPUT_FILE \
    ${MAX_EXAMPLES:+--max_examples $MAX_EXAMPLES}

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
