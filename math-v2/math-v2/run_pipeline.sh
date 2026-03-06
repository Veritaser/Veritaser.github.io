#!/bin/bash
# Full Pipeline Runner
# 完整Pipeline一键运行脚本

set -e

echo "=============================================="
echo "  Math Reasoning Training Pipeline"
echo "  Based on DeepSeekMath-V2 Methodology"
echo "=============================================="

# 配置
PROVIDER=${PROVIDER:-"openai"}
MODEL=${MODEL:-"gpt-4o"}
NUM_SAMPLES=${NUM_SAMPLES:-""}  # 空表示全部
SKIP_PHASE1=${SKIP_PHASE1:-false}
SKIP_PHASE2=${SKIP_PHASE2:-false}
SKIP_PHASE3=${SKIP_PHASE3:-false}

# Phase 1: 数据构建
if [ "$SKIP_PHASE1" != "true" ]; then
    echo ""
    echo "=============================================="
    echo "  Phase 1: Data Construction"
    echo "=============================================="
    cd phase1_data_construction
    PROVIDER=$PROVIDER MODEL=$MODEL NUM_SAMPLES=$NUM_SAMPLES bash run_phase1.sh
    cd ..
fi

# Phase 2: SFT训练
if [ "$SKIP_PHASE2" != "true" ]; then
    echo ""
    echo "=============================================="
    echo "  Phase 2: Supervised Fine-Tuning"
    echo "=============================================="
    cd phase2_sft
    bash run_phase2.sh
    cd ..
fi

# Phase 3: GRPO训练
if [ "$SKIP_PHASE3" != "true" ]; then
    echo ""
    echo "=============================================="
    echo "  Phase 3: GRPO Reinforcement Learning"
    echo "=============================================="
    cd phase3_grpo
    bash run_phase3.sh
    cd ..
fi

# Phase 4: 评估
echo ""
echo "=============================================="
echo "  Phase 4: Evaluation"
echo "=============================================="
cd phase4_inference
bash run_phase4.sh
cd ..

echo ""
echo "=============================================="
echo "  Pipeline Complete!"
echo "=============================================="
echo ""
echo "Results saved to: outputs/"
