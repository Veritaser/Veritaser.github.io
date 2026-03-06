# Windows PowerShell script for Phase 4 Evaluation
# 在GSM8K测试集上评估模型 (Windows版本)

Write-Host "========================================"
Write-Host "Phase 4: Evaluation on GSM8K"
Write-Host "========================================"

# 配置
$MODEL_PATH = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "../outputs/grpo" }
$NUM_SAMPLES = if ($env:NUM_SAMPLES) { $env:NUM_SAMPLES } else { 8 }
$STRATEGY = if ($env:STRATEGY) { $env:STRATEGY } else { "score_weighted_vote" }
$OUTPUT_FILE = if ($env:OUTPUT_FILE) { $env:OUTPUT_FILE } else { "../outputs/eval_results.json" }
$MAX_EXAMPLES = $env:MAX_EXAMPLES

Write-Host ""
Write-Host "Configuration:"
Write-Host "  Model: $MODEL_PATH"
Write-Host "  Samples per question: $NUM_SAMPLES"
Write-Host "  Strategy: $STRATEGY"
Write-Host "  Output: $OUTPUT_FILE"
Write-Host ""

# 构建命令参数
$cmd_args = @(
    "evaluate.py",
    "--model_path", $MODEL_PATH,
    "--num_samples", $NUM_SAMPLES,
    "--strategy", $STRATEGY,
    "--output_file", $OUTPUT_FILE
)

if ($MAX_EXAMPLES) {
    $cmd_args += "--max_examples"
    $cmd_args += $MAX_EXAMPLES
}

# 运行评估
python @cmd_args

Write-Host ""
Write-Host "========================================"
Write-Host "Evaluation Complete!"
Write-Host "========================================"
