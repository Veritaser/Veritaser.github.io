# Windows PowerShell script for Phase 3 GRPO Training
# 使用DeepSpeed进行GRPO强化学习训练 (Windows版本)

Write-Host "========================================"
Write-Host "Phase 3: GRPO Reinforcement Learning"
Write-Host "========================================"

# 配置路径
$MODEL_PATH = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "../outputs/sft" }
$OUTPUT_DIR = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "../outputs/grpo" }
$DEEPSPEED_CONFIG = if ($env:DEEPSPEED_CONFIG) { $env:DEEPSPEED_CONFIG } else { "../config/deepspeed_zero2.json" }

# 训练参数
$NUM_EPOCHS = if ($env:NUM_EPOCHS) { $env:NUM_EPOCHS } else { 1 }
$BATCH_SIZE = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { 2 }
$GRAD_ACCUM = if ($env:GRAD_ACCUM) { $env:GRAD_ACCUM } else { 8 }
$LR = if ($env:LR) { $env:LR } else { "1e-6" }
$NUM_GENERATIONS = if ($env:NUM_GENERATIONS) { $env:NUM_GENERATIONS } else { 8 }
$NUM_SAMPLES = $env:NUM_SAMPLES

# 创建输出目录
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

Write-Host ""
Write-Host "Configuration:"
Write-Host "  Model path: $MODEL_PATH"
Write-Host "  Output: $OUTPUT_DIR"
Write-Host "  Epochs: $NUM_EPOCHS"
Write-Host "  Batch size: $BATCH_SIZE x $GRAD_ACCUM = $($BATCH_SIZE * $GRAD_ACCUM)"
Write-Host "  Learning rate: $LR"
Write-Host "  Generations per prompt: $NUM_GENERATIONS"
Write-Host ""

# 构建命令参数
$cmd_args = @(
    "--num_gpus=1",
    "train_grpo_trl.py",
    "--model_path", $MODEL_PATH,
    "--output_dir", $OUTPUT_DIR,
    "--num_epochs", $NUM_EPOCHS,
    "--batch_size", $BATCH_SIZE,
    "--grad_accum", $GRAD_ACCUM,
    "--lr", $LR,
    "--num_generations", $NUM_GENERATIONS,
    "--max_new_tokens", "1024",
    "--temperature", "0.7",
    "--kl_coef", "0.05",
    "--deepspeed", $DEEPSPEED_CONFIG
)

if ($NUM_SAMPLES) {
    $cmd_args += "--num_samples"
    $cmd_args += $NUM_SAMPLES
}

# 使用DeepSpeed启动GRPO训练
& deepspeed @cmd_args

Write-Host ""
Write-Host "========================================"
Write-Host "GRPO Training Complete!"
Write-Host "========================================"
Write-Host "Model saved to: $OUTPUT_DIR"
