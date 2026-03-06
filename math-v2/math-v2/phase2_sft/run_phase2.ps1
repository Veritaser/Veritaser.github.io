# Windows PowerShell script for Phase 2 SFT Training
# 使用DeepSpeed进行SFT训练 (Windows版本)

Write-Host "========================================"
Write-Host "Phase 2: Supervised Fine-Tuning (SFT)"
Write-Host "========================================"

# 配置路径
$MODEL_NAME = if ($env:MODEL_NAME) { $env:MODEL_NAME } else { "Qwen/Qwen2.5-1.5B-Instruct" }
$TRAIN_FILE = if ($env:TRAIN_FILE) { $env:TRAIN_FILE } else { "../data/sft/train.jsonl" }
$VAL_FILE = if ($env:VAL_FILE) { $env:VAL_FILE } else { "../data/sft/val.jsonl" }
$OUTPUT_DIR = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "../outputs/sft" }
$DEEPSPEED_CONFIG = if ($env:DEEPSPEED_CONFIG) { $env:DEEPSPEED_CONFIG } else { "../config/deepspeed_zero2.json" }

# 训练参数
$NUM_EPOCHS = if ($env:NUM_EPOCHS) { $env:NUM_EPOCHS } else { 3 }
$BATCH_SIZE = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { 4 }
$GRAD_ACCUM = if ($env:GRAD_ACCUM) { $env:GRAD_ACCUM } else { 4 }
$LR = if ($env:LR) { $env:LR } else { "2e-5" }
$MAX_LENGTH = if ($env:MAX_LENGTH) { $env:MAX_LENGTH } else { 2048 }

# 创建输出目录
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

Write-Host ""
Write-Host "Configuration:"
Write-Host "  Model: $MODEL_NAME"
Write-Host "  Train file: $TRAIN_FILE"
Write-Host "  Output: $OUTPUT_DIR"
Write-Host "  Epochs: $NUM_EPOCHS"
Write-Host "  Batch size: $BATCH_SIZE x $GRAD_ACCUM = $($BATCH_SIZE * $GRAD_ACCUM)"
Write-Host "  Learning rate: $LR"
Write-Host ""

# 使用DeepSpeed启动训练
deepspeed --num_gpus=1 train_sft.py `
    --model_name_or_path $MODEL_NAME `
    --train_file $TRAIN_FILE `
    --val_file $VAL_FILE `
    --output_dir $OUTPUT_DIR `
    --num_train_epochs $NUM_EPOCHS `
    --per_device_train_batch_size $BATCH_SIZE `
    --per_device_eval_batch_size $BATCH_SIZE `
    --gradient_accumulation_steps $GRAD_ACCUM `
    --learning_rate $LR `
    --max_length $MAX_LENGTH `
    --warmup_ratio 0.03 `
    --lr_scheduler_type cosine `
    --logging_steps 10 `
    --save_steps 500 `
    --eval_steps 500 `
    --evaluation_strategy steps `
    --save_total_limit 3 `
    --load_best_model_at_end True `
    --bf16 True `
    --gradient_checkpointing True `
    --deepspeed $DEEPSPEED_CONFIG `
    --report_to tensorboard

Write-Host ""
Write-Host "========================================"
Write-Host "SFT Training Complete!"
Write-Host "========================================"
Write-Host "Model saved to: $OUTPUT_DIR"
Write-Host ""
Write-Host "Running sanity check..."
python sanity_check.py --model_path $OUTPUT_DIR
