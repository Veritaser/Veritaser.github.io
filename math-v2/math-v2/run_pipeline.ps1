# Full Pipeline Runner (Windows PowerShell)
# 完整Pipeline一键运行脚本

Write-Host "=============================================="
Write-Host "  Math Reasoning Training Pipeline"
Write-Host "  Based on DeepSeekMath-V2 Methodology"
Write-Host "=============================================="

# 配置
$PROVIDER = if ($env:PROVIDER) { $env:PROVIDER } else { "openai" }
$MODEL = if ($env:MODEL) { $env:MODEL } else { "gpt-4o" }
$NUM_SAMPLES = $env:NUM_SAMPLES
$SKIP_PHASE1 = $env:SKIP_PHASE1 -eq "true"
$SKIP_PHASE2 = $env:SKIP_PHASE2 -eq "true"
$SKIP_PHASE3 = $env:SKIP_PHASE3 -eq "true"

$ErrorActionPreference = "Stop"

# Phase 1: 数据构建
if (-not $SKIP_PHASE1) {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host "  Phase 1: Data Construction"
    Write-Host "=============================================="
    
    Set-Location phase1_data_construction
    
    $env:PROVIDER = $PROVIDER
    $env:MODEL = $MODEL
    if ($NUM_SAMPLES) { $env:NUM_SAMPLES = $NUM_SAMPLES }
    
    python teacher_generate.py --provider $PROVIDER --model $MODEL $(if ($NUM_SAMPLES) { "--num_samples $NUM_SAMPLES" })
    python data_cleaning.py
    python format_dataset.py
    
    Set-Location ..
}

# Phase 2: SFT训练
if (-not $SKIP_PHASE2) {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host "  Phase 2: Supervised Fine-Tuning"
    Write-Host "=============================================="
    
    Set-Location phase2_sft
    .\run_phase2.ps1
    Set-Location ..
}

# Phase 3: GRPO训练
if (-not $SKIP_PHASE3) {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host "  Phase 3: GRPO Reinforcement Learning"
    Write-Host "=============================================="
    
    Set-Location phase3_grpo
    .\run_phase3.ps1
    Set-Location ..
}

# Phase 4: 评估
Write-Host ""
Write-Host "=============================================="
Write-Host "  Phase 4: Evaluation"
Write-Host "=============================================="

Set-Location phase4_inference
.\run_phase4.ps1
Set-Location ..

Write-Host ""
Write-Host "=============================================="
Write-Host "  Pipeline Complete!"
Write-Host "=============================================="
Write-Host ""
Write-Host "Results saved to: outputs/"
