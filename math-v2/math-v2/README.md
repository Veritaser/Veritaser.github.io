# Math Reasoning Pipeline with Self-Verification
# 基于DeepSeekMath-V2的数学推理训练Pipeline

基于 **DeepSeekMath-V2** 论文核心思想（Self-Verifiable Reasoning），专门针对 **Qwen-1.5B** 模型在 **GSM8K** 数据集上进行优化的完整训练Pipeline。

## 📁 项目结构

```
math-v2/
├── config/                         # 配置文件
│   ├── base_config.yaml           # 基础配置
│   ├── deepspeed_zero2.json       # DeepSpeed ZeRO-2配置
│   └── deepspeed_zero3.json       # DeepSpeed ZeRO-3配置
│
├── phase1_data_construction/       # Phase 1: 数据构建
│   ├── prompts.py                 # Prompt模板
│   ├── teacher_generate.py        # Teacher模型数据生成
│   ├── data_cleaning.py           # 数据清洗与验证
│   ├── format_dataset.py          # 格式化为SFT训练数据
│   └── run_phase1.sh              # 运行脚本
│
├── phase2_sft/                     # Phase 2: 监督微调
│   ├── train_sft.py               # SFT训练脚本(DeepSpeed)
│   ├── sanity_check.py            # 格式验证检查
│   ├── sft_config.yaml            # SFT配置
│   ├── run_phase2.sh              # Linux运行脚本
│   └── run_phase2.ps1             # Windows运行脚本
│
├── phase3_grpo/                    # Phase 3: GRPO强化学习
│   ├── reward_functions.py        # 奖励函数设计
│   ├── train_grpo.py              # GRPO训练脚本
│   ├── train_grpo_trl.py          # TRL版本GRPO训练
│   ├── grpo_config.yaml           # GRPO配置
│   ├── run_phase3.sh              # Linux运行脚本
│   └── run_phase3.ps1             # Windows运行脚本
│
├── phase4_inference/               # Phase 4: 推理与评估
│   ├── inference.py               # 自验证推理
│   ├── evaluate.py                # GSM8K评估
│   ├── vllm_inference.py          # vLLM快速推理
│   ├── run_phase4.sh              # Linux运行脚本
│   └── run_phase4.ps1             # Windows运行脚本
│
├── utils/                          # 工具函数
│   └── common.py                  # 共享工具
│
├── data/                           # 数据目录 (运行时生成)
├── outputs/                        # 输出目录 (运行时生成)
├── requirements.txt               # 依赖
├── pipeline.md                    # 原始Pipeline文档
└── README.md                      # 本文件
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
conda create -n math-v2 python=3.10
conda activate math-v2

# 安装依赖
pip install -r requirements.txt

# 如果需要Flash Attention (推荐)
pip install flash-attn --no-build-isolation
```

### 2. 配置API Key (Phase 1需要)

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# 或者 Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# 或者 DeepSeek
export DEEPSEEK_API_KEY="your-api-key"
```

### 3. 运行Pipeline

#### Phase 1: 数据构建
```bash
cd phase1_data_construction

# 使用OpenAI GPT-4o生成数据
PROVIDER=openai MODEL=gpt-4o NUM_SAMPLES=1000 bash run_phase1.sh

# 或者使用DeepSeek
PROVIDER=deepseek MODEL=deepseek-chat bash run_phase1.sh
```

#### Phase 2: SFT训练
```bash
cd phase2_sft

# Linux
bash run_phase2.sh

# Windows PowerShell
.\run_phase2.ps1
```

#### Phase 3: GRPO训练
```bash
cd phase3_grpo

# Linux
bash run_phase3.sh

# Windows PowerShell
.\run_phase3.ps1
```

#### Phase 4: 评估
```bash
cd phase4_inference

# Linux
bash run_phase4.sh

# Windows PowerShell
.\run_phase4.ps1
```

## ⚙️ DeepSpeed配置

项目提供两种DeepSpeed配置：

### ZeRO-2 (推荐用于1.5B模型)
- 优化器状态分片
- 梯度分片
- 适合单卡24GB显存

### ZeRO-3 (用于显存紧张情况)
- 完全分片
- 支持CPU卸载
- 适合显存不足的情况

使用方法：
```bash
# 使用ZeRO-2
deepspeed --num_gpus=1 train_sft.py --deepspeed ../config/deepspeed_zero2.json

# 使用ZeRO-3
deepspeed --num_gpus=1 train_sft.py --deepspeed ../config/deepspeed_zero3.json
```

## 📊 奖励函数设计

基于论文设计的组合奖励函数：

```
R_total = R_outcome + 0.1 * R_format + 0.5 * R_consistency
```

- **R_outcome**: 答案正确性 (0或1)
- **R_format**: 格式正确性 (是否包含Solution和Self Evaluation)
- **R_consistency**: 自评一致性 (自评分数与实际正确性的匹配度)

## 🎯 推理策略

支持多种Best-of-N选择策略：

1. **score_weighted_vote** (默认): 优先选择高分(≥0.9)样本进行投票
2. **majority_vote**: 简单多数投票
3. **highest_score**: 选择自评分数最高的答案

## 📈 预期结果

| 阶段 | 模型 | GSM8K准确率 |
|------|------|-------------|
| Baseline | Qwen-1.5B | ~30% |
| SFT | Qwen-1.5B-SFT | ~45% |
| GRPO + Best-of-8 | Qwen-1.5B-GRPO | ~55% |

*注: 实际结果可能因训练数据质量和超参数而异*

## ⚠️ 常见问题

### 1. 显存不足
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用ZeRO-3配置

### 2. 格式崩坏
- 检查SFT数据质量
- 增加Format Reward权重
- 在GRPO中混入SFT数据

### 3. 幻觉自评
- 确保Consistency Reward正确实现
- 错误答案+高分必须受惩罚

## 📚 参考

- [DeepSeekMath-V2 Paper](https://arxiv.org/abs/...)
- [TRL Library](https://github.com/huggingface/trl)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)

## License

MIT License
