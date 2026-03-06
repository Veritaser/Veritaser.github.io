# 断点续训功能说明

## 🚀 功能特性

本项目现已支持完整的断点续训功能，允许训练过程从任意检查点恢复，无需重新开始训练。

## 📋 功能清单

### ✅ 已实现功能

- **完整状态保存**: 保存模型权重、优化器状态、梯度缩放器状态、训练步数、epoch等
- **自动检查点检测**: 自动查找最新的检查点文件
- **指定检查点恢复**: 从用户指定的检查点文件恢复
- **预训练支持**: ddp_pretrain.py 完全支持断点续训
- **监督微调支持**: ddp_sft_full.py 完全支持断点续训  
- **里程碑检查点**: 每20000步自动保存带步数标记的检查点
- **状态验证**: 恢复时验证训练状态的正确性

## 🎮 使用方法

### 1. 预训练断点续训

#### 自动恢复最新检查点
```bash
python ddp_pretrain.py --auto_resume --out_dir your_model_dir
```

#### 从指定检查点恢复
```bash
python ddp_pretrain.py --resume path/to/checkpoint.pth --out_dir your_model_dir
```

#### 正常训练（会自动保存检查点）
```bash
python ddp_pretrain.py --out_dir your_model_dir --save_interval 1000
```

### 2. 监督微调断点续训

#### 自动恢复最新检查点
```bash
python ddp_sft_full.py --auto_resume --out_dir your_sft_dir
```

#### 从指定检查点恢复
```bash
python ddp_sft_full.py --resume path/to/sft_checkpoint.pth --out_dir your_sft_dir
```

### 3. 检查点文件结构

```
your_output_dir/
├── checkpoint_latest.pth           # 最新的完整检查点
├── checkpoint_step20000.pth        # 第20000步里程碑检查点  
├── checkpoint_step40000.pth        # 第40000步里程碑检查点
├── pretrain_1024_18_6144.pth       # 纯模型权重（用于推理）
└── sft_checkpoint_latest.pth       # SFT最新检查点
```

## 📊 检查点内容

每个完整检查点包含以下信息：

```python
{
    'step': 当前训练步数,
    'epoch': 当前epoch,
    'model_state_dict': 模型权重,
    'optimizer_state_dict': 优化器状态,
    'scaler_state_dict': 梯度缩放器状态,
    'loss': 最后的损失值,
    'lm_config': 模型配置,
    'args': 训练参数
}
```

## ⚙️ 配置参数

### 新增命令行参数

#### 预训练脚本 (`ddp_pretrain.py`)
```bash
--resume PATH           # 从指定检查点恢复，格式: path/to/checkpoint.pth  
--auto_resume          # 自动从最新检查点恢复训练
--save_interval N      # 每N步保存一次检查点（默认1000）
```

#### 监督微调脚本 (`ddp_sft_full.py`)  
```bash
--resume PATH           # 从指定SFT检查点恢复
--auto_resume          # 自动从最新SFT检查点恢复训练
--save_interval N      # 每N步保存一次检查点（默认1000）
```

## 🔧 使用示例

### 场景1：训练中断后恢复

```bash
# 开始训练
python ddp_pretrain.py --out_dir moe_model --epochs 10

# 训练中断（比如Ctrl+C或系统故障）

# 自动恢复训练
python ddp_pretrain.py --out_dir moe_model --epochs 10 --auto_resume
```

### 场景2：从特定里程碑继续

```bash  
# 从第20000步的检查点继续训练
python ddp_pretrain.py --resume moe_model/checkpoint_step20000.pth --epochs 20
```

### 场景3：监督微调断点续训

```bash
# 开始SFT训练
python ddp_sft_full.py --out_dir sft_model --data_path data.jsonl

# 训练中断后恢复
python ddp_sft_full.py --out_dir sft_model --data_path data.jsonl --auto_resume
```

## 🧪 测试断点续训

运行测试脚本验证断点续训功能：

```bash
# 测试预训练断点续训
python test_resume_training.py --test_type pretrain

# 测试SFT断点续训  
python test_resume_training.py --test_type sft

# 测试所有功能
python test_resume_training.py --test_type all
```

## 📈 训练监控

### 日志输出示例

```
从检查点恢复训练: ./moe_model/checkpoint_latest.pth
恢复训练状态 - 步数: 1501, Epoch: 0, 上次损失: 3.2451
开始训练 - 从第1个epoch, 第1501步开始
总共需要训练 5 个epoch, 每个epoch 2000 步
检查点已保存: ./moe_model/checkpoint_latest.pth
```

### 关键信息

- ✅ **无缝恢复**: 从中断点精确继续，不重复训练已完成的步骤
- ✅ **状态一致**: 优化器、学习率调度器状态完全恢复
- ✅ **损失连续**: 损失曲线保持连续，无突变
- ✅ **性能不变**: 恢复后的训练性能与连续训练一致

## 🚨 注意事项

1. **存储空间**: 检查点文件较大，注意磁盘空间
2. **版本兼容**: 确保恢复时的代码版本与保存时兼容
3. **路径正确**: 检查点路径必须正确且文件存在
4. **配置一致**: 恢复时的模型配置应与保存时一致
5. **权限检查**: 确保对检查点目录有读写权限

## 🔄 最佳实践

1. **定期保存**: 设置合理的`save_interval`，平衡存储和安全性
2. **备份重要检查点**: 手动备份关键里程碑检查点
3. **监控日志**: 观察恢复后的损失是否正常连续
4. **测试验证**: 在正式训练前测试断点续训功能
5. **清理策略**: 定期清理过旧的检查点文件

---

断点续训功能让长时间训练更加稳定可靠，有效应对各种中断情况，确保训练资源不被浪费！