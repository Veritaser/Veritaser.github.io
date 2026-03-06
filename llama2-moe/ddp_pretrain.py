# -*- coding: utf-8 -*-
import os
import platform
import argparse
import time
import warnings
import math
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

from transformers import AutoTokenizer

from k_model import ModelConfig, Transformer
from dataset import PretrainDataset

import swanlab

# 忽略警告信息
warnings.filterwarnings('ignore')


def Logger(content):
    """
    简单的日志记录函数
    
    Args:
        content (str): 要打印的内容
    """
    print(content)

def get_lr(it, all):
    """
    计算当前迭代的学习率，使用余弦退火调度策略
    
    学习率调度策略：
    1. Warmup阶段：学习率从0线性增长到目标学习率
    2. 余弦退火阶段：学习率按余弦函数衰减到最小学习率
    3. 超出训练步数后：保持最小学习率
    
    Args:
        it (int): 当前迭代步数
        all (int): 总迭代步数
        
    Returns:
        float: 当前步数对应的学习率
    """
    warmup_iters = args.warmup_iters  # 预热迭代次数
    lr_decay_iters = all  # 学习率衰减的总迭代次数
    min_lr = args.learning_rate / 10  # 最小学习率，为初始学习率的1/10

    # Warmup阶段：线性增长
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # 超出训练步数：保持最小学习率
    if it > lr_decay_iters:
        return min_lr
    
    # 余弦退火阶段
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 余弦系数
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """
    训练一个epoch的函数
    
    实现了完整的训练循环，包括：
    1. 数据加载和设备转移
    2. 动态学习率调整
    3. 前向传播和损失计算
    4. 梯度累积和反向传播
    5. 梯度裁剪和优化器更新
    6. 日志记录和模型保存
    
    Args:
        epoch (int): 当前epoch编号
    """
    start_time = time.time()  # 记录开始时间
    
    # 遍历数据加载器中的每个batch
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 计算全局步数
        global_step = epoch * iter_per_epoch + step
        
        # 检查是否需要跳过已经训练过的步骤（用于断点续训）
        if global_step < start_step:
            continue
            
        # 将数据转移到指定设备（GPU/CPU）
        X = X.to(args.device)  # 输入序列
        Y = Y.to(args.device)  # 目标序列
        loss_mask = loss_mask.to(args.device)  # 损失掩码，用于忽略padding token

        # 计算当前步骤的学习率
        lr = get_lr(global_step, args.epochs * iter_per_epoch)
        # 更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用混合精度训练上下文
        with ctx:
            # 前向传播
            out = model(X, Y)
            # model.last_loss 在 DataParallel 下可能为各设备的张量 (如 [loss_gpu0, loss_gpu1, ...])
            # 或为标量。统一处理为标量：如果有多个元素，则取平均。
            if out.last_loss is None:
                # 没有返回损失（不太可能在训练时发生），跳过
                continue
            loss = out.last_loss
            try:
                # 若为张量且包含多于1个元素（DataParallel 会返回每卡的loss），取平均
                if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                    loss = loss.mean()
            except Exception:
                # 保守处理：如果无法取numel/mean，直接使用原值
                pass

            # 除以累积步数，用于梯度累积
            loss = loss / args.accumulation_steps

        # 使用scaler进行混合精度的反向传播
        scaler.scale(loss).backward()

        # 每accumulation_steps步执行一次优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放，准备梯度裁剪
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行优化器步骤
            scaler.step(optimizer)
            # 更新scaler的缩放因子
            scaler.update()

            # 清零梯度，set_to_none=True可以节省内存
            optimizer.zero_grad(set_to_none=True)

        # 每log_interval步记录一次日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # 打印训练进度信息
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min;'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 恢复真实的loss值
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            
            # 如果启用SwanLab，记录训练指标
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        # 每save_interval步保存一次模型（完整检查点）
        if (global_step + 1) % args.save_interval == 0:
            save_checkpoint(global_step, epoch, model, optimizer, scaler, loss.item() * args.accumulation_steps)
        
        # 每20000步保存一个带步数标记的检查点
        if (global_step + 1) % 20000 == 0:
            save_checkpoint(global_step, epoch, model, optimizer, scaler, loss.item() * args.accumulation_steps, 
                          checkpoint_type='milestone')


def save_checkpoint(step, epoch, model, optimizer, scaler, loss, checkpoint_type='regular'):
    """
    保存完整的训练状态检查点
    
    Args:
        step: 当前训练步数
        epoch: 当前epoch
        model: 模型
        optimizer: 优化器
        scaler: 梯度缩放器
        loss: 当前损失
        checkpoint_type: 检查点类型 ('regular' 或 'milestone')
    """
    model.eval()  # 切换到评估模式
    
    # 构建检查点文件名
    if checkpoint_type == 'milestone':
        ckp_path = f'{args.save_dir}/checkpoint_step{step+1}.pth'
    else:
        ckp_path = f'{args.save_dir}/checkpoint_latest.pth'
        
    # 同时保存一个带配置信息的检查点
    model_only_path = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth'

    # 处理多卡保存：如果是DataParallel模型，需要访问.module属性
    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    
    # 保存完整的训练状态
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'lm_config': lm_config.__dict__,  # 保存配置为字典
        'args': vars(args),               # 保存参数为字典
    }
    
    torch.save(checkpoint, ckp_path)
    # 同时保存纯模型权重（用于推理）
    torch.save(model_state_dict, model_only_path)
    
    Logger(f"检查点已保存: {ckp_path}")
    model.train()  # 切换回训练模式


def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    """
    从检查点恢复训练状态
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 模型
        optimizer: 优化器  
        scaler: 梯度缩放器
        
    Returns:
        tuple: (start_step, start_epoch, loss) 恢复的训练状态
    """
    Logger(f"从检查点恢复训练: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # 恢复模型状态
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 恢复梯度缩放器状态
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_step = checkpoint['step'] + 1  # 从下一步开始
    start_epoch = checkpoint['epoch']
    last_loss = checkpoint.get('loss', 0.0)
    
    Logger(f"恢复训练状态 - 步数: {start_step}, Epoch: {start_epoch}, 上次损失: {last_loss:.6f}")
    
    return start_step, start_epoch, last_loss


def find_latest_checkpoint(save_dir):
    """
    在保存目录中查找最新的检查点
    
    Args:
        save_dir: 检查点保存目录
        
    Returns:
        str: 最新检查点的路径，如果没有找到则返回None
    """
    import glob
    
    # 查找所有检查点文件
    checkpoint_pattern = os.path.join(save_dir, "checkpoint_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # 根据修改时间排序，返回最新的
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint


def init_model():
    """
    初始化模型和分词器
    
    功能包括：
    1. 加载预训练的分词器
    2. 创建Transformer模型
    3. 设置多GPU并行训练（如果可用）
    4. 将模型移动到指定设备
    5. 统计并打印模型参数量
    
    Returns:
        tuple: (model, tokenizer) 初始化后的模型和分词器
    """
    def count_parameters(model):
        """
        统计模型中可训练参数的数量
        
        Args:
            model: PyTorch模型
            
        Returns:
            int: 可训练参数总数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 从本地路径加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # 根据配置创建Transformer模型
    model = Transformer(lm_config)
    
    # 多卡初始化：检查可用GPU数量并设置DataParallel
    # 支持按需使用绝对GPU索引或可见设备映射
    # 说明：
    # - 当 use_absolute=True 且 selected_gpu_ids 提供时，我们尝试把这些绝对物理卡号直接传给 DataParallel(device_ids=...)，
    #   并把 args.device 设为第一个绝对索引（在 GPU 环境里为 cuda:<first_id>）。
    # - 否则采用默认行为：使用 CUDA_VISIBLE_DEVICES（如果之前已被设置）并让 DataParallel 使用可见卡的索引 0..N-1。
    # 这样可以显式控制是按“物理卡号”还是按“可见设备重编号”来选择 GPU，避免索引错位。
    num_gpus = torch.cuda.device_count()
    # 从全局获取用户选择的GPU id和是否使用绝对索引
    selected_gpu_ids = globals().get('selected_gpu_ids', None)
    use_absolute = globals().get('use_absolute_gpus', False)

    if use_absolute and selected_gpu_ids:
        # absolute 模式：不依赖 CUDA_VISIBLE_DEVICES，直接使用系统 GPU 索引
        # 校验可用 GPU 数是否足够
        max_id = max(selected_gpu_ids)
        if not torch.cuda.is_available() or num_gpus == 0:
            # 无可用 GPU，回退到 CPU
            Logger("警告: 系统中没有可用 GPU，已降级为 CPU。")
        elif num_gpus <= max_id:
            # 指定的绝对索引超出系统范围，发出警告并回退到默认 DataParallel（使用所有可见 GPU）
            Logger(f"警告: 要求使用的最大 GPU 索引 {max_id} 超过系统可用 GPU 数量 {num_gpus}，将使用所有可用 GPU。")
            if num_gpus > 1:
                Logger(f"Using {num_gpus} GPUs with DataParallel!")
                model = torch.nn.DataParallel(model)
        else:
            # 将绝对索引传给 DataParallel，注意：这要求进程能直接见到这些物理卡
            Logger(f"Using absolute GPU ids {selected_gpu_ids} with DataParallel!")
            model = torch.nn.DataParallel(model, device_ids=selected_gpu_ids)
    else:
        # 默认模式：使用 CUDA_VISIBLE_DEVICES 做设备映射（如果上面已设置过），并按可见设备数量决定是否启用 DataParallel
        if num_gpus > 1:
            Logger(f"Using {num_gpus} GPUs with DataParallel!")
            model = torch.nn.DataParallel(model)
    
    # 将模型移动到指定设备（GPU或CPU）
    model = model.to(args.device)
    
    # 计算并打印模型参数量（以百万为单位）
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


if __name__ == "__main__":
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    
    # 基础训练参数
    parser.add_argument("--out_dir", type=str, default="base_model_215M", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=5, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    
    # 实验跟踪和数据加载参数
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str, default="./mobvoi_seq_monkey_general_open_corpus.jsonl", help="训练数据路径")
    
    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热迭代次数")
    
    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    
    # 断点续训参数
    parser.add_argument("--resume", type=str, default="", help="从指定检查点恢复训练，格式: path/to/checkpoint.pth")
    parser.add_argument("--auto_resume", action="store_true", help="自动从最新检查点恢复训练")
    
    # 多GPU训练参数
    # 注意：脚本支持两种指定 GPU 的方式：
    # 1) 通过 CUDA_VISIBLE_DEVICES 映射（默认）：传入 --gpus 例如 '4,5,6'，脚本会设置 CUDA_VISIBLE_DEVICES=4,5,6，训练进程内部使用 cuda:0..N-1 对应选择的物理卡。
    # 2) 绝对索引模式（可选）：传入 --gpus 例如 '4,5,6' 并加上 --absolute_gpus，脚本不会更改 CUDA_VISIBLE_DEVICES，直接使用系统中的物理 GPU 索引（例如 cuda:4）。
    # 这样可以在需要按物理卡绝对编号运行训练时避免映射导致的混淆。
    parser.add_argument("--gpus", type=str, default=None, help="使用的GPU ID，用逗号分隔 (例如: '0,1,2' 或 '4,5,6')")
    parser.add_argument("--absolute_gpus", action="store_true", help="如果设置，则认为 --gpus 使用系统中的绝对GPU索引(不会设置 CUDA_VISIBLE_DEVICES)，否则会通过 CUDA_VISIBLE_DEVICES 进行可见设备映射(默认)")

    args = parser.parse_args()

    # ==================== GPU环境设置 ====================
    # 解析 gpus 参数为整型列表（保存在 selected_gpu_ids）
    # selected_gpu_ids 用于两种情况：
    # - absolute_gpus=False（默认）：会把 selected_gpu_ids 写入 CUDA_VISIBLE_DEVICES，使得训练进程内部的 cuda:0 对应第一个选中的物理卡
    # - absolute_gpus=True：不会设置 CUDA_VISIBLE_DEVICES，而是直接把 selected_gpu_ids 作为物理设备索引传递给 DataParallel（device_ids）并把 args.device 设为第一个绝对索引
    selected_gpu_ids = None
    if args.gpus:
        try:
            selected_gpu_ids = [int(x) for x in args.gpus.split(',') if x.strip() != '']
        except Exception:
            Logger(f"无法解析 --gpus 参数: {args.gpus}. 请使用逗号分隔的整数列表，例如 '0,1,2'。")
            selected_gpu_ids = None

    # 说明行为：
    # - 默认行为（不传 --absolute_gpus）：脚本会 set CUDA_VISIBLE_DEVICES，使得在进程内使用 cuda:0..N-1。
    # - 传入 --absolute_gpus：脚本不修改 CUDA_VISIBLE_DEVICES，直接使用系统物理卡编号（例如 cuda:4），适合需要绝对卡索引的场景。
    if selected_gpu_ids:
        if args.absolute_gpus:
            # 绝对索引模式：不改动 CUDA_VISIBLE_DEVICES，直接使用第一个指定的物理 GPU 作为主设备
            if torch.cuda.is_available():
                args.device = f"cuda:{selected_gpu_ids[0]}"
            else:
                args.device = "cpu"
        else:
            # 默认映射模式：通过设置 CUDA_VISIBLE_DEVICES 来隔离设备并重编号
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in selected_gpu_ids)
            if torch.cuda.is_available():
                # 设备会被重编号为 cuda:0..，使用第一个可见设备作为主设备
                args.device = "cuda:0"
            else:
                args.device = "cpu"
    else:
        # 未提供 GPUs 或解析失败：保持原逻辑
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # 将解析结果保存为全局变量，供 init_model 使用（init_model 会根据这两个变量决定 DataParallel 的 device_ids）
    globals()['selected_gpu_ids'] = selected_gpu_ids
    globals()['use_absolute_gpus'] = bool(args.absolute_gpus)

    # ==================== 实验跟踪初始化 ====================
    if args.use_swanlab:
        # 注意：使用前需要先登录 swanlab.login(api_key='your key')
        run = swanlab.init(
            project="Happy-LLM",  # 项目名称
            experiment_name="Pretrain-215M",  # 实验名称
            config=args,  # 保存所有超参数
        )

    # ==================== 模型配置 ====================
    # 定义语言模型的配置参数
    lm_config = ModelConfig(
        dim=1024,      # 模型维度
        n_layers=18,   # Transformer层数
        num_experts=8,              # MoE专家数量
        num_experts_per_tok=2,      # 每个token激活的专家数量
        moe_freq=2,                 # 每隔多少层使用MoE (2表示每隔2层使用MoE)
        aux_loss_coef=0.01,         # 负载均衡损失系数
    )

    # ==================== 训练环境设置 ====================
    max_seq_len = lm_config.max_seq_len  # 最大序列长度
    args.save_dir = os.path.join(args.out_dir)  # 模型保存目录
    
    # 创建必要的目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    
    # 确定设备类型（用于选择合适的上下文管理器）
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置混合精度训练的上下文管理器
    # CPU训练时使用nullcontext，GPU训练时使用autocast
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # ==================== 模型和数据初始化 ====================
    # 初始化模型和分词器
    model, tokenizer = init_model()
    
    # 创建训练数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=max_seq_len)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,  # 批次大小
        pin_memory=True,             # 将数据加载到固定内存中，加速GPU传输
        drop_last=False,             # 不丢弃最后一个不完整的批次
        shuffle=True,                # 随机打乱数据
        num_workers=args.num_workers # 数据加载的并行工作进程数
    )

    # ==================== 优化器和训练组件初始化 ====================
    # 初始化混合精度训练的梯度缩放器
    # 只有在使用float16或bfloat16时才启用
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    
    # 初始化Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # ==================== 断点续训逻辑 ====================
    start_step = 0
    start_epoch = 0
    
    # 检查是否需要从检查点恢复
    if args.resume:
        # 从指定检查点恢复
        if os.path.exists(args.resume):
            start_step, start_epoch, last_loss = load_checkpoint(args.resume, model, optimizer, scaler)
            Logger(f"从指定检查点恢复训练: {args.resume}")
        else:
            Logger(f"指定的检查点不存在: {args.resume}")
            Logger("将从头开始训练...")
            
    elif args.auto_resume:
        # 自动查找最新检查点
        latest_checkpoint = find_latest_checkpoint(args.save_dir)
        if latest_checkpoint:
            start_step, start_epoch, last_loss = load_checkpoint(latest_checkpoint, model, optimizer, scaler)
            Logger(f"自动恢复训练")
        else:
            Logger("未找到检查点，从头开始训练...")
    
    # 将start_step设为全局变量，供train_epoch使用
    globals()['start_step'] = start_step
    
    # ==================== 开始训练 ====================
    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)
    
    Logger(f"开始训练 - 从第{start_epoch+1}个epoch, 第{start_step}步开始")
    Logger(f"总共需要训练 {args.epochs} 个epoch, 每个epoch {iter_per_epoch} 步")
    
    # 开始训练循环 - 从start_epoch开始
    for epoch in range(start_epoch, args.epochs):
        train_epoch(epoch)