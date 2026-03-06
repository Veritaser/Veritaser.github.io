#!/usr/bin/env python3
"""
测试MoE模型功能的脚本
展示MoE模型的特点和性能对比
"""

import torch
import time
from transformers import AutoTokenizer
from k_model import ModelConfig, Transformer

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters())

def count_active_parameters(model):
    """计算MoE模型激活的参数数量（近似）"""
    total_params = 0
    moe_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        # MoE层的专家参数
        if 'experts' in name:
            moe_params += param.numel()
    
    # 假设每次只激活指定数量的专家
    num_experts = model.args.num_experts
    experts_per_tok = model.args.num_experts_per_tok
    active_moe_params = moe_params * experts_per_tok / num_experts
    
    # 总激活参数 = 非MoE参数 + 激活的MoE参数
    active_params = (total_params - moe_params) + active_moe_params
    return active_params, total_params

def test_model_inference(model, tokenizer, test_text):
    """测试模型推理"""
    model.eval()
    
    with torch.no_grad():
        # 编码输入
        input_ids = tokenizer.encode(test_text, return_tensors='pt')
        
        # 推理
        start_time = time.time()
        output = model(input_ids)
        inference_time = time.time() - start_time
        
        # 生成文本
        generated = model.generate(
            input_ids, 
            max_new_tokens=50, 
            temperature=0.7, 
            top_k=50
        )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        
    return inference_time, generated_text

def compare_models():
    """比较MoE模型和传统密集模型"""
    tokenizer = AutoTokenizer.from_pretrained("tokenizer_k")
    
    # 配置1: 传统密集模型
    print("=" * 50)
    print("传统密集模型")
    print("=" * 50)
    dense_config = ModelConfig(
        dim=1024,
        n_layers=18,
        num_experts=1,          # 1个专家=密集模型
        num_experts_per_tok=1,
        moe_freq=999,           # 永不使用MoE
        aux_loss_coef=0.0,
    )
    
    dense_model = Transformer(dense_config)
    dense_params = count_parameters(dense_model)
    print(f"总参数量: {dense_params / 1e6:.2f}M")
    
    # 配置2: MoE模型
    print("\n" + "=" * 50)
    print("MoE模型")
    print("=" * 50)
    moe_config = ModelConfig(
        dim=1024,
        n_layers=18,
        num_experts=8,          # 8个专家
        num_experts_per_tok=2,  # 每次激活2个
        moe_freq=2,             # 每隔2层使用MoE
        aux_loss_coef=0.01,
    )
    
    moe_model = Transformer(moe_config)
    moe_active_params, moe_total_params = count_active_parameters(moe_model)
    moe_layers = sum(1 for layer in moe_model.layers if layer.use_moe)
    
    print(f"总参数量: {moe_total_params / 1e6:.2f}M")
    print(f"激活参数量: {moe_active_params / 1e6:.2f}M")
    print(f"MoE层数量: {moe_layers} / {len(moe_model.layers)}")
    print(f"参数效率: {moe_total_params / moe_active_params:.2f}x")
    
    # 测试文本
    test_text = "你好，我想了解一下"
    
    # 测试密集模型
    print("\n" + "-" * 30)
    print("密集模型推理测试")
    print("-" * 30)
    try:
        dense_time, dense_output = test_model_inference(dense_model, tokenizer, test_text)
        print(f"推理时间: {dense_time:.4f}秒")
        print(f"生成文本: {dense_output}")
    except Exception as e:
        print(f"密集模型测试失败: {e}")
    
    # 测试MoE模型
    print("\n" + "-" * 30)
    print("MoE模型推理测试")
    print("-" * 30)
    try:
        moe_time, moe_output = test_model_inference(moe_model, tokenizer, test_text)
        print(f"推理时间: {moe_time:.4f}秒")
        print(f"生成文本: {moe_output}")
    except Exception as e:
        print(f"MoE模型测试失败: {e}")
    
    # 对比结果
    print("\n" + "=" * 50)
    print("模型对比总结")
    print("=" * 50)
    print(f"密集模型参数量: {dense_params / 1e6:.2f}M")
    print(f"MoE模型总参数量: {moe_total_params / 1e6:.2f}M")
    print(f"MoE模型激活参数量: {moe_active_params / 1e6:.2f}M")
    print(f"参数扩展倍数: {moe_total_params / dense_params:.2f}x")
    print(f"激活参数比例: {moe_active_params / moe_total_params:.2%}")

def test_expert_selection():
    """测试专家选择机制"""
    print("\n" + "=" * 50)
    print("专家选择机制测试")
    print("=" * 50)
    
    tokenizer = AutoTokenizer.from_pretrained("tokenizer_k")
    config = ModelConfig(
        dim=256,  # 较小的模型便于观察
        n_layers=4,
        n_heads=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_freq=1,  # 每层都使用MoE
        aux_loss_coef=0.01,
    )
    
    model = Transformer(config)
    
    # 创建测试输入
    test_inputs = [
        "数学问题：",
        "编程相关：",
        "文学创作：",
        "科学知识：",
    ]
    
    for i, text in enumerate(test_inputs):
        print(f"\n输入 {i+1}: {text}")
        input_ids = tokenizer.encode(text, return_tensors='pt')
        
        # 前向传播并观察专家使用情况
        with torch.no_grad():
            output = model(input_ids)
            print(f"输出形状: {output.logits.shape}")
            if output.last_loss is not None:
                print(f"损失（含辅助损失）: {output.last_loss.item():.4f}")

if __name__ == "__main__":
    print("MoE模型测试开始...")
    
    try:
        # 主要对比测试
        compare_models()
        
        # 专家选择测试
        test_expert_selection()
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()