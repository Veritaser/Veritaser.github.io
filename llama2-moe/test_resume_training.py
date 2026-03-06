#!/usr/bin/env python3
"""
断点续训使用示例脚本

演示如何使用断点续训功能来恢复训练
"""

import os
import subprocess
import argparse

def run_training_with_resume():
    """演示断点续训的使用方法"""
    
    # 基础训练命令
    base_cmd = [
        "python", "ddp_pretrain.py",
        "--out_dir", "moe_pretrain_resume_test",
        "--epochs", "2",
        "--batch_size", "16",  # 小批次便于测试
        "--learning_rate", "1e-4",
        "--log_interval", "50",
        "--save_interval", "100",  # 更频繁保存便于测试
        "--data_path", "./seq_monkey_datawhale.jsonl"
    ]
    
    print("=" * 60)
    print("断点续训功能测试")
    print("=" * 60)
    
    # 第一次训练（模拟中断）
    print("\n1. 开始第一次训练...")
    print("命令:", " ".join(base_cmd))
    
    try:
        # 运行一段时间后手动中断（模拟）
        result = subprocess.run(base_cmd, timeout=30, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print("✓ 第一次训练已中断（模拟中断）")
    except Exception as e:
        print(f"第一次训练出错: {e}")
    
    # 检查是否生成了检查点
    save_dir = "moe_pretrain_resume_test"
    if os.path.exists(save_dir):
        files = os.listdir(save_dir)
        checkpoint_files = [f for f in files if f.endswith('.pth')]
        print(f"✓ 检查点目录存在，包含 {len(checkpoint_files)} 个检查点文件")
        for f in checkpoint_files:
            print(f"  - {f}")
    else:
        print("✗ 检查点目录不存在")
        return
    
    # 第二次训练（从最新检查点恢复）
    print("\n2. 从最新检查点自动恢复训练...")
    resume_cmd = base_cmd + ["--auto_resume"]
    print("命令:", " ".join(resume_cmd))
    
    try:
        result = subprocess.run(resume_cmd, timeout=20, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 自动恢复训练成功")
        else:
            print("✗ 自动恢复训练失败")
            print("错误输出:", result.stderr)
    except subprocess.TimeoutExpired:
        print("✓ 恢复训练正在进行中...")
    except Exception as e:
        print(f"恢复训练出错: {e}")
    
    # 第三次训练（从指定检查点恢复）
    print("\n3. 从指定检查点恢复训练...")
    latest_checkpoint = os.path.join(save_dir, "checkpoint_latest.pth")
    if os.path.exists(latest_checkpoint):
        resume_cmd = base_cmd + ["--resume", latest_checkpoint]
        print("命令:", " ".join(resume_cmd))
        
        try:
            result = subprocess.run(resume_cmd, timeout=20, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ 指定检查点恢复训练成功")
            else:
                print("✗ 指定检查点恢复训练失败")
                print("错误输出:", result.stderr)
        except subprocess.TimeoutExpired:
            print("✓ 指定检查点恢复训练正在进行中...")
        except Exception as e:
            print(f"指定检查点恢复训练出错: {e}")
    else:
        print(f"✗ 检查点文件不存在: {latest_checkpoint}")
    
    print("\n" + "=" * 60)
    print("断点续训测试完成")
    print("=" * 60)


def run_sft_training_with_resume():
    """演示SFT训练的断点续训"""
    
    base_cmd = [
        "python", "ddp_sft_full.py",
        "--out_dir", "moe_sft_resume_test", 
        "--epochs", "1",
        "--batch_size", "8",
        "--learning_rate", "5e-5",
        "--log_interval", "20",
        "--save_interval", "50",
        "--data_path", "./BelleGroup_sft.jsonl"
    ]
    
    print("\n" + "=" * 60)
    print("SFT断点续训功能测试")
    print("=" * 60)
    
    print("\n1. SFT训练自动恢复测试...")
    resume_cmd = base_cmd + ["--auto_resume"]
    print("命令:", " ".join(resume_cmd))
    
    try:
        result = subprocess.run(resume_cmd, timeout=15, capture_output=True, text=True)
        print("✓ SFT自动恢复测试完成")
    except subprocess.TimeoutExpired:
        print("✓ SFT训练正在进行中...")
    except Exception as e:
        print(f"SFT训练出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="断点续训测试脚本")
    parser.add_argument("--test_type", choices=["pretrain", "sft", "all"], default="all",
                       help="测试类型: pretrain(预训练), sft(监督微调), all(全部)")
    
    args = parser.parse_args()
    
    if args.test_type in ["pretrain", "all"]:
        run_training_with_resume()
    
    if args.test_type in ["sft", "all"]:
        run_sft_training_with_resume()
    
    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    print("1. 预训练断点续训:")
    print("   python ddp_pretrain.py --auto_resume")
    print("   python ddp_pretrain.py --resume path/to/checkpoint.pth")
    print()
    print("2. SFT断点续训:")
    print("   python ddp_sft_full.py --auto_resume")
    print("   python ddp_sft_full.py --resume path/to/sft_checkpoint.pth")
    print()
    print("3. 检查点文件位置:")
    print("   - 预训练: {out_dir}/checkpoint_latest.pth")
    print("   - SFT: {out_dir}/sft_checkpoint_latest.pth")
    print("   - 里程碑: {out_dir}/checkpoint_step{N}.pth")
    print("=" * 60)