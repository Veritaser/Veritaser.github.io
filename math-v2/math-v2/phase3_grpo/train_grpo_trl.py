"""
GRPO Training Script with TRL (Alternative Implementation)
使用TRL库的GRPO实现，更简洁稳定
"""

import os
import sys
import json
import logging
import argparse
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer

from reward_functions import (
    extract_answer,
    extract_score,
    check_format,
    verify_answer,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def extract_gsm8k_answer(answer_text: str) -> str:
    """从GSM8K答案中提取数值"""
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    return ""


def load_gsm8k_for_grpo(num_samples: Optional[int] = None) -> HFDataset:
    """加载GSM8K数据集并准备用于GRPO"""
    data_files = {"train": "../gsm8k_dataset/train/data-00000-of-00001.arrow"}
    dataset = load_dataset("arrow", data_files=data_files)["train"]
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    def process_example(example):
        question = example["question"]
        gold_answer = extract_gsm8k_answer(example["answer"])
        
        # 构建system prompt
        system_prompt = """You are a mathematics expert. When solving problems:
1. Show your complete step-by-step reasoning in a "## Solution" section
2. Verify your work in a "## Self Evaluation" section
3. Provide a confidence score between 0.0 and 1.0 at the end, in the format: "Score: \\boxed{score}" """
        
        # 构建完整prompt
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        return {
            "prompt": prompt,
            "question": question,
            "gold_answer": gold_answer,
        }
    
    processed_dataset = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
    )
    
    return processed_dataset


# 全局答案映射，用于奖励函数
ANSWER_MAP = {}


def reward_fn_outcome(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    结果奖励函数 - 答案是否正确
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        # 从prompt中提取问题
        gold_answer = ""
        for question, answer in ANSWER_MAP.items():
            if question in prompt:
                gold_answer = answer
                break
        
        pred_answer = extract_answer(completion)
        is_correct = verify_answer(pred_answer, gold_answer) if pred_answer and gold_answer else False
        rewards.append(1.0 if is_correct else 0.0)
    
    return rewards


def reward_fn_format(completions: List[str], **kwargs) -> List[float]:
    """
    格式奖励函数 - 是否符合要求的格式
    """
    rewards = []
    for completion in completions:
        if check_format(completion):
            rewards.append(0.1)  # 格式正确给小奖励
        else:
            rewards.append(0.0)
    return rewards


def reward_fn_consistency(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    一致性奖励函数 - 自评分数与实际结果是否一致
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        # 获取gold answer
        gold_answer = ""
        for question, answer in ANSWER_MAP.items():
            if question in prompt:
                gold_answer = answer
                break
        
        pred_answer = extract_answer(completion)
        pred_score = extract_score(completion)
        is_correct = verify_answer(pred_answer, gold_answer) if pred_answer and gold_answer else False
        
        if pred_score is not None:
            # R_consistency = 1 - |S_pred - S_real|
            s_real = 1.0 if is_correct else 0.0
            r_consistency = 0.5 * (1.0 - abs(pred_score - s_real))
        else:
            r_consistency = 0.0
        
        rewards.append(r_consistency)
    
    return rewards


def main():
    parser = argparse.ArgumentParser(description="GRPO Training with TRL")
    parser.add_argument("--model_path", type=str, default="./outputs/sft",
                        help="Path to SFT model")
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of training samples")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Number of generations per prompt (G)")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by launcher")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 加载tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    logger.info("Loading dataset...")
    dataset = load_gsm8k_for_grpo(args.num_samples)
    
    # 构建答案映射
    global ANSWER_MAP
    for item in dataset:
        ANSWER_MAP[item["question"]] = item["gold_answer"]
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # 加载模型
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    
    # GRPO配置
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        generation_kwargs={"max_new_tokens": args.max_new_tokens},
        temperature=args.temperature,
        beta=args.kl_coef,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
        report_to="tensorboard",
        seed=args.seed,
        remove_unused_columns=False,
    )
    
    # 创建trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[
            reward_fn_outcome,
            reward_fn_format, 
            reward_fn_consistency,
        ],
    )
    
    # 训练
    logger.info("Starting GRPO training...")
    trainer.train()
    
    # 保存
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("GRPO training complete!")


if __name__ == "__main__":
    main()
