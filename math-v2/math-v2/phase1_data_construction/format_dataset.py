"""
Format Dataset for SFT Training
将清洗后的数据转换为SFT训练格式（ShareGPT/Alpaca格式）
"""

import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import random


def format_to_sharegpt(
    cleaned_data: List[Dict],
    system_prompt: str = None
) -> List[Dict]:
    """
    转换为ShareGPT格式
    
    ShareGPT格式:
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    """
    formatted = []
    
    default_system = """You are a mathematics expert. When solving problems:
1. Show your complete step-by-step reasoning in a "## Solution" section
2. Verify your work in a "## Self Evaluation" section
3. Provide a confidence score between 0.0 and 1.0"""
    
    system = system_prompt or default_system
    
    for item in cleaned_data:
        conversation = {
            "conversations": [
                {"from": "system", "value": system},
                {"from": "human", "value": item["question"]},
                {"from": "gpt", "value": item["response"]}
            ]
        }
        formatted.append(conversation)
    
    return formatted


def format_to_alpaca(cleaned_data: List[Dict]) -> List[Dict]:
    """
    转换为Alpaca格式
    
    Alpaca格式:
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    }
    """
    formatted = []
    
    instruction = """Solve the following math problem step-by-step. After your solution, provide a self-evaluation section where you verify your logic and calculations. Finally, assign a confidence score between 0.0 and 1.0."""
    
    for item in cleaned_data:
        alpaca_item = {
            "instruction": instruction,
            "input": item["question"],
            "output": item["response"]
        }
        formatted.append(alpaca_item)
    
    return formatted


def format_to_messages(cleaned_data: List[Dict]) -> List[Dict]:
    """
    转换为Messages格式 (OpenAI/HuggingFace标准格式)
    
    Messages格式:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    formatted = []
    
    system_content = """You are a mathematics expert. When solving problems:
1. Show your complete step-by-step reasoning in a "## Solution" section
2. Verify your work in a "## Self Evaluation" section  
3. Provide a confidence score between 0.0 and 1.0"""
    
    for item in cleaned_data:
        messages_item = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["response"]}
            ]
        }
        formatted.append(messages_item)
    
    return formatted


def split_dataset(
    data: List[Dict],
    val_ratio: float = 0.05,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    划分训练集和验证集
    
    Args:
        data: 数据列表
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        {"train": [...], "val": [...]}
    """
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    val_size = int(len(shuffled) * val_ratio)
    
    return {
        "train": shuffled[val_size:],
        "val": shuffled[:val_size]
    }


def save_dataset(
    data: List[Dict],
    output_path: str,
    format_type: str = "jsonl"
):
    """
    保存数据集
    
    Args:
        data: 数据列表
        output_path: 输出路径
        format_type: 格式类型 ('jsonl' 或 'json')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format_type == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Format cleaned data for SFT training")
    parser.add_argument("--input", type=str, default="./data/cleaned/verified_data.jsonl",
                        help="Path to cleaned data")
    parser.add_argument("--output_dir", type=str, default="./data/sft",
                        help="Output directory")
    parser.add_argument("--format", type=str, default="messages",
                        choices=["sharegpt", "alpaca", "messages"],
                        help="Output format type")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting")
    parser.add_argument("--only_correct", action="store_true",
                        help="Only include correct answers")
    
    args = parser.parse_args()
    
    # 加载清洗后的数据
    print(f"Loading data from {args.input}...")
    cleaned_data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if args.only_correct and not item.get("is_correct", True):
                continue
            cleaned_data.append(item)
    
    print(f"Loaded {len(cleaned_data)} samples")
    
    # 格式转换
    print(f"Converting to {args.format} format...")
    if args.format == "sharegpt":
        formatted_data = format_to_sharegpt(cleaned_data)
    elif args.format == "alpaca":
        formatted_data = format_to_alpaca(cleaned_data)
    else:
        formatted_data = format_to_messages(cleaned_data)
    
    # 划分数据集
    print("Splitting dataset...")
    splits = split_dataset(formatted_data, args.val_ratio, args.seed)
    
    # 保存数据
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    
    save_dataset(splits["train"], train_path, "jsonl")
    save_dataset(splits["val"], val_path, "jsonl")
    
    # 同时保存一份json格式用于某些框架
    save_dataset(splits["train"], os.path.join(args.output_dir, "train.json"), "json")
    save_dataset(splits["val"], os.path.join(args.output_dir, "val.json"), "json")
    
    print(f"\n{'='*50}")
    print("Dataset Formatting Complete")
    print(f"{'='*50}")
    print(f"Format: {args.format}")
    print(f"Train samples: {len(splits['train'])}")
    print(f"Val samples: {len(splits['val'])}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
