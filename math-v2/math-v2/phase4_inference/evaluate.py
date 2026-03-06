"""
GSM8K Evaluation Script
在GSM8K测试集上评估模型性能
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import SelfVerifyingInference, InferenceConfig


def extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    """从GSM8K标准答案中提取数值"""
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    return None


def normalize_number(num_str: str) -> Optional[float]:
    """标准化数字"""
    if not num_str:
        return None
    try:
        return float(num_str.replace(',', '').strip())
    except:
        return None


def check_answer(pred: str, gold: str, tolerance: float = 1e-6) -> bool:
    """检查答案是否正确"""
    pred_num = normalize_number(pred)
    gold_num = normalize_number(gold)
    
    if pred_num is None or gold_num is None:
        return False
    
    if gold_num == int(gold_num):
        return abs(pred_num - gold_num) < tolerance
    
    return abs(pred_num - gold_num) < tolerance


def evaluate_gsm8k(
    model_path: str,
    num_samples: int = 8,
    temperature: float = 0.7,
    strategy: str = "score_weighted_vote",
    device: str = "cuda",
    max_examples: Optional[int] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    在GSM8K测试集上评估模型
    
    Args:
        model_path: 模型路径
        num_samples: Best-of-N的N
        temperature: 采样温度
        strategy: 答案选择策略
        device: 设备
        max_examples: 最大评估样本数（用于快速测试）
        output_file: 结果输出文件
    
    Returns:
        评估结果
    """
    # 加载测试数据（仅从本地 Arrow 文件加载，失败则直接报错）
    print("Loading GSM8K test set from local Arrow file...")
    # 本地 arrow 文件相对路径（相对于本文件）
    local_arrow = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "gsm8k_dataset", "test", "data-00000-of-00001.arrow")
    )

    if not os.path.exists(local_arrow):
        raise RuntimeError(f"Local arrow file not found: {local_arrow}")

    try:
        # Use Hugging Face `datasets` loader to read the local Arrow shard.
        # This mirrors the loading approach used in phase3_grpo/train_grpo_trl.py
        from datasets import load_dataset

        print(f"Loading local arrow file: {local_arrow}")
        data_files = {"test": local_arrow}
        dataset = load_dataset("arrow", data_files=data_files)["test"]
    except Exception as e:
        raise RuntimeError(f"Failed to load local arrow file ({local_arrow}): {e}")
    
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} examples")
    
    # 初始化推理器
    config = InferenceConfig(
        model_path=model_path,
        device=device,
        num_samples=num_samples,
        temperature=temperature,
        strategy=strategy,
    )
    
    print("Loading model...")
    inferencer = SelfVerifyingInference(config)
    
    # 评估
    results = []
    correct = 0
    total = 0
    
    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        gold_answer = extract_gsm8k_answer(item["answer"])
        
        # 推理
        result = inferencer.infer(question)
        pred_answer = result["answer"]
        
        # 检查正确性
        is_correct = check_answer(pred_answer, gold_answer)
        if is_correct:
            correct += 1
        total += 1
        
        # 记录结果
        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "is_correct": is_correct,
            "selection_info": result["selection_info"],
            "samples": [
                {"answer": s["answer"], "score": s["score"]}
                for s in result["samples"]
            ]
        })
        
        # 实时显示进度
        if total % 50 == 0:
            print(f"  Progress: {total}/{len(dataset)}, Accuracy: {correct/total*100:.2f}%")
    
    # 计算最终结果
    accuracy = correct / total * 100
    
    # 统计分析
    high_score_correct = sum(
        1 for r in results 
        if r["is_correct"] and r["selection_info"].get("method") == "score_weighted_vote"
    )
    
    # 分析自评分数与正确性的相关性
    score_analysis = analyze_scores(results)
    
    eval_results = {
        "model_path": model_path,
        "strategy": strategy,
        "num_samples": num_samples,
        "temperature": temperature,
        "total_examples": total,
        "correct": correct,
        "accuracy": accuracy,
        "score_analysis": score_analysis,
        "timestamp": datetime.now().isoformat(),
    }
    
    # 保存结果
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        
        # 保存详细结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": eval_results,
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
    
    return eval_results


def analyze_scores(results: List[Dict]) -> Dict[str, Any]:
    """分析自评分数与正确性的相关性"""
    high_score_samples = []
    low_score_samples = []
    
    for r in results:
        for s in r["samples"]:
            if s["score"] is not None:
                if s["score"] >= 0.9:
                    high_score_samples.append({
                        "answer": s["answer"],
                        "gold": r["gold_answer"],
                        "correct": check_answer(s["answer"], r["gold_answer"]) if s["answer"] else False,
                    })
                elif s["score"] <= 0.3:
                    low_score_samples.append({
                        "answer": s["answer"],
                        "gold": r["gold_answer"],
                        "correct": check_answer(s["answer"], r["gold_answer"]) if s["answer"] else False,
                    })
    
    high_score_accuracy = sum(1 for s in high_score_samples if s["correct"]) / len(high_score_samples) * 100 if high_score_samples else 0
    low_score_accuracy = sum(1 for s in low_score_samples if s["correct"]) / len(low_score_samples) * 100 if low_score_samples else 0
    
    return {
        "high_score_samples": len(high_score_samples),
        "high_score_accuracy": high_score_accuracy,
        "low_score_samples": len(low_score_samples),
        "low_score_accuracy": low_score_accuracy,
        "calibration_gap": high_score_accuracy - low_score_accuracy,  # 理想情况应该很大
    }


def compare_strategies(
    model_path: str,
    device: str = "cuda",
    max_examples: int = 100,
) -> Dict[str, Any]:
    """
    比较不同选择策略的效果
    """
    strategies = ["score_weighted_vote", "majority_vote", "highest_score"]
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Evaluating strategy: {strategy}")
        print('='*50)
        
        eval_result = evaluate_gsm8k(
            model_path=model_path,
            num_samples=8,
            strategy=strategy,
            device=device,
            max_examples=max_examples,
        )
        
        results[strategy] = eval_result
    
    # 打印比较结果
    print("\n" + "="*60)
    print("Strategy Comparison")
    print("="*60)
    for strategy, result in results.items():
        print(f"  {strategy}: {result['accuracy']:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")
    parser.add_argument("--model_path", type=str, default="./outputs/grpo")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--strategy", type=str, default="score_weighted_vote",
                        choices=["score_weighted_vote", "majority_vote", "highest_score"])
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Max examples for quick testing")
    parser.add_argument("--output_file", type=str, default="./outputs/eval_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compare_strategies", action="store_true",
                        help="Compare different selection strategies")
    
    args = parser.parse_args()
    
    if args.compare_strategies:
        compare_strategies(
            args.model_path,
            args.device,
            args.max_examples or 100,
        )
    else:
        results = evaluate_gsm8k(
            model_path=args.model_path,
            num_samples=args.num_samples,
            temperature=args.temperature,
            strategy=args.strategy,
            device=args.device,
            max_examples=args.max_examples,
            output_file=args.output_file,
        )
        
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"Model: {results['model_path']}")
        print(f"Strategy: {results['strategy']}")
        print(f"Samples per question: {results['num_samples']}")
        print(f"Total examples: {results['total_examples']}")
        print(f"Correct: {results['correct']}")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        
        if results.get('score_analysis'):
            print(f"\nScore Analysis:")
            print(f"  High score (>=0.9) samples: {results['score_analysis']['high_score_samples']}")
            print(f"  High score accuracy: {results['score_analysis']['high_score_accuracy']:.2f}%")
            print(f"  Low score (<=0.3) samples: {results['score_analysis']['low_score_samples']}")
            print(f"  Low score accuracy: {results['score_analysis']['low_score_accuracy']:.2f}%")
            print(f"  Calibration gap: {results['score_analysis']['calibration_gap']:.2f}%")


if __name__ == "__main__":
    main()
