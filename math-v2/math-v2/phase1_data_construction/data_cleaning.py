"""
Data Cleaning and Verification
数据清洗与验证 - 提取答案并与GSM8K标准答案对比
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Tuple, Any
from tqdm import tqdm


def extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    """
    从GSM8K标准答案中提取最终数值
    GSM8K答案格式: "...#### 数字"
    
    Args:
        answer_text: GSM8K原始答案文本
    
    Returns:
        提取的数值字符串
    """
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', answer_text)
    if match:
        # 移除逗号
        return match.group(1).replace(',', '')
    return None


def extract_model_answer(response: str) -> Optional[str]:
    """
    从模型生成的回答中提取最终答案
    支持多种格式:
    - \\boxed{数字}
    - The answer is 数字
    - = 数字 (在最后)
    
    Args:
        response: 模型生成的回答
    
    Returns:
        提取的数值字符串
    """
    if not response:
        return None
    
    # 方法1: 查找 \boxed{} 中非Score的答案
    # 需要排除 Score: \boxed{0.x} 格式
    boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', response)
    for match in boxed_matches:
        # 如果不是0到1之间的小数（Score格式），则认为是答案
        try:
            val = float(match.replace(',', ''))
            if not (0 <= val <= 1 and '.' in match):
                return match.replace(',', '')
        except:
            # 可能包含单位或其他文本
            num_match = re.search(r'(-?[\d,]+(?:\.\d+)?)', match)
            if num_match:
                return num_match.group(1).replace(',', '')
    
    # 方法2: 查找 "The answer is X" 或 "the final answer is X"
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+\$?(-?[\d,]+(?:\.\d+)?)',
        r'(?:therefore|thus|so)[,\s]+(?:the\s+)?answer\s+is[:\s]+\$?(-?[\d,]+(?:\.\d+)?)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '')
    
    # 方法3: 在Solution部分的最后查找等式结果
    solution_match = re.search(r'## Solution(.*?)(?=## Self Evaluation|$)', response, re.DOTALL)
    if solution_match:
        solution_text = solution_match.group(1)
        # 查找最后一个 = 数字
        equals_matches = re.findall(r'=\s*\$?(-?[\d,]+(?:\.\d+)?)\s*(?:\$|$|\n|\.)', solution_text)
        if equals_matches:
            return equals_matches[-1].replace(',', '')
    
    return None


def extract_score(response: str) -> Optional[float]:
    """
    从模型回答中提取自评分数
    格式: Score: \\boxed{0.x}
    
    Args:
        response: 模型生成的回答
    
    Returns:
        提取的分数 (0.0-1.0)
    """
    if not response:
        return None
    
    # 查找 Score: \boxed{x.x} 格式
    score_patterns = [
        r'Score:\s*\\boxed\{([01](?:\.\d+)?)\}',
        r'Score:\s*\*\*([01](?:\.\d+)?)\*\*',
        r'Score:\s*([01](?:\.\d+)?)',
        r'置信度[分数]*[:：]\s*\\boxed\{([01](?:\.\d+)?)\}',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 1:
                    return score
            except:
                pass
    
    return None


def check_format(response: str) -> Tuple[bool, List[str]]:
    """
    检查回答是否符合预期格式
    
    Args:
        response: 模型生成的回答
    
    Returns:
        (是否合格, 缺失的部分列表)
    """
    if not response:
        return False, ["empty_response"]
    
    missing = []
    
    if "## Solution" not in response and "## 解答" not in response:
        missing.append("solution_section")
    
    if "## Self Evaluation" not in response and "## 自我评估" not in response:
        missing.append("self_evaluation_section")
    
    if extract_score(response) is None:
        missing.append("score")
    
    return len(missing) == 0, missing


def normalize_number(num_str: str) -> Optional[float]:
    """
    标准化数字字符串用于比较
    
    Args:
        num_str: 数字字符串
    
    Returns:
        标准化后的浮点数
    """
    if not num_str:
        return None
    try:
        # 移除逗号和空格
        cleaned = num_str.replace(',', '').replace(' ', '')
        return float(cleaned)
    except:
        return None


def verify_answer(pred_answer: str, gold_answer: str, tolerance: float = 1e-6) -> bool:
    """
    验证预测答案是否正确
    
    Args:
        pred_answer: 预测答案
        gold_answer: 标准答案
        tolerance: 浮点数比较容差
    
    Returns:
        是否正确
    """
    pred_num = normalize_number(pred_answer)
    gold_num = normalize_number(gold_answer)
    
    if pred_num is None or gold_num is None:
        return False
    
    # 对于整数结果，精确匹配
    if gold_num == int(gold_num):
        return abs(pred_num - gold_num) < tolerance
    
    # 对于浮点数，允许小误差
    return abs(pred_num - gold_num) < tolerance or abs(pred_num - gold_num) / abs(gold_num) < 1e-4


def clean_and_verify(
    raw_data_path: str,
    gsm8k_answers_path: str,
    output_path: str,
    keep_incorrect: bool = False
) -> Dict[str, Any]:
    """
    清洗和验证数据
    
    Args:
        raw_data_path: Teacher生成的原始数据路径
        gsm8k_answers_path: GSM8K标准答案路径
        output_path: 输出路径
        keep_incorrect: 是否保留错误答案（用于训练模型识别错误）
    
    Returns:
        统计信息
    """
    # 加载数据
    raw_data = []
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw_data.append(json.loads(line))
    
    gsm8k_data = []
    with open(gsm8k_answers_path, 'r', encoding='utf-8') as f:
        for line in f:
            gsm8k_data.append(json.loads(line))
    
    # 创建问题到答案的映射
    question_to_answer = {item["question"]: item["answer"] for item in gsm8k_data}
    
    # 处理数据
    cleaned_data = []
    stats = {
        "total": len(raw_data),
        "success_generation": 0,
        "format_valid": 0,
        "answer_correct": 0,
        "answer_incorrect": 0,
        "missing_format": {"solution_section": 0, "self_evaluation_section": 0, "score": 0}
    }
    
    for item in tqdm(raw_data, desc="Cleaning data"):
        if not item.get("success") or not item.get("response"):
            continue
        
        stats["success_generation"] += 1
        
        question = item["question"]
        response = item["response"]
        
        # 检查格式
        format_valid, missing = check_format(response)
        if format_valid:
            stats["format_valid"] += 1
        else:
            for m in missing:
                if m in stats["missing_format"]:
                    stats["missing_format"][m] += 1
        
        # 提取答案
        gold_answer_text = question_to_answer.get(question, "")
        gold_answer = extract_gsm8k_answer(gold_answer_text)
        pred_answer = extract_model_answer(response)
        pred_score = extract_score(response)
        
        # 验证答案
        is_correct = verify_answer(pred_answer, gold_answer) if pred_answer and gold_answer else False
        
        if is_correct:
            stats["answer_correct"] += 1
        else:
            stats["answer_incorrect"] += 1
        
        # 决定是否保留
        if is_correct or keep_incorrect:
            cleaned_item = {
                "question": question,
                "response": response,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "pred_score": pred_score,
                "is_correct": is_correct,
                "format_valid": format_valid
            }
            cleaned_data.append(cleaned_item)
    
    # 保存清洗后的数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean and verify generated data")
    parser.add_argument("--raw_data", type=str, default="./data/raw/teacher_responses.jsonl",
                        help="Path to raw teacher responses")
    parser.add_argument("--gsm8k_answers", type=str, default="./data/raw/gsm8k_answers.jsonl",
                        help="Path to GSM8K answers")
    parser.add_argument("--output", type=str, default="./data/cleaned/verified_data.jsonl",
                        help="Output path for cleaned data")
    parser.add_argument("--keep_incorrect", action="store_true",
                        help="Keep incorrect answers for advanced training")
    
    args = parser.parse_args()
    
    print("Starting data cleaning and verification...")
    stats = clean_and_verify(
        args.raw_data,
        args.gsm8k_answers,
        args.output,
        args.keep_incorrect
    )
    
    print("\n" + "="*50)
    print("Data Cleaning Statistics")
    print("="*50)
    print(f"Total samples: {stats['total']}")
    print(f"Successful generations: {stats['success_generation']}")
    print(f"Format valid: {stats['format_valid']}")
    print(f"Answer correct: {stats['answer_correct']}")
    print(f"Answer incorrect: {stats['answer_incorrect']}")
    print(f"\nMissing format parts:")
    for key, value in stats['missing_format'].items():
        print(f"  {key}: {value}")
    
    accuracy = stats['answer_correct'] / stats['success_generation'] * 100 if stats['success_generation'] > 0 else 0
    print(f"\nTeacher accuracy: {accuracy:.2f}%")
    print(f"\nCleaned data saved to: {args.output}")


if __name__ == "__main__":
    main()
