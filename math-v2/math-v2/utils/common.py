"""
Common Utilities
共享的工具函数
"""

import re
import json
import os
from typing import Optional, List, Dict, Any, Tuple


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
        return match.group(1).replace(',', '')
    return None


def extract_model_answer(response: str) -> Optional[str]:
    """
    从模型生成的回答中提取最终答案
    
    Args:
        response: 模型生成的回答
    
    Returns:
        提取的数值字符串
    """
    if not response:
        return None
    
    # 方法1: \boxed{} 格式（排除Score）
    boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', response)
    for match in boxed_matches:
        try:
            val = float(match.replace(',', ''))
            if not (0 <= val <= 1 and '.' in match):
                return match.replace(',', '').strip()
        except:
            num_match = re.search(r'(-?[\d,]+(?:\.\d+)?)', match)
            if num_match:
                return num_match.group(1).replace(',', '').strip()
    
    # 方法2: "The answer is X"
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+\$?(-?[\d,]+(?:\.\d+)?)',
        r'(?:therefore|thus|so)[,\s]+(?:the\s+)?answer\s+is[:\s]+\$?(-?[\d,]+(?:\.\d+)?)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '').strip()
    
    # 方法3: Solution部分最后的等式
    solution_match = re.search(r'## Solution(.*?)(?=## Self Evaluation|$)', response, re.DOTALL)
    if solution_match:
        equals_matches = re.findall(r'=\s*\$?(-?[\d,]+(?:\.\d+)?)\s*(?:\$|$|\n|\.)', solution_match.group(1))
        if equals_matches:
            return equals_matches[-1].replace(',', '').strip()
    
    return None


def extract_score(response: str) -> Optional[float]:
    """
    从模型回答中提取自评分数
    
    Args:
        response: 模型生成的回答
    
    Returns:
        提取的分数 (0.0-1.0)
    """
    if not response:
        return None
    
    patterns = [
        r'Score:\s*\\boxed\{([01](?:\.\d+)?)\}',
        r'Score:\s*\*\*([01](?:\.\d+)?)\*\*',
        r'Score:\s*([01](?:\.\d+)?)',
        r'置信度[分数]*[:：]\s*\\boxed\{([01](?:\.\d+)?)\}',
    ]
    
    for pattern in patterns:
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
        cleaned = num_str.replace(',', '').replace(' ', '').strip()
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
    
    if gold_num == int(gold_num):
        return abs(pred_num - gold_num) < tolerance
    
    return abs(pred_num - gold_num) < tolerance or abs(pred_num - gold_num) / abs(gold_num) < 1e-4


def load_jsonl(path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """保存为JSONL文件"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def setup_logging(name: str, level: str = "INFO"):
    """设置日志"""
    import logging
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper()),
    )
    return logging.getLogger(name)
