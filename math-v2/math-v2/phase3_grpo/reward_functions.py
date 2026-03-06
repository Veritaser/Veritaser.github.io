"""
Reward Functions for GRPO Training
GRPO训练的奖励函数设计
"""

import re
from typing import Optional, List, Dict, Tuple, Any


def extract_answer(response: str) -> Optional[str]:
    """
    从模型生成的回答中提取最终答案
    
    Args:
        response: 模型生成的回答
    
    Returns:
        提取的数值字符串
    """
    if not response:
        return None
    
    # 方法1: 查找 \boxed{} 中非Score的答案
    boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', response)
    for match in boxed_matches:
        try:
            val = float(match.replace(',', ''))
            # 如果不是0到1之间的小数（Score格式），则认为是答案
            if not (0 <= val <= 1 and '.' in match):
                return match.replace(',', '').strip()
        except:
            num_match = re.search(r'(-?[\d,]+(?:\.\d+)?)', match)
            if num_match:
                return num_match.group(1).replace(',', '').strip()
    
    # 方法2: 查找 "The answer is X"
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+\$?(-?[\d,]+(?:\.\d+)?)',
        r'(?:therefore|thus|so)[,\s]+(?:the\s+)?answer\s+is[:\s]+\$?(-?[\d,]+(?:\.\d+)?)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '').strip()
    
    # 方法3: 在Solution部分的最后查找等式结果
    solution_match = re.search(r'## Solution(.*?)(?=## Self Evaluation|$)', response, re.DOTALL)
    if solution_match:
        solution_text = solution_match.group(1)
        equals_matches = re.findall(r'=\s*\$?(-?[\d,]+(?:\.\d+)?)\s*(?:\$|$|\n|\.)', solution_text)
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


def check_format(response: str) -> bool:
    """
    检查回答是否符合预期格式
    
    Args:
        response: 模型生成的回答
    
    Returns:
        是否符合格式
    """
    if not response:
        return False
    
    has_solution = "## Solution" in response or "## 解答" in response
    has_eval = "## Self Evaluation" in response or "## 自我评估" in response
    has_score = extract_score(response) is not None
    
    return has_solution and has_eval and has_score


def normalize_number(num_str: str) -> Optional[float]:
    """标准化数字字符串"""
    if not num_str:
        return None
    try:
        cleaned = num_str.replace(',', '').replace(' ', '').strip()
        return float(cleaned)
    except:
        return None


def verify_answer(pred_answer: str, gold_answer: str, tolerance: float = 1e-6) -> bool:
    """验证预测答案是否正确"""
    pred_num = normalize_number(pred_answer)
    gold_num = normalize_number(gold_answer)
    
    if pred_num is None or gold_num is None:
        return False
    
    if gold_num == int(gold_num):
        return abs(pred_num - gold_num) < tolerance
    
    return abs(pred_num - gold_num) < tolerance or abs(pred_num - gold_num) / abs(gold_num) < 1e-4


def compute_reward(
    response: str,
    gold_answer: str,
    outcome_weight: float = 1.0,
    format_weight: float = 0.1,
    consistency_weight: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """
    计算综合奖励分数
    
    根据DeepSeekMath论文设计的奖励函数：
    R_total = R_outcome + w_format * R_format + w_consistency * R_consistency
    
    Args:
        response: 模型生成的回答
        gold_answer: 标准答案
        outcome_weight: 结果奖励权重
        format_weight: 格式奖励权重
        consistency_weight: 一致性奖励权重
    
    Returns:
        (总奖励, 各部分奖励字典)
    """
    # 1. 提取预测答案和自评分数
    pred_answer = extract_answer(response)
    pred_score = extract_score(response)
    
    # 2. 结果奖励 (Outcome Reward)
    is_correct = verify_answer(pred_answer, gold_answer) if pred_answer else False
    r_outcome = 1.0 if is_correct else 0.0
    
    # 3. 格式奖励 (Format Reward)
    r_format = 1.0 if check_format(response) else 0.0
    
    # 4. 一致性奖励 (Consistency Reward) - 增强版 (Rule-based Meta-Verifier)
    # 激励模型：对的时候自信(1.0)，错的时候承认错误(0.0)
    # 惩罚模型：错的时候盲目自信(1.0)
    if pred_score is not None:
        if is_correct:
            if pred_score > 0.9:
                # 正确且自信 -> 完美
                r_consistency = 1.0
            elif pred_score < 0.5:
                # 正确但不自信 -> 小奖励
                r_consistency = 0.1
            else:
                r_consistency = 0.5
        else:
            if pred_score < 0.5:
                # 错误但承认错误 (低分) -> 鼓励 (这是DeepSeekMath-V2的核心思想)
                # 给一个小额正奖励，优于"错误且自信"的负分
                r_consistency = 0.1
            elif pred_score > 0.9:
                # 错误且盲目自信 -> 重罚 (幻觉)
                r_consistency = -1.0
            else:
                # 错误且分数中等 -> 惩罚
                r_consistency = -0.5
    else:
        # 没有输出分数，给予惩罚
        r_consistency = -0.1
    
    # 5. 计算总奖励
    total_reward = (
        outcome_weight * r_outcome +
        format_weight * r_format +
        consistency_weight * r_consistency
    )
    
    reward_breakdown = {
        "outcome": r_outcome,
        "format": r_format,
        "consistency": r_consistency,
        "total": total_reward,
        "is_correct": is_correct,
        "pred_answer": pred_answer,
        "pred_score": pred_score,
    }
    
    return total_reward, reward_breakdown


def batch_compute_rewards(
    prompts: List[str],
    completions: List[str],
    gold_answers: List[str],
    outcome_weight: float = 1.0,
    format_weight: float = 0.1,
    consistency_weight: float = 0.5,
) -> List[float]:
    """
    批量计算奖励
    
    Args:
        prompts: 问题列表
        completions: 模型回答列表
        gold_answers: 标准答案列表
        outcome_weight: 结果奖励权重
        format_weight: 格式奖励权重
        consistency_weight: 一致性奖励权重
    
    Returns:
        奖励列表
    """
    rewards = []
    
    for completion, gold_ans in zip(completions, gold_answers):
        reward, _ = compute_reward(
            completion, gold_ans,
            outcome_weight, format_weight, consistency_weight
        )
        rewards.append(reward)
    
    return rewards


class RewardFunction:
    """
    奖励函数类，用于GRPO训练
    """
    
    def __init__(
        self,
        outcome_weight: float = 1.0,
        format_weight: float = 0.1,
        consistency_weight: float = 0.5,
    ):
        self.outcome_weight = outcome_weight
        self.format_weight = format_weight
        self.consistency_weight = consistency_weight
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        gold_answers: List[str],
        **kwargs
    ) -> List[float]:
        """
        计算奖励
        
        Args:
            prompts: 问题列表
            completions: 模型回答列表
            gold_answers: 标准答案列表
        
        Returns:
            奖励列表
        """
        return batch_compute_rewards(
            prompts, completions, gold_answers,
            self.outcome_weight,
            self.format_weight,
            self.consistency_weight,
        )
    
    def get_detailed_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        gold_answers: List[str],
    ) -> List[Dict[str, Any]]:
        """
        获取详细奖励信息
        
        Returns:
            奖励详情列表
        """
        details = []
        
        for prompt, completion, gold_ans in zip(prompts, completions, gold_answers):
            _, breakdown = compute_reward(
                completion, gold_ans,
                self.outcome_weight,
                self.format_weight,
                self.consistency_weight,
            )
            breakdown["prompt"] = prompt[:100] + "..."
            details.append(breakdown)
        
        return details


# 用于TRL的奖励函数wrapper
def create_reward_fn(
    gold_answers: Dict[str, str],
    outcome_weight: float = 1.0,
    format_weight: float = 0.1,
    consistency_weight: float = 0.5,
):
    """
    创建用于TRL的奖励函数
    
    Args:
        gold_answers: 问题到答案的映射字典
        outcome_weight: 结果奖励权重
        format_weight: 格式奖励权重
        consistency_weight: 一致性奖励权重
    
    Returns:
        奖励函数
    """
    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str], **kwargs):
        rewards = []
        for prompt, output in zip(prompts, outputs):
            gold_ans = gold_answers.get(prompt, "")
            reward, _ = compute_reward(
                output, gold_ans,
                outcome_weight, format_weight, consistency_weight
            )
            rewards.append(reward)
        return rewards
    
    return reward_fn


if __name__ == "__main__":
    # 测试奖励函数
    test_response_correct = """## Solution
Let's solve step by step.
April: 48 clips
May: 48/2 = 24 clips
Total: 48 + 24 = 72 clips

The answer is \\boxed{72}

## Self Evaluation
Let me verify:
- April sales: 48 ✓
- May sales: half of 48 = 24 ✓
- Total: 48 + 24 = 72 ✓

Score: \\boxed{1.0}"""

    test_response_wrong = """## Solution
April: 48 clips
May: 48/2 = 20 clips (wrong)
Total: 48 + 20 = 68 clips

The answer is \\boxed{68}

## Self Evaluation
Calculation seems correct.

Score: \\boxed{1.0}"""

    gold = "72"
    
    print("Test 1: Correct answer with high confidence")
    reward1, breakdown1 = compute_reward(test_response_correct, gold)
    print(f"  Reward: {reward1:.3f}")
    print(f"  Breakdown: {breakdown1}")
    
    print("\nTest 2: Wrong answer with high confidence (should be penalized)")
    reward2, breakdown2 = compute_reward(test_response_wrong, gold)
    print(f"  Reward: {reward2:.3f}")
    print(f"  Breakdown: {breakdown2}")
