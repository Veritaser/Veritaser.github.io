"""
vLLM Fast Inference Server
使用vLLM进行高效批量推理
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any
from collections import Counter
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not installed. Use `pip install vllm` to enable fast inference.")


class VLLMInference:
    """使用vLLM的高效推理器"""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed")
        
        self.model = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
        self.system_prompt = """You are a mathematics expert. When solving problems:
1. Show your complete step-by-step reasoning in a "## Solution" section
2. Verify your work in a "## Self Evaluation" section
3. Provide a confidence score between 0.0 and 1.0"""
    
    def _build_prompt(self, question: str) -> str:
        """构建prompt"""
        return f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """提取答案"""
        if not response:
            return None
        
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
        
        return None
    
    def _extract_score(self, response: str) -> Optional[float]:
        """提取自评分数"""
        if not response:
            return None
        
        patterns = [
            r'Score:\s*\\boxed\{([01](?:\.\d+)?)\}',
            r'Score:\s*([01](?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        return None
    
    def generate_batch(
        self,
        questions: List[str],
        num_samples: int = 8,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[List[Dict[str, Any]]]:
        """
        批量生成多个采样
        
        Args:
            questions: 问题列表
            num_samples: 每个问题的采样数
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: Top-p采样参数
        
        Returns:
            每个问题的采样结果列表
        """
        # 构建所有prompts
        prompts = []
        for q in questions:
            prompt = self._build_prompt(q)
            prompts.extend([prompt] * num_samples)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            n=1,  # 每个prompt生成1个
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        # 批量生成
        outputs = self.model.generate(prompts, sampling_params)
        
        # 整理结果
        all_results = []
        idx = 0
        for _ in questions:
            samples = []
            for _ in range(num_samples):
                response = outputs[idx].outputs[0].text
                samples.append({
                    "response": response,
                    "answer": self._extract_answer(response),
                    "score": self._extract_score(response),
                })
                idx += 1
            all_results.append(samples)
        
        return all_results
    
    def select_best_answer(
        self,
        samples: List[Dict[str, Any]],
        strategy: str = "score_weighted_vote"
    ) -> str:
        """选择最佳答案"""
        if strategy == "highest_score":
            valid = [s for s in samples if s["answer"] and s["score"] is not None]
            if valid:
                return max(valid, key=lambda x: x["score"])["answer"]
        
        elif strategy == "score_weighted_vote":
            high_score = [s for s in samples if s["answer"] and s["score"] and s["score"] >= 0.9]
            if high_score:
                answers = [s["answer"] for s in high_score]
                return Counter(answers).most_common(1)[0][0]
        
        # fallback: majority vote
        answers = [s["answer"] for s in samples if s["answer"]]
        if answers:
            return Counter(answers).most_common(1)[0][0]
        
        return ""
    
    def infer_batch(
        self,
        questions: List[str],
        num_samples: int = 8,
        strategy: str = "score_weighted_vote",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量推理
        
        Returns:
            结果列表
        """
        all_samples = self.generate_batch(questions, num_samples, **kwargs)
        
        results = []
        for question, samples in zip(questions, all_samples):
            answer = self.select_best_answer(samples, strategy)
            results.append({
                "question": question,
                "answer": answer,
                "samples": samples,
            })
        
        return results


def main():
    if not VLLM_AVAILABLE:
        print("vLLM is not installed. Please install with: pip install vllm")
        return
    
    parser = argparse.ArgumentParser(description="vLLM Fast Inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--questions_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="./outputs/vllm_results.json")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--strategy", type=str, default="score_weighted_vote")
    
    args = parser.parse_args()
    
    # 初始化
    inferencer = VLLMInference(args.model_path)
    
    if args.questions_file:
        # 从文件加载问题
        with open(args.questions_file, 'r', encoding='utf-8') as f:
            questions = [json.loads(line)["question"] for line in f]
    else:
        # 使用示例问题
        questions = [
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        ]
    
    # 批量推理
    print(f"Processing {len(questions)} questions...")
    results = []
    
    for i in tqdm(range(0, len(questions), args.batch_size)):
        batch = questions[i:i+args.batch_size]
        batch_results = inferencer.infer_batch(
            batch,
            num_samples=args.num_samples,
            strategy=args.strategy,
        )
        results.extend(batch_results)
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
