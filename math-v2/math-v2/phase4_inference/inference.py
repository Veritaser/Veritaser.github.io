"""
Inference with Self-Verification
利用自验证能力进行推理，支持Best-of-N采样策略
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
from dataclasses import dataclass
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


@dataclass
class InferenceConfig:
    """推理配置"""
    model_path: str = "./outputs/grpo"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    
    # 生成参数
    num_samples: int = 16  # Best-of-N的N (建议16-32)
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # 选择策略
    strategy: str = "score_weighted_vote"  # score_weighted_vote, majority_vote, highest_score


class SelfVerifyingInference:
    """带自验证的推理器"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        # System prompt
        self.system_prompt = """You are a mathematics expert. When solving problems:
1. Show your complete step-by-step reasoning in a "## Solution" section
2. Verify your work in a "## Self Evaluation" section
3. Provide a confidence score between 0.0 and 1.0 at the end, in the format: "Score: \\boxed{score}" """
    
    def _load_model(self):
        """加载模型 (支持LoRA)"""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
        
        # 检查是否是LoRA模型
        is_lora = os.path.exists(os.path.join(self.config.model_path, "adapter_config.json"))
        
        if is_lora:
            print(f"Detected LoRA model at {self.config.model_path}")
            peft_config = PeftConfig.from_pretrained(self.config.model_path)
            base_model_path = peft_config.base_model_name_or_path
            
            print(f"Loading base model: {base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch_dtype,
                device_map=self.config.device,
            )
            
            print(f"Loading LoRA adapters from {self.config.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.config.model_path)
        else:
            print(f"Loading full model from {self.config.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch_dtype,
                device_map=self.config.device,
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
    
    def _build_prompt(self, question: str) -> str:
        """构建推理prompt"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            return f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """从回答中提取答案"""
        if not response:
            return None
        
        # 方法1: \boxed{} 格式
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
        match = re.search(r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+\$?(-?[\d,]+(?:\.\d+)?)', response, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '').strip()
        
        # 方法3: Solution部分最后的等式
        solution_match = re.search(r'## Solution(.*?)(?=## Self Evaluation|$)', response, re.DOTALL)
        if solution_match:
            equals_matches = re.findall(r'=\s*\$?(-?[\d,]+(?:\.\d+)?)\s*(?:\$|$|\n|\.)', solution_match.group(1))
            if equals_matches:
                return equals_matches[-1].replace(',', '').strip()
        
        return None
    
    def _extract_score(self, response: str) -> Optional[float]:
        """从回答中提取自评分数"""
        if not response:
            return None
        
        patterns = [
            r'Score:\s*\\boxed\{([01](?:\.\d+)?)\}',
            r'Score:\s*\*\*([01](?:\.\d+)?)\*\*',
            r'Score:\s*([01](?:\.\d+)?)',
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
    
    @torch.no_grad()
    def generate_samples(self, question: str) -> List[Dict[str, Any]]:
        """
        生成多个采样结果
        
        Args:
            question: 数学问题
        
        Returns:
            采样结果列表，每个包含response, answer, score
        """
        prompt = self._build_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        samples = []
        
        for _ in range(self.config.num_samples):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            answer = self._extract_answer(response)
            score = self._extract_score(response)
            
            samples.append({
                "response": response,
                "answer": answer,
                "score": score,
            })
        
        return samples
    
    def select_best_answer(self, samples: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        从多个采样中选择最佳答案
        
        Args:
            samples: 采样结果列表
        
        Returns:
            (最佳答案, 选择详情)
        """
        if self.config.strategy == "highest_score":
            return self._select_highest_score(samples)
        elif self.config.strategy == "majority_vote":
            return self._select_majority_vote(samples)
        else:  # score_weighted_vote
            return self._select_score_weighted_vote(samples)
    
    def _select_highest_score(self, samples: List[Dict]) -> Tuple[str, Dict]:
        """选择自评分数最高的答案"""
        valid_samples = [s for s in samples if s["answer"] and s["score"] is not None]
        
        if not valid_samples:
            # fallback: 返回第一个有答案的
            for s in samples:
                if s["answer"]:
                    return s["answer"], {"method": "fallback", "score": None}
            return "", {"method": "no_answer", "score": None}
        
        best = max(valid_samples, key=lambda x: x["score"])
        return best["answer"], {"method": "highest_score", "score": best["score"]}
    
    def _select_majority_vote(self, samples: List[Dict]) -> Tuple[str, Dict]:
        """多数投票选择答案"""
        answers = [s["answer"] for s in samples if s["answer"]]
        
        if not answers:
            return "", {"method": "no_answer", "votes": {}}
        
        counter = Counter(answers)
        most_common = counter.most_common(1)[0]
        
        return most_common[0], {
            "method": "majority_vote",
            "votes": dict(counter),
            "winning_votes": most_common[1],
        }
    
    def _select_score_weighted_vote(self, samples: List[Dict]) -> Tuple[str, Dict]:
        """
        分数加权投票
        优先选择高分(>=0.9)的答案进行投票
        """
        # 筛选高分答案
        high_score_samples = [
            s for s in samples 
            if s["answer"] and s["score"] is not None and s["score"] >= 0.9
        ]
        
        if high_score_samples:
            # 对高分答案进行投票
            answers = [s["answer"] for s in high_score_samples]
            counter = Counter(answers)
            most_common = counter.most_common(1)[0]
            
            return most_common[0], {
                "method": "score_weighted_vote",
                "high_score_count": len(high_score_samples),
                "votes": dict(counter),
            }
        
        # 没有高分答案，fallback到分数最高的
        valid_samples = [s for s in samples if s["answer"] and s["score"] is not None]
        
        if valid_samples:
            best = max(valid_samples, key=lambda x: x["score"])
            return best["answer"], {
                "method": "fallback_highest_score",
                "score": best["score"],
            }
        
        # 最后fallback：多数投票
        answers = [s["answer"] for s in samples if s["answer"]]
        if answers:
            counter = Counter(answers)
            return counter.most_common(1)[0][0], {"method": "fallback_majority"}
        
        return "", {"method": "no_answer"}
    
    def infer(self, question: str) -> Dict[str, Any]:
        """
        完整推理流程
        
        Args:
            question: 数学问题
        
        Returns:
            推理结果
        """
        samples = self.generate_samples(question)
        answer, selection_info = self.select_best_answer(samples)
        
        return {
            "question": question,
            "answer": answer,
            "selection_info": selection_info,
            "samples": samples,
            "num_samples": len(samples),
        }


def main():
    parser = argparse.ArgumentParser(description="Self-Verifying Inference")
    parser.add_argument("--model_path", type=str, default="./outputs/grpo")
    parser.add_argument("--question", type=str, default=None,
                        help="Single question to solve")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of samples for Best-of-N")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--strategy", type=str, default="score_weighted_vote",
                        choices=["score_weighted_vote", "majority_vote", "highest_score"])
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    config = InferenceConfig(
        model_path=args.model_path,
        device=args.device,
        num_samples=args.num_samples,
        temperature=args.temperature,
        strategy=args.strategy,
    )
    
    inferencer = SelfVerifyingInference(config)
    
    if args.question:
        # 单个问题推理
        result = inferencer.infer(args.question)
        
        print("\n" + "="*60)
        print("Question:", result["question"])
        print("="*60)
        print(f"\nFinal Answer: {result['answer']}")
        print(f"Selection Method: {result['selection_info']}")
        print(f"\nSamples ({result['num_samples']}):")
        for i, sample in enumerate(result["samples"]):
            print(f"  [{i+1}] Answer: {sample['answer']}, Score: {sample['score']}")
    else:
        # 交互模式
        print("Self-Verifying Inference (type 'quit' to exit)")
        print(f"Model: {args.model_path}")
        print(f"Strategy: {args.strategy}, Samples: {args.num_samples}")
        
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            result = inferencer.infer(question)
            
            print(f"\nAnswer: {result['answer']}")
            print(f"Selection: {result['selection_info']}")
            
            # 显示采样详情
            show_details = input("Show sample details? (y/n): ").strip().lower()
            if show_details == 'y':
                for i, sample in enumerate(result["samples"]):
                    print(f"\n--- Sample {i+1} ---")
                    print(f"Answer: {sample['answer']}, Score: {sample['score']}")
                    print(f"Response: {sample['response'][:500]}...")


if __name__ == "__main__":
    main()
