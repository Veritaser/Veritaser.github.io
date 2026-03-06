"""
Teacher Model Data Generation
使用Teacher模型（GPT-4o/Claude/DeepSeek）生成带自我验证的训练数据
"""

import os
import json
import argparse
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from datasets import load_dataset
from openai import OpenAI

from prompts import TEACHER_PROMPT, TEACHER_SYSTEM_PROMPT


@dataclass
class TeacherConfig:
    """Teacher模型配置"""
    provider: str = "openai"  # openai, anthropic, deepseek, local, vllm
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # 本地模型配置
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1  # vLLM并行度
    gpu_memory_utilization: float = 0.9  # vLLM显存利用率


class TeacherGenerator:
    """Teacher模型数据生成器"""
    
    def __init__(self, config: TeacherConfig):
        self.config = config
        self._init_client()
    
    def _init_client(self):
        """初始化API客户端"""
        if self.config.provider == "openai":
            self.client = OpenAI(
                api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=self.config.base_url
            )
        elif self.config.provider == "deepseek":
            self.client = OpenAI(
                api_key=self.config.api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url=self.config.base_url or "https://api.deepseek.com/v1"
            )
        elif self.config.provider == "local":
            # 使用Transformers本地模型
            self._init_local_transformers()
        elif self.config.provider == "vllm":
            # 使用vLLM本地推理
            self._init_vllm()
        elif self.config.provider == "ollama":
            # 使用Ollama本地服务
            self.client = OpenAI(
                api_key="ollama",  # Ollama不需要真实key
                base_url=self.config.base_url or "http://localhost:11434/v1"
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _init_local_transformers(self):
        """初始化本地Transformers模型"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")
        
        # 确定torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
        
        print(f"Loading local model: {self.config.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=self.config.device,
        )
        self.model.eval()
        
        print(f"Local model loaded successfully on {self.config.device}")
    
    def _init_vllm(self):
        """初始化vLLM推理引擎"""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Please install vllm: pip install vllm")
        
        print(f"Loading vLLM model: {self.config.model_name}...")
        
        self.vllm_model = LLM(
            model=self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
        
        print(f"vLLM model loaded successfully")
    
    def _generate_local_transformers(self, prompt: str) -> str:
        """使用本地Transformers模型生成"""
        import torch
        
        # 构建消息
        messages = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # 使用chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = f"System: {TEACHER_SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response
    
    def _generate_vllm(self, prompt: str) -> str:
        """使用vLLM生成"""
        from vllm import SamplingParams
        
        # 构建完整prompt
        full_prompt = f"system\n{TEACHER_SYSTEM_PROMPT}\nuser\n{prompt}\nassistant\n"
        
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=0.9,
        )
        
        outputs = self.vllm_model.generate([full_prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def generate_single(self, question: str, retry: int = 3) -> Optional[str]:
        """
        生成单个问题的回答
        
        Args:
            question: 数学问题
            retry: 重试次数
        
        Returns:
            生成的回答文本
        """
        prompt = TEACHER_PROMPT.format(question=question)
        
        for attempt in range(retry):
            try:
                if self.config.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.config.model_name,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        system=TEACHER_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text
                elif self.config.provider == "local":
                    # 本地Transformers模型
                    return self._generate_local_transformers(prompt)
                elif self.config.provider == "vllm":
                    # vLLM推理
                    return self._generate_vllm(prompt)
                else:
                    # OpenAI compatible API (OpenAI, DeepSeek, Ollama)
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[
                            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        return None
    
    def generate_batch(
        self, 
        questions: List[str], 
        max_workers: int = 4,
        save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        批量生成数据
        
        Args:
            questions: 问题列表
            max_workers: 并发线程数
            save_path: 中间结果保存路径
        
        Returns:
            生成的数据列表
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate_single, q): i 
                for i, q in enumerate(questions)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(questions), desc="Generating"):
                idx = future_to_idx[future]
                try:
                    response = future.result()
                    result = {
                        "index": idx,
                        "question": questions[idx],
                        "response": response,
                        "success": response is not None
                    }
                    results.append(result)
                    
                    # 增量保存
                    if save_path and len(results) % 100 == 0:
                        self._save_checkpoint(results, save_path)
                        
                except Exception as e:
                    print(f"Error processing question {idx}: {e}")
                    results.append({
                        "index": idx,
                        "question": questions[idx],
                        "response": None,
                        "success": False
                    })
        
        # 按原始顺序排序
        results.sort(key=lambda x: x["index"])
        
        if save_path:
            self._save_checkpoint(results, save_path)
        
        return results
    
    def _save_checkpoint(self, results: List[Dict], path: str):
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_gsm8k_questions(split: str = "train", num_samples: Optional[int] = None) -> List[Dict]:
    """
    加载GSM8K数据集
    
    Args:
        split: 数据集分割 ('train' 或 'test')
        num_samples: 采样数量，None表示全部
    
    Returns:
        问题和答案列表
    """
    # 直接指定数据文件路径来加载数据集
    data_files = {"train": "./gsm8k_dataset/train/data-00000-of-00001.arrow", 
                  "test": "./gsm8k_dataset/test/data-00000-of-00001.arrow"}
    
    dataset = load_dataset("arrow", data_files={split: data_files[split]})[split]
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    data = []
    for item in dataset:
        # GSM8K数据集有固定的question和answer字段
        data.append({
            "question": item["question"],
            "answer": item["answer"]
        })
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate training data using Teacher model")
    parser.add_argument("--provider", type=str, default="openai", 
                        choices=["openai", "anthropic", "deepseek", "local", "vllm", "ollama"])
    parser.add_argument("--model", type=str, default="simaas-deepseek-v3-v1")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Number of samples to process (default: all)")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./data/raw")
    parser.add_argument("--base_url", type=str, default="https://console.siflow.cn/model-api",
                        help="Custom API base URL")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for local model (cuda/cpu)")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Torch dtype for local model")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    
    args = parser.parse_args()
    
    # 初始化配置
    config = TeacherConfig(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        device=args.device,
        torch_dtype=args.torch_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    # 加载数据
    print("Loading GSM8K dataset...")
    gsm8k_data = load_gsm8k_questions("train", args.num_samples)
    questions = [item["question"] for item in gsm8k_data]
    
    print(f"Loaded {len(questions)} questions")
    
    # 生成数据
    generator = TeacherGenerator(config)
    output_path = os.path.join(args.output_dir, "teacher_responses.jsonl")
    
    print(f"Generating responses with {args.provider}/{args.model}...")
    
    # 本地模型使用单线程避免显存问题
    max_workers = 1 if args.provider in ["local", "vllm"] else args.max_workers
    
    results = generator.generate_batch(
        questions, 
        max_workers=max_workers,
        save_path=output_path
    )
    
    # 保存原始GSM8K答案用于后续验证
    gsm8k_answers_path = os.path.join(args.output_dir, "gsm8k_answers.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(gsm8k_answers_path, 'w', encoding='utf-8') as f:
        for item in gsm8k_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 统计
    success_count = sum(1 for r in results if r["success"])
    print(f"\nGeneration complete!")
    print(f"Success: {success_count}/{len(results)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
