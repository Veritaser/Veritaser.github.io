"""
Sanity Check Script for SFT Model
检验SFT训练后的模型是否能正确输出Self Evaluation格式
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device: str = "cuda"):
    """加载模型"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """生成回答"""
    
    messages = [
        {
            "role": "system",
            "content": """You are a mathematics expert. When solving problems:
1. Show your complete step-by-step reasoning in a "## Solution" section
2. Verify your work in a "## Self Evaluation" section
3. Provide a confidence score between 0.0 and 1.0"""
        },
        {"role": "user", "content": question}
    ]
    
    # 使用chat template
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = f"User: {question}\nAssistant:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def check_format(response: str) -> dict:
    """检查输出格式"""
    checks = {
        "has_solution": "## Solution" in response or "## 解答" in response,
        "has_self_evaluation": "## Self Evaluation" in response or "## 自我评估" in response,
        "has_score": "Score:" in response or "分数:" in response,
        "has_boxed": "\\boxed{" in response,
    }
    checks["all_passed"] = all(checks.values())
    return checks


# 测试问题
TEST_QUESTIONS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
]


def main():
    parser = argparse.ArgumentParser(description="Sanity check for SFT model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained SFT model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    args = parser.parse_args()
    
    print("Loading model...")
    model, tokenizer = load_model(args.model_path, args.device)
    
    print("\n" + "="*60)
    print("SFT Model Sanity Check")
    print("="*60)
    
    all_passed = True
    
    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\n{'='*60}")
        print(f"Test {i+1}")
        print(f"{'='*60}")
        print(f"Question: {question[:100]}...")
        
        response = generate_response(
            model, tokenizer, question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        
        print(f"\nResponse:\n{response[:500]}...")
        
        checks = check_format(response)
        print(f"\nFormat Checks:")
        for key, value in checks.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value}")
        
        if not checks["all_passed"]:
            all_passed = False
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    if all_passed:
        print("✓ All format checks passed! Model is ready for Phase 3 (RL).")
    else:
        print("✗ Some format checks failed. Consider:")
        print("  - Training for more epochs")
        print("  - Checking training data quality")
        print("  - Adjusting learning rate")


if __name__ == "__main__":
    main()
