
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/gaorenzhi/math-v2/math-v2/outputs/sft"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

prompt = """<|im_start|>system
You are a mathematics expert. When solving problems:
1. Show your complete step-by-step reasoning in a "## Solution" section
2. Verify your work in a "## Self Evaluation" section
3. Provide a confidence score between 0.0 and 1.0<|im_end|>
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("-" * 20)
print(response)
print("-" * 20)
