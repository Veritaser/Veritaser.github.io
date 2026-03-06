
import torch
from transformers import AutoTokenizer
from dataset import SFTDataset
import numpy as np
import json

def test_mask():
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')
    
    # 创建包含不同格式数据的测试文件
    data = [
        # 格式 1: messages 列表
        [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀"}],
        # 格式 2: Belle 格式
        {"instruction": "你是谁？", "output": "我是AI助手"},
        # 格式 3: 包含 messages key
        {"messages": [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "2"}]}
    ]
    
    with open('test_data.jsonl', 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    ds = SFTDataset('test_data.jsonl', tokenizer, max_length=128)
    
    print(f"Dataset size: {len(ds)}")
    
    for i in range(len(ds)):
        print(f"\n--- Sample {i} ---")
        X, Y, mask = ds[i]
        
        print("Mask sum:", mask.sum())
        
        if mask.sum() > 0:
            masked_Y = Y[mask == 1]
            print("Decoded Masked Y:", tokenizer.decode(masked_Y))
        else:
            print("WARNING: Mask is empty!")
            print("Decoded X:", tokenizer.decode(X))

if __name__ == "__main__":
    test_mask()
