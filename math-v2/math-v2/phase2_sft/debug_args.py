
import sys
from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    trust_remote_code: bool = field(default=True)
    torch_dtype: str = field(default="bfloat16")
    use_flash_attention: bool = field(default=True)

@dataclass
class DataArguments:
    train_file: str = field(default="./data/sft/train.jsonl")
    val_file: str = field(default="./data/sft/val.jsonl")
    max_length: int = field(default=2048)
    preprocessing_num_workers: int = field(default=4)

@dataclass
class SFTTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./outputs/sft")
    # 显式定义 evaluation_strategy 默认值，模拟 train_sft.py
    eval_strategy: str = field(default="steps")

def main():
    print("DEBUG: sys.argv:", sys.argv)
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print(f"DEBUG: training_args.evaluation_strategy = '{training_args.evaluation_strategy}'")
    print(f"DEBUG: training_args.save_strategy = '{training_args.save_strategy}'")
    print(f"DEBUG: training_args.load_best_model_at_end = {training_args.load_best_model_at_end}")

if __name__ == "__main__":
    main()
