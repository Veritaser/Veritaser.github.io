"""
SFT Training Script with DeepSpeed
使用DeepSpeed进行监督微调训练
"""

import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
import datasets
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code"}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for model (float16, bfloat16, float32)"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention 2"}
    )
    # LoRA Arguments
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA"})
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    train_file: str = field(
        default="./data/sft/train.jsonl",
        metadata={"help": "Path to training data"}
    )
    val_file: Optional[str] = field(
        default="./data/sft/val.jsonl",
        metadata={"help": "Path to validation data"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for data preprocessing"}
    )


@dataclass  
class SFTTrainingArguments(TrainingArguments):
    """训练参数"""
    output_dir: str = field(default="./outputs/sft")
    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    eval_strategy: str = field(default="steps")
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    deepspeed: Optional[str] = field(default=None)
    report_to: str = field(default="tensorboard")
    remove_unused_columns: bool = field(default=False)


class SFTDataset(Dataset):
    """SFT训练数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 2048,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        messages = item.get("messages", [])
        
        # 使用tokenizer的chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # 备用方案：手动拼接
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # 创建labels，只对assistant的回复计算loss
        labels = self._create_labels(input_ids, messages)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _create_labels(self, input_ids: List[int], messages: List[Dict]) -> List[int]:
        """
        创建labels，只对assistant回复部分计算loss
        其他部分标记为-100
        """
        labels = input_ids.copy()
        
        # 找到assistant回复的起始位置
        # 简化实现：使用特殊token来定位
        # 对于Qwen，assistant回复在 <|im_start|>assistant 之后
        
        # 这里使用简化方法：全部计算loss
        # 更精确的实现需要根据具体tokenizer调整
        
        # 如果需要mask掉user部分，可以在这里实现
        # 目前保持简单，让模型学习完整对话
        
        return labels


class SFTDataCollator:
    """数据整理器，支持动态padding"""
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 找到batch中最长的序列
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length
        )
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for feature in features:
            input_ids = feature["input_ids"][:max_len]
            attention_mask = feature["attention_mask"][:max_len]
            labels = feature["labels"][:max_len]
            
            # Padding
            padding_length = max_len - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length  # -100会被loss忽略
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        
        # 转换为tensor
        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        
        return batch


def load_model_and_tokenizer(model_args: ModelArguments):
    """加载模型和tokenizer"""
    
    # 确定torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(model_args.torch_dtype, torch.bfloat16)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    
    # 确保pad token存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    model_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    
    # 尝试使用flash attention
    if model_args.use_flash_attention:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            logger.warning("Flash attention not available, using default attention")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )
    
    # 启用gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        
    # 配置LoRA
    if model_args.use_lora:
        logger.info("Configuring LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    # 解析参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, SFTTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 设置日志级别
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    
    # 检查checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming from {last_checkpoint}")
    
    # 加载模型和tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    # 加载数据集
    logger.info("Loading datasets...")
    train_dataset = SFTDataset(
        data_args.train_file,
        tokenizer,
        data_args.max_length
    )
    
    eval_dataset = None
    if data_args.val_file and os.path.exists(data_args.val_file):
        eval_dataset = SFTDataset(
            data_args.val_file,
            tokenizer,
            data_args.max_length
        )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # 数据整理器
    data_collator = SFTDataCollator(tokenizer, data_args.max_length)
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 训练
    logger.info("Starting training...")
    if last_checkpoint is not None:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        train_result = trainer.train()
    
    # 保存最终模型
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 评估
    if eval_dataset:
        logger.info("Running evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
