#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA 微调脚本 - 工地安全VLM
使用 Qwen2-VL-2B-Instruct 进行领域微调
"""

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from transformers import (
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from qwen_vl_utils import process_vision_info


# ============= 配置参数 =============
@dataclass
class TrainConfig:
    # 模型配置
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 训练配置
    output_dir: str = "models/lora_safety"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    
    # 数据配置
    train_data_path: str = "data/vqa/train.jsonl"


class Qwen2VLDataCollator:
    """Qwen2-VL数据整理器"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        """批量处理数据"""
        batch_messages = [json.loads(feature["messages"]) for feature in features]
        
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in batch_messages
        ]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        
        return inputs


def load_vqa_dataset(data_path: str) -> Dataset:
    """加载VQA数据集"""
    samples = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            
            img_path = sample.get('image_path', '')
            if not img_path:
                continue
            
            img_path = img_path.replace('\\', '/')
            
            if not os.path.exists(img_path):
                continue
            
            abs_path = os.path.abspath(img_path).replace('\\', '/')
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{abs_path}"},
                        {"type": "text", "text": sample['question']}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": sample['answer']
                }
            ]
            
            samples.append({"messages": json.dumps(messages, ensure_ascii=False)})
    
    print(f"Loaded {len(samples)} samples")
    return Dataset.from_list(samples)


def setup_lora_model(model_name: str, lora_config: LoraConfig):
    """设置LoRA模型"""
    local_model_path = "models/pretrained/Qwen/Qwen2-VL-2B-Instruct"
    if os.path.exists(local_model_path):
        model_name = local_model_path
        print(f"[INFO] 使用本地模型: {model_name}")
    else:
        print(f"[INFO] 加载基础模型: {model_name}")
    
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.bfloat16
        print(f"[INFO] 使用设备: GPU (CUDA)")
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
        print(f"[INFO] 使用设备: CPU")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor


def main():
    parser = argparse.ArgumentParser(description="LoRA微调工地安全VLM")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/vqa/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/lora_safety")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    args = parser.parse_args()
    
    config = TrainConfig()
    config.model_name = args.model_name
    config.train_data_path = args.data_path
    config.output_dir = args.output_dir
    config.num_train_epochs = args.num_epochs
    config.per_device_train_batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    
    model, processor = setup_lora_model(config.model_name, lora_config)
    
    print(f"[INFO] 加载数据: {config.train_data_path}")
    dataset = load_vqa_dataset(config.train_data_path)
    
    data_collator = Qwen2VLDataCollator(processor)
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    print("[INFO] 开始LoRA微调...")
    trainer.train()
    
    print(f"[INFO] 保存LoRA权重到: {config.output_dir}")
    model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)
    
    print("[INFO] 训练完成!")


if __name__ == "__main__":
    main()
