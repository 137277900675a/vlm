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
from PIL import Image


# ============= 配置参数 =============
@dataclass
class TrainConfig:
    # 模型配置
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_resolution: int = 448  # VLM常用的分辨率
    
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
    max_length: int = 512


def load_vqa_dataset(data_path: str, image_base_dir: str = None) -> Dataset:
    """加载VQA数据集"""
    samples = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            
            # 处理图像路径
            img_path = sample.get('image_path', '')
            if image_base_dir and not os.path.isabs(img_path):
                img_path = os.path.join(image_base_dir, os.path.basename(img_path))
            
            # 构建对话格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": sample['question']}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": [{"type": "text", "text": sample['answer']}]
                }
            ]
            
            samples.append({
                "messages": messages,
                "image_path": img_path,
            })
    
    return Dataset.from_list(samples)


def preprocess_function(examples, processor, max_length: int = 512):
    """预处理数据 - 图像+文本转模型输入"""
    texts = []
    images = []
    
    for msg_list in examples["messages"]:
        # 提取文本和图像
        prompt = ""
        for msg in msg_list:
            if msg["role"] == "user":
                for content in msg["content"]:
                    if content["type"] == "text":
                        prompt += content["text"]
                    elif content["type"] == "image":
                        img_path = content.get("image", "")
                        if os.path.exists(img_path):
                            images.append(Image.open(img_path).convert("RGB"))
                        else:
                            # 使用占位图
                            images.append(Image.new('RGB', (224, 224), color='gray'))
            elif msg["role"] == "assistant":
                for content in msg["content"]:
                    if content["type"] == "text":
                        prompt += "\n" + content["text"]
        
        texts.append(prompt)
    
    # 处理输入
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        max_length=max_length,
        truncation=True,
    )
    
    # 处理标签
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"],
        "labels": labels,
        "attention_mask": inputs["attention_mask"],
    }


def setup_lora_model(model_name: str, lora_config: LoraConfig):
    """设置LoRA模型"""
    print(f"🔄 加载基础模型: {model_name}")
    
    # 加载处理器
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    
    # 加载模型 (使用4bit量化减少显存)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 冻结视觉编码器 (可选择性解冻)
    for param in model.model.visual.parameters():
        param.requires_grad = False
    
    # 应用LoRA
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
    
    # 初始化配置
    config = TrainConfig()
    config.model_name = args.model_name
    config.train_data_path = args.data_path
    config.output_dir = args.output_dir
    config.num_train_epochs = args.num_epochs
    config.per_device_train_batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    
    # 加载模型
    model, processor = setup_lora_model(config.model_name, lora_config)
    
    # 加载数据
    print(f"📂 加载数据: {config.train_data_path}")
    dataset = load_vqa_dataset(config.train_data_path)
    
    # 预处理
    def process(examples):
        return preprocess_function(examples, processor, config.max_length)
    
    processed_dataset = dataset.map(
        process,
        batched=True,
        batch_size=8,
        remove_columns=dataset.column_names,
    )
    
    # 数据整理器 - 使用默认的DataCollator
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        processor=processor,
        padding=True,
        return_tensors="pt",
    )
    
    # 训练参数
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
        bf16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("🚀 开始LoRA微调...")
    trainer.train()
    
    # 保存模型
    print(f"💾 保存LoRA权重到: {config.output_dir}")
    model.save_pretrained(config.output_dir)
    processor.save_pretrained(config.output_dir)
    
    print("✅ 训练完成!")


if __name__ == "__main__":
    main()
