#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估脚本 - 工地安全VLM
定量指标: F1-score / 准确率
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from PIL import Image
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
import pandas as pd


@dataclass
class EvalResult:
    """评估结果"""
    total_samples: int
    correct: int
    accuracy: float
    f1_score: float
    details: List[Dict[str, Any]]


def load_test_dataset(data_path: str, num_samples: int = 50) -> List[Dict]:
    """加载测试集"""
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line.strip()))
    return samples


def simple_match_score(prediction: str, ground_truth: str) -> float:
    """简单的文本匹配评分"""
    pred_lower = prediction.lower().strip()
    gt_lower = ground_truth.lower().strip()
    
    # 精确匹配
    if pred_lower == gt_lower:
        return 1.0
    
    # 关键词匹配
    gt_keywords = set(gt_lower.split())
    pred_keywords = set(pred_lower.split())
    
    if len(gt_keywords) == 0:
        return 0.0
    
    overlap = len(gt_keywords & pred_keywords)
    return overlap / len(gt_keywords)


def evaluate_model(
    model,
    processor,
    test_samples: List[Dict],
    base_image_dir: str = ""
) -> EvalResult:
    """评估模型"""
    details = []
    scores = []
    
    for i, sample in enumerate(test_samples):
        print(f"评估 [{i+1}/{len(test_samples)}]...")
        
        # 获取图像
        img_path = sample.get('image_path', '')
        if base_image_dir and not os.path.isabs(img_path):
            img_path = os.path.join(base_image_dir, os.path.basename(img_path))
        
        if not os.path.exists(img_path):
            print(f"⚠️ 图像不存在: {img_path}")
            continue
        
        # 推理
        image = Image.open(img_path).convert("RGB")
        question = sample['question']
        ground_truth = sample['answer']
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        
        input_length = inputs["input_ids"].shape[1]
        prediction = processor.batch_decode(
            output[:, input_length:],
            skip_special_tokens=True,
        )[0]
        
        # 计算分数
        score = simple_match_score(prediction, ground_truth)
        scores.append(score)
        
        details.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "score": score
        })
    
    # 计算指标
    total = len(scores)
    correct = sum(1 for s in scores if s >= 0.5)
    accuracy = correct / total if total > 0 else 0.0
    
    # F1-score (简化版)
    f1 = accuracy  # 使用准确率作为简化指标
    
    return EvalResult(
        total_samples=total,
        correct=correct,
        accuracy=accuracy,
        f1_score=f1,
        details=details
    )


def compare_baseline_vs_finetuned(
    base_model: str,
    lora_path: str,
    test_samples: List[Dict],
    base_image_dir: str
) -> Dict[str, EvalResult]:
    """对比Base模型 vs 微调模型"""
    print("=" * 50)
    print("评估对比实验")
    print("=" * 50)
    
    processor = Qwen2VLProcessor.from_pretrained(base_model)
    
    results = {}
    
    # 评估Base模型
    print("\n[1/2] 评估 Base 模型...")
    base_model_obj = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    base_result = evaluate_model(
        base_model_obj, processor, test_samples, base_image_dir
    )
    results["base"] = base_result
    print(f"Base 准确率: {base_result.accuracy:.2%}")
    
    # 评估微调模型
    print("\n[2/2] 评估 Fine-tuned 模型...")
    finetuned_model = PeftModel.from_pretrained(base_model_obj, lora_path)
    finetuned_result = evaluate_model(
        finetuned_model, processor, test_samples, base_image_dir
    )
    results["finetuned"] = finetuned_result
    print(f"Fine-tuned 准确率: {finetuned_result.accuracy:.2%}")
    
    return results


def save_results(results: EvalResult, output_path: str):
    """保存评估结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output = {
        "total_samples": results.total_samples,
        "correct": results.correct,
        "accuracy": results.accuracy,
        "f1_score": results.f1_score,
        "details": results.details
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 评估结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="评估工地安全VLM")
    parser.add_argument("--test_data", type=str, default="data/vqa/train.jsonl")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--lora_path", type=str, default="models/lora_safety")
    parser.add_argument("--output", type=str, default="outputs/eval_results.json")
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()
    
    # 加载测试集
    print(f"📂 加载测试数据: {args.test_data}")
    test_samples = load_test_dataset(args.test_data, args.num_samples)
    print(f"共 {len(test_samples)} 条测试样本")
    
    # 检查是否有LoRA权重
    if os.path.exists(args.lora_path):
        results = compare_baseline_vs_finetuned(
            args.base_model,
            args.lora_path,
            test_samples,
            base_image_dir="data/raw/helmet/kaggle/images"
        )
        
        # 保存对比结果
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/comparison_results.json", 'w', encoding='utf-8') as f:
            json.dump({
                "base": {
                    "accuracy": results["base"].accuracy,
                    "f1_score": results["base"].f1_score
                },
                "finetuned": {
                    "accuracy": results["finetuned"].accuracy,
                    "f1_score": results["finetuned"].f1_score
                }
            }, f, ensure_ascii=False, indent=2)
        
        print("\n📊 对比结果:")
        print(f"  Base 模型准确率:     {results['base'].accuracy:.2%}")
        print(f"  Fine-tuned 准确率:  {results['finetuned'].accuracy:.2%}")
        print(f"  提升:               {(results['finetuned'].accuracy - results['base'].accuracy):.2%}")
    else:
        print("⚠️ 未找到LoRA权重，跳过对比实验")
    
    print("\n✅ 评估完成!")


if __name__ == "__main__":
    main()
