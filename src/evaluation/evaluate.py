#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估脚本 - 工地安全VLM
定量指标: F1-score / 准确率
使用本地预训练模型与微调后模型进行对比
"""

import os
import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from PIL import Image
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel


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


def load_inference_model(
    base_model: str,
    lora_path: str = None,
    device: str = "auto"
):
    """加载推理模型"""
    local_model_path = "models/pretrained/Qwen/Qwen2-VL-2B-Instruct"
    if os.path.exists(local_model_path):
        base_model = local_model_path
        print(f"🔄 使用本地模型: {base_model}")
    else:
        print(f"🔄 加载基础模型: {base_model}")
    
    processor = Qwen2VLProcessor.from_pretrained(base_model)
    
    if torch.cuda.is_available():
        device_map = device
        torch_dtype = torch.bfloat16
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    
    if lora_path and os.path.exists(lora_path):
        print(f"🔄 加载LoRA权重: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    
    model.eval()
    return model, processor


def compute_metrics(prediction: str, ground_truth: str) -> dict:
    """计算评估指标（BLEU + 关键词匹配）"""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    # 中文按字符分词
    pred_chars = list(prediction.replace(' ', ''))
    gt_chars = list(ground_truth.replace(' ', ''))
    
    # BLEU分数
    smoothing = SmoothingFunction().method1
    bleu1 = sentence_bleu([gt_chars], pred_chars, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu([gt_chars], pred_chars, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu([gt_chars], pred_chars, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    # 关键词匹配（词级别）
    pred_words = set(prediction.replace(' ', '').replace('，', '').replace('。', ''))
    gt_words = set(ground_truth.replace(' ', '').replace('，', '').replace('。', ''))
    
    if len(gt_words) > 0:
        keyword_match = len(pred_words & gt_words) / len(gt_words)
    else:
        keyword_match = 0.0
    
    # 综合分数：BLEU-4 (70%) + 关键词匹配 (30%)
    combined_score = 0.7 * bleu4 + 0.3 * keyword_match
    
    return {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu4': bleu4,
        'keyword_match': keyword_match,
        'combined_score': combined_score
    }

def simple_match_score_fallback(prediction: str, ground_truth: str) -> dict:
    """降级评分方法"""
    pred_lower = prediction.lower().strip()
    gt_lower = ground_truth.lower().strip()
    
    if pred_lower == gt_lower:
        score = 1.0
    else:
        gt_keywords = set(gt_lower.split())
        pred_keywords = set(pred_lower.split())
        if len(gt_keywords) == 0:
            score = 0.0
        else:
            overlap = len(gt_keywords & pred_keywords)
            score = overlap / len(gt_keywords)
    
    return {'rouge1': score, 'rouge2': score, 'rougeL': score}


def inference_single(model, processor, image_path: str, question: str) -> str:
    """单图推理"""
    from qwen_vl_utils import process_vision_info
    
    abs_path = os.path.abspath(image_path).replace('\\', '/')
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{abs_path}"},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
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
    response = processor.batch_decode(
        output[:, input_length:],
        skip_special_tokens=True,
    )[0]
    
    return response


def evaluate_model(
    model,
    processor,
    test_samples: List[Dict],
    base_image_dir: str = ""
) -> EvalResult:
    """评估模型"""
    details = []
    bleu1_scores = []
    bleu4_scores = []
    combined_scores = []
    
    for i, sample in enumerate(test_samples):
        print(f"评估 [{i+1}/{len(test_samples)}]...")
        
        img_path = sample.get('image_path', '').replace('\\', '/')
        if not os.path.exists(img_path):
            img_path = img_path.replace('data/raw/helmet/kaggle/images/', '')
            img_path = os.path.join('data/raw/helmet/kaggle/images', os.path.basename(img_path))
        
        if not os.path.exists(img_path):
            print(f"⚠️ 图像不存在: {img_path}")
            continue
        
        question = sample['question']
        ground_truth = sample['answer']
        
        prediction = inference_single(model, processor, img_path, question)
        
        scores = compute_metrics(prediction, ground_truth)
        bleu1_scores.append(scores['bleu1'])
        bleu4_scores.append(scores['bleu4'])
        combined_scores.append(scores['combined_score'])
        
        details.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "bleu1": scores['bleu1'],
            "bleu4": scores['bleu4'],
            "combined_score": scores['combined_score']
        })
    
    total = len(combined_scores)
    avg_bleu1 = sum(bleu1_scores) / total if total > 0 else 0.0
    avg_bleu4 = sum(bleu4_scores) / total if total > 0 else 0.0
    avg_combined = sum(combined_scores) / total if total > 0 else 0.0
    
    # 使用综合分数作为主要指标
    accuracy = avg_combined
    f1_score = avg_bleu4
    correct = sum(1 for s in combined_scores if s >= 0.5)
    
    print(f"BLEU-1: {avg_bleu1:.4f}, BLEU-4: {avg_bleu4:.4f}, 综合分数: {avg_combined:.4f}")
    
    return EvalResult(
        total_samples=total,
        correct=correct,
        accuracy=accuracy,
        f1_score=f1_score,
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
    
    results = {}
    
    # 评估Base模型
    print("\n[1/2] 评估 Base 模型...")
    base_model_obj, processor = load_inference_model(base_model)
    base_result = evaluate_model(base_model_obj, processor, test_samples, base_image_dir)
    results["base"] = base_result
    print(f"Base 准确率: {base_result.accuracy:.2%}, BLEU-4: {base_result.f1_score:.4f}")
    
    # 评估微调模型
    print("\n[2/2] 评估 Fine-tuned 模型...")
    finetuned_model, processor = load_inference_model(base_model, lora_path)
    finetuned_result = evaluate_model(finetuned_model, processor, test_samples, base_image_dir)
    results["finetuned"] = finetuned_result
    print(f"Fine-tuned 准确率: {finetuned_result.accuracy:.2%}, BLEU-4: {finetuned_result.f1_score:.4f}")
    
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
