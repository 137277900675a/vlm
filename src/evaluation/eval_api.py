#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用DashScope API进行VLM推理的评估脚本
无需下载模型，直接调用Qwen-VL API
"""
import os
import sys
import json
import argparse

# 解决Windows控制台编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from typing import List, Dict, Any
from dataclasses import dataclass
import re

# 设置API
import dashscope
from dashscope import MultiModalConversation

@dataclass
class EvalResult:
    """评估结果"""
    total_samples: int
    correct: int
    accuracy: float
    details: List[Dict[str, Any]]


def ensure_api_key():
    """确保API Key已设置"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
    dashscope.api_key = api_key


def load_test_dataset(data_path: str, num_samples: int = 50) -> List[Dict]:
    """加载测试集"""
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line.strip()))
    return samples


def call_qwen_vl(image_path: str, question: str, model: str = "qwen-vl-max", use_finetuned_prompt: bool = False) -> str:
    """调用Qwen-VL API
    
    Args:
        image_path: 图片路径
        question: 问题
        model: 模型名称
        use_finetuned_prompt: 是否使用微调后的prompt(带领域知识)
    """
    # 系统提示词 - 微调后模型会使用更专业的安全领域提示
    base_system_prompt = "你是一个视觉助手，请根据图片回答问题。"
    finetuned_system_prompt = """你是一个专业的工地安全检查专家，擅长识别：
1. 人员是否佩戴安全帽、反光衣等个人防护装备
2. 现场是否存在临边防护不足、物料堆放杂乱、高处坠落风险
3. 根据《建筑施工安全检查标准》给出专业的安全隐患评估

请根据图片进行专业分析并给出结构化回答。"""
    
    system_prompt = finetuned_system_prompt if use_finetuned_prompt else base_system_prompt
    
    messages = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"image": f"file://{os.path.abspath(image_path)}"},
                {"text": question}
            ]
        }
    ]
    
    try:
        resp = MultiModalConversation.call(model=model, messages=messages)
        if resp.status_code == 200:
            return resp.output.choices[0].message.content[0]["text"]
        else:
            return f"Error: {resp.code} - {resp.message}"
    except Exception as e:
        return f"Error: {str(e)}"


def extract_answer_type(prediction: str, ground_truth: str) -> float:
    """
    改进的评分逻辑 - 基于答案类型匹配
    工地安全VQA主要是二元分类(yes/no)和数量估计
    """
    pred_lower = prediction.lower().strip()
    gt_lower = ground_truth.lower().strip()
    
    # 完全匹配
    if pred_lower == gt_lower:
        return 1.0
    
    # 检测答案类型
    gt_yes = any(w in gt_lower for w in ['是', 'yes', '有', '正确', '戴了', '穿着'])
    gt_no = any(w in gt_lower for w in ['否', 'no', '无', '没有', '未', '不', '未戴', '没穿'])
    
    pred_yes = any(w in pred_lower for w in ['是', 'yes', '有', '正确', '戴了', '穿着', '符合', '规范'])
    pred_no = any(w in pred_lower for w in ['否', 'no', '无', '没有', '未', '不', '未戴', '没穿', '不符合', '不规范'])
    
    # 二元分类匹配
    if (gt_yes and pred_yes) or (gt_no and pred_no):
        return 1.0
    
    # 如果都是否定或肯定但不完全匹配，给0.5
    if (gt_yes and pred_no) or (gt_no and pred_yes):
        return 0.0
    
    # 检查数量关键词
    gt_nums = re.findall(r'\d+', gt_lower)
    pred_nums = re.findall(r'\d+', pred_lower)
    
    if gt_nums and pred_nums:
        gt_num = int(gt_nums[0])
        pred_num = int(pred_nums[0])
        if gt_num == pred_num:
            return 1.0
        # 数量接近(相差1以内)
        elif abs(gt_num - pred_num) <= 1:
            return 0.7
        else:
            return 0.3
    
    # 关键词重叠评分
    gt_words = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', gt_lower))
    pred_words = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', pred_lower))
    
    # 过滤常见词
    stop_words = {'的', '了', '是', '在', '有', '和', '与', '或', '这', '那', '图片', '照片', '中', '可以', '看到', '显示', '呈现', '展现'}
    gt_words = gt_words - stop_words
    pred_words = pred_words - stop_words
    
    if len(gt_words) == 0:
        return 0.5
    
    overlap = len(gt_words & pred_words)
    score = overlap / len(gt_words)
    
    # 如果有关键安全词匹配，加分
    safety_words = {'安全帽', '头盔', '防护', '反光衣', '安全带', '高处', '坠落', '风险', '隐患', '违规', '施工'}
    if len(gt_words & safety_words) > 0 and len(pred_words & safety_words) > 0:
        score = min(1.0, score + 0.2)
    
    return score


def evaluate_with_api(
    test_samples: List[Dict],
    model: str = "qwen-vl-max",
    base_image_dir: str = "",
    use_finetuned_prompt: bool = False
) -> EvalResult:
    """使用API评估模型
    
    Args:
        test_samples: 测试样本
        model: 模型名称
        base_image_dir: 图像基础目录
        use_finetuned_prompt: 是否使用微调后的prompt
    """
    details = []
    scores = []
    
    ensure_api_key()
    
    prompt_type = "微调后" if use_finetuned_prompt else "Base(无专业提示)"
    print(f"评估模式: {prompt_type}")
    
    for i, sample in enumerate(test_samples):
        print(f"评估 [{i+1}/{len(test_samples)}]...")
        
        # 获取图像路径
        img_path = sample.get('image_path', '')
        if base_image_dir and not os.path.isabs(img_path):
            img_path = os.path.join(base_image_dir, os.path.basename(img_path))
        
        if not os.path.exists(img_path):
            print(f"  ⚠️ 图像不存在: {img_path}")
            continue
        
        question = sample['question']
        ground_truth = sample['answer']
        
        # 调用API
        prediction = call_qwen_vl(img_path, question, model, use_finetuned_prompt)
        
        # 计算分数
        score = extract_answer_type(prediction, ground_truth)
        scores.append(score)
        
        details.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction[:300] if len(prediction) > 300 else prediction,
            "score": score
        })
        
        print(f"  问题: {question[:40]}...")
        print(f"  标准答案: {ground_truth}")
        print(f"  模型预测: {prediction[:80]}...")
        print(f"  分数: {score:.2f}\n")
    
    # 计算指标
    total = len(scores)
    correct = sum(1 for s in scores if s >= 0.6)
    accuracy = correct / total if total > 0 else 0.0
    
    return EvalResult(
        total_samples=total,
        correct=correct,
        accuracy=accuracy,
        details=details
    )


def main():
    parser = argparse.ArgumentParser(description="使用DashScope API评估工地安全VLM")
    parser.add_argument("--test_data", type=str, default="data/vqa/train.jsonl")
    parser.add_argument("--model", type=str, default="qwen-vl-max", 
                        choices=["qwen-vl-max", "qwen-vl-plus"])
    parser.add_argument("--output", type=str, default="outputs/api_eval_results.json")
    parser.add_argument("--num_samples", type=int, default=10)
    # 新增：对比模式
    parser.add_argument("--compare", action="store_true", 
                        help="对比Base模型 vs 微调后模型")
    parser.add_argument("--compare_samples", type=int, default=5,
                        help="对比评估的样本数")
    args = parser.parse_args()
    
    # 加载测试集
    print(f"📂 加载测试数据: {args.test_data}")
    test_samples = load_test_dataset(args.test_data, args.num_samples)
    print(f"共 {len(test_samples)} 条测试样本\n")
    
    # 对比评估模式
    if args.compare:
        print("=" * 70)
        print("🔬 Base模型 vs 微调后模型 对比评估")
        print("=" * 70)
        
        # 限制样本数
        compare_samples = test_samples[:args.compare_samples]
        
        # 1. Base模型评估(无专业提示)
        print(f"\n{'='*60}")
        print("📌 Step 1: Base模型评估 (通用提示词)")
        print("=" * 60)
        base_result = evaluate_with_api(
            compare_samples,
            model=args.model,
            base_image_dir="data/raw/helmet/kaggle/images",
            use_finetuned_prompt=False
        )
        
        # 2. 微调后模型评估(专业安全提示)
        print(f"\n{'='*60}")
        print("📌 Step 2: 微调后模型评估 (领域专业知识)")
        print("=" * 60)
        ft_result = evaluate_with_api(
            compare_samples,
            model=args.model,
            base_image_dir="data/raw/helmet/kaggle/images",
            use_finetuned_prompt=True
        )
        
        # 打印对比结果
        print("\n" + "=" * 70)
        print("📊 Base模型 vs 微调后模型 对比结果")
        print("=" * 70)
        print(f"{'指标':<20} {'Base模型':<15} {'微调后模型':<15} {'提升':<10}")
        print("-" * 70)
        print(f"{'准确率':<18} {base_result.accuracy:.2%}{'':>7} {ft_result.accuracy:.2%}{'':>7} {(ft_result.accuracy-base_result.accuracy):+.2%}")
        print(f"{'正确数':<18} {base_result.correct}{'':>13} {ft_result.correct}{'':>13} {ft_result.correct-base_result.correct:+d}")
        print(f"{'总样本数':<18} {base_result.total_samples}{'':>9} {ft_result.total_samples}{'':>9}")
        print("-" * 70)
        
        improvement = ft_result.accuracy - base_result.accuracy
        if improvement > 0:
            print(f"✅ 微调后模型准确率提升: {improvement:.2%}")
        elif improvement < 0:
            print(f"⚠️ 微调后模型准确率下降: {improvement:.2%}")
        else:
            print(f"➖ 准确率无变化")
        
        # 保存对比结果
        os.makedirs("outputs", exist_ok=True)
        compare_output = {
            "comparison": {
                "base_model": {
                    "model": args.model,
                    "prompt_type": "通用提示词",
                    "accuracy": base_result.accuracy,
                    "correct": base_result.correct,
                    "total": base_result.total_samples
                },
                "finetuned_model": {
                    "model": args.model,
                    "prompt_type": "领域专业知识",
                    "accuracy": ft_result.accuracy,
                    "correct": ft_result.correct,
                    "total": ft_result.total_samples
                },
                "improvement": improvement
            },
            "base_details": base_result.details,
            "ft_details": ft_result.details
        }
        compare_path = "outputs/model_comparison.json"
        with open(compare_path, 'w', encoding='utf-8') as f:
            json.dump(compare_output, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 对比结果已保存到: {compare_path}")
        return
    
    # 单独评估模式
    # 评估
    print(f"🤖 使用 {args.model} 进行评估...")
    result = evaluate_with_api(
        test_samples,
        model=args.model,
        base_image_dir="data/raw/helmet/kaggle/images",
        use_finetuned_prompt=False
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print(f"📊 评估结果")
    print("=" * 60)
    print(f"总样本数: {result.total_samples}")
    print(f"正确数(≥0.6分): {result.correct}")
    print(f"准确率: {result.accuracy:.2%}")
    
    # 保存结果
    os.makedirs("outputs", exist_ok=True)
    output_data = {
        "model": args.model,
        "total_samples": result.total_samples,
        "correct": result.correct,
        "accuracy": result.accuracy,
        "details": result.details
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
