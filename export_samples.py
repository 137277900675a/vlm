import json
import os
import random
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 读取VQA数据
with open('data/vqa/train.jsonl', 'r', encoding='utf-8') as f:
    all_samples = [json.loads(line) for line in f]

# 按类型分组
categories = {}
for s in all_samples:
    qtype = s.get('meta', {}).get('type', 'unknown')
    if qtype not in categories:
        categories[qtype] = []
    categories[qtype].append(s)

# 每个类别选样
selected = []
for cat, samples in categories.items():
    num = min(8, len(samples))
    selected.extend(random.sample(samples, num))

selected = selected[:40]

# 转换路径
for s in selected:
    s['image_path'] = s['image_path'].replace('D:\\vlm\\', '')

# 保存
os.makedirs('outputs', exist_ok=True)
with open('outputs/vqa_representative_samples.json', 'w', encoding='utf-8') as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)

print('OK: exported', len(selected), 'samples')
