# 工地安全 VLM 系统

本项目基于开源视觉语言模型（VLM）与 LoRA 微调，实现对工地安全隐患的视觉问答（VQA）识别与结构化输出。

## 目录结构

```
D:\vlm\
├── data/
│   ├── raw/helmet/kaggle/images/    # 501张原始图片
│   ├── processed/                     # VLM自动标注结果
│   └── vqa/train.jsonl               # 300条VQA训练样本
├── src/
│   ├── config/paths.py               # 路径配置
│   ├── data/                         # 数据处理
│   ├── app/streamlit_app.py          # 前端界面
│   ├── training/finetune_lora.py     # LoRA微调脚本
│   ├── inference/demo.py             # 推理Demo
│   └── evaluation/evaluate.py        # 评估脚本
├── models/lora_safety/               # LoRA权重输出目录
├── outputs/
│   └── vqa_representative_samples.json  # 24条代表性样本
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 启动前端（工地安全多模态助手）

```powershell
cd D:\vlm
streamlit run src/app/streamlit_app.py --server.port 8502
```

访问 http://localhost:8502

### 2. LoRA 微调

```powershell
cd D:\vlm
python -m src.training.finetune_lora ^
  --model_name Qwen/Qwen2-VL-2B-Instruct ^
  --data_path data/vqa/train.jsonl ^
  --output_dir models/lora_safety ^
  --num_epochs 3 ^
  --batch_size 2 ^
  --learning_rate 2e-4
```

### 3. 推理Demo

```powershell
python -m src.inference.demo ^
  --image path\to\site.jpg ^
  --question "这张图中有哪些安全隐患？"
```

### 4. 评估

```powershell
python -m src.evaluation.evaluate ^
  --test_data data/vqa/train.jsonl ^
  --lora_path models/lora_safety
```


| 样本输出 | ✅ | 24条代表性VQA样本(JSON) |

## 核心代码

- `src/training/finetune_lora.py` - LoRA微调脚本
- `src/inference/demo.py` - 推理Demo
- `src/evaluation/evaluate.py` - 评估脚本
- `src/app/streamlit_app.py` - Streamlit前端

## 技术参数

- **基础模型**: Qwen2-VL-2B-Instruct
- **LoRA配置**: Rank=16, Alpha=32, Dropout=0.05
- **训练参数**: Epochs=3, Batch=2, LR=2e-4, BF16
- **数据集**: 501张图片 → 300条VQA样本
