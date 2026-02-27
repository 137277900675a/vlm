"""
使用视觉大模型（优先千问 Qwen-VL）对工地安全相关图片进行自动标注。

设计目标：
- 提供一个统一的自动标注入口，后续可以替换/扩展 Provider；
- 当前实现基于阿里灵积 DashScope 的 Qwen-VL 系列模型；
- 对每张图片输出结构化 JSON，包含安全帽/人员/风险等要素，写入 JSONL。

示例用法（PowerShell）：

    cd D:\vlm
    $env:DASHSCOPE_API_KEY="your-api-key"
    python -m src.data.auto_annotate_vlm `
        --image_dir data\raw\helmet\shwd\JPEGImages `
        --output_path data\processed\helmet_qwen_annotations.jsonl `
        --model qwen-vl-max `
        --max_images 200
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dashscope import MultiModalConversation
import dashscope

from src.config.paths import PATHS


SUPPORTED_QWEN_VL_MODELS = [
    "qwen-vl-max",
    "qwen-vl-plus",
]


@dataclass
class VLMAnnotation:
    image_path: str
    provider: str
    model: str
    annotation: Dict[str, Any]
    raw_text: str


def list_images(image_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths: List[str] = []
    base = Path(image_dir)
    if not base.exists():
        return []
    for p in base.rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(str(p))
    return paths


def ensure_dashscope_api_key() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASH_SCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "未检测到 DASHSCOPE_API_KEY 环境变量，请先在系统/PowerShell 中设置你的 DashScope API Key。"
        )
    dashscope.api_key = api_key


def build_qwen_system_prompt() -> str:
    return (
        "你是一个专门负责工地安全隐患识别的视觉助手，擅长判断人员是否佩戴安全帽、是否穿反光衣，"
        "以及现场是否存在临边防护不足、杂物堆放、高处坠落风险等问题。"
        "请你根据输入图片，给出结构化的 JSON 标注结果，字段包括：\n"
        "- has_helmet_violation: bool，是否存在未按规定佩戴安全帽的人员；\n"
        "- num_persons: int，画面中大致人员数量（估计即可）；\n"
        "- summary: string，对主要安全情况的简要中文描述；\n"
        "- detailed_risks: list[string]，列出你观察到的具体安全隐患点；\n"
        "严格输出一个 JSON 对象，不要输出多余的解释或 Markdown。"
    )


def call_qwen_vl_on_image(
    image_path: str,
    model: str = "qwen-vl-max",
) -> VLMAnnotation:
    if model not in SUPPORTED_QWEN_VL_MODELS:
        raise ValueError(
            f"不支持的 Qwen-VL 模型：{model}，可选：{', '.join(SUPPORTED_QWEN_VL_MODELS)}"
        )

    ensure_dashscope_api_key()

    system_prompt = build_qwen_system_prompt()

    messages = [
        {
            "role": "system",
            "content": [{"text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"image": f"file://{os.path.abspath(image_path)}"},
                {
                    "text": "请根据前述要求返回 JSON 标注，不要输出除 JSON 以外的任何内容。"
                },
            ],
        },
    ]

    resp = MultiModalConversation.call(model=model, messages=messages)

    # DashScope 返回结构中，回答文本通常在 output.choices[0].message.content[0].text
    content_items = (
        getattr(resp, "output", None)
        and getattr(resp.output, "choices", None)
        and resp.output.choices[0].message.content
    )

    if not content_items:
        raise RuntimeError(f"Qwen-VL 返回空内容：{resp}")

    texts = [c.get("text", "") for c in content_items if isinstance(c, dict)]
    raw_text = "\n".join(t.strip() for t in texts if t.strip())

    try:
        annotation = json.loads(raw_text)
        if not isinstance(annotation, dict):
            raise ValueError("解析结果不是 JSON 对象")
    except Exception:
        # 如果解析失败，则把整个文本包在 fallback 字段里，避免数据丢失
        annotation = {"parse_failed": True, "raw_text": raw_text}

    return VLMAnnotation(
        image_path=os.path.abspath(image_path),
        provider="dashscope-qwen-vl",
        model=model,
        annotation=annotation,
        raw_text=raw_text,
    )


def auto_annotate_directory(
    image_dir: str,
    output_path: str,
    model: str = "qwen-vl-max",
    max_images: Optional[int] = None,
) -> None:
    """
    对指定目录下的图片逐张调用 Qwen-VL，生成 JSONL 标注文件。
    """
    imgs = list_images(image_dir)
    if not imgs:
        raise FileNotFoundError(f"在 {image_dir} 下未找到任何图片。")

    if max_images is not None:
        imgs = imgs[: max(0, max_images)]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(
        f"开始使用 Qwen-VL 对 {len(imgs)} 张图片进行自动标注，输出到 {output_path}，模型：{model}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, img_path in enumerate(imgs, start=1):
            try:
                ann = call_qwen_vl_on_image(img_path, model=model)
                f.write(json.dumps(asdict(ann), ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[警告] 处理图片失败：{img_path}，错误：{e}")
            if idx % 10 == 0:
                print(f"已处理 {idx}/{len(imgs)} 张图片...")

    print("自动标注完成。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Qwen-VL 对工地图像目录进行自动标注（输出 JSONL）"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=os.path.join(PATHS.raw_data_dir, "helmet", "shwd", "JPEGImages"),
        help="需要标注的图片目录",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(
            PATHS.processed_data_dir, "helmet_qwen_annotations.jsonl"
        ),
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-vl-max",
        help=f"Qwen-VL 模型名称，可选：{', '.join(SUPPORTED_QWEN_VL_MODELS)}",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=200,
        help="最多处理多少张图片（按文件遍历顺序截断），默认 200",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auto_annotate_directory(
        image_dir=args.image_dir,
        output_path=args.output_path,
        model=args.model,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()

