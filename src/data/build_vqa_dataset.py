"""
构建工地安全 VQA 数据集的脚本。

功能：
- 从本地文件夹读取工地图像，或从公开数据源下载示例数据；
- 为每张图像合成若干 VQA 对（question, answer）；
- 支持固定模板 + 随机问句的方式生成 300-500 条样本；
- 输出 JSONL：每行一个样本，包含 image_path, question, answer, meta。

用法（PowerShell）：

    cd D:\vlm
    python -m src.data.build_vqa_dataset `
        --image_dir data\raw\site_images `
        --output_path data\vqa\train.jsonl `
        --num_samples 400
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image

from src.config.paths import PATHS


SAFETY_CONCEPTS = [
    "安全帽",
    "反光背心",
    "护目镜",
    "安全绳",
    "防护栏杆",
    "安全网",
]

HAZARD_TEMPLATES = [
    "这张图片中有哪些安全隐患？",
    "图中是否有人未按规定佩戴防护用品？",
    "画面里是否存在高处坠落风险？",
    "现场有没有明显违规操作？",
    "请描述施工现场潜在的危险点。",
]

CHECK_TEMPLATES = [
    "图片里所有人都戴了安全帽吗？",
    "现场是否设置了防护栏和警示标志？",
    "施工人员穿着是否符合安全要求？",
    "是否存在杂物堆放影响通行？",
]


@dataclass
class VQASample:
    image_path: str
    question: str
    answer: str
    meta: Dict[str, Any]


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


def probe_image(image_path: str) -> Dict[str, Any]:
    """读取图片尺寸，作为 meta 信息（失败则返回空 dict）。"""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
        return {"width": w, "height": h}
    except Exception:
        return {}


def generate_qas_for_image(image_path: str, max_per_image: int = 3) -> List[VQASample]:
    """
    简单的启发式：仅根据模板构造问句与“较安全/较危险”的文字答案。
    真实项目中可用免费大模型辅助生成更细致的答案，这部分留到 auto-annotate 步骤。
    """
    meta = {"source_image": image_path}
    meta.update(probe_image(image_path))

    samples: List[VQASample] = []

    # hazard-focused QA
    q1 = random.choice(HAZARD_TEMPLATES)
    a1 = random.choice(
        [
            "图片中存在安全隐患，例如部分人员未佩戴安全帽、临边缺少防护栏等。",
            "现场比较混乱，存在材料堆放无序、通道被占用等安全问题。",
            "画面中可以看到高处作业区防护不足，存在坠落风险。",
        ]
    )
    samples.append(
        VQASample(
            image_path=image_path,
            question=q1,
            answer=a1,
            meta={**meta, "type": "hazard_overview"},
        )
    )

    # PPE /规则检查类 QA
    q2 = random.choice(CHECK_TEMPLATES)
    a2 = random.choice(
        [
            "不是所有人都按规定佩戴了安全防护用品，需要加强管理。",
            "现场防护和标识不完善，存在安全管理不到位的问题。",
            "部分区域存在杂物堆放和临边防护不足，需要整改。",
        ]
    )
    samples.append(
        VQASample(
            image_path=image_path,
            question=q2,
            answer=a2,
            meta={**meta, "type": "rule_check"},
        )
    )

    # 额外一条可选 QA
    if max_per_image >= 3:
        q3 = random.choice(
            [
                "这张图中与个人防护装备相关的问题有哪些？",
                "图中有哪些需要立即整改的安全问题？",
            ]
        )
        a3 = random.choice(
            [
                "个人防护装备佩戴不规范，同时现场物料摆放杂乱，需要立即整改。",
                "存在未佩戴安全帽、靠近临边区域作业等，需要立即采取防护措施。",
            ]
        )
        samples.append(
            VQASample(
                image_path=image_path,
                question=q3,
                answer=a3,
                meta={**meta, "type": "ppe_detail"},
            )
        )

    return samples


def build_vqa_dataset(
    image_dir: str, output_path: str, num_samples: int = 400, max_per_image: int = 3
) -> None:
    image_dir = image_dir or PATHS.raw_data_dir
    images = list_images(image_dir)

    if not images:
        raise FileNotFoundError(
            f"未在 {image_dir} 下找到任何图片，请先准备一些工地图片，或从开源数据集中拷贝。"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_samples: List[VQASample] = []
    random.shuffle(images)

    for img_path in images:
        if len(all_samples) >= num_samples:
            break
        qas = generate_qas_for_image(img_path, max_per_image=max_per_image)
        all_samples.extend(qas)

    # 截断到指定条数
    all_samples = all_samples[:num_samples]

    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    print(f"已生成 {len(all_samples)} 条 VQA 样本到 {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建工地安全 VQA 数据集 JSONL")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=os.path.join(PATHS.raw_data_dir, "site_images"),
        help="工地图像所在文件夹（递归扫描）",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(PATHS.vqa_data_dir, "train.jsonl"),
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=400,
        help="期望生成的样本数（会截断）",
    )
    parser.add_argument(
        "--max_per_image",
        type=int,
        default=3,
        help="每张图片最多生成多少条 QA",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_vqa_dataset(
        image_dir=args.image_dir,
        output_path=args.output_path,
        num_samples=args.num_samples,
        max_per_image=args.max_per_image,
    )


if __name__ == "__main__":
    main()

