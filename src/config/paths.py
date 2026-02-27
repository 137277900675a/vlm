import os
from dataclasses import dataclass


@dataclass
class PathConfig:
    """
    所有与路径相关的默认配置，基于项目根目录 D:\\vlm。
    方便在脚本中统一引用。
    """

    project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # 原始与中间数据
    raw_data_dir: str = os.path.join(project_root, "data", "raw")
    processed_data_dir: str = os.path.join(project_root, "data", "processed")
    vqa_data_dir: str = os.path.join(project_root, "data", "vqa")

    # 模型与输出
    models_dir: str = os.path.join(project_root, "models")
    lora_output_dir: str = os.path.join(models_dir, "lora")
    logs_dir: str = os.path.join(project_root, "logs")
    outputs_dir: str = os.path.join(project_root, "outputs")


PATHS = PathConfig()

