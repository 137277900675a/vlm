"""
下载工地安全帽相关开源数据集的脚本（默认接入 SHWD）。

目标：
- 把公开数据集下载并解压到 data/raw/helmet/shwd 目录下；
- 仅关心原始图片（JPEGImages）与简单的 train/val/test 划分文件；
- 为后续自动标注与 VQA 构建提供统一的本地图片目录入口。

说明：
- 为避免第三方站点变更导致脚本失效，这里使用一个可配置的 URL；
- 如果默认链接失效，你可以在命令行中通过 --url 参数覆盖。
"""

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import requests

from src.config.paths import PATHS


DEFAULT_SHWD_ZIP_URL = (
    # 该链接可能会变动，如失效请在命令行传入新的下载地址。
    "https://github.com/sticktn/detect_helmet/releases/download/v1.0/SHWD.zip"
)


def download_file(url: str, dst_path: Path, chunk_size: int = 8192) -> None:
    """简单的 HTTP 下载封装。"""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        # 如果远端返回 404/403 等，直接抛异常，由上层统一处理
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def extract_zip(zip_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)


def prepare_shwd_dataset(
    base_raw_dir: Optional[Path] = None,
    url: str = DEFAULT_SHWD_ZIP_URL,
    keep_archives: bool = False,
) -> Path:
    """
    下载并解压 SHWD 数据集到 data/raw/helmet/shwd。

    返回值：包含图片的根目录（通常为 data/raw/helmet/shwd/JPEGImages）。
    """
    if base_raw_dir is None:
        base_raw_dir = Path(PATHS.raw_data_dir) / "helmet" / "shwd"

    base_raw_dir.mkdir(parents=True, exist_ok=True)

    # 如果已经存在 JPEGImages 目录，则认为已准备好，直接返回。
    jpeg_dir = base_raw_dir / "JPEGImages"
    if jpeg_dir.exists() and any(jpeg_dir.glob("*.jpg")):
        print(f"检测到已存在的 SHWD 数据集，跳过下载：{jpeg_dir}")
        return jpeg_dir

    zip_path = base_raw_dir / "shwd.zip"

    # 如果本地已经有 shwd.zip，则直接解压使用，避免重复下载
    if zip_path.exists():
        print(f"检测到本地已存在压缩包，跳过下载，直接解压：{zip_path}")
        print(f"解压到 {base_raw_dir}")
        extract_zip(zip_path, base_raw_dir)
        # 解压后再尝试找到 JPEGImages 目录
        if jpeg_dir.exists() and any(jpeg_dir.glob("*.jpg")):
            return jpeg_dir

    # 下载到临时文件再移动，避免中途失败产生半成品。
    print(f"开始下载 SHWD 数据集：{url}")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_zip = Path(tmpdir) / "shwd.zip"
            download_file(url, tmp_zip)
            shutil.move(str(tmp_zip), zip_path)
    except requests.HTTPError as e:
        # 友好提示，方便用户手动下载并放置到 zip_path
        print(
            f"[错误] 无法从默认地址下载 SHWD（HTTP {e.response.status_code}）。\n"
            f"你可以：\n"
            f"1) 手动从公开站点下载 SHWD.zip，并重命名为 shwd.zip 放到：\n"
            f"   {zip_path}\n"
            f"2) 然后重新运行本脚本；\n"
            f"或者：\n"
            f"3) 在命令行中通过 --url 参数传入一个可用的下载链接。\n"
        )
        raise

    print(f"解压到 {base_raw_dir}")
    extract_zip(zip_path, base_raw_dir)

    if not keep_archives and zip_path.exists():
        zip_path.unlink()

    # 尝试推断 JPEGImages 目录位置
    if jpeg_dir.exists():
        return jpeg_dir

    # 有些压缩包会多一层目录，做一次自动探测
    candidates = list(base_raw_dir.rglob("JPEGImages"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"已解压 SHWD，但未找到 JPEGImages 目录，请手动检查 {base_raw_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并准备 SHWD 安全帽数据集")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_SHWD_ZIP_URL,
        help="SHWD 压缩包下载地址，如默认失效可自行指定",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PATHS.raw_data_dir, "helmet", "shwd"),
        help="数据集解压根目录（默认 data/raw/helmet/shwd）",
    )
    parser.add_argument(
        "--keep_archives",
        action="store_true",
        help="是否保留下载的压缩包文件（默认删除以节省空间）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jpeg_dir = prepare_shwd_dataset(
        base_raw_dir=Path(args.output_dir),
        url=args.url,
        keep_archives=args.keep_archives,
    )
    print(f"SHWD 数据集已准备完成，图像目录：{jpeg_dir}")


if __name__ == "__main__":
    main()

