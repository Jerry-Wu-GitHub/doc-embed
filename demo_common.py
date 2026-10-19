"""demo 共用的图片收集、向量化与距离展示工具。

本模块只服务于手动演示，不属于包的公开 API。
"""

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # 作为包导入时使用相对导入
    from .constants import DISTANCE_METRIC
    from .embedding import embed_image
    from .metrics import cosine_distance
except ImportError:  # 直接把项目根目录加入 sys.path 后导入
    from constants import DISTANCE_METRIC
    from embedding import embed_image
    from metrics import cosine_distance


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Page:
    """一张已完成向量化的图片。"""

    number: str = field(metadata={"description": "文件名去掉扩展名后的编号。"})
    vector: np.ndarray = field(metadata={"description": "图片的 embedding 向量。"})
    elapsed_milliseconds: float = field(
        metadata={"description": "向量化耗时（毫秒）。"}
    )


def collect_image_files(paths: Sequence[str]) -> list[Path]:
    """收集待处理的图片文件。

    Args:
        paths: 图片文件或目录路径列表；目录会被递归扫描。

    Returns:
        按路径排序的图片文件列表。
    """
    image_files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            image_files.extend(
                file
                for file in sorted(path.rglob("*"))
                if file.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file():
            image_files.append(path)
        else:
            print(f"[跳过] 找不到路径: {path}", file=sys.stderr)
    return image_files


def vectorize_directory(directory: Path, hybrid_mode: bool) -> list[Page]:
    """向量化目录中的所有图片，并打印逐张耗时和平均耗时。

    Args:
        directory: 图片目录。
        hybrid_mode: 是否使用 hybrid embedding。

    Returns:
        按文件名排序的 :class:`Page` 列表。
    """
    image_files = sorted(
        file
        for file in directory.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        return []

    print(f"向量化 {directory.name}：")
    print("".join(file.stem.rjust(8) for file in image_files))
    pages: list[Page] = []
    total_milliseconds = 0.0
    for image_file in image_files:
        start_time = time.perf_counter()
        vector = embed_image(image_file.read_bytes(), hybrid_mode=hybrid_mode)
        elapsed_milliseconds = (time.perf_counter() - start_time) * 1000.0
        total_milliseconds += elapsed_milliseconds
        pages.append(Page(image_file.stem, vector, elapsed_milliseconds))
        print(f"{elapsed_milliseconds:8.1f}", end="")
    print(f"\n平均耗时：{total_milliseconds / len(pages):.1f} ms")
    return pages


def build_distance_matrix(
    model_pages: Sequence[Page],
    real_pages: Sequence[Page],
) -> np.ndarray:
    """计算 real 图片到 model 图片的余弦距离矩阵。

    Args:
        model_pages: 候选图片列表。
        real_pages: 查询图片列表。

    Returns:
        形状为 ``(len(real_pages), len(model_pages))`` 的距离矩阵。
    """
    model_matrix = np.stack([page.vector for page in model_pages])
    real_matrix = np.stack([page.vector for page in real_pages])
    return 1.0 - real_matrix @ model_matrix.T


def print_distance_matrix(
    model_pages: Sequence[Page],
    real_pages: Sequence[Page],
    distance_matrix: np.ndarray,
) -> None:
    """打印距离矩阵。

    Args:
        model_pages: 候选图片列表。
        real_pages: 查询图片列表。
        distance_matrix: 由 :func:`build_distance_matrix` 计算出的矩阵。
    """
    print(f"距离（{DISTANCE_METRIC}）：")
    print("model:" + "".join(page.number.rjust(8) for page in model_pages))
    for page, row in zip(real_pages, distance_matrix):
        print(page.number.rjust(8) + "".join(f"{value:8.3f}" for value in row))


def match_pages(
    model_pages: Sequence[Page],
    real_pages: Sequence[Page],
    threshold: float,
) -> list[tuple[Page, Page | None]]:
    """为每张 real 图片查找距离小于阈值且距离最近的 model 图片。

    Args:
        model_pages: 候选图片列表。
        real_pages: 查询图片列表。
        threshold: 判定为同一页的余弦距离阈值。

    Returns:
        ``(real 图片, 匹配到的 model 图片或 None)`` 的列表。
    """
    matches: list[tuple[Page, Page | None]] = []
    for real_page in real_pages:
        best_distance = threshold
        best_page: Page | None = None
        for model_page in model_pages:
            distance = cosine_distance(real_page.vector, model_page.vector)
            if distance < best_distance:
                best_distance = distance
                best_page = model_page
        matches.append((real_page, best_page))
    return matches


def print_matches(
    model_pages: Sequence[Page],
    matches: Sequence[tuple[Page, Page | None]],
) -> None:
    """打印匹配结果和正确率。

    Args:
        model_pages: 候选图片列表。
        matches: 由 :func:`match_pages` 得到的匹配结果。
    """
    model_numbers = {page.number for page in model_pages}
    print("\n匹配：")
    correct_count = 0
    for real_page, matched_page in matches:
        expected_number = real_page.number if real_page.number in model_numbers else None
        matched_number = matched_page.number if matched_page is not None else None
        correct = matched_number == expected_number
        correct_count += int(correct)
        status = "OK" if correct else "错误"
        print(
            f"  {real_page.number:>5} -> {str(matched_number):<5} "
            f"期望: {str(expected_number):<5} {status}"
        )
    print(
        f"正确率：{correct_count}/{len(matches)} = "
        f"{correct_count / len(matches) * 100:.1f}%"
    )
