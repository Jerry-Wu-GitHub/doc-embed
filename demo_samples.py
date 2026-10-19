"""test_samples 示例组的评测逻辑。

本模块遍历 ``test_samples/<组名>/model`` 与 ``test_samples/<组名>/real``，
演示 hybrid/CNN-only 向量的距离计算、匹配结果和 strict 模式校验。
"""

import sys
from pathlib import Path

try:  # 作为包导入时使用相对导入
    from .constants import (
        CNN_DIFFERENCE_THRESHOLD,
        DIFFERENCE_THRESHOLD,
        DISTANCE_METRIC,
    )
    from .demo_common import (
        IMAGE_EXTENSIONS,
        build_distance_matrix,
        match_pages,
        print_distance_matrix,
        print_matches,
        vectorize_directory,
    )
    from .hybrid import SimilarityMode
    from .matching import HybridMatcher
except ImportError:  # 直接把项目根目录加入 sys.path 后导入
    from constants import (
        CNN_DIFFERENCE_THRESHOLD,
        DIFFERENCE_THRESHOLD,
        DISTANCE_METRIC,
    )
    from demo_common import (
        IMAGE_EXTENSIONS,
        build_distance_matrix,
        match_pages,
        print_distance_matrix,
        print_matches,
        vectorize_directory,
    )
    from hybrid import SimilarityMode
    from matching import HybridMatcher


def evaluate_hybrid_matching(
    model_directory: Path,
    real_directory: Path,
    *,
    mode: SimilarityMode,
) -> None:
    """使用 hybrid 特征的 soft 或 strict 模式匹配，并打印正确率。

    Args:
        model_directory: 候选图片目录。
        real_directory: 查询图片目录。
        mode: 判定模式，``soft`` 或 ``strict``。
    """
    model_files = sorted(
        file
        for file in model_directory.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )
    real_files = sorted(
        file
        for file in real_directory.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )
    matcher = HybridMatcher(
        [file.read_bytes() for file in model_files],
        mode=mode,
    )
    model_numbers = {file.stem for file in model_files}
    correct_count = 0
    print(f"\n{mode} 匹配（hybrid）：")
    for real_file in real_files:
        result = matcher.match(real_file.read_bytes())
        if result is None:
            matched_number = None
        else:
            matched_index, _ = result
            matched_number = model_files[matched_index].stem
        expected_number = real_file.stem if real_file.stem in model_numbers else None
        correct = matched_number == expected_number
        correct_count += int(correct)
        status = "OK" if correct else "错误"
        print(
            f"  {real_file.stem:>5} -> {str(matched_number):<5} "
            f"期望: {str(expected_number):<5} {status}"
        )
    print(
        f"{mode} 正确率：{correct_count}/{len(real_files)} = "
        f"{correct_count / len(real_files) * 100:.1f}%"
    )


def run_sample_suite(
    sample_root: Path,
    *,
    hybrid_mode: bool,
    strict: bool,
) -> int:
    """运行 ``test_samples`` 中的全部样例组。

    Args:
        sample_root: 包含多个图片组的根目录。
        hybrid_mode: 是否使用 hybrid embedding。
        strict: 是否使用 strict 模式匹配；仅在 ``hybrid_mode`` 为 True 时生效。

    Returns:
        进程退出码；0 表示成功，1 表示没有可处理的样例。
    """
    if not sample_root.is_dir():
        print(f"找不到测试样例目录: {sample_root}", file=sys.stderr)
        return 1

    group_directories = sorted(
        directory for directory in sample_root.iterdir() if directory.is_dir()
    )
    if not group_directories:
        print(f"测试样例目录为空: {sample_root}", file=sys.stderr)
        return 1

    threshold = DIFFERENCE_THRESHOLD if hybrid_mode else CNN_DIFFERENCE_THRESHOLD
    mode_name = "hybrid" if hybrid_mode else "cnn-only"
    decision_mode: SimilarityMode = (
        "strict" if strict and hybrid_mode else "soft"
    )
    print(
        f"模式：{mode_name}，判定：{decision_mode}，"
        f"距离：{DISTANCE_METRIC}，阈值：{threshold:.4f}\n"
    )

    processed_count = 0
    for group_directory in group_directories:
        model_directory = group_directory / "model"
        real_directory = group_directory / "real"
        if not model_directory.is_dir() or not real_directory.is_dir():
            continue

        print(f"\n==== 测试 {group_directory.name} ====\n")
        model_pages = vectorize_directory(model_directory, hybrid_mode)
        print()
        real_pages = vectorize_directory(real_directory, hybrid_mode)
        if not model_pages or not real_pages:
            continue

        distance_matrix = build_distance_matrix(model_pages, real_pages)
        print()
        print_distance_matrix(model_pages, real_pages, distance_matrix)

        if hybrid_mode:
            evaluate_hybrid_matching(
                model_directory,
                real_directory,
                mode=decision_mode,
            )
        else:
            matches = match_pages(model_pages, real_pages, threshold)
            print_matches(model_pages, matches)
        processed_count += 1

    if processed_count == 0:
        print("没有找到包含 model/real 的样例组", file=sys.stderr)
        return 1
    return 0
