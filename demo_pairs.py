"""命令行传入图片时的成对比较演示。

本模块计算每张图片的 embedding，打印相似度矩阵和两两判定，并额外展示
CNN + pHash/dHash 的 soft/strict 特征比较与图库匹配。
"""

import sys
from collections.abc import Sequence

import numpy as np

try:  # 作为包导入时使用相对导入
    from .constants import (
        CNN_DIFFERENCE_THRESHOLD,
        DIFFERENCE_THRESHOLD,
        DISTANCE_METRIC,
    )
    from .demo_common import collect_image_files
    from .embedding import embed_image
    from .hybrid import SimilarityMode
    from .matching import HybridMatcher, compare_images
    from .metrics import cosine_distance
except ImportError:  # 直接把项目根目录加入 sys.path 后导入
    from constants import (
        CNN_DIFFERENCE_THRESHOLD,
        DIFFERENCE_THRESHOLD,
        DISTANCE_METRIC,
    )
    from demo_common import collect_image_files
    from embedding import embed_image
    from hybrid import SimilarityMode
    from matching import HybridMatcher, compare_images
    from metrics import cosine_distance


def run_pair_demo(
    paths: Sequence[str],
    *,
    hybrid_mode: bool,
    strict: bool,
) -> int:
    """对命令行传入的图片运行向量化和混合特征演示。

    Args:
        paths: 图片文件或目录路径列表。
        hybrid_mode: 是否使用 hybrid embedding。
        strict: 是否使用 hybrid strict 模式；为 False 时使用 soft 模式。

    Returns:
        进程退出码；0 表示成功，1 表示没有找到图片。
    """
    image_files = collect_image_files(paths)
    if not image_files:
        print("没有找到图片", file=sys.stderr)
        return 1

    image_bytes_list = [file.read_bytes() for file in image_files]
    threshold = DIFFERENCE_THRESHOLD if hybrid_mode else CNN_DIFFERENCE_THRESHOLD
    embeddings: list[np.ndarray] = []
    for image_file, image_bytes in zip(image_files, image_bytes_list):
        vector = embed_image(image_bytes, hybrid_mode=hybrid_mode)
        embeddings.append(vector)
        print(
            f"{image_file.name}: shape={vector.shape}, dtype={vector.dtype}, "
            f"L2={np.linalg.norm(vector):.6f}, first5={vector[:5]}"
        )

    embedding_matrix = np.stack(embeddings)
    similarity_matrix = embedding_matrix @ embedding_matrix.T
    print(f"\n阈值：{threshold:.4f}（{DISTANCE_METRIC} 距离）")
    print("余弦相似度矩阵：")
    print(np.array2string(similarity_matrix, precision=4, suppress_small=True))

    print("\n向量两两比较（soft 参考）：")
    for first_index in range(len(image_files)):
        for second_index in range(first_index + 1, len(image_files)):
            distance = cosine_distance(embeddings[first_index], embeddings[second_index])
            print(
                f"  ({first_index}, {second_index}) "
                f"similarity={1.0 - distance:.4f} distance={distance:.4f} "
                f"same={distance < threshold}"
            )

    decision_mode: SimilarityMode = "strict" if strict else "soft"
    print(f"\nhybrid 两两比较（{decision_mode}）：")
    for first_index in range(len(image_files)):
        for second_index in range(first_index + 1, len(image_files)):
            result = compare_images(
                image_bytes_list[first_index],
                image_bytes_list[second_index],
                mode=decision_mode,
            )
            print(
                f"  ({first_index}, {second_index}) "
                f"cnn={result.cnn_similarity:.4f} "
                f"hash={result.average_hash_similarity:.4f} "
                f"hybrid={result.hybrid_score:.4f} same={result.same}"
            )

    matcher = HybridMatcher(image_bytes_list, mode=decision_mode)
    print("\n图库匹配：")
    for index, image_file in enumerate(image_files):
        result = matcher.match(image_bytes_list[index])
        if result is None:
            print(f"  {index}: {image_file.name} -> None")
            continue
        matched_index, similarity_result = result
        print(
            f"  {index}: {image_file.name} -> "
            f"{matched_index}: {image_files[matched_index].name} "
            f"hybrid={similarity_result.hybrid_score:.4f}"
        )
    return 0
