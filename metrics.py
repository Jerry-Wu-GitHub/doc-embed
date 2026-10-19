"""向量相似度与距离计算工具。

本模块提供余弦相似度和余弦距离的计算。hybrid embedding 与 CNN-only
embedding 都已做 L2 归一化，因此余弦相似度可以直接用点积计算，余弦距离
与相似度满足 ``distance = 1 - similarity``。
"""

import numpy as np


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """计算两个向量的余弦相似度。

    Args:
        vector_a: 第一个向量，会被展开为一维。
        vector_b: 第二个向量，会被展开为一维。

    Returns:
        余弦相似度，范围为 -1 到 1。

    Raises:
        ValueError: 两个向量形状不一致或包含零向量。
    """
    array_a = np.asarray(vector_a, dtype=np.float32).reshape(-1)
    array_b = np.asarray(vector_b, dtype=np.float32).reshape(-1)
    if array_a.shape != array_b.shape:
        raise ValueError(f"向量形状不一致: {array_a.shape} vs {array_b.shape}")

    denominator = float(np.linalg.norm(array_a) * np.linalg.norm(array_b))
    if denominator == 0:
        raise ValueError("向量不能为零向量")
    return float(np.dot(array_a, array_b) / denominator)


def cosine_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """计算两个向量的余弦距离。

    Args:
        vector_a: 第一个向量。
        vector_b: 第二个向量。

    Returns:
        余弦距离，等于 ``1 - cosine_similarity``，范围为 0 到 2。

    Raises:
        ValueError: 两个向量形状不一致或包含零向量。
    """
    return 1.0 - cosine_similarity(vector_a, vector_b)
