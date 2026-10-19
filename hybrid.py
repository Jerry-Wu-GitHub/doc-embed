"""hybrid embedding 与两阶段相似度计算。

hybrid embedding 由三部分组成：

1. CNN embedding（128 维，L2 归一化）；
2. pHash 位向量（默认 16x16 = 256 bit，映射为 ±1）；
3. dHash 位向量（默认 16x16 = 256 bit，映射为 ±1）。

拼接时给 CNN 分量加权 ``cnn_weight``（默认 0.9），剩余的权重平均分配给
pHash 和 dHash，并额外加入一个常数坐标来表示哈希相似度中的常数项。
这样构造出的向量 L2 范数恰好为 1，且两个向量的余弦相似度严格等于
``cnn_weight * s_cnn + (1 - cnn_weight) * (0.5 * s_phash + 0.5 * s_dhash)``，
也就是模型报告中的 soft 融合分数。因此 hybrid embedding 可以直接用余弦
距离比较，阈值与 soft 融合阈值一一对应。
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

try:  # 作为包导入时使用相对导入
    from .config import get_config
    from .constants import (
        HYBRID_SIMILARITY_THRESHOLD,
        STRICT_CNN_HIGH,
        STRICT_CNN_LOW,
        STRICT_HASH_MINIMUM,
    )
    from .signature import ImageSignature, hash_similarity
except ImportError:  # 直接把项目根目录加入 sys.path 后导入
    from config import get_config
    from constants import (
        HYBRID_SIMILARITY_THRESHOLD,
        STRICT_CNN_HIGH,
        STRICT_CNN_LOW,
        STRICT_HASH_MINIMUM,
    )
    from signature import ImageSignature, hash_similarity


SimilarityMode = Literal["soft", "strict"]


@dataclass(frozen=True)
class SimilarityResult:
    """两个图片特征的相似度计算结果。"""

    cnn_similarity: float = field(
        metadata={"description": "两个 CNN embedding 的余弦相似度。"}
    )
    phash_similarity: float = field(
        metadata={"description": "两个 pHash 位数组的相似度。"}
    )
    dhash_similarity: float = field(
        metadata={"description": "两个 dHash 位数组的相似度。"}
    )
    average_hash_similarity: float = field(
        metadata={"description": "pHash 与 dHash 相似度的平均值。"}
    )
    hybrid_score: float = field(
        metadata={"description": "CNN 与 hash 加权融合后的分数。"}
    )
    same: bool = field(
        metadata={"description": "按照指定模式判定两张图片是否相同。"}
    )
    mode: SimilarityMode = field(
        metadata={"description": "判定模式：soft 或 strict。"}
    )


def build_hybrid_embedding(
    signature: ImageSignature,
    *,
    cnn_weight: float | None = None,
) -> np.ndarray:
    """把图片特征拼接为 L2 归一化的 hybrid embedding。

    Args:
        signature: 图片的 CNN embedding 与感知哈希特征。
        cnn_weight: CNN 分量权重；为空时使用全局配置，默认 0.9。

    Returns:
        形状为 ``HYBRID_EMBEDDING_DIMENSION`` 的 float32 向量，L2 范数为 1。
        两个向量点积等于 soft 融合分数。

    Raises:
        ValueError: 权重越界，或 pHash/dHash 位数组长度不一致。
    """
    config = get_config()
    actual_cnn_weight = config.cnn_weight if cnn_weight is None else cnn_weight
    if not 0.0 <= actual_cnn_weight <= 1.0:
        raise ValueError("cnn_weight 必须在 0 到 1 之间")

    phash_bits = np.asarray(signature.phash_bits, dtype=np.float32).reshape(-1)
    dhash_bits = np.asarray(signature.dhash_bits, dtype=np.float32).reshape(-1)
    if phash_bits.shape != dhash_bits.shape:
        raise ValueError(
            f"pHash 与 dHash 位数不一致: {phash_bits.shape} vs {dhash_bits.shape}"
        )
    bit_count = phash_bits.size
    if bit_count == 0:
        raise ValueError("哈希位数组不能为空")

    per_hash_weight = (1.0 - actual_cnn_weight) / 2.0
    constant_coordinate = np.array([np.sqrt(per_hash_weight)], dtype=np.float32)
    hash_coefficient = np.sqrt(per_hash_weight / (2.0 * bit_count))
    phash_signs = phash_bits * 2.0 - 1.0
    dhash_signs = dhash_bits * 2.0 - 1.0

    embedding = np.asarray(signature.embedding, dtype=np.float32).reshape(-1)
    return np.concatenate(
        [
            np.sqrt(actual_cnn_weight) * embedding,
            constant_coordinate,
            hash_coefficient * phash_signs,
            hash_coefficient * dhash_signs,
        ]
    ).astype(np.float32)


def hybrid_similarity(
    signature_a: ImageSignature,
    signature_b: ImageSignature,
    *,
    mode: SimilarityMode = "soft",
    cnn_weight: float | None = None,
    threshold: float | None = None,
    cnn_high: float | None = None,
    cnn_low: float | None = None,
    hash_minimum: float | None = None,
) -> SimilarityResult:
    """计算两个图片特征的混合相似度并给出是否相同的判定。

    soft 模式：融合分数大于等于阈值即判定为相同。

    strict 模式：CNN 相似度大于等于 ``cnn_high`` 时直接通过；否则要求
    CNN 相似度不低于 ``cnn_low``、平均 hash 相似度不低于 ``hash_minimum``，
    且融合分数不低于阈值。

    Args:
        signature_a: 第一个图片特征。
        signature_b: 第二个图片特征。
        mode: 判定模式，``soft`` 或 ``strict``。
        cnn_weight: CNN 分量权重；为空时使用全局配置。
        threshold: 融合分数阈值；为空时使用模型的 soft 阈值。
        cnn_high: strict 模式的 CNN 高置信阈值。
        cnn_low: strict 模式的 CNN 低置信下限。
        hash_minimum: strict 模式的平均 hash 相似度下限。

    Returns:
        包含各项相似度和最终判定的结果对象。

    Raises:
        ValueError: ``mode`` 未知，或 CNN 权重超出有效范围。
    """
    if mode not in ("soft", "strict"):
        raise ValueError(f"未知的判定模式: {mode}")

    config = get_config()
    actual_cnn_weight = config.cnn_weight if cnn_weight is None else cnn_weight
    actual_threshold = HYBRID_SIMILARITY_THRESHOLD if threshold is None else threshold
    actual_cnn_high = STRICT_CNN_HIGH if cnn_high is None else cnn_high
    actual_cnn_low = STRICT_CNN_LOW if cnn_low is None else cnn_low
    actual_hash_minimum = (
        STRICT_HASH_MINIMUM if hash_minimum is None else hash_minimum
    )
    if not 0.0 <= actual_cnn_weight <= 1.0:
        raise ValueError("cnn_weight 必须在 0 到 1 之间")

    cnn_similarity = float(
        np.dot(
            np.asarray(signature_a.embedding, dtype=np.float32).reshape(-1),
            np.asarray(signature_b.embedding, dtype=np.float32).reshape(-1),
        )
    )
    phash_similarity = hash_similarity(signature_a.phash_bits, signature_b.phash_bits)
    dhash_similarity = hash_similarity(signature_a.dhash_bits, signature_b.dhash_bits)
    average_hash_similarity = 0.5 * (phash_similarity + dhash_similarity)
    hybrid_score = (
        actual_cnn_weight * cnn_similarity
        + (1.0 - actual_cnn_weight) * average_hash_similarity
    )

    if mode == "soft":
        same = hybrid_score >= actual_threshold
    else:
        same = bool(
            cnn_similarity >= actual_cnn_high
            or (
                cnn_similarity >= actual_cnn_low
                and average_hash_similarity >= actual_hash_minimum
                and hybrid_score >= actual_threshold
            )
        )

    return SimilarityResult(
        cnn_similarity=cnn_similarity,
        phash_similarity=phash_similarity,
        dhash_similarity=dhash_similarity,
        average_hash_similarity=average_hash_similarity,
        hybrid_score=hybrid_score,
        same=bool(same),
        mode=mode,
    )
