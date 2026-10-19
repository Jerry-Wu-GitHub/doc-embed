"""基于混合特征的两图比较与图库匹配工具。

本模块提供：

- :func:`compare_images`：比较两张图片并返回 soft/strict 判定结果；
- :func:`match_signature`：在一组候选特征中查找匹配项；
- :class:`HybridMatcher`：缓存候选图片特征并支持增量匹配。

:func:`compare_images` 和 :class:`HybridMatcher` 由包的顶层导出，
:func:`match_signature` 作为更底层的接口保留在模块内。
"""

from collections.abc import Sequence

try:  # 作为包导入时使用相对导入
    from .hybrid import SimilarityMode, SimilarityResult, hybrid_similarity
    from .image_utils import ImageBytes
    from .signature import ImageSignature, compute_signature
except ImportError:  # 直接把项目根目录加入 sys.path 后导入
    from hybrid import SimilarityMode, SimilarityResult, hybrid_similarity
    from image_utils import ImageBytes
    from signature import ImageSignature, compute_signature


def compare_images(
    image_bytes_a: ImageBytes,
    image_bytes_b: ImageBytes,
    *,
    mode: SimilarityMode = "soft",
    cnn_weight: float | None = None,
    threshold: float | None = None,
    cnn_high: float | None = None,
    cnn_low: float | None = None,
    hash_minimum: float | None = None,
    hash_size: int | None = None,
) -> SimilarityResult:
    """比较两张图片并返回 hybrid 相似度与是否相同的判定。

    soft 模式使用 CNN 与 hash 的加权融合分数；strict 模式在融合分数之外
    还要求 hash 校验通过，适合手机拍照、近重复页面等高精度场景。

    Args:
        image_bytes_a: 第一张图片的原始字节。
        image_bytes_b: 第二张图片的原始字节。
        mode: 判定模式，``soft`` 或 ``strict``。
        cnn_weight: CNN 分量权重；为空时使用全局配置。
        threshold: 融合分数阈值；为空时使用模型的 soft 阈值。
        cnn_high: strict 模式的 CNN 高置信阈值。
        cnn_low: strict 模式的 CNN 低置信下限。
        hash_minimum: strict 模式的平均 hash 相似度下限。
        hash_size: pHash/dHash 的边长；为空时使用全局配置。

    Returns:
        包含 CNN、pHash、dHash、融合分数和最终判定的结果对象。

    Raises:
        TypeError: 输入不是字节类型。
        ValueError: 输入为空、无法解码为图片，或参数越界。
        RuntimeError: 模型加载或下载失败。
    """
    signature_a = compute_signature(image_bytes_a, hash_size=hash_size)
    signature_b = compute_signature(image_bytes_b, hash_size=hash_size)
    return hybrid_similarity(
        signature_a,
        signature_b,
        mode=mode,
        cnn_weight=cnn_weight,
        threshold=threshold,
        cnn_high=cnn_high,
        cnn_low=cnn_low,
        hash_minimum=hash_minimum,
    )


def match_signature(
    query_signature: ImageSignature,
    gallery_signatures: Sequence[ImageSignature],
    *,
    mode: SimilarityMode = "soft",
    cnn_weight: float | None = None,
    threshold: float | None = None,
    cnn_high: float | None = None,
    cnn_low: float | None = None,
    hash_minimum: float | None = None,
) -> tuple[int, SimilarityResult] | None:
    """在候选特征中查找与查询特征相同的项。

    只有判定为相同的候选才会参与比较，最终返回融合分数最高的一项。

    Args:
        query_signature: 查询图片的特征。
        gallery_signatures: 候选图片特征的序列。
        mode: 判定模式，``soft`` 或 ``strict``。
        cnn_weight: CNN 分量权重；为空时使用全局配置。
        threshold: 融合分数阈值；为空时使用模型的 soft 阈值。
        cnn_high: strict 模式的 CNN 高置信阈值。
        cnn_low: strict 模式的 CNN 低置信下限。
        hash_minimum: strict 模式的平均 hash 相似度下限。

    Returns:
        ``(索引, 相似度结果)`` 二元组；没有匹配项时返回 None。
    """
    best_index: int | None = None
    best_result: SimilarityResult | None = None

    for index, gallery_signature in enumerate(gallery_signatures):
        result = hybrid_similarity(
            query_signature,
            gallery_signature,
            mode=mode,
            cnn_weight=cnn_weight,
            threshold=threshold,
            cnn_high=cnn_high,
            cnn_low=cnn_low,
            hash_minimum=hash_minimum,
        )
        if result.same and (
            best_result is None or result.hybrid_score > best_result.hybrid_score
        ):
            best_index = index
            best_result = result

    if best_index is None or best_result is None:
        return None
    return best_index, best_result


class HybridMatcher:
    """可增量添加候选图片的混合特征匹配器。

    匹配器在内部缓存每张候选图片的特征，适合服务端把图库预热一次后，
    对多张查询图片重复匹配。
    """

    def __init__(
        self,
        gallery_bytes: Sequence[ImageBytes] | None = None,
        *,
        mode: SimilarityMode = "soft",
        cnn_weight: float | None = None,
        threshold: float | None = None,
        cnn_high: float | None = None,
        cnn_low: float | None = None,
        hash_minimum: float | None = None,
    ) -> None:
        """初始化匹配器。

        Args:
            gallery_bytes: 可选的候选图片字节序列，会在初始化时计算特征。
            mode: 判定模式，``soft`` 或 ``strict``。
            cnn_weight: CNN 分量权重；为空时使用全局配置。
            threshold: 融合分数阈值；为空时使用模型的 soft 阈值。
            cnn_high: strict 模式的 CNN 高置信阈值。
            cnn_low: strict 模式的 CNN 低置信下限。
            hash_minimum: strict 模式的平均 hash 相似度下限。
        """
        if mode not in ("soft", "strict"):
            raise ValueError(f"未知的判定模式: {mode}")

        self.mode = mode
        self.cnn_weight = cnn_weight
        self.threshold = threshold
        self.cnn_high = cnn_high
        self.cnn_low = cnn_low
        self.hash_minimum = hash_minimum
        self.signatures: list[ImageSignature] = []

        if gallery_bytes is not None:
            for image_bytes in gallery_bytes:
                self.add(image_bytes)

    def add(self, image_bytes: ImageBytes) -> int:
        """向图库添加一张图片。

        Args:
            image_bytes: 图片文件的原始字节。

        Returns:
            新图片在图库中的索引。
        """
        signature = compute_signature(image_bytes)
        self.signatures.append(signature)
        return len(self.signatures) - 1

    def match(
        self,
        query_bytes: ImageBytes,
        *,
        mode: SimilarityMode | None = None,
    ) -> tuple[int, SimilarityResult] | None:
        """在已缓存的图库中匹配查询图片。

        Args:
            query_bytes: 查询图片的原始字节。
            mode: 临时覆盖判定模式；为空时使用初始化时的模式。

        Returns:
            ``(索引, 相似度结果)`` 二元组；没有匹配项时返回 None。
        """
        actual_mode = self.mode if mode is None else mode
        return match_signature(
            compute_signature(query_bytes),
            self.signatures,
            mode=actual_mode,
            cnn_weight=self.cnn_weight,
            threshold=self.threshold,
            cnn_high=self.cnn_high,
            cnn_low=self.cnn_low,
            hash_minimum=self.hash_minimum,
        )
