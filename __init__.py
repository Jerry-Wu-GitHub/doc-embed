"""doc-embed：书页/插画图片向量化。

对外提供以下接口：

- :func:`embed_image`：输入图片字节，输出 embedding 向量。默认使用
  hybrid 模式（CNN + pHash/dHash），也可关闭 hybrid 模式只输出 CNN
  embedding。
- :func:`compare_images`：比较两张图片，按 soft 或 strict 模式返回相似度
  和是否相同的判定。
- :class:`HybridMatcher`：在候选图片集合中做增量匹配。
- :func:`get_config`：获取全局配置。

同时导出 :class:`SimilarityResult`、距离计算方法、相似度阈值和 embedding
维度等常量和类型，便于调用方直接做向量比较。所有可配置项都支持通过
环境变量或 ``.env`` 文件覆盖。

示例::

    from doc_embed import compare_images, embed_image, DIFFERENCE_THRESHOLD, METRIC

    with open("page.jpg", "rb") as file:
        vector = embed_image(file.read())

    result = compare_images(page_a_bytes, page_b_bytes, mode="strict")
    print(result.hybrid_score, result.same)
"""

try:  # 作为包导入时使用相对导入
    from .config import get_config
    from .constants import (
        CNN_DIFFERENCE_THRESHOLD,
        CNN_SIMILARITY_THRESHOLD,
        DIFFERENCE_THRESHOLD,
        DISTANCE_METRIC,
        EMBEDDING_DIMENSION,
        HYBRID_DIFFERENCE_THRESHOLD,
        HYBRID_EMBEDDING_DIMENSION,
        HYBRID_SIMILARITY_THRESHOLD,
        SAME_THRESHOLD,
        STRICT_CNN_HIGH,
        STRICT_CNN_LOW,
        STRICT_HASH_MINIMUM,
    )
    from .embedding import embed_image
    from .hybrid import SimilarityResult
    from .matching import HybridMatcher, compare_images
except ImportError:  # 直接把项目根目录加入 sys.path 后导入
    from config import get_config
    from constants import (
        CNN_DIFFERENCE_THRESHOLD,
        CNN_SIMILARITY_THRESHOLD,
        DIFFERENCE_THRESHOLD,
        DISTANCE_METRIC,
        EMBEDDING_DIMENSION,
        HYBRID_DIFFERENCE_THRESHOLD,
        HYBRID_EMBEDDING_DIMENSION,
        HYBRID_SIMILARITY_THRESHOLD,
        SAME_THRESHOLD,
        STRICT_CNN_HIGH,
        STRICT_CNN_LOW,
        STRICT_HASH_MINIMUM,
    )
    from embedding import embed_image
    from hybrid import SimilarityResult
    from matching import HybridMatcher, compare_images


# 距离计算方法：余弦距离。
METRIC: str = DISTANCE_METRIC

__all__ = [
    "embed_image",
    "compare_images",
    "HybridMatcher",
    "SimilarityResult",
    "get_config",
    "METRIC",
    "SAME_THRESHOLD",
    "DIFFERENCE_THRESHOLD",
    "CNN_SIMILARITY_THRESHOLD",
    "CNN_DIFFERENCE_THRESHOLD",
    "HYBRID_SIMILARITY_THRESHOLD",
    "HYBRID_DIFFERENCE_THRESHOLD",
    "EMBEDDING_DIMENSION",
    "HYBRID_EMBEDDING_DIMENSION",
    "STRICT_CNN_HIGH",
    "STRICT_CNN_LOW",
    "STRICT_HASH_MINIMUM",
]
