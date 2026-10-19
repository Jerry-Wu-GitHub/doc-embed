"""图片向量化的对外接口。

本模块只暴露一个业务函数 :func:`embed_image`：输入图片字节，输出图片的
embedding 向量。默认使用 hybrid 模式，把 CNN embedding 与 pHash/dHash
特征拼接成一个 L2 归一化的向量；也可以关闭 hybrid 模式，只返回 CNN
embedding。
"""

import numpy as np

try:  # 作为包导入时使用相对导入
    from .config import get_config
    from .embedder import compute_cnn_embedding
    from .hybrid import build_hybrid_embedding
    from .image_utils import ImageBytes
    from .signature import compute_signature
except ImportError:  # 直接把项目根目录加入 sys.path 后导入
    from config import get_config
    from embedder import compute_cnn_embedding
    from hybrid import build_hybrid_embedding
    from image_utils import ImageBytes
    from signature import compute_signature


def embed_image(
    image_bytes: ImageBytes,
    *,
    hybrid_mode: bool | None = None,
    cnn_weight: float | None = None,
    hash_size: int | None = None,
) -> np.ndarray:
    """计算图片的 embedding 向量。

    hybrid 模式（默认）返回 CNN embedding、pHash 和 dHash 拼接后的向量，
    默认维度为 641（``HYBRID_EMBEDDING_DIMENSION``），L2 范数为 1；两个
    向量的余弦相似度等于 CNN 与 hash 的 soft 融合分数。非 hybrid 模式只
    返回 128 维的 CNN embedding。

    soft/strict 是两张图片比较时的判定模式，不影响单张图片的 embedding：
    两种模式返回相同的 hybrid 向量；strict 还需要 pHash/dHash 原始位进行
    门控，因此由 :func:`compare_images` 或 :class:`HybridMatcher` 完成。

    Args:
        image_bytes: 图片文件的原始字节，支持 bytes、bytearray 或 memoryview。
        hybrid_mode: 是否使用 hybrid 模式；为空时使用全局配置。
        cnn_weight: hybrid 模式中 CNN 分量的权重；为空时使用全局配置。
        hash_size: pHash/dHash 的边长；为空时使用全局配置。

    Returns:
        L2 归一化的 float32 embedding 向量。hybrid 模式形状为
        ``(HYBRID_EMBEDDING_DIMENSION,)``，CNN-only 模式形状为
        ``(EMBEDDING_DIMENSION,)``。

    Raises:
        TypeError: 输入不是字节类型。
        ValueError: 输入为空、无法解码为图片，或参数越界。
        RuntimeError: 模型加载或下载失败。
    """
    config = get_config()
    actual_hybrid_mode = config.hybrid_mode if hybrid_mode is None else hybrid_mode
    if not actual_hybrid_mode:
        return compute_cnn_embedding(image_bytes)

    signature = compute_signature(image_bytes, hash_size=hash_size)
    return build_hybrid_embedding(signature, cnn_weight=cnn_weight)
