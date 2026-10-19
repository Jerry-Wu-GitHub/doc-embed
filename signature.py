"""图片签名：CNN embedding 与感知哈希特征。

本模块负责从图片字节中提取三类特征：

- CNN embedding（128 维，L2 归一化）；
- pHash 位数组；
- dHash 位数组。

这些特征既可以单独使用，也可以在 ``hybrid`` 模块中拼接成 hybrid
embedding，或用于两阶段相似度校验。
"""

from dataclasses import dataclass, field

import imagehash
import numpy as np
from PIL import Image

try:  # 作为包导入时使用相对导入
    from .config import get_config
    from .embedder import compute_cnn_embedding_from_image
    from .image_utils import ImageBytes, decode_image
except ImportError:  # 直接把项目根目录加入 sys.path 后导入
    from config import get_config
    from embedder import compute_cnn_embedding_from_image
    from image_utils import ImageBytes, decode_image


@dataclass(frozen=True)
class ImageHashes:
    """一张图片的感知哈希位数组。"""

    phash_bits: np.ndarray = field(
        metadata={"description": "pHash 位数组，展开为一维 0/1 序列。"}
    )
    dhash_bits: np.ndarray = field(
        metadata={"description": "dHash 位数组，展开为一维 0/1 序列。"}
    )


@dataclass(frozen=True)
class ImageSignature:
    """一张图片的 CNN embedding 与感知哈希特征。"""

    embedding: np.ndarray = field(
        metadata={"description": "128 维、L2 归一化的 CNN embedding。"}
    )
    phash_bits: np.ndarray = field(
        metadata={"description": "pHash 位数组，展开为一维 0/1 序列。"}
    )
    dhash_bits: np.ndarray = field(
        metadata={"description": "dHash 位数组，展开为一维 0/1 序列。"}
    )


def compute_phash_bits(image: Image.Image, hash_size: int) -> np.ndarray:
    """计算图片的 pHash 位数组。

    Args:
        image: Pillow 图片对象。
        hash_size: 哈希矩阵的边长，总位数为 ``hash_size * hash_size``。

    Returns:
        展开为一维的 uint8 位数组，元素为 0 或 1。
    """
    hash_matrix = imagehash.phash(image, hash_size=hash_size).hash
    return hash_matrix.flatten().astype(np.uint8)


def compute_dhash_bits(image: Image.Image, hash_size: int) -> np.ndarray:
    """计算图片的 dHash 位数组。

    Args:
        image: Pillow 图片对象。
        hash_size: 哈希矩阵的边长，总位数为 ``hash_size * hash_size``。

    Returns:
        展开为一维的 uint8 位数组，元素为 0 或 1。
    """
    hash_matrix = imagehash.dhash(image, hash_size=hash_size).hash
    return hash_matrix.flatten().astype(np.uint8)


def compute_image_hashes_from_image(
    image: Image.Image,
    hash_size: int,
) -> ImageHashes:
    """计算 Pillow 图片的 pHash 与 dHash 位数组。

    Args:
        image: Pillow 图片对象。
        hash_size: 哈希矩阵的边长，总位数为 ``hash_size * hash_size``。

    Returns:
        包含 pHash 位和 dHash 位的特征对象。

    Raises:
        ValueError: ``hash_size`` 不是正整数。
    """
    if hash_size <= 0:
        raise ValueError("hash_size 必须是正整数")
    return ImageHashes(
        phash_bits=compute_phash_bits(image, hash_size),
        dhash_bits=compute_dhash_bits(image, hash_size),
    )


def compute_image_hashes(
    image_bytes: ImageBytes,
    *,
    hash_size: int | None = None,
) -> ImageHashes:
    """只计算图片的感知哈希（pHash + dHash），不加载 CNN 模型。

    该函数只做图片解码和哈希计算，适合在向量检索前做快速预筛，或在不需要
    CNN embedding 的场景下单独使用。

    Args:
        image_bytes: 图片文件的原始字节。
        hash_size: pHash/dHash 的边长；为空时使用全局配置，默认 16。

    Returns:
        包含 pHash 位和 dHash 位的特征对象。

    Raises:
        TypeError: 输入不是字节类型。
        ValueError: 输入为空、无法解码为图片，或 ``hash_size`` 不是正整数。
    """
    config = get_config()
    actual_hash_size = config.hash_size if hash_size is None else hash_size
    image = decode_image(image_bytes)
    return compute_image_hashes_from_image(image, actual_hash_size)


def compute_signature(
    image_bytes: ImageBytes,
    *,
    hash_size: int | None = None,
) -> ImageSignature:
    """计算图片字节的 CNN embedding 与感知哈希特征。

    Args:
        image_bytes: 图片文件的原始字节。
        hash_size: pHash/dHash 的边长；为空时使用全局配置。

    Returns:
        包含 CNN embedding、pHash 位和 dHash 位的特征对象。

    Raises:
        ValueError: ``hash_size`` 不是正整数。
    """
    config = get_config()
    actual_hash_size = config.hash_size if hash_size is None else hash_size
    image = decode_image(image_bytes)
    image_hashes = compute_image_hashes_from_image(image, actual_hash_size)
    return ImageSignature(
        embedding=compute_cnn_embedding_from_image(image),
        phash_bits=image_hashes.phash_bits,
        dhash_bits=image_hashes.dhash_bits,
    )


def hash_similarity(bits_a: np.ndarray, bits_b: np.ndarray) -> float:
    """计算两个位数组的相似度。

    相似度定义为相同位的比例，等价于 ``1 - 汉明距离 / 位数``。

    Args:
        bits_a: 第一个位数组。
        bits_b: 第二个位数组。

    Returns:
        0 到 1 之间的相似度。

    Raises:
        ValueError: 两个位数组形状不一致或为空。
    """
    array_a = np.asarray(bits_a, dtype=np.uint8).reshape(-1)
    array_b = np.asarray(bits_b, dtype=np.uint8).reshape(-1)
    if array_a.shape != array_b.shape:
        raise ValueError(f"位数组形状不一致: {array_a.shape} vs {array_b.shape}")
    if array_a.size == 0:
        raise ValueError("位数组不能为空")
    return float(1.0 - np.mean(array_a != array_b))
