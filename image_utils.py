"""图片解码与预处理工具。

本模块只负责把图片字节解码为 Pillow 图片，以及把图片缩放为模型输入
所需的 RGB 数组；不涉及具体的 TFLite 推理。
"""

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


ImageBytes = bytes | bytearray | memoryview


def decode_image(image_bytes: ImageBytes) -> Image.Image:
    """把图片字节解码为 RGB Pillow 图片。

    解码时会根据 EXIF 信息自动纠正拍摄方向，并把所有颜色模式（灰度、
    RGBA、CMYK 等）统一转换为 RGB，保证后续模型输入格式一致。

    Args:
        image_bytes: 图片文件的原始字节，支持 bytes、bytearray 或 memoryview。

    Returns:
        已加载到内存的 RGB 图片对象。

    Raises:
        TypeError: 输入不是字节类型。
        ValueError: 输入为空，或字节流无法解码为有效图片。
    """
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("image_bytes 必须是 bytes、bytearray 或 memoryview")
    if len(image_bytes) == 0:
        raise ValueError("image_bytes 不能为空")

    try:
        with Image.open(io.BytesIO(bytes(image_bytes))) as image:
            transposed_image = ImageOps.exif_transpose(image)
            rgb_image = transposed_image.convert("RGB")
            rgb_image.load()
            return rgb_image
    except UnidentifiedImageError as error:
        raise ValueError("image_bytes 不是可解码的图片") from error
    except OSError as error:
        raise ValueError(f"图片解码失败: {error}") from error


def resize_to_array(image: Image.Image, size: int) -> np.ndarray:
    """把图片缩放为指定边长的 RGB uint8 数组。

    Args:
        image: 待缩放的 Pillow 图片。
        size: 输出图片的边长，输出形状为 ``(size, size, 3)``。

    Returns:
        形状为 ``(size, size, 3)``、类型为 ``uint8`` 的 RGB 数组。

    Raises:
        ValueError: ``size`` 不是正整数。
    """
    if size <= 0:
        raise ValueError("size 必须是正整数")
    resized_image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized_image, dtype=np.uint8)
