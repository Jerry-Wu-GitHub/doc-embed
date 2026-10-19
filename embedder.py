"""TFLite CNN embedding 推理核心。

本模块封装模型的懒加载、输入预处理、量化参数处理与推理流程：

- 首次调用时才解析并加载模型，避免导入包时触发下载；
- 模型文件通过 ``model_store`` 从魔搭缓存或远程仓库获取，不硬编码路径；
- 输入图片统一缩放为模型声明的尺寸，输出反量化后做 L2 归一化；
- 解释器不是线程安全的，因此所有推理调用通过一把锁串行化。
"""

import threading

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

try:  # 作为包导入时使用相对导入
    from .config import get_config
    from .constants import EMBEDDING_DIMENSION
    from .image_utils import ImageBytes, decode_image, resize_to_array
    from .model_store import load_interpreter, resolve_model_path
except ImportError:  # 直接把项目根目录加入 sys.path 后导入
    from config import get_config
    from constants import EMBEDDING_DIMENSION
    from image_utils import ImageBytes, decode_image, resize_to_array
    from model_store import load_interpreter, resolve_model_path


class ModelRuntime:
    """已加载的 TFLite 模型及其输入输出张量信息。

    该类在 :func:`build_runtime` 中创建，集中保存解释器、输入尺寸、输入
    量化参数和输出量化参数，避免每次推理都重复查询模型信息。
    """

    def __init__(
        self,
        interpreter: Interpreter,
        input_detail: dict,
        output_detail: dict,
    ) -> None:
        """初始化模型运行时信息。

        Args:
            interpreter: 已完成张量分配的 TFLite 解释器。
            input_detail: 输入张量详情，包含形状、类型和量化参数。
            output_detail: 输出张量详情，包含形状、类型和量化参数。

        Raises:
            RuntimeError: 模型输入或输出形状与当前代码不兼容。
        """
        input_shape = tuple(int(size) for size in input_detail["shape"])
        if (
            len(input_shape) != 4
            or input_shape[0] != 1
            or input_shape[1] != input_shape[2]
            or input_shape[1] <= 0
            or input_shape[3] != 3
        ):
            raise RuntimeError(f"不支持的模型输入形状: {input_shape}")

        output_shape = tuple(int(size) for size in output_detail["shape"])
        output_dimension = int(np.prod(output_shape[1:])) if len(output_shape) > 1 else 1
        if output_dimension != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"模型输出维度为 {output_dimension}，但代码期望 {EMBEDDING_DIMENSION}"
            )

        self.interpreter = interpreter
        self.input_detail = input_detail
        self.output_detail = output_detail
        self.input_size = input_shape[1]
        self.input_dtype = input_detail["dtype"]
        self.output_dtype = output_detail["dtype"]
        self.input_scale, self.input_zero_point = input_detail["quantization"]
        self.output_scale, self.output_zero_point = output_detail["quantization"]


_RUNTIME: ModelRuntime | None = None
_INITIALIZATION_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def build_runtime() -> ModelRuntime:
    """根据全局配置解析模型路径并创建运行时。

    Returns:
        已完成初始化的 :class:`ModelRuntime` 实例。

    Raises:
        FileNotFoundError: 模型文件不存在。
        ValueError: 模型文件无效。
        RuntimeError: 模型下载失败或模型结构不兼容。
    """
    config = get_config()
    model_path = resolve_model_path(config)
    interpreter = load_interpreter(model_path, config.thread_count)
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return ModelRuntime(interpreter, input_detail, output_detail)


def get_runtime() -> ModelRuntime:
    """获取进程内共享的模型运行时。

    使用双重检查锁保证模型只加载一次，并且多个线程同时首次调用时不会
    重复创建解释器。

    Returns:
        全局共享的 :class:`ModelRuntime` 实例。
    """
    global _RUNTIME
    if _RUNTIME is None:
        with _INITIALIZATION_LOCK:
            if _RUNTIME is None:
                _RUNTIME = build_runtime()
    return _RUNTIME


def prepare_input(array: np.ndarray, runtime: ModelRuntime) -> np.ndarray:
    """把 RGB uint8 数组转换为模型输入张量。

    Args:
        array: 形状为 ``(size, size, 3)`` 的 RGB uint8 数组。
        runtime: 提供输入尺寸、类型和量化参数的模型运行时。

    Returns:
        带有 batch 维度的模型输入数组。

    Raises:
        ValueError: 数组形状与模型输入尺寸不匹配。
        TypeError: 模型输入类型不受支持。
    """
    array = np.asarray(array, dtype=np.uint8)
    expected_shape = (runtime.input_size, runtime.input_size, 3)
    if array.shape != expected_shape:
        raise ValueError(f"期望输入形状 {expected_shape}，实际为 {array.shape}")

    if runtime.input_dtype == np.uint8:
        model_input = array
    elif runtime.input_dtype == np.int8:
        input_scale = runtime.input_scale or 1.0
        quantized = np.rint(array.astype(np.float32) / input_scale)
        quantized = quantized + runtime.input_zero_point
        model_input = np.clip(quantized, -128, 127).astype(np.int8)
    elif runtime.input_dtype == np.float32:
        model_input = array.astype(np.float32)
    else:
        raise TypeError(f"不支持的模型输入类型: {runtime.input_dtype}")

    return model_input[None, ...]


def dequantize_output(values: np.ndarray, runtime: ModelRuntime) -> np.ndarray:
    """把模型输出反量化为 float32 向量。

    Args:
        values: 解释器返回的原始输出数组。
        runtime: 提供输出类型和量化参数的模型运行时。

    Returns:
        形状为 ``(EMBEDDING_DIMENSION,)`` 的 float32 向量。
    """
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if runtime.output_dtype in (np.int8, np.uint8):
        vector = (vector - runtime.output_zero_point) * runtime.output_scale
    return vector


def embed_array(array: np.ndarray) -> np.ndarray:
    """计算 224x224x3 RGB uint8 数组的 CNN embedding。

    Args:
        array: 形状为 ``(size, size, 3)`` 的 RGB uint8 数组。

    Returns:
        形状为 ``(EMBEDDING_DIMENSION,)``、L2 范数约为 1 的 float32 向量。

    Raises:
        ValueError: 数组形状与模型输入尺寸不匹配。
        RuntimeError: 模型加载失败。
    """
    runtime = get_runtime()
    model_input = prepare_input(array, runtime)
    with _INFERENCE_LOCK:
        runtime.interpreter.set_tensor(runtime.input_detail["index"], model_input)
        runtime.interpreter.invoke()
        raw_output = runtime.interpreter.get_tensor(runtime.output_detail["index"])

    vector = dequantize_output(raw_output, runtime)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def compute_cnn_embedding_from_image(image: Image.Image) -> np.ndarray:
    """计算一张 Pillow 图片的 CNN embedding。

    Args:
        image: Pillow 图片对象，内部会统一转换为 RGB 并缩放。

    Returns:
        形状为 ``(EMBEDDING_DIMENSION,)``、L2 归一化的 float32 向量。
    """
    runtime = get_runtime()
    array = resize_to_array(image, runtime.input_size)
    return embed_array(array)


def compute_cnn_embedding(image_bytes: ImageBytes) -> np.ndarray:
    """计算图片字节的 CNN embedding。

    Args:
        image_bytes: 图片文件的原始字节。

    Returns:
        形状为 ``(EMBEDDING_DIMENSION,)``、L2 归一化的 float32 向量。

    Raises:
        TypeError: 输入不是字节类型。
        ValueError: 输入为空或无法解码为图片。
    """
    image = decode_image(image_bytes)
    return compute_cnn_embedding_from_image(image)
