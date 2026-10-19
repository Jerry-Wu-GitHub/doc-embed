"""模型文件的本地缓存、下载与解释器加载。

本模块负责解析 TFLite 模型文件的位置：

1. 若配置中显式指定了 ``model_path``，直接使用该本地文件；
2. 否则先查询 ModelScope 的本地缓存；
3. 缓存不存在时，通过 ``modelscope`` 库从模型仓库下载。

模型仓库中的文件路径由 :class:`~config.Config` 的 ``model_file`` 字段给出，
默认是 ``doc_embed_int8.tflite``。
"""

from pathlib import Path

from ai_edge_litert.interpreter import Interpreter

try:  # 作为包导入时使用相对导入
    from .config import Config
    from .constants import MINIMUM_MODEL_SIZE, TFLITE_MAGIC
except ImportError:  # 直接把项目根目录加入 sys.path 后导入
    from config import Config
    from constants import MINIMUM_MODEL_SIZE, TFLITE_MAGIC

try:
    from modelscope.hub.file_download import model_file_download
except ImportError:  # 允许在没有 modelscope 时使用显式指定的本地模型
    model_file_download = None


def normalize_model_file(model_file: str) -> str:
    """去掉模型仓库路径开头的斜杠。

    ModelScope 的 ``model_file_download`` 要求文件路径相对于仓库根目录，
    不能以 ``/`` 或 ``\\`` 开头；而文档中常用 ``/doc_embed_int8.tflite``
    这种写法表示仓库根目录下的文件。

    Args:
        model_file: 模型仓库中的文件路径。

    Returns:
        去掉开头斜杠后的相对路径。
    """
    return model_file.lstrip("/\\")


def load_interpreter(model_path: Path, thread_count: int) -> Interpreter:
    """创建并初始化 TFLite 解释器。

    这里先用 Python 读取模型字节，再通过 ``model_content=`` 传入解释器，
    而不是使用 ``model_path=``。原因是 LiteRT/TFLite 的 C++ 层使用窄字符
    ``fopen`` 打开模型，在 Windows 上遇到中文路径或空格时会失败。

    Args:
        model_path: 本地 ``.tflite`` 模型文件路径。
        thread_count: 推理线程数。

    Returns:
        已完成张量分配的解释器对象。

    Raises:
        FileNotFoundError: 模型文件不存在或无法读取。
    """
    try:
        raw_content = model_path.read_bytes()
    except OSError as error:
        raise FileNotFoundError(f"无法读取模型文件: {model_path} ({error})") from error

    interpreter = Interpreter(model_content=raw_content, num_threads=thread_count)
    interpreter.allocate_tensors()
    return interpreter


def validate_model_file(model_path: Path) -> None:
    """校验模型文件是否存在且是有效的 TFLite 文件。

    Args:
        model_path: 待校验的模型文件路径。

    Raises:
        FileNotFoundError: 模型文件不存在。
        ValueError: 模型文件为空、过小或不是 TFLite flatbuffer。
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    raw_content = model_path.read_bytes()
    if len(raw_content) < MINIMUM_MODEL_SIZE or raw_content[4:8] != TFLITE_MAGIC:
        raise ValueError(
            f"模型文件无效或为空: {model_path} (size={len(raw_content)} bytes)。"
            "请确认文件是有效的 .tflite 模型。"
        )


def find_cached_model(config: Config) -> Path | None:
    """在 ModelScope 本地缓存中查找模型文件。

    Args:
        config: 全局配置，提供模型仓库 ID、文件路径和 revision。

    Returns:
        缓存中的模型文件路径；未找到或未安装 modelscope 时返回 None。
    """
    if model_file_download is None:
        return None

    try:
        cached_path = model_file_download(
            config.model_repository_id,
            normalize_model_file(config.model_file),
            revision=config.model_revision,
            local_files_only=True,
        )
    except ValueError:
        return None
    if cached_path is None:
        return None
    return Path(cached_path)


def download_model(config: Config) -> Path:
    """从 ModelScope 模型仓库下载模型文件。

    下载结果会写入 ModelScope 的本地缓存，后续调用可以直接命中缓存。

    Args:
        config: 全局配置，提供模型仓库 ID、文件路径和 revision。

    Returns:
        下载完成后的本地模型文件路径。

    Raises:
        RuntimeError: 未安装 modelscope 或下载失败。
    """
    if model_file_download is None:
        raise RuntimeError(
            "未安装 modelscope，无法自动下载模型。"
            "请安装 modelscope，或通过 MODEL_PATH 指定本地模型文件。"
        )

    model_file = normalize_model_file(config.model_file)
    try:
        downloaded_path = model_file_download(
            config.model_repository_id,
            model_file,
            revision=config.model_revision,
        )
    except Exception as error:
        raise RuntimeError(
            "无法从模型仓库下载模型: "
            f"{config.model_repository_id}/{model_file} ({error})"
        ) from error

    if downloaded_path is None:
        raise RuntimeError(
            "模型下载失败: "
            f"{config.model_repository_id}/{model_file}"
        )
    return Path(downloaded_path)


def resolve_model_path(config: Config) -> Path:
    """解析并校验可用的本地模型文件路径。

    解析顺序：显式配置的本地路径、ModelScope 本地缓存、远程下载。

    Args:
        config: 全局配置，提供模型位置相关设置。

    Returns:
        已验证的本地 TFLite 模型文件路径。

    Raises:
        FileNotFoundError: 显式指定的模型文件不存在。
        ValueError: 模型文件无效。
        RuntimeError: 自动下载失败。
    """
    if config.model_path is not None:
        model_path = config.model_path
        validate_model_file(model_path)
        return model_path

    cached_path = find_cached_model(config)
    if cached_path is not None:
        validate_model_file(cached_path)
        return cached_path

    downloaded_path = download_model(config)
    validate_model_file(downloaded_path)
    return downloaded_path
