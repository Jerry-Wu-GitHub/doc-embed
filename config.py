"""项目全局配置模块。

本模块定义不可变的 :class:`Config` 数据类，并通过 :func:`get_config`
提供全局访问入口。所有配置项均支持通过环境变量（包括项目目录下的
``.env`` 文件）覆盖，环境变量名与 ``constants`` 中的常量去掉
``DEFAULT_`` 前缀同名，例如 ``MODEL_REPOSITORY_ID``、``MODEL_PATH``、
``THREAD_COUNT``、``HYBRID_MODE``、``CNN_WEIGHT``、``HASH_SIZE``。

环境变量的具体解析规则见 :mod:`environment`。
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

try:  # 作为包导入时使用相对导入
    from .constants import (
        DEFAULT_CNN_WEIGHT,
        DEFAULT_HASH_SIZE,
        DEFAULT_HYBRID_MODE,
        DEFAULT_MODEL_FILE,
        DEFAULT_MODEL_PATH,
        DEFAULT_MODEL_REPOSITORY_ID,
        DEFAULT_MODEL_REVISION,
        DEFAULT_THREAD_COUNT,
        PROJECT_ROOT,
    )
    from .environment import (
        read_environment_bool,
        read_environment_float,
        read_environment_int,
        read_environment_optional_path,
        read_environment_string,
    )
except ImportError:  # 直接把项目根目录加入 sys.path 后导入
    from constants import (
        DEFAULT_CNN_WEIGHT,
        DEFAULT_HASH_SIZE,
        DEFAULT_HYBRID_MODE,
        DEFAULT_MODEL_FILE,
        DEFAULT_MODEL_PATH,
        DEFAULT_MODEL_REPOSITORY_ID,
        DEFAULT_MODEL_REVISION,
        DEFAULT_THREAD_COUNT,
        PROJECT_ROOT,
    )
    from environment import (
        read_environment_bool,
        read_environment_float,
        read_environment_int,
        read_environment_optional_path,
        read_environment_string,
    )


# 在模块首次导入时加载 .env 文件。先加载项目根目录下的文件，再按
# python-dotenv 的默认规则从当前工作目录向上查找；已存在的环境变量
# 优先级更高，不会被 .env 覆盖。
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()


# 模块加载时统一解析一次环境变量，作为 Config 各字段的默认值。
_MODEL_REPOSITORY_ID: str = read_environment_string(
    "MODEL_REPOSITORY_ID", DEFAULT_MODEL_REPOSITORY_ID
)
_MODEL_FILE: str = read_environment_string("MODEL_FILE", DEFAULT_MODEL_FILE)
_MODEL_REVISION: str = read_environment_string("MODEL_REVISION", DEFAULT_MODEL_REVISION)
_MODEL_PATH: Path | None = read_environment_optional_path("MODEL_PATH")
if _MODEL_PATH is None and DEFAULT_MODEL_PATH is not None:
    _MODEL_PATH = Path(DEFAULT_MODEL_PATH).expanduser().resolve()
_THREAD_COUNT: int = read_environment_int("THREAD_COUNT", DEFAULT_THREAD_COUNT)
_HYBRID_MODE: bool = read_environment_bool("HYBRID_MODE", DEFAULT_HYBRID_MODE)
_CNN_WEIGHT: float = read_environment_float("CNN_WEIGHT", DEFAULT_CNN_WEIGHT)
_HASH_SIZE: int = read_environment_int("HASH_SIZE", DEFAULT_HASH_SIZE)


@dataclass(frozen=True)
class Config:
    """项目全局配置。

    所有字段的默认值来自环境变量，环境变量未设置时回退到 ``constants``
    中定义的常量。该类为不可变对象，请通过 :func:`get_config` 获取全局
    唯一实例。
    """

    model_repository_id: str = field(
        default=_MODEL_REPOSITORY_ID,
        metadata={"description": "魔搭社区的模型仓库 ID。"},
    )

    model_file: str = field(
        default=_MODEL_FILE,
        metadata={"description": "模型仓库中的模型文件路径。"},
    )

    model_revision: str = field(
        default=_MODEL_REVISION,
        metadata={"description": "模型仓库的 revision（分支、标签或提交号）。"},
    )

    model_path: Path | None = field(
        default=_MODEL_PATH,
        metadata={"description": "显式指定的本地模型文件路径；为空时自动获取。"},
    )

    thread_count: int = field(
        default=_THREAD_COUNT,
        metadata={"description": "TFLite 推理使用的线程数。"},
    )

    hybrid_mode: bool = field(
        default=_HYBRID_MODE,
        metadata={"description": "是否默认输出 hybrid（CNN + pHash/dHash）embedding。"},
    )

    cnn_weight: float = field(
        default=_CNN_WEIGHT,
        metadata={"description": "hybrid embedding 中 CNN 分量的权重（0 到 1）。"},
    )

    hash_size: int = field(
        default=_HASH_SIZE,
        metadata={"description": "pHash/dHash 的边长，总位数为边长的平方。"},
    )

    def __post_init__(self) -> None:
        """校验配置项的取值范围。

        Raises:
            ValueError: 线程数或哈希边长不是正整数，或 CNN 权重不在 0 到 1
                之间。
        """
        if self.thread_count <= 0:
            raise ValueError("THREAD_COUNT 必须是正整数")
        if self.hash_size <= 0:
            raise ValueError("HASH_SIZE 必须是正整数")
        if not 0.0 <= self.cnn_weight <= 1.0:
            raise ValueError("CNN_WEIGHT 必须在 0 到 1 之间")


@lru_cache(maxsize=1)
def get_config() -> Config:
    """获取全局唯一的 :class:`Config` 实例。

    该函数通过 :func:`functools.lru_cache` 保证进程内仅创建一次配置对象，
    多次调用返回同一个实例，避免重复读取环境变量。

    Returns:
        全局配置单例。

    Raises:
        ValueError: 某个配置项超出有效范围。
    """
    return Config()
