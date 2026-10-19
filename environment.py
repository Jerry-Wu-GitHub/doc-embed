"""环境变量读取工具。

本模块提供一组带默认值和容错处理的读取函数，供 ``config`` 模块在导入时
解析环境变量。解析失败时统一回退到调用方提供的默认值，避免因为一个
非法配置项导致整个模块无法导入。

支持的解析类型：

- 字符串：未设置或为空字符串时回退到默认值；
- 整数、浮点数：解析失败时回退到默认值；
- 布尔值：支持 ``1/0``、``true/false``、``yes/no``、``on/off``；
- 路径：会展开 ``~`` 并解析为绝对路径。
"""

import os
from pathlib import Path


def read_environment_string(name: str, default: str) -> str:
    """从环境变量读取字符串值。

    Args:
        name: 环境变量名。
        default: 环境变量未设置或为空字符串时返回的默认值。

    Returns:
        环境变量对应的字符串值；若环境变量未设置或为空则返回默认值。
    """
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def read_environment_int(name: str, default: int) -> int:
    """从环境变量读取整数值。

    Args:
        name: 环境变量名。
        default: 环境变量未设置、为空或解析失败时返回的默认值。

    Returns:
        解析后的整数值；无法解析时返回默认值。
    """
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def read_environment_float(name: str, default: float) -> float:
    """从环境变量读取浮点数值。

    Args:
        name: 环境变量名。
        default: 环境变量未设置、为空或解析失败时返回的默认值。

    Returns:
        解析后的浮点数值；无法解析时返回默认值。
    """
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_environment_bool(name: str, default: bool) -> bool:
    """从环境变量读取布尔值。

    Args:
        name: 环境变量名。
        default: 环境变量未设置、为空或取值无法识别时返回的默认值。

    Returns:
        解析后的布尔值；无法识别时返回默认值。
    """
    value = os.getenv(name)
    if value is None or value == "":
        return default
    normalized_value = value.strip().lower()
    if normalized_value in ("1", "true", "yes", "on"):
        return True
    if normalized_value in ("0", "false", "no", "off"):
        return False
    return default


def read_environment_optional_path(name: str) -> Path | None:
    """从环境变量读取可选的路径。

    Args:
        name: 环境变量名。

    Returns:
        环境变量已设置且非空时返回展开 ``~`` 后的绝对路径；否则返回 None。
    """
    raw_path = os.getenv(name)
    if raw_path is None or raw_path == "":
        return None
    return Path(raw_path).expanduser().resolve()
