#!/usr/bin/env python3
"""doc-embed 命令行示例。

在项目根目录下运行::

    python -m demo
    python -m demo --cnn-only
    python -m demo --soft

也可以直接运行脚本::

    python demo.py
    python demo.py page_a.jpg page_b.jpg
    python demo.py test_samples --soft

不传图片路径时，默认读取 ``test_samples`` 目录，按 ``model``/``real``
两组图片计算距离、匹配结果和正确率。默认使用 hybrid 的 strict 判定；
传入 ``--soft`` 可以改用 soft 融合判定。首次运行会自动从魔搭社区下载
模型，之后直接使用本地缓存。
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parent
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

try:  # 作为包的一部分导入时使用相对导入
    from .config import get_config
    from .demo_pairs import run_pair_demo
    from .demo_samples import run_sample_suite
except ImportError:  # 作为脚本或模块直接运行
    from config import get_config
    from demo_pairs import run_pair_demo
    from demo_samples import run_sample_suite


def main() -> int:
    """运行命令行示例。

    Returns:
        进程退出码；0 表示成功，1 表示没有找到可处理的图片。
    """
    argument_parser = argparse.ArgumentParser(description="doc-embed 手动测试")
    argument_parser.add_argument(
        "images",
        nargs="*",
        help="图片文件或目录；不传时使用 test_samples 示例集",
    )
    argument_parser.add_argument(
        "--samples",
        default=str(PROJECT_DIRECTORY / "test_samples"),
        help="示例集根目录（默认 test_samples）",
    )
    argument_parser.add_argument(
        "--cnn-only",
        action="store_true",
        help="只使用 128 维 CNN embedding，不拼接 pHash/dHash",
    )
    argument_parser.add_argument(
        "--soft",
        action="store_true",
        help="使用 soft 融合判定（默认使用 hybrid strict 判定）",
    )
    arguments = argument_parser.parse_args()

    config = get_config()
    hybrid_mode = config.hybrid_mode and not arguments.cnn_only
    strict_mode = hybrid_mode and not arguments.soft
    print(f"模型仓库：{config.model_repository_id}/{config.model_file}")
    print(f"hybrid 模式：{'开启' if hybrid_mode else '关闭'}")
    print(f"判定模式：{'strict' if strict_mode else 'soft'}")
    print(f"CNN 权重：{config.cnn_weight}，哈希边长：{config.hash_size}\n")

    if arguments.images:
        return run_pair_demo(
            arguments.images,
            hybrid_mode=hybrid_mode,
            strict=strict_mode,
        )
    return run_sample_suite(
        Path(arguments.samples),
        hybrid_mode=hybrid_mode,
        strict=strict_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
