"""项目常量定义。

本模块集中定义模型位置、推理参数和判定阈值等常量。``config`` 模块会
导入其中以 ``DEFAULT_`` 开头的常量，作为全局配置各字段的默认值；业务
代码统一通过全局配置对象读取可配置项，模型标定得到的阈值则直接从包
的顶层导出。

常量命名约定：

- ``DEFAULT_`` 前缀表示可被同名（去掉前缀）环境变量覆盖的默认值；
- 不带前缀的常量表示模型的固有参数，不随环境变化。
"""

from pathlib import Path


# 项目根目录（doc-embed/）
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# 魔搭社区上的模型仓库 ID
DEFAULT_MODEL_REPOSITORY_ID: str = "JerryWuModelScope/doc-embed"

# 模型仓库中的默认模型文件（相对于仓库根目录）
DEFAULT_MODEL_FILE: str = "doc_embed_int8.tflite"

# 模型仓库的默认 revision
DEFAULT_MODEL_REVISION: str = "master"

# 显式指定的本地模型文件；为 None 时从魔搭缓存或远程仓库获取
DEFAULT_MODEL_PATH: Path | None = None

# TFLite 推理线程数
DEFAULT_THREAD_COUNT: int = 1

# 是否默认使用 hybrid（CNN + pHash/dHash）embedding
DEFAULT_HYBRID_MODE: bool = True

# hybrid embedding 中 CNN 分量的权重；其余权重平均分给 pHash 和 dHash
DEFAULT_CNN_WEIGHT: float = 0.9

# pHash/dHash 的边长；16 表示 16x16 = 256 bit
DEFAULT_HASH_SIZE: int = 16

# 模型的输入图片边长（224x224 RGB）
DEFAULT_MODEL_INPUT_IMAGE_SIZE: int = 224

# CNN embedding 的维度
EMBEDDING_DIMENSION: int = 128

# CNN-only 模式的余弦相似度阈值（在验证集上按 F1 标定）
CNN_SIMILARITY_THRESHOLD: float = 0.63936

# CNN + pHash/dHash soft 融合的余弦相似度阈值
HYBRID_SIMILARITY_THRESHOLD: float = 0.6535

# strict 模式：CNN 高置信直接通过，低置信时要求 hash 校验
STRICT_CNN_HIGH: float = 0.9
STRICT_CNN_LOW: float = 0.45
STRICT_HASH_MINIMUM: float = 0.65

# 向量距离计算方法；向量均已 L2 归一化，余弦距离与相似度一一对应
DISTANCE_METRIC: str = "cosine"

# 默认模式（hybrid）的“相同”相似度阈值
SAME_THRESHOLD: float = HYBRID_SIMILARITY_THRESHOLD

# 余弦距离阈值（距离 = 1 - 余弦相似度）
HYBRID_DIFFERENCE_THRESHOLD: float = 1.0 - HYBRID_SIMILARITY_THRESHOLD
CNN_DIFFERENCE_THRESHOLD: float = 1.0 - CNN_SIMILARITY_THRESHOLD

# 默认距离阈值：默认输出 hybrid embedding，因此等于 hybrid 距离阈值
DIFFERENCE_THRESHOLD: float = HYBRID_DIFFERENCE_THRESHOLD

# hybrid embedding 的维度：CNN + 1 个常数坐标 + pHash + dHash
HYBRID_EMBEDDING_DIMENSION: int = (
    EMBEDDING_DIMENSION + 1 + 2 * DEFAULT_HASH_SIZE * DEFAULT_HASH_SIZE
)

# TFLite flatbuffer 文件的魔数（位于文件第 4 至第 8 字节）
TFLITE_MAGIC: bytes = b"TFL3"

# 有效 TFLite 文件的最小长度（字节）
MINIMUM_MODEL_SIZE: int = 1024
