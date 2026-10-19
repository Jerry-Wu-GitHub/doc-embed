# doc-embed

面向绘本、书页和插画图片的轻量向量化库。输入图片字节，输出 L2 归一化的
embedding 向量；内容相同或相近的图片，向量余弦相似度高。

默认使用 **hybrid 模式**：把 128 维 CNN embedding 与 16×16 的 pHash、
dHash 位向量拼接成一个 641 维向量。两个 hybrid 向量的余弦相似度严格等于
CNN 与感知哈希的 soft 融合分数，因此可以直接用余弦距离做匹配，也可以使用
内置的 strict 两阶段判定。

## 特性

- 输入图片字节，输出 `numpy.ndarray` 向量。
- 默认 hybrid embedding（641 维），也支持纯 CNN embedding（128 维）。
- 模型托管在魔搭社区，首次使用自动下载，之后直接命中本地缓存。
- 支持环境变量和 `.env` 文件覆盖模型位置、线程数、融合权重等配置。
- 公开 API 提供 `embed_image`、`compare_images`、`HybridMatcher`，支持 soft / strict 判定。

## 模型

| 项目 | 值 |
| --- | --- |
| 魔搭模型 ID | [`JerryWuModelScope/doc-embed`](https://www.modelscope.cn/models/JerryWuModelScope/doc-embed) |
| 默认文件 | `doc_embed_int8.tflite` |
| 输入 | `uint8 [1, 224, 224, 3]`，RGB |
| 输出 | `int8 [1, 128]`，反量化后做 L2 归一化 |
| 参数量 / 体积 | 144,720 / 191,360 字节 |
| CNN-only 相似度阈值 | `0.63936` |
| hybrid soft 相似度阈值 | `0.6535` |
| strict hash 最低相似度 | `0.65` |

模型查找顺序：

1. 配置项 `MODEL_PATH`（显式指定本地文件）；
2. ModelScope 本地缓存；
3. 缓存不存在时通过 `modelscope` 自动下载。

## 安装

需要 Python 3.10 或更高版本。

```bash
git clone https://github.com/Jerry-Wu-GitHub/doc-embed.git doc_embed
cd doc_embed

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

python -m pip install -r requirements.txt
```

> 目录名建议使用下划线 `doc_embed`，这样可以直接 `import doc_embed`。
> 如果克隆成了 `doc-embed`，请把目录重命名为 `doc_embed`；也可以把项目
> 根目录加入 `sys.path`，直接导入其中的模块用于运行 demo。

## 快速开始

```python
from doc_embed import embed_image, METRIC, DIFFERENCE_THRESHOLD

with open("page.jpg", "rb") as file:
    image_bytes = file.read()

# 默认 hybrid embedding，形状为 (641,)，L2 范数为 1
vector = embed_image(image_bytes)

with open("another.jpg", "rb") as file:
    other_vector = embed_image(file.read())

# 向量已 L2 归一化，余弦相似度就是点积
similarity = float(vector @ other_vector)
distance = 1.0 - similarity

print(METRIC)                    # cosine
print(similarity, distance)
print(distance < DIFFERENCE_THRESHOLD)  # 是否判定为同一张图片
```

只使用 128 维 CNN embedding：

```python
vector = embed_image(image_bytes, hybrid_mode=False)

# CNN-only 使用另一组阈值
from doc_embed import CNN_DIFFERENCE_THRESHOLD
```

直接比较两张图片（推荐用于 soft/strict 判定）：

```python
from doc_embed import compare_images

with open("a.jpg", "rb") as file:
    image_a = file.read()
with open("b.jpg", "rb") as file:
    image_b = file.read()

result = compare_images(image_a, image_b, mode="strict")
print(result.cnn_similarity)
print(result.average_hash_similarity)
print(result.hybrid_score)
print(result.same)
```

## API

### `embed_image`

```python
def embed_image(
    image_bytes: bytes | bytearray | memoryview,
    *,
    hybrid_mode: bool | None = None,
    cnn_weight: float | None = None,
    hash_size: int | None = None,
) -> numpy.ndarray
```

计算图片的 embedding。

- `image_bytes`：图片原始字节，支持 JPG、PNG、BMP、WebP 等 Pillow 可解码格式。
- `hybrid_mode`：是否使用 hybrid 模式；为空时读取全局配置，默认 `True`。
- `cnn_weight`：hybrid 模式中 CNN 分量的权重；为空时读取全局配置，默认 `0.9`。
- `hash_size`：pHash/dHash 的边长；为空时读取全局配置，默认 `16`。

返回 L2 归一化的 `float32` 向量：

- hybrid 模式默认形状为 `(641,)`；
- CNN-only 模式形状为 `(128,)`。

异常约定：

- `TypeError`：输入不是 bytes / bytearray / memoryview；
- `ValueError`：输入为空、无法解码为图片，或参数越界；
- `FileNotFoundError` / `RuntimeError`：模型文件缺失、无效或下载失败。

### `compare_images`

```python
def compare_images(
    image_bytes_a: bytes | bytearray | memoryview,
    image_bytes_b: bytes | bytearray | memoryview,
    *,
    mode: Literal["soft", "strict"] = "soft",
    cnn_weight: float | None = None,
    threshold: float | None = None,
    cnn_high: float | None = None,
    cnn_low: float | None = None,
    hash_minimum: float | None = None,
    hash_size: int | None = None,
) -> SimilarityResult
```

比较两张图片并返回 hybrid 相似度与是否相同的判定。`mode="soft"` 使用
CNN 与 hash 的加权融合分数；`mode="strict"` 在融合分数之外还要求 hash
校验通过，适合手机拍照、近重复页面等高精度场景。

参数为空时使用全局配置和模型标定阈值；`hash_size`、`cnn_weight` 等参数
可以临时覆盖默认值。

返回的 `SimilarityResult` 包含：

| 字段 | 说明 |
| --- | --- |
| `cnn_similarity` | 两个 CNN embedding 的余弦相似度 |
| `phash_similarity` | pHash 位数组相似度 |
| `dhash_similarity` | dHash 位数组相似度 |
| `average_hash_similarity` | pHash 与 dHash 相似度的平均值 |
| `hybrid_score` | CNN 与 hash 加权融合后的分数 |
| `same` | 按指定模式判定的“是否相同” |
| `mode` | 实际使用的判定模式 |

### `HybridMatcher`

```python
from doc_embed import HybridMatcher

matcher = HybridMatcher(gallery_bytes, mode="strict")
new_index = matcher.add(another_gallery_bytes)  # 增量添加候选图片
result = matcher.match(query_bytes)             # (索引, SimilarityResult) 或 None
```

在候选图片集合中做增量匹配。匹配器会缓存每张候选图片的特征，适合服务端
把图库预热一次后，对多张查询图片重复匹配。

### `get_config`

```python
from doc_embed import get_config

config = get_config()
print(config.model_repository_id)
print(config.hybrid_mode, config.cnn_weight, config.hash_size)
```

返回进程内唯一的只读 `Config` 实例。配置在首次调用时解析，之后复用。

### 导出的常量

| 常量 | 默认值 | 说明 |
| --- | ---: | --- |
| `METRIC` | `"cosine"` | 距离计算方法 |
| `SAME_THRESHOLD` | `0.6535` | 默认 hybrid 模式的相似度阈值 |
| `DIFFERENCE_THRESHOLD` | `0.3465` | 默认 hybrid 模式的余弦距离阈值 |
| `CNN_SIMILARITY_THRESHOLD` | `0.63936` | CNN-only 相似度阈值 |
| `CNN_DIFFERENCE_THRESHOLD` | `0.36064` | CNN-only 余弦距离阈值 |
| `HYBRID_SIMILARITY_THRESHOLD` | `0.6535` | hybrid soft 相似度阈值 |
| `HYBRID_DIFFERENCE_THRESHOLD` | `0.3465` | hybrid soft 余弦距离阈值 |
| `EMBEDDING_DIMENSION` | `128` | CNN embedding 维度 |
| `HYBRID_EMBEDDING_DIMENSION` | `641` | 默认 hybrid embedding 维度 |
| `STRICT_CNN_HIGH` | `0.9` | strict：CNN 高置信阈值 |
| `STRICT_CNN_LOW` | `0.45` | strict：CNN 低置信下限 |
| `STRICT_HASH_MINIMUM` | `0.65` | strict：hash 平均相似度下限 |

距离与相似度满足 `distance = 1 - similarity`。

## hybrid 模式

hybrid embedding 由三部分组成：

1. CNN embedding `e`（128 维，L2 归一化）；
2. pHash 位向量 `p`（默认 256 bit，映射为 ±1）；
3. dHash 位向量 `d`（默认 256 bit，映射为 ±1）。

设 `W = cnn_weight`、`h = (1 - W) / 2`、`n = hash_size²`，拼接方式为：

```text
H = concat( sqrt(W) * e, sqrt(h), sqrt(h / (2n)) * p, sqrt(h / (2n)) * d )
```

这样 `H` 的 L2 范数为 1，且：

```text
H_a · H_b = W * s_cnn + h * s_phash + h * s_dhash
          = W * s_cnn + (1 - W) * (0.5 * s_phash + 0.5 * s_dhash)
```

正好等于 soft 融合分数，因此可以直接用余弦距离比较。

strict 模式在 soft 分数之外增加门控：

```text
s_cnn >= 0.9
或 (s_cnn >= 0.45 且 s_hash >= 0.65 且 s_hybrid >= 0.6535)
```

高置信 CNN 匹配直接通过；CNN 处于模糊区间时，要求 hash 校验通过。

> soft/strict 是**两张图片比较时**的判定模式，不影响单张图片的 embedding。
> `embed_image` 在两种模式下返回相同的 hybrid 向量；strict 还需要 pHash/dHash
> 原始位进行门控，因此由 `compare_images()` 或 `HybridMatcher` 完成。

## 配置

所有配置都支持环境变量和项目根目录下的 `.env` 文件。环境变量名与常量
去掉 `DEFAULT_` 前缀同名；真实环境变量优先于 `.env`。

示例 `.env`：

```dotenv
# 显式指定本地模型；为空时自动从魔搭缓存或远程仓库获取
MODEL_PATH=

# 模型仓库设置
MODEL_REPOSITORY_ID=JerryWuModelScope/doc-embed
MODEL_FILE=doc_embed_int8.tflite
MODEL_REVISION=master

# 推理与 hybrid 参数
THREAD_COUNT=1
HYBRID_MODE=true
CNN_WEIGHT=0.9
HASH_SIZE=16
```

| 配置项 | 环境变量 | 默认值 |
| --- | --- | --- |
| 模型仓库 ID | `MODEL_REPOSITORY_ID` | `JerryWuModelScope/doc-embed` |
| 模型文件 | `MODEL_FILE` | `doc_embed_int8.tflite` |
| 模型 revision | `MODEL_REVISION` | `master` |
| 本地模型路径 | `MODEL_PATH` | 空（自动获取） |
| 推理线程数 | `THREAD_COUNT` | `1` |
| 是否默认 hybrid | `HYBRID_MODE` | `true` |
| CNN 权重 | `CNN_WEIGHT` | `0.9` |
| 哈希边长 | `HASH_SIZE` | `16` |

布尔值支持 `1/0`、`true/false`、`yes/no`、`on/off`。

## Demo

在项目根目录运行：

```bash
# 使用 test_samples 示例集，默认 hybrid strict 判定
python -m demo

# 改用 soft 融合判定
python -m demo --soft

# 只使用 128 维 CNN embedding
python -m demo --cnn-only

# 指定示例集目录
python -m demo --samples path/to/test_samples

# 直接比较多张图片
python -m demo page_a.jpg page_b.jpg page_c.jpg
```

也可以直接运行脚本：

```bash
python demo.py
```

首次运行会自动下载模型，之后直接使用本地缓存。`test_samples` 包含
`model`（候选）和 `real`（查询）两组图片，demo 会打印距离矩阵、匹配结果、
耗时和正确率。

参考结果（本仓库自带样例，strict 模式）：

| 样例组 | 正确率 |
| --- | ---: |
| 哆啦A梦 | 10/10 |
| 波西和皮普 | 13/16 |

## 线程安全

模型解释器在进程内只加载一次，所有推理调用通过锁串行化，因此可以在
多线程环境中复用。模型加载本身也使用双重检查锁，避免并发首次调用重复
创建解释器。

## 许可

Apache License 2.0，详见 [LICENSE](LICENSE)。
