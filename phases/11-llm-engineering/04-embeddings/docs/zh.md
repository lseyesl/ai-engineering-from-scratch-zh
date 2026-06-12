# 嵌入与向量表示

> 文本是离散的。数学是连续的。每次你让 LLM 查找"相似"文档、比较含义或进行超越关键词的搜索时，你都在依赖这两个世界之间的桥梁。那座桥梁就是嵌入。如果你不理解嵌入，你就不理解现代 AI。你只是在使用它。

**类型：** Build
**语言：** Python
**前置要求：** 阶段 11，课程 01（提示词工程）
**预计时间：** ~75 分钟
**关联：** 阶段 5 · 22（嵌入模型深度解析）涵盖了稠密 vs 稀疏 vs 多向量、Matryoshka 截断和按轴模型选择。本课程侧重于生产管道（向量数据库、HNSW、相似度计算）。在选择模型之前请阅读阶段 5 · 22。

## 学习目标

- 使用 API 提供商和开源模型生成文本嵌入，并计算它们之间的余弦相似度
- 解释为什么嵌入解决了关键词搜索无法处理的词汇不匹配问题
- 构建一个语义搜索索引，按含义而非精确关键词匹配检索文档
- 使用检索基准（precision@k、recall）评估嵌入质量，并为你的任务选择合适的嵌入模型

## 问题

你有 10,000 个支持工单。客户写道"我的付款没有通过。"你需要找到类似的过往工单。关键词搜索找到包含"付款"和"没有通过"的工单。它漏掉了"交易失败"、"扣款被拒绝"和"账单错误"。这些工单用完全不同的词语描述了完全相同的问题。

这就是词汇不匹配问题。人类语言有几十种方式说同一件事。关键词搜索将每个词视为独立符号，没有含义。它无法知道"被拒绝"和"没有通过"指的是同一概念。

你需要一种文本表示，其中含义（而非拼写）决定了相似性。你需要一种方式，将"我的付款没有通过"和"交易被拒绝"在某些数学空间中放置在一起，同时将"我的付款按时到达"推得很远，尽管它们共享词语"付款"。

这种表示就是嵌入。

## 概念

### 什么是嵌入？

嵌入是一个稠密的浮点数向量，表示文本的含义。"稠密"这个词很重要——每个维度都携带信息，而稀疏表示（词袋、TF-IDF）中大多数维度为零。

"The cat sat on the mat" 变成类似 `[0.023, -0.041, 0.087, ..., 0.012]` 的东西——根据模型不同，包含 768 到 3072 个数字的列表。这些数字编码了含义。你从不直接检查它们。你比较它们。

### Word2Vec 突破

2013 年，Tomas Mikolov 及其在 Google 的同事发表了 Word2Vec。核心洞见：训练一个神经网络从上下文预测一个词（或从一个词预测上下文），隐藏层权重就变成了有意义的向量表示。

著名的结果：

```
king - man + woman = queen
```

词嵌入上的向量算术捕捉语义关系。从"man"到"woman"的方向大致与从"king"到"queen"的方向相同。这是该领域认识到几何可以编码含义的时刻。

Word2Vec 产生 300 维向量。每个词无论上下文如何都只有一个向量。"river bank"和"bank account"中的"bank"有相同的嵌入。这个限制驱动了接下来十年的研究。

### 从词到句子

词嵌入表示单个 token。生产系统需要嵌入整个句子、段落或文档。出现了四种方法：

**平均**：取句子中所有词向量的均值。廉价、有损，对于短文本出奇地好。完全丢失了词序——"dog bites man"和"man bites dog"得到相同的嵌入。

**CLS token**：transformer 模型（BERT，2018）输出一个特殊的 [CLS] token 嵌入，表示整个输入。比平均好，但 [CLS] token 是为下一句预测训练的，而非相似性。

**对比学习**：显式训练模型将相似对推近，不相似对推远。Sentence-BERT（Reimers & Gurevych，2019）使用了这种方法，并成为现代嵌入模型的基础。给定"如何重置密码？"和"我需要更改密码"，模型学习这些应该有几乎相同的向量。

**指令微调嵌入**：最新方法。像 E5 和 GTE 这样的模型接受任务前缀（"search_query:"、"search_document:"），告诉模型要产生什么类型的嵌入。这让一个模型服务于多个任务。

```mermaid
graph LR
    subgraph "2013: Word2Vec"
        W1["king"] --> V1["[0.2, -0.1, ...]"]
        W2["queen"] --> V2["[0.3, -0.2, ...]"]
    end

    subgraph "2019: Sentence-BERT"
        S1["如何重置密码？"] --> E1["[0.04, 0.12, ...]"]
        S2["我需要更改密码"] --> E2["[0.05, 0.11, ...]"]
    end

    subgraph "2024: 指令微调"
        I1["search_query: 密码重置"] --> T1["[0.08, 0.09, ...]"]
        I2["search_document: 要重置密码，点击..."] --> T2["[0.07, 0.10, ...]"]
    end
```

### 现代嵌入模型

市场已稳定在少数几个生产级选项上（截至 2026 年初的 MTEB 分数，MTEB v2）：

| 模型 | 提供商 | 维度 | MTEB | 上下文 | 成本 / 1M tokens |
|-------|----------|-----------|------|---------|------------------|
| Gemini Embedding 2 | Google | 3072（Matryoshka） | 67.7（检索） | 8192 | $0.15 |
| embed-v4 | Cohere | 1024（Matryoshka） | 65.2 | 128K | $0.12 |
| voyage-4 | Voyage AI | 1024/2048（Matryoshka） | 66.8 | 32K | $0.12 |
| text-embedding-3-large | OpenAI | 3072（Matryoshka） | 64.6 | 8192 | $0.13 |
| text-embedding-3-small | OpenAI | 1536（Matryoshka） | 62.3 | 8192 | $0.02 |
| BGE-M3 | BAAI | 1024（稠密+稀疏+ColBERT） | 63.0 多语言 | 8192 | 开源权重 |
| Qwen3-Embedding | 阿里巴巴 | 4096（Matryoshka） | 66.9 | 32K | 开源权重 |
| Nomic-embed-v2 | Nomic | 768（Matryoshka） | 63.1 | 8192 | 开源权重 |

MTEB（Massive Text Embedding Benchmark）v2 涵盖 100+ 个任务，涵盖检索、分类、聚类、重排序和摘要。分数越高越好。到 2026 年，开源权重模型（Qwen3-Embedding、BGE-M3）在大多数指标上匹敌或超越封闭托管模型。Gemini Embedding 2 在纯检索方面领先；Voyage/Cohere 在特定领域（金融、法律、代码）领先。在承诺使用前，始终在你自己的查询上进行基准测试。

### 相似度指标

给定两个嵌入向量，有三种方式衡量它们的相似程度：

**余弦相似度**：两个向量之间夹角的余弦。范围从 -1（相反）到 1（相同方向）。忽略幅度——一个 10 词的句子和一个 500 词的文档如果指向相同方向可以得 1.0。这是 90% 用例的默认选择。

```
cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
```

**点积**：两个向量的原始内积。当向量被归一化（单位长度）时，与余弦相似度相同。计算更快。OpenAI 的嵌入已归一化，因此点积和余弦给出相同的排序。

```
dot(a, b) = sum(a_i * b_i)
```

**欧几里得（L2）距离**：向量空间中的直线距离。越小 = 越相似。对幅度差异敏感。当空间中的绝对位置重要而不仅仅是方向时使用。

```
L2(a, b) = sqrt(sum((a_i - b_i)^2))
```

何时使用哪种：

| 指标 | 何时使用 | 何时避免 |
|--------|----------|------------|
| 余弦相似度 | 比较不同长度的文本；大多数检索任务 | 幅度携带信息时 |
| 点积 | 嵌入已经归一化；追求最大速度 | 向量的幅度不同时 |
| 欧几里得距离 | 聚类；空间最近邻问题 | 比较长度差异很大的文档时 |

### 向量数据库与 HNSW

暴力相似度搜索将查询与每个存储的向量进行比较。在 100 万个向量、1536 维时，每次查询需要 15 亿次乘加运算。太慢了。

向量数据库使用近似最近邻（ANN）算法解决这个问题。主导算法是 HNSW（分层可导航小世界）：

1. 构建一个向量的多层图
2. 顶层是稀疏的——远距离簇之间的长程连接
3. 底层是稠密的——附近向量之间的细粒度连接
4. 搜索从顶层开始，贪婪地向下细化
5. 以 O(log n) 时间而非 O(n) 的时间返回近似 top-k 结果

HNSW 以微小的准确率损失（通常 95-99% 召回率）换取巨大的速度提升。在 1000 万向量时，暴力搜索需要数秒。HNSW 需要毫秒。

```mermaid
graph TD
    subgraph "HNSW 层"
        L2["第 2 层（稀疏）"] -->|"长跳"| L1["第 1 层（中等）"]
        L1 -->|"短跳"| L0["第 0 层（稠密，所有向量）"]
    end

    Q["查询向量"] -->|"从顶层进入"| L2
    L0 -->|"最近邻"| R["Top-k 结果"]
```

生产选项：

| 数据库 | 类型 | 最适合 | 最大规模 |
|----------|------|----------|-----------|
| Pinecone | 托管 SaaS | 零运维生产 | 数十亿 |
| Weaviate | 开源 | 自托管，混合搜索 | 1 亿+ |
| Qdrant | 开源 | 高性能，过滤 | 1 亿+ |
| ChromaDB | 嵌入式 | 原型开发，本地开发 | 100 万 |
| pgvector | Postgres 扩展 | 已在用 Postgres | 1000 万 |
| FAISS | 库 | 进程内，研究 | 10 亿+ |

### 分块策略

文档太长，无法作为单个向量嵌入。一份 50 页的 PDF 涵盖数十个主题——它的嵌入变成了所有内容的平均值，对任何特定内容都不相似。你将文档分割成块并分别嵌入每个块。

**固定大小分块**：每 N 个 token 分割一次，有 M 个 token 的重叠。简单且可预测。在文档没有清晰结构时效果良好。128 token 的块，12 token 的重叠：块 1 是 token 0-127，块 2 是 token 115-242。

**基于句子的分块**：在句子边界分割，将句子分组直到达到 token 限制。每个块至少包含一个完整句子。比固定大小好，因为你永远不会把一个想法切成两半。

**递归分块**：首先尝试在最大的边界（章节标题）分割。如果仍然太大，尝试段落边界。然后是句子边界。然后是字符限制。这是 LangChain 的 `RecursiveCharacterTextSplitter`，适用于混合格式的语料库。

**语义分块**：嵌入每个句子，然后分组嵌入相似的连续句子。当嵌入相似度低于阈值时，开始一个新的块。昂贵（需要单独嵌入每个句子）但产生最连贯的块。

| 策略 | 复杂度 | 质量 | 最适合 |
|----------|-----------|---------|----------|
| 固定大小 | 低 | 尚可 | 非结构化文本，日志 |
| 基于句子 | 低 | 好 | 文章，邮件 |
| 递归 | 中 | 好 | Markdown，HTML，混合文档 |
| 语义 | 高 | 最佳 | 关键检索质量 |

大多数系统的甜点：256-512 token 的块，50 token 重叠。

### 双编码器 vs 交叉编码器

双编码器独立嵌入查询和文档，然后比较向量。快速——你嵌入查询一次，然后与预先计算的文档嵌入比较。这就是你用于检索的方式。

交叉编码器将查询和文档作为单个输入，输出相关性分数。慢——它通过完整模型处理每个查询-文档对。但远比双编码器准确，因为它可以同时关注查询和文档 token 之间的交互。

生产模式：双编码器检索 top-100 候选，交叉编码器将它们重新排序为 top-10。这就是检索-然后-重排序管道。

```mermaid
graph LR
    Q["查询"] --> BE["双编码器：嵌入查询"]
    BE --> VS["向量搜索：前 100"]
    VS --> CE["交叉编码器：重排序"]
    CE --> R["前 10 结果"]
```

重排序模型：Cohere Rerank 3.5（每 1000 次查询 $2），BGE-reranker-v2（免费，开源），Jina Reranker v2（免费，开源）。

### Matryoshka 嵌入

传统嵌入是全有或全无的。一个 1536 维向量使用 1536 个浮点数。你不能在不重新训练的情况下截断到 256 维。

Matryoshka 表示学习（Kusupati 等，2022）解决了这个问题。模型被训练为前 N 个维度捕捉最重要的信息，就像俄罗斯套娃一样。将 1536 维的 Matryoshka 嵌入截断到 256 维会损失一些准确率，但仍然可用。

OpenAI 的 text-embedding-3-small 和 text-embedding-3-large 通过 `dimensions` 参数支持 Matryoshka 截断。请求 256 维而非 1536 维可将存储减少 6 倍，在 MTEB 基准上准确率损失约 3-5%。

### 二值量化

一个以 float32 存储的 1536 维嵌入使用 6,144 字节。乘以 1000 万文档：仅向量就需要 61 GB。

二值量化将每个浮点数转换为单个位：正值变为 1，负值变为 0。存储从 6,144 字节降到 192 字节——32 倍减少。相似度使用汉明距离（计算不同位的数量）计算，CPU 可以在单条指令中完成。

准确率损失大约为检索召回率的 5-10%。常见模式：对百万级向量的初轮搜索使用二值量化，然后用全精度向量对 top-1000 重新评分。这让你以 32 倍更少的内存获得 95%+ 的全精度准确率。

```figure
cosine-similarity
```

## 构建它

我们从头构建一个语义搜索引擎。没有向量数据库。没有外部嵌入 API。纯 Python 加 numpy 处理数学计算。

### 步骤 1：文本分块

```python
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_by_sentences(text, max_chunk_tokens=200):
    sentences = text.replace("\n", " ").split(".")
    sentences = [s.strip() + "." for s in sentences if s.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_chunk_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```

### 步骤 2：从头构建嵌入

我们实现一个使用 TF-IDF 和 L2 归一化的简单稠密嵌入。这不是神经网络嵌入，但它遵循相同的契约：输入文本，输出固定大小的向量，相似的文本产生相似的向量。

```python
import math
import numpy as np
from collections import Counter

class SimpleEmbedder:
    def __init__(self):
        self.vocab = []
        self.idf = []
        self.word_to_idx = {}

    def fit(self, documents):
        vocab_set = set()
        for doc in documents:
            vocab_set.update(doc.lower().split())
        self.vocab = sorted(vocab_set)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        n = len(documents)
        self.idf = np.zeros(len(self.vocab))
        for i, word in enumerate(self.vocab):
            doc_count = sum(1 for doc in documents if word in doc.lower().split())
            self.idf[i] = math.log((n + 1) / (doc_count + 1)) + 1

    def embed(self, text):
        words = text.lower().split()
        count = Counter(words)
        total = len(words) if words else 1
        vec = np.zeros(len(self.vocab))
        for word, freq in count.items():
            if word in self.word_to_idx:
                tf = freq / total
                vec[self.word_to_idx[word]] = tf * self.idf[self.word_to_idx[word]]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
```

### 步骤 3：相似度函数

```python
def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def dot_product(a, b):
    return float(np.dot(a, b))


def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))
```

### 步骤 4：带暴力搜索的向量索引

```python
class VectorIndex:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add(self, vector, text, meta=None):
        self.vectors.append(vector)
        self.texts.append(text)
        self.metadata.append(meta or {})

    def search(self, query_vector, top_k=5, metric="cosine"):
        scores = []
        for i, vec in enumerate(self.vectors):
            if metric == "cosine":
                score = cosine_similarity(query_vector, vec)
            elif metric == "dot":
                score = dot_product(query_vector, vec)
            elif metric == "euclidean":
                score = -euclidean_distance(query_vector, vec)
            else:
                raise ValueError(f"未知指标：{metric}")
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            results.append({
                "text": self.texts[idx],
                "score": score,
                "metadata": self.metadata[idx],
                "index": idx
            })
        return results

    def size(self):
        return len(self.vectors)
```

### 步骤 5：语义搜索引擎

```python
class SemanticSearchEngine:
    def __init__(self, chunk_size=200, overlap=50):
        self.embedder = SimpleEmbedder()
        self.index = VectorIndex()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def index_documents(self, documents, source_names=None):
        all_chunks = []
        all_sources = []
        for i, doc in enumerate(documents):
            chunks = chunk_text(doc, self.chunk_size, self.overlap)
            all_chunks.extend(chunks)
            name = source_names[i] if source_names else f"doc_{i}"
            all_sources.extend([name] * len(chunks))
        self.embedder.fit(all_chunks)
        for chunk, source in zip(all_chunks, all_sources):
            vec = self.embedder.embed(chunk)
            self.index.add(vec, chunk, {"source": source})
        return len(all_chunks)

    def search(self, query, top_k=5, metric="cosine"):
        query_vec = self.embedder.embed(query)
        return self.index.search(query_vec, top_k, metric)

    def search_with_scores(self, query, top_k=5):
        results = self.search(query, top_k)
        return [
            {
                "text": r["text"][:200],
                "source": r["metadata"].get("source", "unknown"),
                "score": round(r["score"], 4)
            }
            for r in results
        ]
```

### 步骤 6：比较相似度指标

```python
def compare_metrics(engine, query, top_k=3):
    results = {}
    for metric in ["cosine", "dot", "euclidean"]:
        hits = engine.search(query, top_k=top_k, metric=metric)
        results[metric] = [
            {"score": round(h["score"], 4), "preview": h["text"][:80]}
            for h in hits
        ]
    return results
```

## 使用它

使用生产级嵌入 API 时，架构保持不变。只有嵌入器发生变化：

```python
from openai import OpenAI

client = OpenAI()

def openai_embed(texts, model="text-embedding-3-small", dimensions=None):
    kwargs = {"model": model, "input": texts}
    if dimensions:
        kwargs["dimensions"] = dimensions
    response = client.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]
```

使用 OpenAI 的 Matryoshka 截断——相同的模型，更少的维度，更低的存储：

```python
full = openai_embed(["语义搜索查询"], dimensions=1536)
compact = openai_embed(["语义搜索查询"], dimensions=256)
```

256 维向量使用 6 倍更少的存储。对于 1000 万文档，那是 10 GB 对比 61 GB。准确率损失在标准基准上约为 3-5%。

使用 Cohere 进行重排序：

```python
import cohere

co = cohere.ClientV2()

results = co.rerank(
    model="rerank-v3.5",
    query="退款政策是什么？",
    documents=["30 天内全额退款...", "90 天后不退款..."],
    top_n=3
)
```

使用无需 API 依赖的本地嵌入：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(["语义搜索查询", "另一个文档"])
```

我们构建的 VectorIndex 类适用于以上任何方法。替换嵌入函数，保留搜索逻辑。

## 交付物

本课程产出：
- `outputs/prompt-embedding-advisor.md` —— 一个为特定用例选择嵌入模型和策略的提示词
- `outputs/skill-embedding-patterns.md` —— 一个教授代理如何在生产中有效使用嵌入的技能

## 练习

1. **指标比较**：使用余弦相似度、点积和欧几里得距离对样本文档运行相同的 5 个查询。记录每种方法的 top-3 结果。哪些查询的指标不一致？为什么？

2. **分块大小实验**：以 50、100、200 和 500 词的块大小索引样本文档。对每种块大小运行 5 个查询，记录 top-1 相似度分数。绘制块大小与检索质量的关系图。找到较大块开始有害的临界点。

3. **Matryoshka 模拟**：构建一个产生 500 维向量的 SimpleEmbedder。截断到 50、100、200 和 500 维。测量每次截断后检索召回率如何下降。这模拟了 Matryoshka 行为，无需真正的训练技巧。

4. **二值量化**：从搜索引擎获取嵌入，将它们转换为二值（正数为 1，负数为 0），实现汉明距离搜索。将 top-10 结果与全精度余弦相似度进行比较。衡量重叠百分比。

5. **基于句子的分块**：将固定大小分块替换为 `chunk_by_sentences`。运行相同的查询并比较检索分数。尊重句子边界是否改善了结果？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| 嵌入 | "文本转数字" | 一个稠密向量，其中几何邻近性编码语义相似性 |
| Word2Vec | "最初的嵌入" | 2013 年模型，通过预测上下文词学习词向量；证明向量算术编码含义 |
| 余弦相似度 | "两个向量有多相似" | 向量之间夹角的余弦；1 = 方向相同，0 = 正交，-1 = 相反 |
| HNSW | "快速向量搜索" | 分层可导航小世界图——多层结构实现 O(log n) 近似最近邻搜索 |
| 双编码器 | "分别嵌入，快速比较" | 将查询和文档独立编码为向量；支持预计算和快速检索 |
| 交叉编码器 | "慢但准确的重排序器" | 通过完整模型联合处理查询-文档对；准确率更高，无法预计算 |
| Matryoshka 嵌入 | "可截断的向量" | 嵌入被训练为前 N 个维度捕捉最重要信息，支持可变大小存储 |
| 二值量化 | "1 位嵌入" | 将浮点向量转换为二值（仅符号位），存储减少 32 倍，配合汉明距离搜索 |
| 分块 | "分割文档以便嵌入" | 将文档分割为 256-512 token 的段，使每段可以被独立嵌入和检索 |
| 向量数据库 | "面向嵌入的搜索引擎" | 为存储向量和执行大规模近似最近邻搜索而优化的数据存储 |
| 对比学习 | "通过比较训练" | 将相似对的嵌入推近、不相似对的嵌入推远的训练方法 |
| MTEB | "嵌入基准" | Massive Text Embedding Benchmark——涵盖 8 个任务的 56 个数据集；比较嵌入模型的标准 |

## 延伸阅读

- Mikolov 等，"向量空间中词表示的高效估计"（2013）——以 king-queen 类比开始嵌入革命的 Word2Vec 论文
- Reimers & Gurevych，"Sentence-BERT：使用孪生 BERT 网络的句子嵌入"（2019）——如何训练用于句子级别相似性的双编码器，现代嵌入模型的基础
- Kusupati 等，"Matryoshka 表示学习"（2022）——OpenAI 为 text-embedding-3 采用的可变维度嵌入背后的技术
- Malkov & Yashunin，"使用分层可导航小世界图的高效稳健近似最近邻搜索"（2018）——HNSW 论文，大多数生产向量搜索背后的算法
- OpenAI 嵌入指南（platform.openai.com/docs/guides/embeddings）——text-embedding-3 模型的实践参考，包括 Matryoshka 维度缩减
- MTEB 排行榜（huggingface.co/spaces/mteb/leaderboard）——跨任务和语言比较所有嵌入模型的实时基准
- [Muennighoff 等，"MTEB：Massive Text Embedding Benchmark"（EACL 2023）](https://arxiv.org/abs/2210.07316) —— 定义了排行榜报告的 8 个任务类别（分类、聚类、配对分类、重排序、检索、语义文本相似度、摘要、双语文本挖掘）的基准；在信任任何单个 MTEB 分数之前请先阅读。
- [Sentence Transformers 文档](https://www.sbert.net/) —— 双编码器 vs 交叉编码器、池化策略以及本课程实现的摄取-分割-嵌入-存储 RAG 管道的权威参考。
