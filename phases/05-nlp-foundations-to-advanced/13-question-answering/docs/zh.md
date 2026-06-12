# 信息检索与搜索 — 找到正确的文档

> TF-IDF 说：这个词出现在这个文档中。BM25 说：这个词以比预期的更高频率出现在这个文档中。密集检索说：这个词所代表的概念出现在这个文档中。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 02（TF-IDF）
**时间：** 约 50 分钟

## 问题

信息检索（IR）是搜索的基础。给定一个查询，从数十亿文档中找到最相关的文档。所有搜索系统—— Elasticsearch、Google、专用搜索引擎——其核心都是 IR。

经典的 IR 使用基于倒排索引的词汇匹配。现代 IR 使用基于嵌入的语义搜索。在大多数生产系统中，两者都使用：BM25 提供精确的词汇匹配，密集搜索提供同义词和概念匹配，两者通过混合搜索融合在一起。

## 概念

**倒排索引**是 IR 的基础数据结构。它不是将文档映射到单词，而是将单词映射到包含这些单词的文档列表。"cat" → [doc1, doc3, doc7, ...]。在查询时，你查找查询词中的每一个单词，合并文档列表，并对结果进行排序。

**TF-IDF 变体（BM25）** 是词袋模型排名的标准。它改进 TF-IDF 的方式有：(1) 对词频进行饱和处理而非线性减少（防止一个文档中出现 100 次"cat"就压到出现 5 次的情况）(2) 按文档长度进行归一化，使较长的文档不会因为长度而获得更高的排名。

**密集检索** 将查询和文档嵌入到同一空间中。相似度是嵌入向量之间的余弦距离。它可以处理词汇不匹配问题——即使查询包含"automobile"而文档只说"car"，它们仍然能够匹配。

## 构建

### 步骤 1：概念性倒排索引

```python
from collections import defaultdict

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)

    def add_document(self, doc_id, text):
        for word in text.lower().split():
            self.index[word].add(doc_id)

    def search(self, query):
        words = query.lower().split()
        if not words:
            return set()
        # 交集：包含所有查询词的文档
        result = self.index[words[0]]
        for word in words[1:]:
            result = result.intersection(self.index.get(word, set()))
        return result
```

交集搜索返回包含所有查询词的文档（AND 语义）。对于实际使用，你需要 OR 语义（包含任一查询词的文档）以及一个排名函数（TF-IDF 或 BM25），以确定哪些文档最匹配。

### 步骤 2：BM25 排名函数

```python
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.index = defaultdict(dict)
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.num_docs = 0

    def fit(self, documents):
        self.num_docs = len(documents)
        total_length = 0
        for doc_id, text in documents.items():
            words = text.lower().split()
            self.doc_lengths[doc_id] = len(words)
            total_length += len(words)
            word_counts = Counter(words)
            for word, count in word_counts.items():
                self.index[word][doc_id] = count
        self.avg_doc_length = total_length / self.num_docs

    def idf(self, word):
        n = len(self.index.get(word, {}))
        return math.log((self.num_docs - n + 0.5) / (n + 0.5) + 1)

    def score(self, query, doc_id):
        words = query.lower().split()
        score = 0
        doc_len = self.doc_lengths.get(doc_id, 0)
        for word in words:
            tf = self.index.get(word, {}).get(doc_id, 0)
            if tf > 0:
                idf = self.idf(word)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * numerator / denominator
        return score
```

BM25 有两个超参数：`k1`（控制词频饱和——`k1=0` 意味着不考虑 TF，`k1` 越大，TF 的影响就越持久）和 `b`（控制长度归一化——`b=0` 禁用归一化，`b=1` 完全归一化）。Elasticsearch 的默认值是 `k1=1.2`，`b=0.75`。

### 步骤 3：密集检索——概念性搜索

```python
class DenseRetriever:
    def __init__(self, encoder):
        self.encoder = encoder
        self.doc_embeddings = {}
        self.documents = {}

    def index_documents(self, documents):
        for doc_id, text in documents.items():
            self.doc_embeddings[doc_id] = self.encoder.encode(text)
            self.documents[doc_id] = text

    def search(self, query, k=5):
        q_emb = self.encoder.encode(query)
        scores = {}
        for doc_id, doc_emb in self.doc_embeddings.items():
            scores[doc_id] = self._cosine_sim(q_emb, doc_emb)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, self.documents[doc_id], score) for doc_id, score in ranked[:k]]
```

密集检索的关键组件是将文本编码到嵌入空间的编码器。`SentenceTransformer`（阶段 5 · 23）提供了针对搜索进行优化的预训练模型：`all-MiniLM-L6-v2` 是一个不错的多功能选择。

### 步骤 4：混合搜索

```python
def hybrid_search(query, doc_ids, bm25_scores, dense_scores, alpha=0.5):
    """对 BM25 和密集检索分数进行加权融合。"""
    combined = {}
    for doc_id in doc_ids:
        bm25 = bm25_scores.get(doc_id, 0)
        dense = dense_scores.get(doc_id, 0)
        combined[doc_id] = alpha * bm25 + (1 - alpha) * dense
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

对 BM25 和密集分数进行归一化是必不可少的（通常使用 min-max 或 z-score），这样它们就处于相同的范围内。`alpha` 控制词汇匹配与语义匹配的权重。对于包含大量命名实体和精确术语的查询，BM25 更强。对于概念性查询，密集检索更强。

## 使用

### Elasticsearch（BM25）

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

# BM25 是默认的相似度算法
result = es.search(
    index="documents",
    query={"match": {"text": "machine learning transformer architecture"}},
    size=5
)
```

Elasticsearch 在索引时分词并计算 `k1` 和 `b` 的统计信息。查询词匹配按 BM25 评分。相关性评分在返回前按索引统计进行归一化。

### 使用 FAISS 进行密集检索

```python
import faiss
import numpy as np

# 构建索引
embeddings = np.array([encoder.encode(doc) for doc in documents]).astype("float32")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # 内积 = 归一化向量的余弦相似度
index.add(embeddings)

# 搜索
query_emb = np.array([encoder.encode(query)]).astype("float32")
distances, indices = index.search(query_emb, k=5)
```

FAISS 不是包到包地搜索，而是使用近似最近邻（ANN）算法。`IndexFlatIP` 是穷举搜索（准确但缓慢，不适合上百万文档）。`IndexIVFFlat` 使用聚类进行分桶搜索（更快，但召回率略有下降）。

## 发布

用于日志分析的搜索质量调试提示。

保存为 `outputs/prompt-search-debug.md`：

```markdown
---
name: search-debug
description: 提示：调试搜索质量问题，比较 BM25 与密集检索。
phase: 5
lesson: 13
---

分析搜索排名问题。给定查询、返回的前 k 个结果和预期结果：

1. 词汇匹配：查询词是否出现在排名靠前的结果中？如果没有，BM25 可能会失败。
2. 同义词："car" 是否能匹配到包含 "automobile" 的文档？密集检索可以做到，BM25 不行。
3. 意图：查询是导航式的（"OpenAI 登录"）还是信息式的（"什么是变压器"）？导航式查询需要精确的词汇匹配；信息式查询则需要密集检索。
4. 缺失的顶级结果：为什么期望的文档没有排在前列？它是缺少查询词（BM25 问题）还是与查询含义不同（密集检索问题）？

建议使用混合搜索或调整 BM25 参数以优化结果。
```

## 练习

1. **简单。** 构建一个 BM25 索引，包含 10 篇文档并运行 3 个查询。得分排名是否符合你的预期？
2. **中等。** 使用 Sentence-BERT 实现密集检索。在同一个查询集上比较 BM25 与密集检索的结果——哪些查询在每种方法下表现更好？为什么？
3. **困难。** 实现混合搜索（BM25 + 密集检索），使用归一化评分和可调的 `alpha` 参数。找到你所在领域的 `alpha` 最佳值。绘制不同 `alpha` 值下的精确度-召回率曲线。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 倒排索引 | 词到文档的映射 | 对于每个词，该词出现的文档 ID 列表。 |
| BM25 | 排名函数 | TF-IDF 的饱和版，带有长度归一化。概率公式。 |
| 密集检索 | 嵌入搜索 | 在语义嵌入空间中进行搜索。 |
| 混合搜索 | 两者结合 | BM25 + 密集检索，采用加权融合。 |
| FAISS | 高效相似度搜索 | 用于大规模 ANN 搜索的库。 |
