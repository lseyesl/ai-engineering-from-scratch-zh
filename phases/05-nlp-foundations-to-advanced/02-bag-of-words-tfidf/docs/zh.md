# 词袋模型与 TF-IDF

> 统计词汇，构建信号，忽略音调。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 01（文本处理）
**时间：** 约 45 分钟

## 问题

你有一个包含 10000 封客户投诉的语料库。你如何从语义上找到最相似的一对，而不需要任何预先训练的模型？

你获取所有文档中的所有单词，统计频率，将它们转化为向量。任何具有相同词频分布的文档都被认为是语义相似的。这个假设既愚蠢又有效。我们将深入探讨它为何有效、何时失效，以及 TF-IDF 是做什么的。

## 概念

```figure
bow-tfidf
```

### 词袋模型（Bag of Words, BoW）

词袋模型就是：你获取所有文档中的所有唯一单词并构建一个词汇表。对于每个文档，你统计每个单词出现的次数。结果是一个矩阵，其中行 = 文档，列 = 单词，值 = 计数。

它被称为"词袋"是因为语法被丢弃了——我们只知道文档中有哪些词，不知道它们的顺序。"狗咬人"和"人咬狗"具有相同的 BoW 向量。

### TF-IDF

词频-逆文档频率（Term Frequency — Inverse Document Frequency）通过两个机制对计数进行加权。

**词频（TF）：** 一个词在给定文档中出现的频率。通常计算为对数缩放计数：`TF(t, d) = log(1 + count(t, d))`。如果某个词在一个文档中出现 1000 次，你得到的是 log(1001) 而不是 1000。这防止了冗长的文档主宰相似度计算。

**逆文档频率（IDF）：** 一个词在语料库中的稀有程度。`IDF(t) = log(N / DF(t))`，其中 N 是文档总数，DF(t) 是包含词 t 的文档数量。出现在所有文档中的词（"the"、"a"、"is"）获得接近零的 IDF。出现在少数文档中的特定词获得高 IDF。

最终得分是两者的乘积：`TF-IDF(t, d) = TF(t, d) × IDF(t)`。

**直觉：** TF-IDF 放大了仅在少数文档中高频出现的词的信号，同时压制了到处出现的词的噪声。它是三种冲动之间的平衡——词频、稀有度和文档长度——每一种都会相互干扰。

## 构建

### 步骤 1：词袋向量化器

```python
import math
from collections import Counter

class BoWVectorizer:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    def fit(self, documents):
        words = set()
        for doc in documents:
            words.update(w.lower() for w in doc.split())
        self.vocab = {w: i for i, w in enumerate(sorted(words))}
        self.vocab_size = len(self.vocab)

    def transform(self, document):
        words = [w.lower() for w in document.split()]
        counts = Counter(words)
        vec = [0] * self.vocab_size
        for word, count in counts.items():
            if word in self.vocab:
                vec[self.vocab[word]] = count
        return vec

    def fit_transform(self, documents):
        self.fit(documents)
        return [self.transform(d) for d in documents]
```

```python
>>> vec = BoWVectorizer()
>>> vec.fit_transform(["cat sat mat", "dog ran"])  # doctest: +SKIP
```

`fit` 处理文档收集的遍历。`transform` 是一个独立的调用，因此你可以将训练期间学习的词汇表复用在新的文档上。如果你在推理时 `fit` 新的数据，向量将包含不同维度的特征且无法相互比较。永远不要这样做。

### 步骤 2：从边边角角添加平滑

空的 BoW 向量表示未知词。如果一个除了一行新词之外完全相同的文档得到一个全零向量，余弦相似度会告诉你它与其他所有文档正交。这对于精确匹配来说是正确的行为，但意味着 BoW 无法处理未见过的数据。

第一个平滑：为未知词添加一个 `OOV`（Out Of Vocabulary）槽。在词汇表中保留索引 0，并将所有未见过的词映射到它。这不是一个很好的解决方案——它只是防止了崩溃——但它承认了分布外数据的存在。

```python
class BoWVectorizerOOV:
    def __init__(self):
        self.vocab = {"<OOV>": 0}
        self.vocab_size = 1

    def fit(self, documents):
        words = set()
        for doc in documents:
            words.update(w.lower() for w in doc.split())
        for w in sorted(words):
            if w not in self.vocab:
                self.vocab[w] = self.vocab_size
                self.vocab_size += 1

    def transform(self, document):
        words = [w.lower() for w in document.split()]
        counts = Counter(words)
        vec = [0] * self.vocab_size
        for word, count in counts.items():
            idx = self.vocab.get(word, 0)
            vec[idx] += count
        return vec
```

### 步骤 3：TF-IDF 变换器

```python
class TfidfTransformer:
    def __init__(self):
        self.idf = {}
        self.N = 0

    def fit(self, count_matrix):
        self.N = len(count_matrix)
        n_docs = [0] * len(count_matrix[0])
        for row in count_matrix:
            for col, val in enumerate(row):
                if val > 0:
                    n_docs[col] += 1
        self.idf = {
            col: math.log((1 + self.N) / (1 + n)) + 1
            for col, n in enumerate(n_docs)
        }

    def transform(self, count_matrix):
        result = []
        for row in count_matrix:
            tf = [1 + math.log(c) if c > 0 else 0 for c in row]
            tfidf = [tf[i] * self.idf[i] for i in range(len(row))]
            result.append(tfidf)
        return result
```

IDF 公式使用 `(1+N)/(1+DF)` 而不是 `N/DF`，并添加了 `+1` 平滑。这确保了 IDF 永远不会是零。一个出现在所有文档中的词得到 `IDF ≈ log(1) + 1 = 1`，而不是零。一个仅出现在一个文档中的词得到 `IDF ≈ log((1+N)/2) + 1`，范围大约在 2-4 之间，具体取决于 N。TF 使用 `1 + log(count)` 而不是原始计数。这是 scikit-learn 的默认行为，也是大多数生产代码的行为。

```python
>>> vec = BoWVectorizerOOV()
>>> counts = vec.fit_transform(["cat sat mat", "dog ran"])
>>> tfidf = TfidfTransformer()
>>> tfidf.fit(counts)
>>> tfidf.transform(counts)  # doctest: +ELLIPSIS
[[...], [...]]
```

### 步骤 4：余弦相似度

```python
def cosine_similarity(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

余弦相似度测量的是向量之间的角度，而不是长度。两个文档如果包含相似的比例的词，即使一个文档是另一个的两倍长，也能达到 1.0 的相似度。这就是为什么 TF-IDF 缩放长度差异——没有它，较长的文档总会产生更大的点积。当所有值非负时（就像 BoW 和 TF-IDF 的情况），范围是 0 到 1。

## 使用

### scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "cat sat mat dog ran",
]

count_vec = CountVectorizer()
counts = count_vec.fit_transform(docs)

tfidf_vec = TfidfVectorizer()
tfidf = tfidf_vec.fit_transform(docs)

sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
```

### 你能获取到的

| 方面 | 你能获取到的 | 缺陷 |
|--------|-------------|------|
| 简单性 | 三行代码，任何数据集 |  |
| 可解释性 | 每个权重对应一个单词；你可以展示为何两个文档相似 | 没有交互作用项——"not good" 和 "good" 都贡献给 "good" |
| 扩展性 | 仅需一次扫描语料库即可学习 IDF；适用于数百万文档 | 词汇表增长失控；大规模场景下 N×V 矩阵无法放入内存 |
| 无参数 | 不需要训练循环或 GPU | 无法处理未见过词的组合 |

### 三大失败模式

**大小写和噪声：** `"Apple"`（公司）和 `"apple"`（水果）映射到同一个词元。拼写错误完全无法匹配。小写化和词形还原在 BoW 之前使用并不能解决同形异义词问题（银行河岸 vs 金融银行）。每个词只能得到一个向量，丢失了所有上下文。

**维度诅咒：** 具有 100000 个文档的英语语料库轻松产生 500000 个独特词元。你的相似度矩阵大小是 100000 × 100000。你实际上无法将其放到内存中。减少维度（修剪罕见词、使用主题模型、使用嵌入）是不可避免的。

**上下文盲区：** 词袋模型在其名称中包含了"袋"。顺序被完全丢弃了。`"not good"` 对于 `common_noun = true + positive = true` 的求和，结果与 `"good not"` 完全相同。TF-IDF 无济于事——它重新加权相同的求和。n-gram（成对词组、三词组）通过在词汇表中添加 `"not_good"` 作为一个单独的词元来部分修复这个问题，但它与你所使用的 n-gram 数量呈指数扩展关系。

### 经验法则

- 使用 BoW 进行：聚类、快速原型设计、大规模搜索基线
- 使用 TF-IDF 进行：信息检索、关键词抽取、文档分类
- 切换到嵌入当：你需要同义词匹配（"car" ↔ "automobile"）、你在处理多语言内容、或者你需要理解否定和强度

## 发布

一种直接用于分类和搜索的 TF-IDF 基线配方。

保存为 `outputs/skill-tfidf-baseline.md`：

```markdown
---
name: tfidf-baseline
description: 用于文本分类和信息检索的 TF-IDF 基线。
phase: 5
lesson: 02
---

作为第一步，在投入嵌入或微调之前，运行一个 TF-IDF 基线。
它设置了一个可达到的性能标准，揭示了数据泄露问题，并在 30 分钟内运行完毕。

1. 预处理：小写化、标点移除、可选词形还原。使用 `TfidfVectorizer` 的 `token_pattern` 和 `stop_words` 参数。
2. 向量化：`TfidfVectorizer(max_features=10000, sublinear_tf=True)`。
3. 建模：搜索 `LogisticRegression(C=[0.01, 0.1, 1.0])`。
4. 评估：准确性、F1 和分类报告。
5. 对照基线检查：如果 TF-IDF 接近最佳模型（<5% 的差距），你可能不需要更多的数据或更大的模型。差距较大则需要更好的表示或更好的数据。
```

## 练习

1. **简单。** 使用你的 BoWVectorizer 计算 `["i love this movie", "i hate this movie"]` 的余弦相似度。解释结果。
2. **中等。** 用你的 TfidfTransformer 运行相同的实验。为什么相似度不同？IDF 如何改变结果？
3. **困难。** 实现 `NgramVectorizer`，在词汇表中包含大小为 1、2 和 3 的 n-gram。你还能在解析具有否定词的句子（`"not good"` vs `"good"`）方面做得更好吗？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 词袋 | 统计单词 | 不保留顺序的词频矩阵。 |
| TF | 词频 | 文档内频率，进行对数缩放以防止文档长度偏差。 |
| IDF | 稀有度权重 | 全局对数缩放，压制常见词，提升特定词。 |
| 余弦相似度 | 向量之间的角度 | 受长度无关的相似度，对 BoW/TF-IDF 向量的范围是 [0, 1]。 |
| 词汇表 | 所有文档中所有单词的集合 | 填充矩阵列的唯一标记列表。 |
