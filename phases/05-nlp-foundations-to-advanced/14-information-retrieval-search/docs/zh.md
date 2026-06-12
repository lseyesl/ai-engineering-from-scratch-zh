# 主题建模 — 在词汇背后发现主题

> 100000 个文档。50000 个维度（词）。降维到 10 个主题。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 02（BoW/TF-IDF）
**时间：** 约 50 分钟

## 问题

你有 10000 封客户支持邮件。你需要知道它们在谈论什么，而不需要阅读每一封。"退款政策"、"产品缺陷"、"账户问题"——这些主题在你大规模构建自动路由系统之前就已经存在了。

主题模型发现文档集合中的潜在主题。它在寻找每篇文档中隐藏的"为什么"——这些词的统计模式背后所揭示的主题。

## 概念

**潜在狄利克雷分配（LDA）** 是最广泛使用的主题模型。它假设每篇文档是多个主题的混合体，每个主题是多个词的分布。LDA 逆向工程出主题：给定文档，找出最能解释词频率分布的主题。

**非负矩阵分解（NMF）** 是一个更简单的替代方案。它对 TF-IDF 矩阵进行分解：`A ≈ W × H`，其中 W（文档-主题）和 H（主题-词）都是非负的。W 的列是主题；H 的行是主题。非负性约束强制 NMF 学习各部分的总和——非常符合主题的构成直觉。

## 构建

### 步骤 1：LDA——主题-词分配

```python
import random
from collections import Counter

class LDAByHand:
    def __init__(self, num_topics=5, alpha=0.1, beta=0.01):
        self.num_topics = num_topics
        self.alpha = alpha  # 文档-主题 Dirichlet 先验
        self.beta = beta    # 主题-词 Dirichlet 先验

    def fit(self, documents, vocab, iterations=100):
        n_docs = len(documents)
        n_words = sum(len(doc) for doc in documents)

        # 初始化随机主题分配
        self.doc_topic_counts = Counter()
        self.topic_word_counts = Counter()
        self.topic_counts = [0] * self.num_topics
        doc_topics = []

        for d_idx, doc in enumerate(documents):
            doc_topic = []
            for word in doc:
                topic = random.randrange(self.num_topics)
                doc_topic.append(topic)
                self.doc_topic_counts[(d_idx, topic)] += 1
                self.topic_word_counts[(topic, word)] += 1
                self.topic_counts[topic] += 1
            doc_topics.append(doc_topic)

        # Gibbs 采样迭代
        for iteration in range(iterations):
            for d_idx, doc in enumerate(documents):
                for w_idx, word in enumerate(doc):
                    topic = doc_topics[d_idx][w_idx]
                    # 移除当前主题分配
                    self.doc_topic_counts[(d_idx, topic)] -= 1
                    self.topic_word_counts[(topic, word)] -= 1
                    self.topic_counts[topic] -= 1

                    # 使用条件概率采样新主题
                    topic_probs = []
                    for t in range(self.num_topics):
                        dt_prob = (self.doc_topic_counts[(d_idx, t)] + self.alpha)
                        tw_prob = ((self.topic_word_counts[(t, word)] + self.beta)
                                   / (self.topic_counts[t] + len(vocab) * self.beta))
                        topic_probs.append(dt_prob * tw_prob)

                    total = sum(topic_probs)
                    r = random.random() * total
                    cum = 0
                    new_topic = 0
                    for t, prob in enumerate(topic_probs):
                        cum += prob
                        if r <= cum:
                            new_topic = t
                            break

                    # 分配新主题
                    doc_topics[d_idx][w_idx] = new_topic
                    self.doc_topic_counts[(d_idx, new_topic)] += 1
                    self.topic_word_counts[(new_topic, word)] += 1
                    self.topic_counts[new_topic] += 1

        # 提取主题词
        self.topics = []
        for t in range(self.num_topics):
            word_probs = {}
            for (topic, word), count in self.topic_word_counts.items():
                if topic == t:
                    word_probs[word] = (count + self.beta) / (self.topic_counts[t] + len(vocab) * self.beta)
            self.topics.append(word_probs)
```

这是 LDA 的一个教学简化版，省略了词-主题分布的共轭计算，但保留了核心的 Gibbs 采样思想：每个词被分配到一个主题，更新计数，然后重新采样。经过足够的迭代后，主题在语义上趋于一致。

### 步骤 2：NMF——通过矩阵分解进行主题建模

```python
import math
import random

class NMF:
    def __init__(self, num_topics=5, max_iter=100):
        self.num_topics = num_topics
        self.max_iter = max_iter

    def fit(self, tfidf_matrix):
        n_docs = len(tfidf_matrix)
        n_words = len(tfidf_matrix[0])

        # 初始化非负矩阵
        W = [[random.random() for _ in range(self.num_topics)] for _ in range(n_docs)]
        H = [[random.random() for _ in range(n_words)] for _ in range(self.num_topics)]

        for iteration in range(self.max_iter):
            # 乘法更新规则
            for i in range(n_docs):
                for j in range(self.num_topics):
                    num = 0
                    den = 0
                    for k in range(n_words):
                        recon = sum(W[i][t] * H[t][k] for t in range(self.num_topics))
                        num += H[j][k] * tfidf_matrix[i][k] / (recon + 1e-10)
                        den += H[j][k]
                    W[i][j] *= num / (den + 1e-10)

            for j in range(self.num_topics):
                for k in range(n_words):
                    num = 0
                    den = 0
                    for i in range(n_docs):
                        recon = sum(W[i][t] * H[t][k] for t in range(self.num_topics))
                        num += W[i][j] * tfidf_matrix[i][k] / (recon + 1e-10)
                        den += W[i][j]
                    H[j][k] *= num / (den + 1e-10)

        self.W = W  # 文档-主题权重
        self.H = H  # 主题-词权重
```

NMF 损失是 `||A - WH||²`（Frobenius 范数），约束条件：`W >= 0`，`H >= 0`。乘法更新规则保持了非负性。这在主题建模方面是一个很好的教学工具，因为主题可以被解释为词的加权组合。

## 使用

### scikit-learn LDA

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=5000, stop_words="english")
doc_term = vectorizer.fit_transform(documents)

lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(doc_term)

# 每个主题的前 10 个词
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")
```

`n_components` 是主题的数量。这是 LDA 中唯一真正重要的超参数（除了 `learning_method="online"`，它可以流式处理大型语料库）。主题数量太少会合并不同的主题；主题数量太多会分裂它们。使用 `model.log_likelihood()` 或人类可读性评估来调整。

### BERTopic（现代方法）

```python
from bertopic import BERTopic

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(documents)
```

BERTopic 使用句子嵌入（Sentence-BERT）对文档进行聚类，然后用 c-TF-IDF（在每个聚类的文档中将词的重要性相对于整个语料库进行排名）提取主题。与 LDA 相比，它生成的输出更容易解释，并且不需要预先指定主题的数量。

## 发布

主题模型可解释性检查清单。

保存为 `outputs/prompt-topic-interpretation.md`：

```markdown
---
name: topic-interpretation
description: 提示：解释和验证主题模型输出。
phase: 5
lesson: 14
---

评估主题模型输出。每个主题应包含语义上彼此相关的词。

1. 主题一致性：主题中的前 10 个词是否在概念上相关？如果不相关，主题只是一个词汇包，而不是真正的主题。
2. 主题区分度：不同主题的前 10 个词之间是否有显著重叠？如果有，主题没有很好地分离。
3. 文档分配：检查每个主题中得分最高的 5 篇文档——它们是否真的属于同一个主题？
4. 离群值：检查所有主题中得分最低的文档——它们是否包含了独特的主题，还是只是噪声？
5. 稳定性：在不同随机种子下运行模型。主题是否一致地出现？如果不同的运行产生不同的主题，数据中的信号可能太弱了。
```

## 练习

1. **简单。** 在 20 Newsgroups 数据集上训练 LDA。打印每个主题的前 10 个词。这些主题在语义上是否能被解释？
2. **中等。** 在一堆文档上比较 LDA 和 NMF。哪个模型产生了更清晰的分离主题？使用主题一致性评分（通过 `gensim.models.CoherenceModel`）进行量化评估。
3. **困难。** 以 BERTopic 作为起始，基于主题建模实现一个文档聚类系统。比较 BERTopic 与 LDA 在你自己的数据上的可解释性——哪个对你领域中的人类评估者更有用？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| LDA | 潜在狄利克雷分配 | 将每篇文档建模为主题混合，将每个主题建模为词混合的生成模型。 |
| NMF | 非负矩阵分解 | 将 TF-IDF 矩阵分解为文档-主题（W）和主题-词（H）矩阵的乘积。 |
| Gibbs 采样 | 迭代重新分配 | 通过按条件分布重新对词的主题分配进行采样来近似联合分布。 |
| 一致度 | 主题质量指标 | 词在主题中共同出现的频率，与外部语料库无关。 |
| BERTopic | 嵌入聚类 | 现代主题建模：嵌入 → 降维（UMAP） → 聚类（HDBSCAN） → c-TF-IDF。 |
