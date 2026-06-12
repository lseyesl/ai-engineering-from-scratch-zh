# GloVe 与 FastText — 计数、子词和对比

> Word2Vec 查看窗口。GloVe 查看整个语料库。FastText 查看子词。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 03（Word2Vec）
**时间：** 约 50 分钟

## 问题

Word2Vec 使用一个滑动窗口，并且仅查看局部上下文。"猫"和"狗"很相似，因为它们共享"宠物"、"喂"、"兽医"这些上下文词。但某个词是英语中罕见的外来词怎么办？一个在 1000 万文档语料库中只出现两次的词，Word2Vec 无法学到有用的表示。

GloVe 和 FastText 以两种不同的方式修复这个问题：GloVe 加入了全局统计信息（全文语料库计数），FastText 加入了子词信息（"cat"包含"ca"、"at"……以及"cats"共享"cat"）。

## 概念

### GloVe（全局向量）

GloVe 不是在嵌入滑动窗口对上运行，而是首先构建一个全局共现矩阵。它使用所有文档进行计数——"猫"与"兽医"在全文中共现的次数——而不仅仅是在相邻窗口内。

然后，Glove 通过最小化单词嵌入向量和共现统计量之间的差异来学习嵌入。目标很简单：对于任意一对单词 i 和 j，`w_i · w_j + b_i + b_j ≈ log(X_ij)`，其中 X_ij 是这两个单词在整个语料库中共现的次数。

主要的权衡是，GloVe 需要对整个语料库进行一次预处理传递（一次 O(V²) 操作，具体取决于窗口大小），然后才能开始训练。Word2Vec 则是在读取句子时在线生成训练对。GloVe 因此在训练开始时较慢，但一旦进入训练阶段，通常会收敛得更快。

### FastText

FastText 不是将一个单词表示为单个向量，而是将其表示为其组成部分字符 n-gram 向量的和。"apple" 在字符三元组下被分解为 `["<ap", "app", "ppl", "ple", "le>"]`（尖括号表示边界）。最终向量是所有这些子词向量的总和。

由于每个单词都由其子模式的总和表示，FastText 可以为未见过词生成嵌入——通过将其分解为字符 n-gram 并求和。GloVe 和 Word2Vec 对于未见过词汇会返回错误或产生垃圾结果。

这对形态丰富的语言（土耳其语：同一个动词有数百种形式）和包含拼写错误或罕见术语的文本至关重要。它也是大多数用于生产的嵌入方法的基础，尽管 Word2Vec 仍然是教学基础的选择。

## 构建

### 步骤 1：Glove 共现矩阵（简化版）

```python
from collections import defaultdict, Counter

class GloveCooccurrence:
    def __init__(self, window=5):
        self.window = window
        self.cooccur = defaultdict(Counter)

    def fit(self, sentences):
        for tokens in sentences:
            for i, center in enumerate(tokens):
                start = max(0, i - self.window)
                end = min(len(tokens), i + self.window + 1)
                for j in range(start, end):
                    if i != j:
                        self.cooccur[center][tokens[j]] += 1.0 / (abs(i - j))
```

距离加权：距离当前词 5 个位置的词与紧邻的词相比，其共现计数的一半。这引入了距离衰减——距离较近的词在嵌入中具有更强的影响。

```python
>>> c = GloveCooccurrence()
>>> c.fit([["cat", "sat", "on", "the", "mat"]])
>>> c.cooccur["cat"]
Counter({"sat": 1.0})
>>> c.cooccur["the"]
Counter({"sat": 0.5, "on": 0.5, "mat": 0.5})
```

### 步骤 2：GloVe 损失函数

```python
import math

def glove_loss(w_i, w_j, b_i, b_j, X_ij, f(X_ij)):
    diff = w_i @ w_j + b_i + b_j - math.log(X_ij)
    return f(X_ij) * diff * diff
```

加权函数 `f(X_ij)` 控制着每个共现对在总损失中的权重。罕见共现（X < 100）得到较低的权重（方差不稳定）。高频共现（X > 100）得到饱和的权重（它们过于主导）。满 Gewichtungsfunktion `f(x) = (x / x_max)^alpha` 当 `x < x_max` 时生效，否则为 1——这是一个精心设计的剪辑，用于平衡频率分布。

### 步骤 3：FastText 子词生成

```python
def char_ngrams(word, n=3):
    word = "<" + word + ">"
    return {word[i:i+n] for i in range(len(word) - n + 1)}
```

```python
>>> char_ngrams("cat", n=3)
{'<ca', 'cat', 'at>'}
>>> char_ngrams("cats", n=3)
{'<ca', 'cat', 'ats', 'ts>'}
```

`"cat"` 和 `"cats"` 共享字符三元组 `'<ca'` 和 `'cat'`。它们相应的向量将共享这些子词向量的信号。在 Word2Vec 中，`"cat"` 和 `"cats"` 是完全不同的、无关的词。在 FastText 中，它们是重叠的。

### 步骤 4：FastText 词向量推导

```python
def word_vector(self, word):
    ngrams = char_ngrams(word, self.n)
    vec = [0.0] * self.dim
    count = 0
    for ng in ngrams:
        if ng in self.ngram_vectors:
            for d in range(self.dim):
                vec[d] += self.ngram_vectors[ng][d]
            count += 1
    if count > 0:
        for d in range(self.dim):
            vec[d] /= count
    return vec
```

取平均而不是求和可以防止长词（产生更多 n-gram）的向量比短词具有更大的范数。在推理时，我们可以为任何字符串生成向量——甚至是在训练中从未见过的字符串——因为任何字符串都可以分解为字符 n-gram。

## 使用

### GloVe

```python
import numpy as np

def load_glove_embeddings(path):
    embeddings = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
    return embeddings
```

GloVe 向量以文本格式分发，大约为 800MB（100 维）。可以通过 numpy 进行内存映射以避免一次性加载加载所有内容。预训练向量的常见来源是 `nlp.stanford.edu/projects/glove/`。

### FastText

```python
import fasttext

model = fasttext.train_unsupervised("corpus.txt", model="skipgram", dim=100)
vector = model.get_word_vector("cat")
vector_oov = model.get_word_vector("kat")  # 拼写错误也有向量
```

`fasttext.train_unsupervised` 在单机上数分钟内处理数十亿词。它输出一个 `.bin` 模型文件（用于进一步训练/查询）和一个 `.vec` 文件（用于导入 numpy）。`model.get_word_vector("kat")` 为未见过词返回一个合理的向量——这是 FastText 相对于 Word2Vec 和 GloVe 的最大优势。

## 发布

用于选择静态嵌入方法的决策提示。

保存为 `outputs/prompt-embedding-advisor.md`：

```markdown
---
name: embedding-advisor
description: 根据语料库规模、语言和任务推荐嵌入方法。
phase: 5
lesson: 04
---

给定 NLP 任务和文本语料库，推荐一种静态嵌入方法。

1. 如果语料库 < 1M 词 → 使用预训练的 GloVe 或 FastText。从头训练 W2V 没什么帮助。
2. 如果语料库 ≥ 1M 词且为英语 → Word2Vec（速度）或 GloVe（多义词）。注意单数：Word2Vec 更快，GloVe 的全局损失更快收敛。
3. 如果语料库 ≥ 1M 词且为非英语或含噪声 → FastText。子词提供形态覆盖和 OCR/拼写错误鲁棒性。
4. 如果需要 OOV 支持 → FastText。W2V/GloVe 对未见过词返回零向量。
5. 如果任务受益于多义词感知 → 需要上下文嵌入（ELMo/BERT/阶段 9）。静态嵌入为每个词提供一个向量。

对于简单分类，嵌入 + 逻辑回归通常与 GRU/LSTM 性能相当，但训练速度快 10 倍。
```

## 练习

1. **简单。** 加载 GloVe 向量。计算 `king - man + woman` 并找到最接近的匹配。它与你的 Word2Vec 方法结果相比如何？
2. **中等。** 使用 FastText 为一个从未在训练数据中出现的词生成一个向量。证明即使对于 OOV 词，`model.get_word_vector(OOV)` 也具有有意义的值。
3. **困难。** 在形态丰富的语言（土耳其语、芬兰语或德语）上训练 FastText 和 Word2Vec。测量两种方法下名词屈折变化的词类比准确率。FastText 在多大程度上提升效果？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 共现矩阵 | 计数配对 | 在整个语料库中哪些词出现在一起的统计信息。 |
| 加权函数 | 平衡罕见/常见词 | 剪切极端频率的权重函数（GloVe 的 `f(X)`）。 |
| 子词 | 单词碎片 | 字符 n-gram——一个词由其内部模式表示。 |
| OOV | 未登录词 | 训练数据中未出现的词。FastText 通过子词处理 OOV。 |
| 静态嵌入 | 每个词=一个向量 | 无论上下文如何，每个词形一个向量。与 BERT 的上下文化向量相对。 |
