# Word2Vec — 从计数到嵌入

> 你无法从"生活"中学到任何东西，除非你读了它前后所有的句子。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 01（文本处理），阶段 2 · 14（Softmax）
**时间：** 约 60 分钟

## 问题

BoW 和 TF-IDF 生成的向量长度为 V——词汇表大小。英语有超过一百万个词汇。在 100 万维空间中，所有向量都是稀疏的且相距甚远。苹果和橙子的余弦相似度为零，因为这两个词从未在同一个文档中出现过。而它们应该是相关的。

Word2Vec 解决了两个问题：维度（从 1M 降到 300）和语义（捕获"苹果" ≈ "橙子"）。解决方案是一个巧妙的博弈：通过其相邻的上下文词来预测一个词。

## 概念

Word2Vec 有两种风格。

**Skip-gram（SG）：** 给定一个中心词，预测其相邻的上下文词。"king" → ["a", "noble", "who", "rules"]。在大型数据集上较慢但更好，尤其是在训练数据有限时对罕见词效果更好。

**CBOW（Continuous Bag of Words）：** 给定上下文词，预测中心词。["a", "noble", "who", "rules"] → "king"。训练速度更快，但对罕见词效果较差。

两种风格都将词嵌入到低维空间中，其中语义上相似的词彼此靠近。该空间是连续的（每条轴上都是浮点数），并且密集的（每个维度都承载意义）。

学习过程基于分布假设：**出现在相似上下文中的词具有相似的含义。** 这是现代 NLP 中最重要的假设。如果你理解了这一句话，你就理解了嵌入。

```figure
word-vector-arithmetic
```

## 构建

### 步骤 1：从句子到训练对

```python
import random
from collections import Counter

def skipgram_pairs(tokens, window=2):
    pairs = []
    for i, center in enumerate(tokens):
        start = max(0, i - window)
        end = min(len(tokens), i + window + 1)
        for j in range(start, end):
            if i != j:
                pairs.append((center, tokens[j]))
    return pairs
```

```python
>>> skipgram_pairs("the cat sat".split())
[('the', 'cat'), ('cat', 'the'), ('cat', 'sat'), ('sat', 'cat')]
```

中心词总是第一个元素。上下文词总是第二个。这个方向在损失函数中很重要。窗口大小控制信号与噪声——较小的窗口捕获句法关系（形容词紧邻名词出现），较大的窗口捕获主题关系（"医生"和"医院"相距几个词出现）。

### 步骤 2：用代码行实现权重和 Softmax

```python
import math
import random

class Word2VecSkipGram:
    def __init__(self, vocab_size, embedding_dim=100):
        self.W = [[random.gauss(0, 0.01) for _ in range(embedding_dim)]
                  for _ in range(vocab_size)]
        self.W_out = [[random.gauss(0, 0.01) for _ in range(vocab_size)]
                      for _ in range(embedding_dim)]

    def softmax(self, scores):
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]
```

`W` 是嵌入矩阵——形状为(V, D)，其中 D 是嵌入维度。第 i 行是词汇表中第 i 个词的嵌入向量。`W_out` 是输出投影——形状为(D, V)，它从嵌入空间投影回词汇表以获得预测分数。

训练过程是：获取一个中心词的嵌入，将其与 W_out 矩阵相乘，进行 softmax 处理以获得概率分布，然后相对于 ground truth 上下文词计算交叉熵损失。

### 步骤 3：一次前向/反向传播

```python
def forward(self, center_idx, context_idx):
    # 中心词的嵌入
    h = self.W[center_idx]  # 形状：(D,)
    # 投影到词汇空间
    scores = [sum(h[j] * self.W_out[j][k] for j in range(len(h)))
              for k in range(len(self.W_out[0]))]
    probs = self.softmax(scores)
    loss = -math.log(probs[context_idx] + 1e-10)
    return loss, (h, probs, scores)
```

`loss = -log(p(context|center))`。如果模型完美地预测了正确的上下文词，p 接近 1，损失接近 0。如果模型犯了错误，p 很小，损失很大。梯度通过 `W_out` 回传以更新嵌入。

### 步骤 4：负采样

词汇表上的 Softmax 需要在词汇表大小上求和。对于 V = 1,000,000 来说，这太慢了。负采样通过将多类问题转化为二分类问题来绕过这个问题：模型学习区分真正的（中心词，上下文词）对和随机选择的噪声对。

```python
def negative_samples(self, center_idx, context_idx, k=5):
    noise = random.sample(
        [i for i in range(len(self.W)) if i != context_idx], k
    )
    # 正例：log(sigma(dot(h, v_context)))
    # 负例：log(sigma(-dot(h, v_noise))) 对每个噪声样本
    h = self.W[center_idx]
    pos_score = sum(h[j] * self.W_out[j][context_idx] for j in range(len(h)))
    pos_loss = -math.log(self._sigmoid(pos_score) + 1e-10)
    neg_loss = 0
    for neg_idx in noise:
        neg_score = sum(h[j] * self.W_out[j][neg_idx] for j in range(len(h)))
        neg_loss += -math.log(self._sigmoid(-neg_score) + 1e-10)
    return pos_loss + neg_loss
```

每次更新仅使用 k+1 个输出神经元（k 通常为 5-20）而不是全部 V 个。这使计算成本从 O(V) 降至 O(k)。这就是 Word2Vec 可以扩展到数百万词汇的原因。

### 步骤 5：组合成训练循环

```python
def train(self, sentences, epochs=5, lr=0.01):
    for epoch in range(epochs):
        for sentence in sentences:
            tokens = sentence.split()
            for center, context in skipgram_pairs(tokens):
                center_idx = self.word_to_idx[center]
                context_idx = self.word_to_idx[context]
                loss = self.negative_samples(center_idx, context_idx)
                # 梯度下降（省略了梯度计算细节）
```

在单线程笔记本电脑上对 10000 个句子运行此代码比 scikit-learn 的 TF-IDF 慢约 100 倍。这是 Word2Vec 的通常体现——你需要大规模数据才能学到出色的向量，而大规模需要 C 实现。这就是 gensim 和原始 C 代码的实际用途。

## 使用

### gensim

```python
from gensim.models import Word2Vec

sentences = [["the", "cat", "sat", "on", "the", "mat"],
             ["the", "dog", "ran", "in", "the", "park"]]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
vector = model.wv["cat"]
similar = model.wv.most_similar("cat")
```

`vector_size=100` 是你想要的维度。`window=5` 的默认值对主题相似度有效，但如果你需要句法相似度，可以降低到 2-3。`min_count=5` 丢弃出现少于 5 次的词——显著的降噪效果，并且将你的词汇量削减 90% 而不损失准确性。`sg=1` 选择 skip-gram；`sg=0` 选择 CBOW。

### 词嵌入的向量运算

```python
# king - man + woman ≈ queen
result = model.wv.most_similar(positive=["king", "woman"],
                                negative=["man"])
```

这之所以有效，是因为训练过程诱导出一个具有线性关系的空间：方向"王权"的向量加上方向"女性"的向量，指向最接近"女王"的向量。这并不是天生就设计好的——它来自于学习分布模式所产生的意外副作用。类比推理大约有 60-75% 的准确率，具体取决于数据集和超参数。

### 当你需要更多时

| 需求 | 怎么做 |
|------|-------------|
| 更大语料库 | 使用 gensim 的 `corpora` 迭代器；一次仅将一行保留在内存中。 |
| 罕见词 | 增加 `min_count=1`，但要知道会带来噪声。使用更多的训练数据。 |
| 子词信息 | 切换到 FastText（阶段 5 · 04）。 |
| 句子级向量 | 对词向量取平均（弱），使用 Sentence-BERT（强）。 |
| 多义词 | Word2Vec 为"苹果"（水果和公司）提供一个向量。使用 ELMo/BERT（阶段 9）。 |

## 发布

生产就绪的 Word2Vec 训练流程。

保存为 `outputs/skill-word2vec-training.md`：

```markdown
---
name: word2vec-training
description: 用于生产文本相似度的 Word2Vec 训练流程。
phase: 5
lesson: 03
---

使用 gensim 构建领域特定的嵌入。

1. 预处理：小写化，标点移除，可选词形还原。保留 `min_count=5`——噪声词无法学到好的向量。
2. 训练：
   ```python
   model = Word2Vec(sentences, vector_size=100, window=5,
                    min_count=5, workers=4, sg=1, epochs=10)
   ```
3. 评估：内部类比测试（`model.wv.most_similar`）+ 外部下游任务。
4. 保存/加载：`model.save("w2v.model")` / `Word2Vec.load("w2v.model")`。
5. 陷阱：调用 `model.build_vocab(sentences)` 然后调用 `model.train(sentences, total_examples=..., epochs=...)` ——不要调用 `model.train` 两次而不重置学习率或忽略学习率衰减。

如果相似度看起来不对，检查你的预处理。带有停用词的 `"not good"` 清洗为 `"good"` ——你的"否定"嵌入已经消失。
```

## 练习

1. **简单。** 使用 gensim 在 `sentences = ["king strong male", "queen strong female", ...]` 上训练一个模型。打印 `king` 和 `queen` 之间的余弦相似度。
2. **中等。** 实现 CBOW 前向传播——给定上下文词，预测中心词。与你的 skip-gram 实现相比，训练速度有何不同？
3. **困难。** 实现负采样，而不仅仅是在 `forward` 中使用完整 softmax。在 100 个句子的语料库上测量训练时间差异。解释为什么负采样有效，尽管它从未直接计算完整的概率分布。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 嵌入 | 词的向量表示 | 学习到的、密集的、固定维度的浮点数数组（通常为 100-300）。 |
| Skip-gram | 用中心词预测上下文 | 为每个中心词输出 k 个训练对。 |
| CBOW | 对上下文词取平均 | 将多个上下文向量合并为一个表示。 |
| 负采样 | 加快训练速度 | 对 k 个噪声词进行二分类，而不是对 V 个词进行多类分类。 |
| 分布假设 | 单词由其上下文定义 | 嵌入有效的最重要的理论基础。 |
