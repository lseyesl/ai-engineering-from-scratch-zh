# 情感分类 — 从规则到上下文

> "不好"是否定的。"不好，真的很好"是肯定的。语境决定一切。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 2 · 14（逻辑回归），阶段 5 · 02（BoW/TF-IDF）
**时间：** 约 50 分钟

## 问题

情感分类是 NLP 的"Hello World"。给定一段文本，确定它是正面、负面还是中性。简单到可以用一行代码解决，复杂到你可以写一篇博士论文。

棘手之处在于否定："不好"转向负面，"不差"转向正面。"我本来打算喜欢这部电影，但……"是一段充满情感的文本，带有强烈的二元标签，但情感轨迹复杂。普通的词袋模型无法胜任此任务——它将"好"和"不好"都计入"好"的计数。

在本课中，你在一个完整的情感数据集上构建一个从词袋模型到逻辑回归再到 LSTM 的流程，观察每一次表示变化如何改变决策边界。

## 概念

情感分类的三种方法随着表示复杂度的增加而呈现分层结构。

**基于规则的**使用词典（例如，AFINN 为"好"分配 +2，为"坏"分配 -2，并进行求和）。速度极快，但忽略了结构——"不好"与"好"得到的分数相同。适用于冷启动，但在需要可靠性的场景下无法投入生产。

**基于特征（词袋模型）的**学习每个词的权重。逻辑回归在每个词与其极性之间建立线性关联。比基于规则的方法更好，但仍然独立的处理词——"不好"和"非常好"都仅以"好"来衡量。

**上下文方法（RNN、LSTM、Transformer）**按顺序处理词。"好"前面是"不"还是"非常"决定了其方向。捕获否定范围和程度变化是为什么基于向量的上下文模型设定了基线。

```figure
sentiment-logits
```

## 构建

### 步骤 1：基于规则的情感——AFINN 风格

```python
AFINN = {
    "good": 2, "great": 3, "excellent": 4, "wonderful": 4,
    "bad": -2, "terrible": -3, "awful": -4, "horrible": -4,
    "not": -1, "no": -1, "never": -2, "hate": -3,
    "love": 3, "amazing": 4, "poor": -2, "boring": -2,
}

def rule_sentiment(text):
    words = text.lower().split()
    score = sum(AFINN.get(w, 0) for w in words)
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"
```

```python
>>> rule_sentiment("This movie is good")
'positive'
>>> rule_sentiment("This movie is not good")
'positive'  # 错误！"not good" 是负面的
>>> rule_sentiment("This movie is terrible")
'negative'
```

第一个错误（`"not good"`）是文本中基于规则的情感分类的核心失败模式。词典不知道否定词。如果 AFINN 包含 `"not_good": -1` 作为二元组特征，可以部分缓解这个问题，但这需要手动列出所有可能的否定词修饰搭配。

### 步骤 2：使用 BoW + 逻辑回归学习权重

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("vectorize", CountVectorizer(max_features=5000)),
    ("classify", LogisticRegression(C=1.0, max_iter=1000)),
])

pipeline.fit(train_texts, train_labels)
predictions = pipeline.predict(test_texts)
```

这学习了每个词的权重，而不是硬编码它们。`"good"` 可能获得 +0.5 的权重，`"bad"` 获得 -0.6。否定词如 `"not"` 可能获得轻微的负权重，但 BoW 无法捕获组合效应——线性模型无法推断出 `"good"` 的正面权重在 `"not"` 出现时应被撤销。

```python
# 查看学习到的权重
weights = pipeline.named_steps["classify"].coef_[0]
vocab = pipeline.named_steps["vectorize"].get_feature_names_out()
word_weights = dict(sorted(
    [(v, w) for v, w in zip(vocab, weights)],
    key=lambda x: x[1], reverse=True
))
```

前几个词通常是：`"excellent"` (+1.2)、`"amazing"` (+1.1)、`"worst"` (-1.4)、`"terrible"` (-1.3)。如果 `"not"` 的权重接近零，那是因为它大致均匀地分布在正面和负面文档中——它对"好"和"坏"的使用频率一样高。

### 步骤 3：二元组特征捕获局部否定

```python
from sklearn.feature_extraction.text import CountVectorizer

bigram_vec = CountVectorizer(ngram_range=(1, 2), max_features=10000)
```

`ngram_range=(1, 2)` 在词汇表中添加了像 `"not_good"`、`"not_bad"`、`"very_good"` 这样的二元组。模型可以学习 `"not_good"` 的负权重，即使 `"good"` 本身具有正权重。二元组扩展大大提高了情感准确性——将 RoTTEN IMDB 数据集上的准确率从 82% 提高到 86%。

代价是特征爆炸：10 万个词在包含二元组时很容易变成 500 万个特征。使用 `min_df=5` 过滤掉罕见的二元组。

### 步骤 4：序列模型捕获范围

```python
# 概念性 PyTorch 情感 LSTM
# 嵌入 → LSTM → 池化 → 线性 → softmax

class SentimentLSTM:
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        self.embedding = ...  # (vocab_size, embedding_dim)
        self.lstm = ...      # (embedding_dim, hidden_dim)
        self.classifier = ... # (hidden_dim, 3)
```

完整代码在阶段 7（RNN 与 LSTM）中，但概念很清晰：LSTM 按顺序读取词，维护一个状态。当它读取 `"not"` 时更新状态，当读取 `"good"` 时使用该状态来翻转极性。否定范围被隐式地捕获，因为 `"good"` 的状态向量已经包含前一个词方向。

## 使用

### HuggingFace 情感分类

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("This movie is not good")
```

```python
[{'label': 'NEGATIVE', 'score': 0.998}]
```

`pipeline` 内部使用一个在大型情感数据集上微调的 BERT 模型。它处理否定词、程度变化，以及否定范围——"不好"被准确地归类为负面。不需要特征工程，没有词典，没有数据加载。只需一行代码。

参数：使用 `model="distilbert-base-uncased-finetuned-sst-2-english"` 以获得更小的模型，或使用 `cardiffnlp/twitter-roberta-base-sentiment-latest` 以获得最新的社交网络数据。

### 检查是否有空字符串

```python
def safe_classify(text, fallback="neutral"):
    if not text or not text.strip():
        return fallback
    return classifier(text)[0]
```

情感分类器在空字符串或仅含空格的文本上的行为是不可预测的。一些返回 `POSITIVE`（因为空字符串与训练分布正交），一些抛出异常。事先检查可以消除这种不确定性。

## 发布

情感分类验证检查清单。

保存为 `outputs/prompt-sentiment-validation.md`：

```markdown
---
name: sentiment-validation
description: 验证情感分类准备是否就绪。
phase: 5
lesson: 05
---

验证情感分类系统。输入：标签为正面/负面/中性的样本。检查：

1. 否定处理：`"not good"` 应为负面。`"not bad"` 应为正面。如果两者都失败，则否定未被捕获。
2. 程度变化：`"good"` < `"very good"` < `"excellent"`。如果程度未被区分，则模型未利用强度。
3. 中性检测：`"the sky is blue"` 应为中性。如果模型始终将中性内容归类为某一极性，则存在类别偏差。
4. 空输入：覆盖空/空白输入的输出。不存在漏洞。
5. 讽刺阻力：`"great, another meeting"` — 预期为负面。纯文本讽刺检测是一个开放问题；指明当前限制。
```

## 练习

1. **简单。** 在 AFINN 词汇表中添加 `"not_good": -1`、`"not_bad": 1` 等条目。重新运行之前的失败案例。修复了哪些问题？还剩下哪些问题？
2. **中等。** 在 IMDB 评论数据集的子集上训练一个 TF-IDF + 逻辑回归情感分类器。报告训练/测试准确率和混淆矩阵。比较一元组和二元组特征。
3. **困难。** 实现否定范围检测：给定一个带有否定词（"not"、"never"）的句子，识别从否定词到下一个标点符号的所有词。翻转这些词的情感极性分配，与原始逻辑回归相比，是否改善了结果？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 否定范围 | "not" 之后的内容 | 从否定词到下一个标点符号的词序列，其极性会被反转。 |
| 情感词典 | 极性与得分 | 硬编码单词-情感映射（AFINN、SentiWordNet、VADER）。 |
| 程度变化 | 强度级 | "good" → "very good" → "excellent"。LSTM 通过顺序捕获这一点；BoW 无法处理。 |
| 极性 | 正面/负面/中性 | 情感的基础三向分类。多分类系统可以添加"混合"或"愤怒的"类别。 |
