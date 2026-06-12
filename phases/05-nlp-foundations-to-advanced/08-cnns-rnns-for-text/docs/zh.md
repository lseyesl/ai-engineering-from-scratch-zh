# 用于文本的 CNN 与 RNN — 位置与顺序

> 卷积捕获模式。循环捕获顺序。文本两者都需要。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 3（CNN），阶段 3 · 17（RNN）
**时间：** 约 55 分钟

## 问题

文本分类通常是这样的：获取词嵌入，通过池化或递归层处理它们，然后输出一个标签。但如何从嵌入到标签是模型架构的选择。

CNN 和 RNN 对文本进行两种不同的假设。CNN 假设局部模式（"not good" 作为一个 n-gram）是分类所需的全部。RNN 假设长距离依存关系（"not … but actually"）很重要。两者都可以用于文本——一个适合文档级分类，另一个适合序列标注——但如果你选择了错误的那个，你就会与你的数据性质对抗。

## 概念

**用于文本的 CNN** 沿着序列维度滑动卷积滤波器。滤波器宽度（内核大小）控制 n-gram 的覆盖范围。宽度为 3 的滤波器同时查看三个连续的嵌入——相当于三元组。多个滤波器学习不同的 n-gram 模式。最大池化选择每个滤波器的最大激活值，识别句子中最重要的模式。

**用于文本的 RNN** 按顺序处理 token，每一步更新隐藏状态。最终隐藏状态（或所有状态的池化结果）是句子表示。双向 RNN 向前和向后读取句子，为每个位置捕获两个方向的上下文。适合序列标注（NER、POS），因为每个位置都有上下文感知的表示。

主要的经验法则是：CNN 用于分类，RNN 用于标记。实际上，Transformer 在做这两方面都更好，但理解差异解释了为什么 Transformer 使用自注意力（它结合了 CNN 的并行性和 RNN 的顺序建模能力）。

```figure
rnn-unroll
```

## 构建

### 步骤 1：用于文本分类的 CNN

```python
class TextCNN:
    """概念性 TextCNN，使用三个不同的滤波器大小。"""
    def __init__(self, vocab_size, embedding_dim=100,
                 filter_sizes=[3, 4, 5], num_filters=100, num_classes=2):
        self.embedding = [[random.gauss(0, 0.01) for _ in range(embedding_dim)]
                         for _ in range(vocab_size)]
        self.filters = {}
        for f_size in filter_sizes:
            self.filters[f_size] = [
                [[random.gauss(0, 0.01) for _ in range(embedding_dim)]
                 for _ in range(f_size)]
                for _ in range(num_filters)
            ]
        self.classifier = [[random.gauss(0, 0.01)
                           for _ in range(len(filter_sizes) * num_filters)]
                          for _ in range(num_classes)]
```

TextCNN 为每种滤波器大小学习一组卷积滤波器。宽度为 3 的滤波器捕获三元组模式，宽度为 5 的滤波器捕获五元组模式。每个滤波器生成一个特征图（滑动点积的序列），然后对其进行最大池化以产生一个标量。所有标量连接成一个向量并输入分类器。

```python
# 概念性卷积 + 池化
def conv_and_pool(self, sentence_indices, filter_size):
    embeddings = [self.embedding[idx] for idx in sentence_indices]
    results = []
    for f_idx, kernel in enumerate(self.filters[filter_size]):
        feature_map = []
        for i in range(len(embeddings) - filter_size + 1):
            # 滑动窗口点积
            value = 0
            for j in range(filter_size):
                for d in range(len(kernel[j])):
                    value += embeddings[i+j][d] * kernel[j][d]
            feature_map.append(value)
        # 最大池化
        results.append(max(feature_map) if feature_map else 0)
    return results
```

### 步骤 2：用于文本的 RNN（前向）

```python
class SimpleRNN:
    """概念性 RNN，用于文本的隐藏状态更新。"""
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        self.embedding = [[random.gauss(0, 0.01) for _ in range(embedding_dim)]
                         for _ in range(vocab_size)]
        self.W_hh = [[random.gauss(0, 0.01) for _ in range(hidden_dim)]
                    for _ in range(hidden_dim)]
        self.W_xh = [[random.gauss(0, 0.01) for _ in range(embedding_dim)]
                    for _ in range(hidden_dim)]

    def forward(self, sentence_indices):
        h = [0.0] * len(self.W_hh)
        for idx in sentence_indices:
            x = self.embedding[idx]
            # h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t)
            h_new = [0.0] * len(h)
            for i in range(len(h)):
                for j in range(len(h)):
                    h_new[i] += self.W_hh[i][j] * h[j]
                for j in range(len(x)):
                    h_new[i] += self.W_xh[i][j] * x[j]
                h_new[i] = math.tanh(h_new[i])
            h = h_new
        return h  # 最终隐藏状态 = 句子表示
```

在每个步骤中，新隐藏状态是前一个状态和当前嵌入的函数。最终隐藏状态包含从句子开头到结尾的聚合信息。问题：梯度消失。第 1 步的信号在第 20 步之后变得非常弱。LSTM 和 GRU 通过门控机制解决这个问题（阶段 7）。

### 步骤 3：为什么方向很重要

```python
# "The movie was not good at all." — 负面情感
# 从左到右的 RNN 逐步处理直到结尾：
# "good" 出现在步骤 5，但到那时否定范围（步骤 3-4）已经被编码进隐藏状态。
# 方向性很重要：向前 + 向后 → 双向上下文。
```

这就是为什么在文本中使用双向 RNN 更常见——每个位置的表示既包含左侧上下文的总结（向前 RNN），也包含右侧上下文的总结（向后 RNN）。

## 使用

### PyTorch TextCNN

```python
import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100,
                 filter_sizes=[3, 4, 5], num_filters=100, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, f_size) for f_size in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        pooled = [nn.functional.max_pool1d(
            nn.functional.relu(conv(x)), conv(x).shape[-1]
        ).squeeze(-1) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Conv1d 在嵌入维度上操作，沿序列长度滑动。`permute` 将形状从 (batch, seq_len, embed_dim) 转换为 (batch, embed_dim, seq_len)，因为 PyTorch 的 Conv1d 期望通道维度在第二个位置。

## 发布

用于将 CNN 与 RNN 应用于文本的决策提示。

保存为 `outputs/prompt-text-architecture-comparison.md`：

```markdown
---
name: text-architecture-comparison
description: 根据任务和资源比较 CNN、RNN 和 Transformer 在文本上的应用。
phase: 5
lesson: 08
---

给定文本任务、数据集大小和硬件，推荐架构：

1. 如果仅有 CPU + 分类 → TextCNN。速度极快，在长文档上表现良好，可并行化。
2. 如果序列标注（NER、POS） → BiLSTM + CRF。每个位置都需要双向上下文。
3. 如果长序列（>512 tokens） → BiLSTM 或 Longformer。Transformer 在二次方扩展下失效。
4. 如果资源充足 + 最大精度 → Transformer（BERT/RoBERTa）。在概念上融合了 CNN（并行注意力）和 RNN（序列建模）。
5. 如果推理延迟 < 10ms → 蒸馏 Transformer 或将 CNN 知识蒸馏到小模型中。

对于情感、主题分类等标准文本分类，TextCNN 在速度和质量之间提供了最佳平衡。对于标记任务，BiLSTM 仍然很强势。
```

## 练习

1. **简单。** 使用词嵌入 + 平均池化创建文档向量。对于 `["good movie", "bad movie", "great film"]` 计算余弦相似度。它们是否按照你的期望聚类？
2. **中等。** 实现一个 TextCNN，比较 `filter_sizes=[2]` 和 `filter_sizes=[5]` 之间的最大池化激活情况。如果所有激活都相似，说明什么？
3. **困难。** 训练一个 RNN 进行情感分类。比较它在 `"not good"` 和 `"good"` 上的表现与你之前的逻辑回归基线。向 RNN 添加多少参数才会有显著变化？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| TextCNN | 文本的 CNN | 在词嵌入序列上使用 1D 卷积滤波器的 CNN。 |
| 滤波器大小 | 滑动窗口宽度 | 也被称为内核大小。定义 n-gram 覆盖范围。 |
| 特征图 | 卷积输出 | 滑动点积的序列——每个位置处模式匹配强度。 |
| 最大池化 | 最显著特征 | 通过取最大值从特征图中提取最强信号。 |
| 隐藏状态 | RNN 的记忆 | 在每一步中，该步骤所有先前输入的摘要。 |
| 双向 | 前后方向 | 两个 RNN 分别沿序列的向前和向后方向读取并拼接输出。 |
