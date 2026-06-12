# Seq2Seq 与注意力机制 — 从序列到序列

> 编码器读取。注意力选择。解码器写入。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 3 · 17（RNN），阶段 5 · 08（用于文本的 CNN/RNN）
**时间：** 约 60 分钟

## 问题

机器翻译面临一个基本挑战：输入序列和输出序列具有不同的长度。"The cat sat on the mat" 可能翻译成"猫坐在垫子上"（5 个词到 6 个词）。你如何将相同的信息投影到不同的长度上？

Seq2Seq（编码器-解码器）通过两个 RNN——一个读取输入序列，另一个生成输出序列——并通过一个固定维度的向量（瓶颈）连接它们来解决这个问题。瓶颈是主要问题：无论输入序列有多长，信息都必须压缩到最后一个隐藏状态中。注意力机制通过在每一步允许解码器访问所有编码器隐藏状态来修复这个问题。

## 概念

**编码器**是一个 RNN，它按顺序读取输入 token，输出隐藏状态的序列（每个输入位置一个）。最后一个隐藏状态是整个输入序列的摘要。

**解码器**是另一个 RNN，它以编码器的最后一个隐藏状态为初始状态，并逐步生成输出 token。在每一步，它接收前一步的输出 token 及其隐藏状态。

**注意力机制**不是在每一步传递相同的最后编码器状态，而是计算编码器所有隐藏状态的加权和，并选择与当前解码器步骤最相关的隐藏状态。这使梯度能够通过瓶颈直接传播到相关的编码器位置。

```figure
lstm-gates
```

## 构建

### 步骤 1：概念性编码器

```python
class EncoderRNN:
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        self.embedding = [[random.gauss(0, 0.01) for _ in range(embedding_dim)]
                         for _ in range(vocab_size)]
        self.W_hh = [[random.gauss(0, 0.01) for _ in range(hidden_dim)]
                    for _ in range(hidden_dim)]
        self.W_xh = [[random.gauss(0, 0.01) for _ in range(embedding_dim)]
                    for _ in range(hidden_dim)]

    def forward(self, input_indices):
        hidden_states = []
        h = [0.0] * len(self.W_hh)
        for idx in input_indices:
            x = self.embedding[idx]
            h_new = [0.0] * len(h)
            for i in range(len(h)):
                for j in range(len(h)):
                    h_new[i] += self.W_hh[i][j] * h[j]
                for j in range(len(x)):
                    h_new[i] += self.W_xh[i][j] * x[j]
                h_new[i] = math.tanh(h_new[i])
            h = h_new
            hidden_states.append(h)
        return hidden_states, h  # 所有状态 + 最终状态
```

编码器为每个输入位置输出一个隐藏状态。没有注意力机制的解码器只接收最后的 `h`。

### 步骤 2：概念性注意力

```python
class Attention:
    def __init__(self, hidden_dim=512):
        self.W_a = [[random.gauss(0, 0.01) for _ in range(hidden_dim)]
                   for _ in range(hidden_dim)]

    def score(self, decoder_h, encoder_h):
        """计算单个编码器状态与解码器状态的相关性。"""
        value = 0
        for i in range(len(decoder_h)):
            for j in range(len(encoder_h)):
                value += decoder_h[i] * self.W_a[i][j] * encoder_h[j]
        return value

    def forward(self, decoder_hidden, encoder_hiddens):
        scores = [self.score(decoder_hidden, h) for h in encoder_hiddens]
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        sum_exps = sum(exps)
        weights = [e / sum_exps for e in exps]

        # 通过注意力权重对编码器隐藏状态进行加权求和
        context = [0.0] * len(encoder_hiddens[0])
        for i, h in enumerate(encoder_hiddens):
            for d in range(len(h)):
                context[d] += weights[i] * h[d]
        return context, weights
```

`score` 测量解码器的当前隐藏状态与每个编码器状态之间的对齐程度。softmax 将得分转化为权重。上下文向量是编码器隐藏状态的加权和——注意力选择的内容。权重可以可视化：注意力图突出显示解码器在每一步关注输入序列中的哪些位置。

### 步骤 3：概念性注意力解码器

```python
class AttnDecoderRNN:
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        self.embedding = [[random.gauss(0, 0.01) for _ in range(embedding_dim)]
                         for _ in range(vocab_size)]
        self.attention = Attention(hidden_dim)
        self.W_hh = [[random.gauss(0, 0.01) for _ in range(hidden_dim)]
                    for _ in range(hidden_dim)]
        self.W_xh = [[random.gauss(0, 0.01) for _ in range(embedding_dim)]
                    for _ in range(hidden_dim)]
        self.W_c = [[random.gauss(0, 0.01) for _ in range(hidden_dim * 2)]
                   for _ in range(hidden_dim)]

    def forward(self, target_indices, encoder_hiddens, start_token=0):
        h = encoder_hiddens[-1]  # 初始化解码器状态
        x = self.embedding[start_token]
        outputs = []

        for idx in target_indices:
            context, weights = self.attention.forward(h, encoder_hiddens)
            # 将上下文和嵌入结合
            combined = context + x  # 简化：连接 + 投影
            h_new = [0.0] * len(h)
            for i in range(len(h)):
                for j in range(len(h)):
                    h_new[i] += self.W_hh[i][j] * h[j]
                for j in range(len(combined)):
                    h_new[i] += self.W_xh[i][j] * combined[j] if j < len(self.W_xh[0]) else 0
                    h_new[i] += context[j] * self.W_c[i][j] if j < len(context) else 0
                h_new[i] = math.tanh(h_new[i])
            h = h_new
            outputs.append(h)
            x = self.embedding[idx]

        return outputs
```

注意力上下文在每一步都添加到解码器输入中。这就是无注意力 Seq2Seq 总是在瓶颈处压缩信息与注意力 Seq2Seq 在每一步都可以直接查询相关信息之间的区别。

## 使用

### PyTorch（概念性框架）

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        return self.rnn(self.embed(x))

class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRUCell(embed_dim + hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, encoder_outputs, hidden):
        # 注意力分数
        attn_scores = self.attn(torch.cat(
            (encoder_outputs, hidden.expand(encoder_outputs.shape[0], -1)),
            dim=-1
        )).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=0)
        context = (attn_weights.unsqueeze(-1) * encoder_outputs).sum(0)
        rnn_input = torch.cat((self.embed(x), context), dim=-1)
        hidden = self.rnn(rnn_input, hidden)
        return self.out(hidden), hidden, attn_weights
```

`nn.GRU` 比手动 RNN 更高效，并内置了门控机制以解决梯度消失问题。注意力分数计算为 `W @ concat(encoder_states, decoder_state)`——这就是加性注意力（Bahdanau），与乘法注意力（Luong）相对。

## 发布

机器翻译评估检查清单。

保存为 `outputs/prompt-mt-eval.md`：

```markdown
---
name: mt-eval
description: 检查 MT 系统 output 质量，而不仅仅是 BLEU。
phase: 5
lesson: 09
---

评估机器翻译。给定源文本、系统译文和参考译文：

1. 忠实度：所有源信息都存在于译文中吗？查找遗漏的实体或数字。
2. 流畅度：译文是惯用表达吗？查找不自然的措辞。
3. 词序：对于具有不同语序的语言对（例如英语 → 日语），词序是否正确？
4. 罕见词：生僻词和命名实体是否被正确翻译？有无回退到未翻译的源语言词？
5. 长句子：在 20+ 词长的句子上，质量是否会下降（注意力衰减的迹象）？

BLEU 评分：在 40 分以上表示良好；30 分以下表示存在严重问题。但始终检查样本——BLEU 喜欢与参考译文相似的译文，即使参考译文不好。
```

## 练习

1. **简单。** 可视化注意力权重。从你的注意力解码器中获取权重矩阵，并叠加热力图。注意力是否与合理的词对齐？
2. **中等。** 实现无注意力的 Seq2Seq（仅使用编码器的最后一个隐藏状态）。比较添加注意力前后，BLEU 分数在长句（10+ 词）上下降了多少。
3. **困难。** 实施教师强制（teacher forcing）与计划采样（scheduled sampling）。在同一个数据集上训练两种变体，并比较推理时的误差累积。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| Seq2Seq | 序列到序列 | 将输入序列映射到不同长度输出序列的编码器-解码器框架。 |
| 编码器 | 读取序列 | 处理输入 token 并产生隐藏状态的 RNN。 |
| 解码器 | 生成序列 | 根据编码的表示预测输出 token 的 RNN。 |
| 注意力 | 对齐机制 | 在每个解码步骤中允许解码器关注输入序列不同部分的加权和。 |
| 教师强制 | 真实输入训练 | 在训练期间将真实目标 token 而非预测值输入解码器；加速收敛，但会导致推理时暴露偏差。 |
