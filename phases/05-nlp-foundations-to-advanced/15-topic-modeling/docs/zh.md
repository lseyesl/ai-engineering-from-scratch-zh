# 预 Transformer 时代的文本生成 — 统计语言模型

> "the" 之后最常见的是什么？"cat"比"run"常见 10 倍。语言模型就是编码这个信息的。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 2 · 13（概率与贝叶斯）
**时间：** 约 50 分钟

## 问题

在 GPT 出现之前，语言模型只是一个计算词序列概率的概率函数。P("the cat sat") > P("sat the cat")，因为英语更喜欢特定的词序。

语言模型是所有生成任务的基础：机器翻译（我们希望译入语是流畅的）、语音识别（"recognize speech" vs "wreck a nice beach"）、拼写纠正、文本补全。在 Transformers 出现之前，N-gram 语言模型是标准方案。

## 概念

**N-gram 语言模型** 计算 P(w_n | w_{n-1}, ..., w_{n-N+1})——根据前 N-1 个词预测下一个词的条件概率。如果 N=2（bigram 模型），P(cat | the) = count("the cat") / count("the")。如果 N=3（trigram 模型），P(sat | the, cat) = count("the cat sat") / count("the cat")。

计数是通过在训练语料库中统计所有 N-gram 的频率来估计的。基本的训练过程不过是计数和除法，但平滑技术解决了零概率的问题。

**困惑度（Perplexity）** 是语言模型的主要评估指标。它是测试集上负对数似然的指数：PPL = exp(-1/N * sum(log P(w_i | context)))。较低的困惑度意味着更好的模型。对于均匀随机词，困惑度 = 词汇量大小。

## 构建

### 步骤 1：Bigram 计数

```python
from collections import defaultdict, Counter
import math
import random

class BigramLM:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.total = Counter()

    def train(self, sentences):
        for sentence in sentences:
            tokens = ["<S>"] + sentence.split() + ["</S>"]
            for i in range(len(tokens) - 1):
                self.counts[tokens[i]][tokens[i+1]] += 1
                self.total[tokens[i]] += 1
```

Bigram 模型将 "the cat sat" 处理为：P(cat|<S>)、P(the|cat)、P(sat|the)、P(</S>|sat) 的学习。`<S>` 和 `</S>` 是句子边界标记，允许模型学习句子开头和结尾的概率。

### 步骤 2：概率计算与加一平滑

```python
def prob(self, word, prev_word):
    """带加一平滑的条件概率。"""
    numerator = self.counts[prev_word][word] + 1
    denominator = self.total[prev_word] + self.vocab_size
    return numerator / denominator
```

加一平滑（拉普拉斯平滑）为每个可能的 2-gram 添加一次计数。这可以防止模型输出零概率，但会使概率分布趋于均匀——罕见词的可能性比应有可能更高。更好的平滑方法包括 Good-Turing 估计和 Kneser-Ney 平滑。

### 步骤 3：文本生成

```python
def generate(self, max_length=20):
    tokens = ["<S>"]
    for _ in range(max_length):
        prev = tokens[-1]
        next_words = list(self.counts.keys())
        probs = [self.prob(w, prev) for w in next_words]
        chosen = random.choices(next_words, weights=probs, k=1)[0]
        if chosen == "</S>":
            break
        tokens.append(chosen)
    return " ".join(tokens[1:])
```

从分布中采样（而非取最可能的词）会产生多样化的输出，但输出的流畅性较差。取最可能的词（贪婪解码）会陷入重复模式，因为大语言模型往往对少数高频词赋予高概率。在统计质量与采样多样性之间权衡是语言模型生成的核心问题。

### 步骤 4：困惑度

```python
def perplexity(self, sentence):
    tokens = ["<S>"] + sentence.split() + ["</S>"]
    log_prob = 0
    n = len(tokens) - 1
    for i in range(n):
        p = self.prob(tokens[i+1], tokens[i])
        log_prob += math.log(p + 1e-10)
    return math.exp(-log_prob / n)
```

困惑度可以被认为是有效词汇量大小。如果 PPL = 100，则模型在每个步骤中从大约 100 个可能的下一个词中进行选择。如果 PPL = 5000，则模型基本上是随机猜测的——接近词汇表大小。

## 使用

### KenLM（生产级 N-gram 模型）

```python
import kenlm

model = kenlm.Model("wiki.arpa")
log_prob = model.score("the cat sat", bos=True, eos=True)
perplexity = model.perplexity("the cat sat")
```

KenLM 支持回退（backoff）和插值（interpolation），使用修剪来平衡内存和准确性。一个包含 20 万词汇的 5-gram 模型大约占用 5GB 内存——对于生产使用来说仍然是可行的。

### 与 Transformer 的比较

| 方面 | N-gram | Transformer |
|------|--------|-------------|
| 上下文窗口 | N 个词 | 512-8192 个词元 |
| 零概率 | 需要平滑 | 从未见过零（softmax 保证非零） |
| 词汇表 | 固定，常为 50000 | 固定，常为 32000-128000 个词元 |
| 训练 | 单次扫描（计数） | 在 GPU 上进行多次 epoch |
| 困惑度（维基百科） | ~250（5-gram） | ~20（GPT-2） |
| 生成质量 | 在 3-4 个词后开始偏离 | 在 30+ 个词内保持连贯 |

## 发布

N-gram 与神经网络语言模型的决策矩阵。

保存为 `outputs/prompt-lm-choice.md`：

```markdown
---
name: lm-choice
description: 提示：根据约束条件选择语言模型。
phase: 5
lesson: 15
---

根据以下条件在 N-gram 和神经网络语言模型之间进行选择：

1. 数据规模：如果数据 < 1M 词，N-gram 加平滑表现和神经网络模型一样好，且训练速度快 100 倍。
2. 延迟：如果需要 < 1ms 的词生成，使用 N-gram。神经网络模型即使在 GPU 上也需要更多时间。
3. 上下文长度：如果任务需要 > 5 个词的上下文，使用神经网络模型。N-gram 无法扩展。
4. 资源：如果只有 CPU，N-gram 可以处理，神经网络模型可能太慢。
5. 罕见词：如果很多词在训练集中只出现一次，N-gram 几乎无法处理。神经网络模型使用嵌入来泛化。
```

## 练习

1. **简单。** 在《爱丽丝梦游仙境》上训练一个 bigram 语言模型。生成 5 个句子。其中有多少个是语法正确的？
2. **中等。** 比较 bigram 和 trigram 模型的困惑度。困惑度随着 N 增加而下降了多少？增加 n-gram 的 N 值什么时候开始出现回报递减？
3. **困难。** 实现 Kneser-Ney 平滑并替换加一平滑。在测试集困惑度上，Kneser-Ney 比加一平滑改进了多少？为什么？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| N-gram | N 个连续词 | 训练语料中发现的大小为 N 的词序列。 |
| 平滑 | 防止零概率 | 将概率质量从已知事件重新分配给未见过的 N-gram 的技术。 |
| 困惑度 | 评估指标 | exp(负对数似然)。有效词汇量大小。越低越好。 |
| 回退 | 缩短上下文 | 当 N-gram 不可见时，使用 (N-1)-gram 概率作为底层估计。 |
| KenLM | N-gram 工具包 | 生产级 N-gram 模型构建与评分。 |
