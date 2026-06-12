# 子词分词 — BPE、WordPiece、SentencePiece

> 没有单独的"猫"这个词元。"猫"是 "ca" + "t"。这就是 Transformer 如何处理所有语言的方式。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 01（分词）
**时间：** 约 50 分钟

## 问题

英语将词用空格分隔开。中文、日文和泰文没有词边界。土耳其语将一个英语句子用一个词来表达。统一的语言无关的分词方案是现代 NLP 的基础。

子词分词通过在出现时拆分罕见词来平衡词汇表大小和覆盖范围。"lower" 可以被拆分为 "low" + "er"。"low" 仍然是一个词元。罕见词被拆分为频繁已知子词的序列。每个 Transformer 模型都使用子词分词。

## 概念

**字节对编码（BPE）** 从单个字符的词汇表开始，并迭代地合并最频繁出现的相邻对。每次合并都会创建一个新的词元。合并重复进行，直到达到目标词表大小。"l" + "o" 经常相邻出现 → BPE 创建词元 "lo"。"lo" + "w" 经常出现 → BPE 创建词元 "low"。

**WordPiece** 与 BPE 类似，但不是按频率合并，而是按互信息合并——计算一个对的似然比。"lo" + "w" 是否比单独出现这两个词元时更频繁地共同出现？如果是，就合并。

**SentencePiece** 将原始文本（含空格）作为输入，并直接从原始文本学习 BPE 或 unigram 分词模型。它不假设空格是分词边界，因此它原生地适用于中文和日语。

## 构建

### 步骤 1：BPE 训练

```python
from collections import defaultdict
import re

class BPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}

    def train(self, corpus):
        # 初始词汇表：字符 + 词尾标记
        word_counts = defaultdict(int)
        for text in corpus:
            for word in text.split():
                word_counts[" ".join(list(word)) + " </w>"] += 1

        # 字符词汇表
        chars = set()
        for word in word_counts:
            for char in word.split():
                chars.add(char)

        self.vocab = {c: i for i, c in enumerate(sorted(chars))}
        next_idx = len(self.vocab)

        # 合并
        for _ in range(self.vocab_size - len(chars)):
            pairs = defaultdict(int)
            for word, count in word_counts.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += count

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = next_idx
            next_idx += 1

            new_vocab = defaultdict(int)
            for word, count in word_counts.items():
                new_word = word.replace(
                    " ".join(best_pair),
                    "".join(best_pair)
                )
                new_vocab[new_word] += count
            word_counts = new_vocab
```

BPE 通过统计子词对的频率从训练语料库中学习合并规则。学习到的合并规则构成了分词器——一个查找表，用于以相同的方式拆分新的文本。词汇表大小是一个超参数。

### 步骤 2：BPE 分词

```python
def tokenize(self, text):
    words = text.split()
    tokens = []
    for word in words:
        word = " ".join(list(word)) + " </w>"
        while len(word.split()) > 1:
            pairs = []
            symbols = word.split()
            for i in range(len(symbols) - 1):
                if (symbols[i], symbols[i+1]) in self.merges:
                    pairs.append(((symbols[i], symbols[i+1]), i))
            if not pairs:
                break
            # 应用最早（最低优先级）的合并
            pair, idx = min(pairs, key=lambda x: self.merges.get(x[0], float("inf")))
            word = " ".join(symbols[:idx]) + " " + "".join(pair) + " " + " ".join(symbols[idx+2:])
            word = word.strip()
        tokens.extend(word.split())
    return [t for t in tokens if t != "</w>"]
```

BPE 分词通过查找学习到的合并对序列来工作。它总是执行最早的合并——BPE 规则以确定性方式应用，输入文本的分词方式是固定的。

### 步骤 3：Byte-level BPE（GPT-2 的方法）

```python
def bytes_to_unicode():
    """GPT-2 的 bytes-to-unicode 映射。"""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
```

字节级 BPE 将每个字节视为一个字符，而不是将 UTF-8 码点作为字符。这意味着它可以在任何语言的任何文本上进行分词，而不需要"未知字符"的概念。每个文本，无论使用什么语言，都可以表示为字节序列。这是像 GPT-2、GPT-4、LLaMA 和许多其他现代模型的标准。

## 使用

### HuggingFace 分词器

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=5000, special_tokens=["<UNK>", "<BOS>", "<EOS>"])
tokenizer.train(["corpus.txt"], trainer)

output = tokenizer.encode("The cat sat on the mat.")
print(output.tokens)
```

```python
['The', 'cat', 'sat', 'on', 'the', 'mat', '.']
```

对于像"mat"这样的常见词，BPE 将其保留为单个词元。对于拼写错误的词"mkat"，它可能会拆分为 "mk" + "at" + "."

### SentencePiece（LLaMA、Gemma）

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="m",
    vocab_size=32000,
    character_coverage=0.9995,
    model_type="bpe"
)

sp = spm.SentencePieceProcessor()
sp.load("m.model")
tokens = sp.encode("The cat sat on the mat.", out_type=str)
```

SentencePiece 将输入视为原始字节流，包括空格。词汇表大小决定了分词器的压缩率。词汇表 = 32000（标准）会产生约 1.3 个词元/词的比率。词汇表 = 128000（GPT-4）会产生约 0.7 个词元/词的比率——更长的上下文窗口，但更大的词汇表。

## 发布

分词器选择的决策提示。

保存为 `outputs/prompt-tokenizer-choice.md`：

```markdown
---
name: tokenizer-choice
description: 提示：为 NLP 任务选择子词分词器。
phase: 5
lesson: 18
---

为文本任务推荐分词器：

1. 如果目标是最大覆盖范围 → 使用字节级 BPE（GPT-2 风格）。覆盖所有 Unicode，没有未知字符。
2. 如果目标是中文/日语 → 使用 SentencePiece unigram。不假定空格边界。
3. 如果目标是语言保持不变性 → 使用 SentencePiece BPE。跨模型可重现。
4. 如果目标是嵌入对齐 → 使用与模型相同的分词器。如果你使用 BERT，使用 WordPiece。
5. 如果目标是效率 → 针对压缩率优化词汇表大小。32000 词元适用于通用任务，64000 词元可提高罕见词覆盖。

始终检查：你的词汇表中 `<UNK>` 的频率。如果超过 0.1%，词汇表太小或训练语料库不足以覆盖你的领域。
```

## 练习

1. **简单。** 训练一个词汇量为 500 的 BPE 分词器。在一个包含 10 个句子的语料库上对其进行训练，并观察合并的过程。
2. **中等。** 将训练好的 BPE 应用于"mkaitypingerror"和"低资源语言中的罕见词长词形变化"——检查这些词是如何被分割的。
3. **困难。** 比较词汇表大小为 1000、32000 和 128000 的 BPE 分词器。测量一个 1MB 测试集的分词精度、词元计数和恢复率。在词汇表大小与分词质量之间作出取舍。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| BPE | 字节对编码 | 子词分词：从字符开始，合并最频繁出现的相邻对。 |
| WordPiece | 概率合并 | 类似 BPE 但不是按频率合并，而是按互信息合并。 |
| SentencePiece | 原始输入分词器 | 将原始文本（含空格）作为输入，不预设空格分界。 |
| 字节级 BPE | 字节覆盖 | 将每个字节视为一个字符，而非 Unicode 码点。 |
| 词元 | 输出单元 | 模型实际处理的原子单元。 |
