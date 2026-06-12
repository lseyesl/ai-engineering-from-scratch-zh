# 机器翻译 — 从规则到端到端

> 每种语言都是一个独立的程序。翻译是最高级别的逆向工程。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 09（Seq2Seq 与注意力机制）
**时间：** 约 50 分钟

## 问题

机器翻译是 Seq2Seq 的旗舰任务。输入：一种语言的文本。输出：另一种语言的文本。方法经历了三个时代：基于规则的（语言学家编写的语法）、统计的（基于短语的对齐模型）和神经的（端到端 Seq2Seq）。

本课将统计机器翻译（SMT）和神经机器翻译（NMT）并排对比。SMT 更易于理解——它可以逐个组件地进行检查——且在小众语言对上仍在使用。NMT 是当前的标准——更好的流畅度，但主要缺点是在罕见词上会"凭空捏造"（hallucinate）。

## 概念

**基于短语的 SMT** 将源句子分割成短语，翻译每个短语，然后重新排列。短语是在训练语料库中学习到的对齐：英语中的 "cat" 在法语中对齐到 "chat"。重新排序模型学习译入语的词序。语言模型确保输出流畅。

**NMT** 使用一个单一的神经网络来翻译。没有单独的组件。编码器读取源句子；解码器生成目标句子。注意力机制处理对齐。语言建模由解码器隐式完成。

## 构建

```figure
attention-heatmap
```

### 步骤 1：SMT 短语提取（概念性）

```python
from collections import defaultdict, Counter
import math

class PhraseExtractor:
    def __init__(self):
        self.phrase_table = defaultdict(Counter)

    def extract(self, bitext):
        for src, tgt in bitext:
            # 从词对齐到短语对齐
            for i in range(len(src)):
                for j in range(i + 1, len(src) + 1):
                    src_phrase = " ".join(src[i:j])
                    for k in range(len(tgt)):
                        for l in range(k + 1, len(tgt) + 1):
                            tgt_phrase = " ".join(tgt[k:l])
                            # 检查一致性：短语边界应与词对齐对齐
                            self.phrase_table[(src_phrase, tgt_phrase)] += 1

    def phrase_prob(self, src_phrase, tgt_phrase):
        counts = self.phrase_table.get((src_phrase, tgt_phrase), 0)
        total = sum(c for (sp, _), c in self.phrase_table.items() if sp == src_phrase)
        return counts / total if total > 0 else 0
```

短语提取扫描并行的双语句对的所有可能短语对。短语边界必须与词对齐对齐——这是数据稀疏性的来源（一致的对齐比随机切割更少），但确保了基于短语模型的可信度。

### 步骤 2：NMT 数据准备

```python
def prepare_nmt_data(src_sentences, tgt_sentences, src_vocab, tgt_vocab):
    src_tokenized = [s.strip().split() for s in src_sentences]
    tgt_tokenized = [s.strip().split() for s in tgt_sentences]

    src_indices = [[src_vocab.get(w, src_vocab["<UNK>"]) for w in s]
                   for s in src_tokenized]
    tgt_indices = [[tgt_vocab.get(w, tgt_vocab["<UNK>"]) for w in s]
                   for s in tgt_tokenized]

    return src_indices, tgt_indices
```

序列被转换为整数索引。常见的预处理：小写化、标点规范化、替换罕见词（`< freq` 的词）为 `<UNK>`。子词分词（BPE/WordPiece）通过将罕见词分割成更小的已知单位来消除 `<UNK>`（阶段 5 · 17）。

### 步骤 3：束搜索

在推理时，解码器在每个步骤从整个词汇表中采样，但贪婪解码（始终选择最高概率的词）会导致次优结果。束搜索跟踪 k 个部分翻译序列，并选择总体概率最高的序列。

```python
def beam_search(decoder, encoder_output, beam_size=4, max_len=50):
    # 束中的每个项目：(sequence, log_prob, decoder_state)
    beam = [([start_token], 0.0, initial_state)]

    for step in range(max_len):
        candidates = []
        for seq, log_prob, state in beam:
            next_token, new_state = decoder.step(seq[-1], encoder_output, state)
            for token_id, token_prob in enumerate(next_token):
                if token_prob > 0:
                    candidates.append((
                        seq + [token_id],
                        log_prob + math.log(token_prob),
                        new_state
                    ))
        # 按概率排序并保留前 k 个
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_size]

        # 如果所有束都生成了 <EOS>，则提前停止
        if all(seq[-1] == eos_token for seq, _, _ in beam):
            break

    return beam[0][0]
```

`beam_size` 控制质量与速度的权衡。较大的束更准确但更慢。对于大多数 MT 任务，束大小为 4-8 是最佳点。当束大小增大时，NMT 有时会出现质量下降（"束搜索越宽，输出越差"的悖论），因为较长的假设得到不成比例的惩罚。

## 使用

### MarianMT（HuggingFace）

```python
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

source = ["The cat sat on the mat."]
inputs = tokenizer(source, return_tensors="pt", padding=True)
translated = model.generate(**inputs)
result = tokenizer.decode(translated[0], skip_special_tokens=True)
```

Helsinki-NLP 的 Opus 模型覆盖 1000+ 种语言对。`opus-mt-en-zh` 针对英语到中文进行了优化。如果你需要中译英，使用 `opus-mt-zh-en`。语言对特定模型通常比多语言模型在某一个方向上的表现更好。

### 陷阱：长度归一化

```python
# 对长度进行归一化以防止对短序列的偏好
translated = model.generate(
    **inputs,
    max_length=64,
    num_beams=4,
    length_penalty=0.6,  # < 1.0 奖励较短的翻译，> 1.0 奖励较长的翻译
    early_stopping=True
)
```

`length_penalty` 修正了束搜索对短序列的固有偏好——因为束通过对数概率求和，较长的序列通常具有更低的概率，即使它们在意义上是较好的翻译。对其进行归一化：`score(y) = log P(y|x) / |y|^alpha`。

## 发布

翻译质量评估的验证提示。

保存为 `outputs/prompt-translation-quality.md`：

```markdown
---
name: translation-quality
description: SMT 与 NMT 系统的质量评估。
phase: 5
lesson: 10
---

评估给定的翻译系统输出。针对每个输出：

1. 流畅度：译入语是否自然的？查找不自然的措辞或词序。
2. 忠实度：源语言中的所有信息都保留了吗？检查命名实体、数字、否定句。
3. 罕见词处理：生僻词是否被翻译？NMT 有时会跳过它们，SMT 会照搬它们。
4. 长度偏好：NMT 是否在不截断的情况下输出合理的长度？如果输出比源语言短得多，则信息已丢失。
5. Hallucination：NMT 是否添加了源语言中没有的词？最危险的 NMT 失败模式——它会自信地输出不存在的虚假信息。

报告 BLEU 分数和人工质量评分。
```

## 练习

1. **简单。** 加载 MarianMT 模型并翻译三个句子。分析输出是否保留了所有实体。
2. **中等。** 实现一个基于短语的 SMT 系统，从一个小的平行语料库中提取短语表，并用它来翻译一个新句子。与 NMT 基线相比如何？
3. **困难。** 比较在你选择的语言对上添加注意力与不添加注意力的 Seq2Seq MT 的 BLEU 分数。当句子超过 15 个词时，没有注意力的模型性能下降了多少？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| SMT | 统计机器翻译 | 使用基于短语的对齐、重新排序和语言模型的翻译。 |
| NMT | 神经机器翻译 | 使用单一端到端序列到序列神经网络的翻译。 |
| 束搜索 | 多路径解码 | 在推理时跟踪 k 个候选序列。 |
| 短语对齐 | 跨语言的短语对应 | 平行语料库中的短语配对，是 SMT 的基础。 |
| BLEU | 双语评估替换 | 计算系统输出与参考译文之间的 n-gram 重叠。范围 0-100。40+ = 好。 |
