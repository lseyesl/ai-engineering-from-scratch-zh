# 多语言 NLP — 一种模型，多种语言

> 如果你只针对英语构建 NLP 产品，你忽略了世界上 75% 的内容。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 03（嵌入），阶段 5 · 13（IR）
**时间：** 约 50 分钟

## 问题

大多数 NLP 工具都是针对英语构建的。英语分词器。英语停用词列表。英语语料库。但世界上有 7000 多种语言，互联网内容中只有 25% 是英语的。

多语言 NLP 解决语言之间的差距。解决方案在于跨语言嵌入——将不同语言中的相似词映射到嵌入空间中的同一区域——以及大规模的多语言模型。

## 概念

**跨语言嵌入** 将来自不同语言的词映射到共享嵌入空间。"猫"和"cat"在共享空间中彼此接近，即使它们来自不同的语言。这通过在平行语料库（相同的句子，不同的语言）上进行训练，或通过映射单语嵌入（使用像 MUSE、VecMap 这样的跨语言映射）来实现。

**多语言 BERT（mBERT）** 在 104 种语言的维基百科文章上进行了预训练。它为所有语言使用同一个共享词元表（尽管这个词元表必然由高资源语言主导）。**XLM-R** 通过大规模扩展（100 种语言，2.5TB 数据）来改进 mBERT，并使用更大的词汇表来平衡覆盖率。

## 构建

### 步骤 1：概念性跨语言词映射

```python
import random
import math

class CrossLingualMapper:
    def __init__(self, dim=300):
        self.W = [[random.gauss(0, 0.01) for _ in range(dim)] for _ in range(dim)]

    def fit(self, src_vectors, tgt_vectors, bilingual_dict):
        """学习从源语言到目标语言的线性映射。"""
        # 双语词典中的成对词提供了训练数据
        X, Y = [], []
        for src_word, tgt_word in bilingual_dict.items():
            if src_word in src_vectors and tgt_word in tgt_vectors:
                X.append(src_vectors[src_word])
                Y.append(tgt_vectors[tgt_word])

        # 求解 W 以最小化 ||WX - Y||（Procrustes 问题）
        # 使用 SVD 求解闭式解
        C = [[0.0 for _ in range(len(X[0]))] for _ in range(len(X[0]))]
        for i in range(len(X)):
            for j in range(len(X[0])):
                for k in range(len(X[0])):
                    C[j][k] += X[i][j] * Y[i][k]

        # 简化：用最小二乘法代替 SVD
        # 教学性——真正的 MUSE 使用 Wasserstein-GAN 进行无监督映射
        for i in range(len(self.W)):
            for j in range(len(self.W[0])):
                num, den = 0, 0
                for k in range(len(X)):
                    num += Y[k][i] * X[k][j]
                    den += X[k][j] * X[k][j]
                if den > 0:
                    self.W[i][j] = num / den

    def transform(self, src_vector):
        result = [0.0] * len(self.W[0])
        for i in range(len(self.W)):
            for j in range(len(self.W[0])):
                result[i] += self.W[i][j] * src_vector[j]
        return result
```

映射步骤假设源语言和目标语言的嵌入空间是近似同构的——即语言之间的语义关系以类似的方式排列。实际上，语言的形态学差异（英语孤立语 vs 土耳其语黏着语）意味着同构性只是近似成立。

### 步骤 2：零样本跨语言迁移

```python
from transformers import pipeline

# 使用多语言 NER 模型——在英语上训练，用于其他语言
ner = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english")
result = ner("Elon Musk a visité Paris en juin.")
```

```python
[{'entity': 'B-PER', 'score': 0.99, 'word': 'Elon'},
 {'entity': 'I-PER', 'score': 0.99, 'word': 'Musk'},
 {'entity': 'B-LOC', 'score': 0.99, 'word': 'Paris'}]
```

零样本迁移之所以有效，是因为多语言表示在所有语言中是共享的。"Elon Musk" 在法语中的上下文与英语中的上下文具有相同的分布——嵌入了人名的主题、使用了大写字母，实体模式也类似。XLM-R 在法语句子上运行得和英语一样好，即使它在英语句子上进行了微调。

### 步骤 3：语言识别

```python
def detect_language(text, language_profiles):
    """基于字符 n-gram 频率的语言识别。"""
    scores = {}
    for lang, profile in language_profiles.items():
        score = 0
        for char in text.lower():
            if char in profile:
                score += profile[char]
        scores[lang] = score / len(text)
    return max(scores, key=scores.get)
```

基于字符的语言识别检查文本中某一语言的字符频率是否高于其他语言。如果文本中包含字母"ñ"，则很可能是西班牙语。如果包含字母"æ"，则很可能是丹麦语。常见字母在每种语言中会出现不同的分布——"a" 在英语中比 "z" 更常见。

## 使用

### spaCy 多语言模型

```python
import spacy

nlp = spacy.load("xx_ent_wiki_sm")  # 多语言 NER
doc = nlp("Apple Inc. est une entreprise américaine.")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

spaCy 的多语言模型（`xx_ent_wiki_sm`、`xx_sent_ud_sm`）支持 55 种以上的语言，且性能几乎没有下降。不支持的语言回退到基于规则的模型。与英语模型每个部分相比，多语言模型的质量在 80-95% 之间。

### 语言覆盖范围

| 模型 | 语言数量 | 训练数据 | 在英语上的表现 |
|-------|-----------|-------------|----------------|
| mBERT | 104 | 维基百科 | BERT-base 的 ~97% |
| XLM-R | 100 | 2.5TB CommonCrawl | BERT-base 的 ~99% |
| LaBSE | 109 | 平行语料库 | 双语句子嵌入 |
| mT5 | 101 | mC4 | T5 的 ~95% |

## 发布

多语言 NLP 项目检查清单。

保存为 `outputs/prompt-multilingual-checklist.md`：

```markdown
---
name: multilingual-checklist
description: 检查多语言 NLP 项目的准备工作，包括对低资源语言的特定挑战。
phase: 5
lesson: 17
---

启动多语言 NLP 项目。检查：

1. 语言覆盖范围：目标模型是否在目标语言上进行过训练？mBERT 只覆盖了维基百科上存在的语言。
2. 分词：语言需要空格分割（英语）还是需要亚字符分词（中文、日语、泰语）？检查模型的分词器覆盖率。
3. 评估数据：在目标语言中，是否有特定于任务的标记数据？如果没有，计划零样本或少量样本评估。
4. 脚本：是否是同一种书写系统（拉丁语、西里尔语、阿拉伯语、汉语）？不同脚本需要不同的预处理。
5. 陷阱：低资源语言在模型中往往被高资源语言掩盖。法语 + 德语在 mBERT 中很好，但斯瓦希里语 + 祖鲁语有显著的性能差异。

对于低资源语言：考虑回退到翻译（将目标语言翻译回英语并在英语系统上进行处理）。对于高资源语言，可以优先使用 mBERT 或 XLM-R，而不是英语专用模型。
```

## 练习

1. **简单。** 在一个非英语句子上加载 `pipeline("ner", model="xlm-roberta-base")`。模型检测到了哪些实体？准确性如何？
2. **中等。** 在两种不同语言（例如法语和德语）中对同一概念进行句子嵌入（使用 Sentence-BERT 的多语言变体）。测量跨语言嵌入对之间的距离。嵌入是否将相似的概念映射到一起？
3. **困难。** 收集一个 100 个英语句子的数据集，人工翻译成另一种语言。使用 XLM-R 为每种语言生成句子嵌入。计算跨语言对的余弦相似度——嵌入空间在多大程度上是一致的？报告同一种语言和跨语言之间的平均相似度。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 跨语言 | 跨语言 | 表示在不同语言之间映射（"cat" ≈ "猫"）。 |
| 多语言 | 多种语言 | 一次处理多种语言的单个模型。 |
| mBERT | 多语言 BERT | 在 104 种语言上进行预训练的 BERT。 |
| XLM-R | 鲁棒跨语言表示 | 更大、更好的 mBERT，用 100 种语言、2.5TB 数据进行训练。 |
| 零样本迁移 | 跨语言无需训练 | 在英语上微调模型，在法语上使用——用于跨语言嵌入空间。 |
