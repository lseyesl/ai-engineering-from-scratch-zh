# 文本摘要 — 压缩而不丢失关键信息

> 7000 个词的论文。700 个词的摘要。一篇好的摘要能捕捉所有关键发现，而不只是开头段落的复述。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 02（TF-IDF），阶段 5 · 08（用于文本的 CNN/RNN）
**时间：** 约 50 分钟

## 问题

为一篇 10 页的论文写一段摘要。你需要理解它、确定关键点，然后用更少的词重新表达它。这在认知上很困难，且对机器来说也极其困难。

文本摘要有两种范式：抽取式（从原文中选择最重要的句子）和生成式（合成新的文本）。传统的 NLP 倾向于抽取式（更容易，更安全）。深度学习在生成式上取得了进展（更灵活，但容易产生幻觉）。

## 概念

**抽取式摘要**将摘要视为一个句子排序问题。对句子进行评分——通过 TF-IDF、TextRank（PageRank 的一个变体）或基于 BERT 的句子嵌入——并选择得分最高的 k 个句子作为摘要。句子源自原文，这保证了事实准确性，但生成的内容可能不连贯。

**生成式摘要**将摘要视为一个 Seq2Seq 问题。编码器读取文档，解码器生成摘要。流畅且连贯，但模型可能会凭空捏造出原文中没有的事实。

```figure
seq2seq-alignment
```

## 构建

### 步骤 1：基于 TF-IDF 的文本摘要

```python
import math
from collections import Counter

def tfidf_summarize(doc, sentences, top_k=3):
    # 基于句子与文档 TF-IDF 向量的余弦相似度进行评分
    tf = Counter(doc.lower().split())
    doc_len = len(doc.split())
    idf = {}

    # 简化：使用文档本身的 IDF
    # 在单个文档上，IDF 无法很好地估计，但这里的意图是有教学性的
    n_sentences = len(sentences)
    all_words = set(doc.lower().split())
    for word in all_words:
        df = sum(1 for s in sentences if word in s.lower())
        idf[word] = math.log((1 + n_sentences) / (1 + df)) + 1

    sent_scores = []
    for sent in sentences:
        words = sent.lower().split()
        tfidf = 0
        for word in words:
            weight = tf.get(word, 0) / doc_len * idf.get(word, 0)
            tfidf += weight
        sent_scores.append((sent, tfidf))

    sent_scores.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in sent_scores[:top_k]]
```

在单文档摘要中，文档充当了全局语料库的角色。IDF 在文档级别计算——在整个文档中出现频率低的词会得到较高的 IDF（它们更具区分性）。跨多个文档的 IDF（新闻摘要）更准确，但需要较大的文档集合。

### 步骤 2：句子嵌入与聚类

```python
# 概念性：使用平均词嵌入进行句子评分
def sentence_embedding(sent, word_vectors, dim=100):
    words = sent.lower().split()
    vec = [0.0] * dim
    count = 0
    for w in words:
        if w in word_vectors:
            for d in range(dim):
                vec[d] += word_vectors[w][d]
            count += 1
    if count > 0:
        for d in range(dim):
            vec[d] /= count
    return vec

def cosine_sim(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a))
    nb = math.sqrt(sum(bi * bi for bi in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0
```

将句子与文档中心（所有句子嵌入的平均值）进行比较的相似度评分，识别最能代表文本的句子。基于 Transformer 的嵌入（Sentence-BERT）在语义相似度方面将这一思路提升到了更高的水平。

### 步骤 3：生成式摘要的数据准备

```python
def prepare_summarization_data(articles, summaries, max_src_len=512, max_tgt_len=128):
    """为 Seq2Seq 摘要准备数据。"""
    src_tokens = [tokenizer.tokenize(a)[:max_src_len] for a in articles]
    tgt_tokens = [tokenizer.tokenize(s)[:max_tgt_len] for s in summaries]
    return src_tokens, tgt_tokens
```

生成式摘要所需的标记数据比抽取式更多，也更昂贵。抽取式摘要只需要在句子级别进行标注；生成式摘要需要完整的参考摘要。这就是为什么即使生成式方法可以得到更流畅的结果，基于抽取式摘要仍然广泛存在的原因。

## 使用

### HuggingFace 生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
result = summarizer(article, max_length=130, min_length=30)
```

`facebook/bart-large-cnn` 是目前最通用的基于 Transformer 的摘要模型之一。它在 CNN/DailyMail 数据集上进行了微调——该数据集包含新闻文章及其多句参考摘要。注意：BART 已经在新闻文章上进行了微调，所以它在领域外的文本（如研究论文）上表现会显著下降。

### 段落级 vs 句子级

```python
# 长文档：按段落处理
paragraphs = doc.split("\n\n")
summaries = [summarizer(p, max_length=50, min_length=10)[0]["summary_text"]
             for p in paragraphs]
```

长文档的常用技巧是：先运行抽取式摘要以减少长度，然后对抽取出的文本运行生成式摘要。这个两步过程——"提取然后压缩"——结合了抽取式方法的忠实度和生成式方法的流畅性。

## 发布

摘要质量检查清单。

保存为 `outputs/prompt-summary-quality.md`：

```markdown
---
name: summary-quality
description: 事实准确性、覆盖范围和冗余度的摘要评估。
phase: 5
lesson: 11
---

评估一个文本摘要系统。给定源文档和生成的摘要：

1. 事实准确性：摘要中的所有陈述都可以从源文本中追溯到吗？标记幻觉。
2. 覆盖范围：摘要是否涵盖所有关键点？与作者明确强调的内容进行核对。
3. 简洁性：摘要去除了哪些冗余内容？是否保留了核心信息？
4. 流畅性：摘要读起来连贯吗？检查指代清晰度和过渡。
5. 冗余度：摘要中是否存在在源文本基础上重复自身的句子？（抽取式摘要中常见）。

如果同时评估多个系统，加入人工偏好评估。
```

## 练习

1. **简单。** 在一篇短新闻文章上运行 `pipeline("summarization")`。将生成的摘要与你自己的摘要进行比较——缺失了什么？
2. **中等。** 实现 TextRank 用于抽取式摘要：构建句子相似度图，运行 PageRank，并提取得分最高的 k 个句子。与随机选择句子作为摘要进行对比。
3. **困难。** 将抽取式步骤与生成式模型相结合：用句子图提取 top-5 句子，然后通过 BART 运行提取出的文本。将结果与纯生成式摘要进行对比——哪一步贡献更大？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式 | 选择句子 | 选择现有句子作为摘要。准确但可能不连贯。 |
| 生成式 | 生成文本 | 合成新文本作为摘要。流畅但可能产生幻觉。 |
| TextRank | PageRank 用于句子 | 句子相似度图上的图排序算法。 |
| ROUGE | 召回率导向的摘要评估 | 根据参考摘要衡量 n-gram 重叠的指标。 |
| 提取然后压缩 | 两步管道 | 先通过抽取减少范围，然后进行生成式压缩以优化流畅性。 |
