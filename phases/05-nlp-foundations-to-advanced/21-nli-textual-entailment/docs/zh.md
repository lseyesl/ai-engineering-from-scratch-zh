# 嵌入模型深度解析

> 嵌入是 NLP 的货币。理解它们是如何创建的、在哪里失效的，以及如何为你的数据选择正确的嵌入。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 03（Word2Vec），阶段 5 · 13（IR）
**时间：** 约 55 分钟

## 问题

嵌入无处不在：搜索、聚类、分类、RAG。但嵌入的质量取决于你的数据与模型训练数据的匹配程度。在一个领域（新闻）上训练的嵌入模型在另一个领域（医学、法律）上可能会失效。

本课深入探讨嵌入模型的内部工作原理，理解它们是如何在对比学习中训练的，并建立一个评估框架来衡量你的数据的嵌入质量。

## 概念

**对比学习**是现代嵌入训练的基础。目标：在嵌入空间中将相似的文本对拉近，将不相似的文本对推远。使用三重损失或 InfoNCE（噪声对比估计）损失——这是如今 Sentence-BERT 和其他嵌入模型使用的目标。

**池化策略**决定了如何将 token 级嵌入合并为句子级嵌入。平均池化是最常见的——对最后隐藏层的所有 token 向量求平均。CLS 池化用于 BERT。权重平均（通过注意力权重加权）是最新的模型使用的方法。

**维度**在质量与效率之间权衡。384（MiniLM）适合通用用途。768（BERT-base）在质量上稍好一些。1024（BERT-large）——质量最高，但代价是更大的维度。

## 构建

### 步骤 1：概念性对比损失（InfoNCE）

```python
import math

def info_nce_loss(anchor_emb, positive_emb, negative_embs, temperature=0.05):
    """
    对比损失：将锚点与正例拉近，与负例推远。
    anchor_emb: (D,) 维
    positive_emb: (D,) 维
    negative_embs: (N, D) 维
    """
    # 正例相似度
    pos_sim = cosine_sim(anchor_emb, positive_emb) / temperature

    # 负例相似度
    neg_sims = [cosine_sim(anchor_emb, neg) / temperature
                for neg in negative_embs]

    # 所有相似度——正例在最前
    all_sims = [pos_sim] + neg_sims

    # softmax：正例概率
    max_sim = max(all_sims)
    exps = [math.exp(s - max_sim) for s in all_sims]
    total = sum(exps)
    prob_pos = exps[0] / total

    # 损失：负对数似然
    return -math.log(prob_pos + 1e-10)
```

InfoNCE 损失将正例概率（锚点-正例对）与所有负例概率进行比较。温度控制标签平滑——温度越低，分布越尖锐，正例对的区分度越高。

### 步骤 2：概念性句子嵌入生成

```python
class SentenceEmbedder:
    def __init__(self, model, pool="mean"):
        self.model = model
        self.pool = pool

    def encode(self, sentence):
        tokens = self.model.tokenize(sentence)
        hidden_states = self.model.forward(tokens)

        if self.pool == "mean":
            # 对所有 token 向量取平均
            emb = [0.0] * len(hidden_states[0])
            for token_vec in hidden_states:
                for i, v in enumerate(token_vec):
                    emb[i] += v
            n = len(hidden_states)
            return [v / n for v in emb]

        elif self.pool == "cls":
            # 使用 [CLS] token 向量
            return hidden_states[0]
```

对于大多数模型，平均池化优于 CLS 池化。对于 BERT 而言，CLS token 被专门训练为分类表示，因此它包含句子级信息。但对于其他模型，CLS token 只是第一个 token，并不具有特殊用途。

### 步骤 3：嵌入质量评估

```python
def evaluate_embeddings(embedder, eval_pairs):
    """在相似度评估集上评估嵌入质量。"""
    correct = 0
    total = 0

    for text1, text2, expected_similar in eval_pairs:
        emb1 = embedder.encode(text1)
        emb2 = embedder.encode(text2)
        sim = cosine_sim(emb1, emb2)

        # 如果相似度超过阈值，则预测为相似
        predicted = sim > 0.5
        if predicted == expected_similar:
            correct += 1
        total += 1

    return correct / total
```

## 使用

### Sentence-BERT

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = ["The cat sat on the mat.", "A dog ran in the park."]
embeddings = model.encode(sentences)
similarity = model.similarity(embeddings[0], embeddings[1])
```

`all-MiniLM-L6-v2`：在超过 10 亿对数据上训练的通用嵌入模型。小巧（80MB）、快速、适用于大多数领域。对于中文，使用 `paraphrase-multilingual-MiniLM-L12-v2`——覆盖 50+ 种语言。

### 嵌入模型比较

| 模型 | 维度 | 大小 | MTEB（平均） | 最佳用途 |
|-------|-----------|------|-------------|-----------|
| all-MiniLM-L6-v2 | 384 | 80MB | 56 | 通用、快速 |
| BAAI/bge-base-en-v1.5 | 768 | 130MB | 58 | 知识库 |
| intfloat/e5-mistral-7b-instruct | 4096 | 14GB | 66 | 最高质量 |
| voyage-2 | 1024 | API | 62 | 托管 API |

### 关于嵌入维度的警告

```python
# 维度削减——牺牲质量以减小存储
from sklearn.decomposition import PCA

def reduce_dimension(embeddings, target_dim=128):
    pca = PCA(n_components=target_dim)
    return pca.fit_transform(embeddings)
```

将嵌入维度从 768 降至 128 可以减小 6 倍的存储，同时保留约 90-95% 的检索质量。当嵌入被编入索引用于生产时这一点很重要——每个额外维度都会增加 FAISS 索引的内存占用和搜索时间。

## 发布

嵌入模型的选择引导器。

保存为 `outputs/prompt-embedding-selector.md`：

```markdown
---
name: embedding-selector
description: 提示：为不同任务和领域推荐嵌入模型。
phase: 5
lesson: 21
---

根据任务和领域推荐嵌入模型：

1. 通用（新闻、网页、社交媒体） → all-MiniLM-L6-v2 或 BAAI/bge-base-en-v1.5。
2. 特定领域（医学、法律、金融） → 检查是否有特定领域的嵌入模型（PubMedBERT、LegalBERT），或使用 intfloat/e5-mistral-7b-instruct。
3. 多语言 → paraphrase-multilingual-MiniLM-L12-v2 或 intfloat/multilingual-e5-large。
4. 大规模检索（>1000 万文档） → 降低嵌入维度（在 128-256 之间），使用 OPQ 或 HNSW 索引的 FAISS。
5. 最高质量 → intfloat/e5-mistral-7b-instruct（注意：14GB 模型，比 MiniLM 慢 20 倍）。

如果在一个新的领域中使用，始终预先评估。嵌入质量可能与你预期的不同。
```

## 练习

1. **简单。** 使用 `all-MiniLM-L6-v2` 嵌入 5 个句子。计算余弦相似度矩阵。相似的句子是否彼此接近？
2. **中等。** 通过实现余弦相似度评估来评估 `e5-mistral-7b-instruct` 与 `all-MiniLM-L6-v2` 在你自己的数据（10 对相似/不相似文本）上的表现。较大的模型带来了多大的改善？
3. **困难。** 实现一个嵌入模型选择实验。在 3 个不同的领域（新闻、代码、医学）上比较 3 个嵌入模型。绘制每个模型在 MTEB 基准测试上的性能与你自己标注数据上的关系图。哪个嵌入模型在哪个领域表现最佳？原因是什么？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 嵌入 | 文本向量 | 文本的密集向量表示。 |
| 对比学习 | 拉近正例，推远负例 | 通过对比相似/不相似文本对来训练嵌入模型的损失函数。 |
| 池化 | 合并 token 向量 | 将所有 token 向量合并为单个句子向量的方法。 |
| MTEB | 大规模文本嵌入基准 | 在 8 个任务（分类、聚类、对句分类、重排序、检索、STS、摘要、消歧）上评估嵌入模型的基准。 |
| 温度 | 对比学习中的超参数 | 控制对比损失中 logit 的缩放。温度越低，对比越强。 |
