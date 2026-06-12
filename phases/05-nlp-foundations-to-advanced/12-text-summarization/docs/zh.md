# 问答 — 检索答案，而非文档

> 搜索找到页面。QA 找到句子。提取式 QA 找到精确的范围。生成式 QA 在训练数据中从未见过的情况下合成它们。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 02（TF-IDF），阶段 5 · 09（注意力机制）
**时间：** 约 55 分钟

## 问题

你向系统提问："OpenAI 的总部在哪里？"它不应该给你一篇关于 OpenAI 的文章；它应该返回"旧金山，加利福尼亚州"。问答是信息检索的细分：不只是找到相关的文档，而是提取精确的答案。

QA 以三种形式出现。**抽取式 QA** 返回文本中作为答案的一个连续范围。"埃隆·马斯克于 1971 年出生" → 问题："埃隆·马斯克什么时候出生？" → 答案："1971 年"。**生成式 QA** 合成文本，这对于合并多个来源的答案或回答训练范围之外的问题至关重要。**检索增强生成（RAG）** 将检索（搜索相关片段）与生成（用 LLM 合成答案）相结合。

## 概念

**抽取式 QA** 是一种序列标注问题。模型在上下文中为答案范围的开始和结束位置输出概率。跨度是（开始，结束），且 `开始 < 结束`。模型在 SQuAD 等数据集上进行训练。

**开放式 QA** 不提供上下文。模型必须从自己的知识中回答，或检索支持文档。检索器（通常是 TF-IDF 或密集检索）获取相关文本。阅读器从这些文本中提取答案。

**封闭式 QA** 限制领域（如"关于我们公司的常见问题"）。更适合基于规则的方法或使用标准数据集上的微调模型。

## 构建

### 步骤 1：抽取式 QA 评分

```python
class ExtractiveQA:
    def __init__(self, model):
        self.model = model

    def answer(self, question, context):
        # 模型将问题 + 上下文编码为单个序列
        # 输出开始和结束 logits
        start_logits, end_logits = self.model(question, context)

        # 在有效跨度内寻找最佳（开始，结束）对
        best_score = float("-inf")
        best_span = (0, 0)
        for start in range(len(start_logits)):
            for end in range(start, min(start + max_answer_len, len(end_logits))):
                score = start_logits[start] + end_logits[end]
                if score > best_score:
                    best_score = score
                    best_span = (start, end)

        return context[best_span[0]:best_span[1]+1]
```

在所有 `N²` 个可能的跨度上进行全搜索，是复杂度为 O(N) 的约束算法的教学版本（通过跟踪最佳开始/结束而非所有组合）。在 Transformer 中，令牌可能为数十个；在大型文档上，使用滑动窗口将上下文限制为 384 个令牌的分块。

### 步骤 2：限制跨度长度

```python
def find_best_span(start_probs, end_probs, max_len=30):
    """找到概率最高的有效跨度，受最大长度限制。"""
    n = len(start_probs)
    best_start, best_end, best_score = 0, 0, 0.0

    for start in range(n):
        for end in range(start, min(n, start + max_len)):
            score = start_probs[start] + end_probs[end]
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    return best_start, best_end, best_score
```

`max_len` 约束对抽取式问答是必要的：如果没有长度限制，模型会选择重叠度较高但跨度很长的开始和结束位置（开始 = 0，结束 = n-1），这适用于许多输入，但对任何实际操作来说都太粗粒度了。

## 使用

### HuggingFace 管道

```python
from transformers import pipeline

qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

result = qa(
    question="Where is OpenAI headquartered?",
    context="OpenAI is an AI research organization headquartered in San Francisco, California."
)
```

```python
{"score": 0.98, "start": 52, "end": 66, "answer": "San Francisco"}
```

`score` 大致为 `softmax(start_logits)[start] * softmax(end_logits)[end]`。它根据不同问题和上下文进行了粗略的校准，可以用于大致比较答案置信度。

### 密集检索（RAG 风格）

```python
from sentence_transformers import SentenceTransformer

retriever = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, documents, k=3):
    q_emb = retriever.encode(query)
    doc_embs = retriever.encode(documents)
    scores = [cosine_sim(q_emb, d_emb) for d_emb in doc_embs]
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [documents[i] for i in top_k]
```

这是 RAG 的核心：检索到文档，将它们作为上下文提供给阅读器，然后阅读器从中提取答案。密集检索之所以被称为"密集"，是因为它使用密集的嵌入向量而不是稀疏的 TF-IDF 向量来进行相似度计算。

### 何时使用生成式 QA

当出现以下情况时，可使用生成式 QA（使用 LLM 根据上下文进行回答）而非提取式：
- 答案不准确地对应于一个连续的文本片段（需要推理）
- 需要多文档整合
- 需要"我不知道"的选项
- 自由形式的答案更合适

代价：产生幻觉的风险、更长的延迟、更高的成本。

## 发布

QA 系统评估检查清单。

保存为 `outputs/prompt-qa-validation.md`：

```markdown
---
name: qa-validation
description: QA 系统评估——准确性、鲁棒性和覆盖范围。
phase: 5
lesson: 12
---

评估给定的问答系统：

1. 答案正确性：提供的答案正确吗？针对参考答案进行核对。
2. 空回答：当系统不确定时，它是空回答（没有返回任何内容）还是产生幻觉？空回答比幻觉更安全。
3. 交叉来源：如果同一问题有多个上下文，输出是否一致？报告不一致的情况。
4. 长度策略：答案是否被截断？系统是返回精确的跨度还是多余的文本？
5. 领域适应：在训练领域 vs 未见过领域上的表现。量化下降。

对于 RAG 系统，还要检查检索质量：检索到的文本中是否包含答案？
```

## 练习

1. **简单。** 将 `pipeline("question-answering")` 应用于 SQuAD 拆分区中的样本。提取的跨度是否准确？
2. **中等。** 使用上述密集检索器实现一个 RAG 系统。根据维基百科文章片段回答三个问题，并测量答案准确率。
3. **困难。** 比较抽取式 QA 与生成式 QA（使用 LLM 根据检索到的上下文进行回答）。对于需要从多个句子中推理的问题（如"所有的创始成员是谁？"），哪种方法更好？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式 QA | 选择答案 | 从上下文中定位连续文本跨度作为答案。 |
| 生成式 QA | 合成答案 | 基于上下文生成自由形式的答案文本。 |
| SQuAD | 斯坦福问答数据集 | 流行的抽取式 QA 基准测试，包含 10 万个问题。 |
| RAG | 检索增强生成 | 检索 + 生成：在 LLM 生成答案之前搜索相关文本。 |
| 密集检索 | 嵌入搜索 | 使用嵌入在语义空间中搜索，使用 TF-IDF 用于 BM25。 |
