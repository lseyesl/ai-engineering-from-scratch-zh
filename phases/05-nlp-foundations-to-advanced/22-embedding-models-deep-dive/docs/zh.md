# RAG 的文本分块策略

> 检索质量取决于文本分块质量。分块质量取决于你的文档结构。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 13（IR），阶段 5 · 21（嵌入）
**时间：** 约 50 分钟

## 问题

RAG 系统从文档中检索相关片段，并将其输入 LLM。但什么样的片段才是"最优"的呢？

文本分块是将文档分割成可检索单元的过程。复杂之处在于：粒度太粗会导致多个主题合并到一个分块中，从而降低检索的相关性。粒度太细会导致上下文碎片化，使得 LLM 无法理解整体脉络。每个文档都有不同的最优分块大小——而你不知道它,直到你尝试之后。

## 概念

**固定大小分块**：按字符数或词元数切割，通常带有重叠。简单、可预测、可扩展。无论文档结构如何，在相同位置上切割。缺点是可能在句子中间切割，从而切断上下文。

**递归分块**：按层次结构分割——先分段，然后按句子，然后按词元。按照最自然的边界进行切割。如果段落太短，可以合并相邻的段落。

**基于文档结构的分块**：使用文档的标记结构（Markdown 标题、HTML 标签、LaTeX 部分）。保留标题层次结构和上下文。最适合结构化文档。

**语义分块**：使用嵌入相似度来检测主题边界。当两个相邻句子的嵌入差异很大时，它们属于不同的分块。计算成本高，但从概念上讲是最纯粹的。

## 构建

### 步骤 1：固定大小分块

```python
def fixed_size_chunks(text, chunk_size=500, overlap=50):
    """将文本分割为固定大小的分块，相邻分块之间有重叠。"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
```

重叠有助于前一个分块中切割掉的上下文浮入下一个分块的开头。如果没有重叠，一个跨越分块边界的主题在两个分块中都会丢失上下文。50 个词元的重叠足够用于大多数用途。

### 步骤 2：递归分块

```python
def recursive_chunk(text, max_chunk_size=500):
    """递归分割：段落 → 句子 → 固定大小。"""
    # 首先按段落分割
    paragraphs = text.split("\n\n")

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < max_chunk_size:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            # 如果段落本身太大，按句子分割
            if len(para) > max_chunk_size:
                sentences = para.replace("! ", "!|").replace("? ", "?|").split("|")
                for sent in sentences:
                    if len(current) + len(sent) < max_chunk_size:
                        current += sent + " "
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = sent + " "
            else:
                current = para + "\n\n"

    if current:
        chunks.append(current.strip())
    return chunks
```

递归分块首先尝试段落，因为段落是自然的分隔单位。然后使用句子，最后使用固定大小。构建一个 `RecursiveCharacterTextSplitter`（来自 LangChain）正是使用这种策略。

### 步骤 3：基于文档结构的 Markdown 分块

```python
def markdown_chunk(markdown_text, max_chunk_size=500):
    """根据 Markdown 标题结构进行分块。"""
    lines = markdown_text.split("\n")
    chunks = []
    current_header = ""
    current_chunk = ""

    for line in lines:
        if line.startswith("#"):
            # 如果 current_chunk 非空，将其保存并开始新的分块
            if current_chunk.strip():
                chunks.append({
                    "header": current_header,
                    "content": current_chunk.strip()
                })
                current_chunk = ""
            current_header = line
        current_chunk += line + "\n"

    if current_chunk.strip():
        chunks.append({
            "header": current_header,
            "content": current_chunk.strip()
        })

    # 将小分块合并到前一个分块中
    merged = []
    for chunk in chunks:
        if merged and len(merged[-1]["content"]) < max_chunk_size * 0.5:
            merged[-1]["content"] += "\n" + chunk["content"]
        else:
            merged.append(chunk)

    return merged
```

Markdown 标题是天然的分块边界。标题下的文本通常形成一个连贯的主题单元。当标题层级改变时，开始一个新的分块。在 RAG 系统中，保留标题信息可以大大提高答案质量——LLM 可以看到它正在回答的问题位于哪个标题下。

## 使用

### LangChain 文本分割器

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)
chunks = splitter.split_text(document)
```

LangChain 的 `RecursiveCharacterTextSplitter` 尝试按 `separators` 中的顺序进行分割。首先尝试 `\n\n`（段落）。如果得到的片段仍然过大，就尝试 `\n`（行）。然后尝试句子结尾。如果强制分割，最后尝试空格。这种层级化的方法在全面性（保留上下文）和细粒度（分割长文档）之间取得了平衡。

### 分块对检索的影响

```python
import numpy as np

def evaluate_chunk_strategy(embedder, chunks, questions, relevant_chunks):
    """评估分块策略的检索质量。"""
    chunk_embs = embedder.encode(chunks)
    correct = 0
    total = len(questions)

    for q, relevant in zip(questions, relevant_chunks):
        q_emb = embedder.encode([q])
        scores = np.dot(chunk_embs, q_emb.T).squeeze()
        top_idx = np.argmax(scores)
        if top_idx in relevant:
            correct += 1

    return correct / total
```

## 发布

分块策略优化提示。

保存为 `outputs/prompt-chunk-strategy.md`：

```markdown
---
name: chunk-strategy
description: 提示：为 RAG 推荐分块策略。
phase: 5
lesson: 22
---

根据文档类型推荐分块策略。

1. 结构化文档（手册、规范、书籍）→ 基于结构的分块（Markdown 标题、章节）。
2. 非结构化文档（报告、文章）→ 递归分块，`chunk_size=500`，`overlap=50`。
3. 代码文档——→ 尝试在函数/类边界进行分割（基于语法）。
4. 多语言文档——→ 使用 Unicode 文本分割器；避免按字符长度分块。
5. 短文档（< 1000 个字符）→ 当作单个分块处理；不进行分割。

评估分块策略：针对领域特定查询测试检索召回率。如果分块包含多个主题，检索器可能返回一个包含问题答案的分块，但也会返回不相关的信息，从而降低最终答案的质量。
```

## 练习

1. **简单。** 在一个 2000 词的文本文档上比较 `fixed_size_chunks(500)` 与 `fixed_size_chunks(200)`。在较小的分块下，有多少分块包含不完整的句子？
2. **中等。** 实现 `markdown_chunk` 并应用于一个带有标题的 Markdown 文档。检查块边界是否与段落边界对齐。
3. **困难。** 用 3 种策略（固定大小、递归、语义）构建一个 RAG 检索管道。测量每个策略在 10 个问题上的检索准确率。哪种策略在你的领域表现最好？原因是什么？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 分块 | 文档分割 | 将文档分割成较小的统一片段。 |
| 重叠 | 上下文连续 | 分块之间重叠的部分——在前一个分块的末尾与后一个分块的开头之间保持上下文连续性。 |
| 递归分块 | 分层法 | 按层级顺序（段落→句子→词元）进行分割。 |
| 语义分块 | 嵌入边界 | 通过嵌入相似度变化来检测主题边界。 |
| 块大小 | 段长度 | 每个块的目标词元数。越大的块保留更多上下文，但引入更多噪声。 |
