# RAG（检索增强生成）

> 你的 LLM 知道训练截止日期之前的一切。它不知道你公司的文档、你的代码库或上周的会议记录。RAG 通过检索相关文档并将其填入提示词来解决这个问题。它是生产 AI 中部署最多的模式。如果你从本课程中只构建一件事，那就构建一个 RAG 管道。

**类型：** Build
**语言：** Python
**前置要求：** 阶段 10（LLMs from Scratch），阶段 11 课程 01-05
**预计时间：** ~90 分钟
**关联：** 阶段 5 · 23（分块策略）涵盖六种分块算法及适用场景。阶段 5 · 22（嵌入模型深入）帮助选择嵌入器。阶段 11 · 07（高级 RAG）涵盖混合搜索、重排序和查询转换。

## 学习目标

- 构建一个完整的 RAG 管道：文档加载、分块、嵌入、向量存储、检索和生成
- 使用向量数据库（ChromaDB、FAISS 或 Pinecone）实现带有适当索引的语义搜索
- 解释为什么在知识驱动的应用中 RAG 优于微调（成本、新鲜度、归因能力）
- 使用检索指标（精确度、召回率）和生成指标（忠实度、相关性）评估 RAG 质量

## 问题

你为你的公司构建了一个聊天机器人。客户问"企业计划的退款政策是什么？"LLM 给出了一个关于典型 SaaS 退款政策的通用回答。实际政策埋藏在一个 200 页的内部维基中，说企业客户有 60 天的按比例退款窗口。LLM 从未见过这份文档。它无法知道它没有训练过的东西。

微调是一种解决方案。拿 LLM，在你的内部文档上训练它，然后部署更新后的模型。这有效但有严重问题。微调花费数千美元的计算成本。文档一变，模型就过时了。你无法知道模型从哪个来源得出结论。如果公司下个月收购了另一个产品线，你又得重新微调。

RAG 是另一种解决方案。保持模型不变。当问题来了，搜索你的文档库获取相关段落，在问题之前将它们粘贴到提示词中，让模型使用这些段落作为上下文来回答。文档库可以在几分钟内更新。你可以确切地看到哪些文档被检索到。模型本身从不改变。这就是为什么 RAG 是生产中的主导模式：它更便宜、更新鲜、更可审计，并且适用于任何 LLM。

## 概念

### RAG 模式

整个模式分为四个步骤：

```mermaid
graph LR
    Q["用户查询"] --> R["检索"]
    R --> A["增强提示词"]
    A --> G["生成"]
    G --> Ans["答案"]

    subgraph "检索"
        R --> Embed["嵌入查询"]
        Embed --> Search["搜索向量存储"]
        Search --> TopK["返回 top-k 块"]
    end

    subgraph "增强"
        TopK --> Format["将块格式化为提示词"]
        Format --> Combine["与用户问题组合"]
    end

    subgraph "生成"
        Combine --> LLM["LLM 生成答案"]
        LLM --> Cite["基于检索文档的答案"]
    end
```

查询 -> 检索 -> 增强提示词 -> 生成。每个 RAG 系统都遵循这个模式。生产 RAG 系统之间的差异在于每一步的细节：如何分块、如何嵌入、如何搜索以及如何构建提示词。

### 为什么 RAG 胜过微调

| 关注点 | 微调 | RAG |
|---------|------------|-----|
| 成本 | 每次训练 $1,000-$100,000+ | 每次查询 $0.01-$0.10（嵌入 + LLM） |
| 新鲜度 | 过时直到重新训练 | 通过重新索引文档在几分钟内更新 |
| 可审计性 | 无法追溯答案来源 | 可以展示确切的检索段落 |
| 幻觉 | 仍然自由产生幻觉 | 基于检索到的文档 |
| 数据隐私 | 训练数据烘焙到权重中 | 文档留在你的向量存储中 |

微调永久性地改变模型的权重。RAG 临时性地改变模型的上下文。对于大多数应用，临时上下文就是你想要的。

微调取胜的唯一情况是：当你需要模型采用无法通过提示词实现的特定风格、语气或推理模式时。对于事实性知识检索，RAG 每次都能赢。

### 嵌入模型

嵌入模型将文本转换为稠密向量。相似的文本产生在高维空间中彼此接近的向量。"如何重置密码？"和"我需要更改密码"尽管共享很少的词，但产生几乎相同的向量。"猫坐在垫子上"产生一个非常不同的向量。

常见的嵌入模型（2026 年的阵容——参见阶段 5 · 22 的完整分析）：

| 模型 | 维度 | 提供商 | 备注 |
|-------|-----------|----------|-------|
| text-embedding-3-small | 1536（Matryoshka） | OpenAI | 大多数用例的最佳性价比 |
| text-embedding-3-large | 3072（Matryoshka） | OpenAI | 更高精度，可截断至 256/512/1024 |
| Gemini Embedding 2 | 3072（Matryoshka） | Google | 顶级 MTEB 检索；8K 上下文 |
| voyage-4 | 1024/2048（Matryoshka） | Voyage AI | 领域变体（代码、金融、法律） |
| Cohere embed-v4 | 1024（Matryoshka） | Cohere | 强大多语言，128K 上下文 |
| BGE-M3 | 1024（dense + sparse + ColBERT） | BAAI（开源权重） | 一个模型三种视图 |
| Qwen3-Embedding | 4096（Matryoshka） | Alibaba（开源权重） | 顶级开源检索分数 |
| all-MiniLM-L6-v2 | 384 | 开源权重（Sentence Transformers） | 原型基线 |

在本课程中，我们使用 TF-IDF 构建自己的简单嵌入。不是因为 TF-IDF 是生产系统使用的，而是因为它让概念变得具体：文本输入，向量输出，相似的文本产生相似的向量。

### 向量相似度

给定两个向量，如何衡量相似度？三种选择：

**余弦相似度**：两个向量之间夹角的余弦值。范围从 -1（相反）到 1（相同）。忽略幅度，只关心方向。这是 RAG 的默认选择。

```
cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
```

**点积**：原始内积。较大的向量获得较高分数。当幅度携带信息时有用（较长的文档可能更相关）。

```
dot(a, b) = sum(a_i * b_i)
```

**L2（欧几里得）距离**：向量空间中的直线距离。距离越小 = 越相似。对幅度差异敏感。

```
L2(a, b) = sqrt(sum((a_i - b_i)^2))
```

余弦相似度是标准选择。它优雅地处理不同长度的文档，因为它通过幅度进行归一化。当有人说"向量搜索"时，他们几乎总是指余弦相似度。

### 分块策略

文档太长，不能作为单个向量嵌入。一份 50 页的 PDF 可能产生糟糕的嵌入，因为它包含数十个主题。相反，你将文档分割成块并分别嵌入每个块。

**固定大小分块**：每 N 个 token 分割一次。简单且可预测。一个 512 token 的块带有 50 token 的重叠意味着块 1 是 token 0-511，块 2 是 token 462-973，以此类推。重叠确保你不会在不幸的边界处分割句子。

**语义分块**：在自然边界处分割。段落、章节或 markdown 标题。每个块是一个连贯的意义单元。实现更复杂但产生更好的检索效果。

**递归分块**：尝试先在最大的边界处分割（章节标题）。如果章节仍然太大，在段落边界分割。如果段落仍然太大，在句子边界分割。这是 LangChain RecursiveCharacterTextSplitter 的方法，在实践中效果很好。

块大小比人们想象的更重要：

- 太小（64-128 tokens）：每个块缺乏上下文。"上个季度增长了 15%"在不知道"它"指的是什么的情况下毫无意义。
- 太大（2048+ tokens）：每个块覆盖多个主题，稀释了相关性。当你搜索收入数据时，你得到的是 10% 关于收入和 90% 关于员工人数的块。
- 最佳选择（256-512 tokens）：足够的上下文以自我包含，足够聚焦以保持相关。

大多数生产 RAG 系统使用 256-512 token 的块，带有 50 token 的重叠。Anthropic 的 RAG 指南推荐这个范围。

### 向量数据库

一旦你有了嵌入，你需要一个地方来存储和搜索它们。选项：

| 数据库 | 类型 | 最适合 |
|----------|------|----------|
| FAISS | 库（进程内） | 原型设计，中小型数据集 |
| Chroma | 轻量级数据库 | 本地开发，小型部署 |
| Pinecone | 托管服务 | 无需运维开销的生产 |
| Weaviate | 开源数据库 | 自托管生产 |
| pgvector | Postgres 扩展 | 已在用 Postgres |
| Qdrant | 开源数据库 | 高性能自托管 |

在本课程中，我们构建一个简单的内存向量存储。它将向量存储在列表中并执行暴力余弦相似度搜索。这相当于带有平面索引的 FAISS。它在达到约 100,000 个向量之前还算快。生产系统使用近似最近邻（ANN）算法（如 HNSW）在毫秒内搜索数百万个向量。

### 完整管道

```mermaid
graph TD
    subgraph "索引（离线）"
        D["文档"] --> C["分块"]
        C --> E["嵌入每个块"]
        E --> S["存储向量 + 文本"]
    end

    subgraph "查询（在线）"
        Q["用户查询"] --> QE["嵌入查询"]
        QE --> VS["向量搜索（top-k）"]
        VS --> P["使用块构建提示词"]
        P --> LLM["LLM 生成答案"]
    end

    S -.->|"同一向量空间"| VS
```

索引阶段每个文档运行一次（或文档更新时）。查询阶段在每个用户请求时运行。在生产中，索引可能在数小时内处理数百万个文档。查询必须在一秒内响应。

### 真实数据

大多数生产 RAG 系统使用这些参数：

- **k = 5 到 10** 每个查询检索的块数
- **块大小 = 256 到 512 tokens**，带有 50 token 重叠
- **上下文预算**：每个查询 2,500-5,000 tokens 的检索内容
- **总提示词**：~8,000-16,000 tokens（系统提示词 + 检索块 + 对话历史 + 用户查询）
- **嵌入维度**：384-3072，取决于模型
- **索引吞吐量**：使用 API 嵌入每秒 100-1,000 个文档
- **查询延迟**：检索 50-200ms，生成 500-3000ms

```figure
rag-chunking
```

## 构建它

### 步骤 1：文档分块

```python
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
```

### 步骤 2：TF-IDF 嵌入

我们构建一个简单的嵌入函数。TF-IDF（词频-逆文档频率）不是神经嵌入，但它以一种捕获词重要性的方式将文本转换为向量。文档中的高频词获得更高的 TF。整个语料库中的稀有词获得更高的 IDF。乘积给出了重要、独特词汇具有高值的向量。

```python
import math
from collections import Counter

def build_vocabulary(documents):
    vocab = set()
    for doc in documents:
        vocab.update(doc.lower().split())
    return sorted(vocab)

def compute_tf(text, vocab):
    words = text.lower().split()
    count = Counter(words)
    total = len(words)
    return [count.get(word, 0) / total for word in vocab]

def compute_idf(documents, vocab):
    n = len(documents)
    idf = []
    for word in vocab:
        doc_count = sum(1 for doc in documents if word in doc.lower().split())
        idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
    return idf

def tfidf_embed(text, vocab, idf):
    tf = compute_tf(text, vocab)
    return [t * i for t, i in zip(tf, idf)]
```

### 步骤 3：余弦相似度搜索

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def search(query_embedding, stored_embeddings, top_k=5):
    scores = []
    for i, emb in enumerate(stored_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
```

### 步骤 4：提示词构建

这就是 RAG 中"增强"发生的地方。取出检索到的块，将它们格式化为提示词，并要求 LLM 基于提供的上下文回答。

```python
def build_rag_prompt(query, retrieved_chunks):
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    return f"""Answer the question based ONLY on the following context.
If the context doesn't contain enough information, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""
```

### 步骤 5：完整的 RAG 管道

```python
class RAGPipeline:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.vocab = []
        self.idf = []

    def index(self, documents):
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunk_text(doc))
        self.chunks = all_chunks
        self.vocab = build_vocabulary(all_chunks)
        self.idf = compute_idf(all_chunks, self.vocab)
        self.embeddings = [
            tfidf_embed(chunk, self.vocab, self.idf)
            for chunk in all_chunks
        ]

    def query(self, question, top_k=5):
        query_emb = tfidf_embed(question, self.vocab, self.idf)
        results = search(query_emb, self.embeddings, top_k)
        retrieved = [(self.chunks[i], score) for i, score in results]
        prompt = build_rag_prompt(
            question, [chunk for chunk, _ in retrieved]
        )
        return prompt, retrieved
```

### 步骤 6：生成（模拟）

在生产中，这是你调用 LLM API 的地方。在本课程中，我们通过从检索到的上下文中提取最相关的句子来模拟生成。

```python
def simple_generate(prompt, retrieved_chunks):
    query_words = set(prompt.lower().split("question:")[-1].split())
    best_sentence = ""
    best_score = 0
    for chunk in retrieved_chunks:
        for sentence in chunk.split("."):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = set(sentence.lower().split())
            overlap = len(query_words & words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
    return best_sentence if best_sentence else "I don't have enough information."
```

## 使用它

使用真实的嵌入模型和 LLM，代码几乎不变：

```python
from openai import OpenAI

client = OpenAI()

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
```

或者使用 Anthropic：

```python
import anthropic

client = anthropic.Anthropic()

def generate(prompt):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

管道是一样的。替换嵌入函数。替换生成函数。检索逻辑、分块、提示词构建——无论使用哪种模型，都是相同的。

对于大规模向量存储，用真正的向量数据库替换暴力搜索：

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

results = collection.query(
    query_texts=["What is the refund policy?"],
    n_results=5
)
```

Chroma 内部处理嵌入（默认使用 all-MiniLM-L6-v2）并将向量存储在本地数据库中。相同的模式，不同的底层实现。

## 交付物

本课程产出：
- `outputs/prompt-rag-architect.md` —— 为特定用例设计 RAG 系统的提示词
- `outputs/skill-rag-pipeline.md` —— 教授代理如何构建和调试 RAG 管道的技能

## 练习

1. 将 TF-IDF 嵌入替换为简单的词袋方法（二元：词出现为 1，否则为 0）。在样本文档上比较检索质量。TF-IDF 应该更优，因为它对稀有词赋予更高权重。

2. 实验分块大小：在同一份文档集上尝试 50、100、200 和 500 个词。对每种大小，运行相同的 5 个查询，统计在 top-3 中返回相关块的次数。找到检索质量峰值的最佳位置。

3. 为每个块添加元数据（源文档名称、块位置）。修改提示词模板以包含来源归属，使 LLM 引用其来源。

4. 实现一个简单的评估：给定 10 个问答对，将每个问题通过 RAG 管道运行，并衡量检索到的块中包含答案的百分比。这是 k 处的检索召回率。

5. 构建一个对话感知的 RAG 管道：维护最近 3 次交换的历史记录，并将其与检索到的块一起包含在提示词中。用诸如"那企业用户呢？"之类的后续问题在询问定价之后进行测试。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| RAG | "会读你文档的 AI" | 检索相关文档，粘贴到提示词中，生成基于这些文档的答案 |
| 嵌入 | "将文本转数字" | 文本的稠密向量表示，相似含义产生相似向量 |
| 向量数据库 | "AI 的搜索引擎" | 优化用于存储向量并按相似度查找最近邻的数据存储 |
| 分块 | "将文档切碎" | 将文档分割成较小的片段（通常 256-512 tokens），使每个片段可以独立嵌入和检索 |
| 余弦相似度 | "两个向量有多相似" | 两个向量之间夹角的余弦值；1=方向相同，0=正交，-1=相反 |
| Top-k 检索 | "获取 k 个最佳匹配" | 从向量存储中返回与查询最相似的 k 个块 |
| 上下文窗口 | "LLM 能看到多少文本" | LLM 在单个请求中能处理的最大 token 数；检索到的块必须适合此限制 |
| 增强生成 | "使用给定上下文回答" | 使用检索到的文档作为上下文生成响应，而非仅依赖训练知识 |
| TF-IDF | "词重要性评分" | 词频乘以逆文档频率；根据词在语料库中的独特性赋予权重 |
| 索引 | "准备文档以供搜索" | 对文档进行分块、嵌入和存储的离线过程，以便在查询时可搜索 |

## 延伸阅读

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020) —— 来自 Facebook AI Research 的原始 RAG 论文，形式化了检索-生成模式
- Anthropic 的 RAG 文档 (docs.anthropic.com) —— 关于分块大小、提示词构建和评估的实用指南
- Pinecone 学习中心，"What is RAG?" —— RAG 管道的清晰视觉解释，包含生产考虑因素
- Sentence-BERT: Reimers & Gurevych (2019) —— all-MiniLM 嵌入模型背后的论文，展示了如何训练用于语义相似度的双编码器
- [Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering" (EMNLP 2020)](https://arxiv.org/abs/2004.04906) —— DPR 论文，证明了稠密双编码器检索在开放域问答上胜过 BM25，为现代 RAG 检索器设定了模式。
- [LlamaIndex 高层概念](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html) —— 构建 RAG 管道时需要了解的主要概念：数据加载器、节点解析器、索引、检索器、响应合成器。
- [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/) —— 另一种风格的编排器；从可运行链的角度看待相同的检索-生成模式。