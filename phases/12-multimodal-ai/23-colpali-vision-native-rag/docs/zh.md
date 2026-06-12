# ColPali：面向 RAG 的原生视觉多模态文档检索

> 文档检索的标准做法是：将 PDF 解析成文本，将文本嵌入到向量存储中，并在向量之间进行相似度搜索。但正如我们在课程 22 中了解到的，文档不仅仅是文本——还有表格、图表和布局。ColPali（Faysse 等人，2024 年 7 月）去掉了解析步骤，直接检索图像。Pali（视觉语言模型）对文档页面进行编码，修改 ColBERT 风格的后期交互，以支持视觉 token-视觉 token 的相似度匹配。结果是在保留的文档检索基准上，召回率比文本密集检索（BM25 + ColBERT-v2）高出 10-15 个点。本课程构建一个玩具 ColPali 检索器：通过后期交互相似度来匹配文档图像和文本查询。

**类型：** 构建
**语言：** Python（标准库，ViT 编码器 + 后期交互 + 检索前端）
**前置要求：** Phase 12 · 22（文档），Phase 10（RAG）
**时间：** ~180 分钟

## 学习目标

- 解释为什么文档解析是检索的瓶颈，以及为什么视觉检索绕过了这一限制。
- 在 ColBERT 的后期交互和视觉 token 之间实现后期交互相似度。
- 描述 ColPali 的架构：Pali（ViT + 编码器文本解码器）→ 后期交互 → 最大相似度（MaxSim）检索。
- 将 ColPali 与基于文本的检索基线（BM25、ColBERT-v2、SPLADE）进行比较。

## 问题

标准文档检索（RAG 管道）由以下步骤组成：

1. 文档解析：检测布局、OCR 文本提取、表格提取、图表提取。文本以特殊 token 连接。
2. 文本嵌入：文本片段被馈入 BERT 或 BGE 嵌入器。
3. 向量搜索：查询嵌入与片段嵌入进行匹配（余弦相似度）。

每个步骤都会丢失信息。解析器遗漏了文本，错误读取了表格，并且完全忽略了视觉信息。文本嵌入器在数值推理上表现不佳。

ColPali 设问：如果我们直接检索文档图像，完全跳过解析呢？

## 概念

### Pali 模型

Pali（Google，2023）是一个视觉语言模型，专门设计用于理解文档。它由一个 ViT 编码器和一个仅解码器的语言模型组成。它不像 LLaVA 那样对自然图像高度优化，而是用于高分辨率、文本密集的文档图像。

ColPali 使用了 Pali 的一个变体，在文档检索数据进行微调。Pali 编码文档页面并输出每个 patch 的嵌入向量。这个嵌入矩阵直接用于后期交互，而不是被投影到 LLM token。

### 后期交互（ColBERT 风格）

ColBERT 论文（Khattab & Zaharia，2020）引入了一种检索方法，该方法不是将文档压缩成一个向量，而是保留每个 token 的嵌入。查询：由 BERT 编码的 N 个 token。文档：由 BERT 编码的 M 个 token。相似度是通过对查询 token 和文档 token 之间的最大余弦相似度求和来计算的：

```
S(q, d) = sum_{i in query_tokens} max_{j in doc_tokens} cos(E_q[i], E_d[j])
```

实际上，这是对每个查询 token，在文档中找到最相似的 token，并求和。

ColPali 将这种后期交互应用于 Pali 的视觉 patch token。查询被编码为文本 token 的词汇表（如同 ColBERT中）。文档是图像 patch token（Pali 编码器将文档页面编码为视觉 patch token 的一个序列）。

视觉 token 和文本 token 被投影到一个共享的嵌入空间。然后应用 MaxSim：

```
S(q, d) = sum_{i in query_tokens} max_{j in image_patches} cos(E_query[i], E_patch[d][j])
```

视觉 patch token 密集地捕捉表格结构、字体和图表细节，而文本编码器则捕捉语义。

### 为什么视觉检索击败了文本检索

在保留的文档检索基准上（DocVQA、InfographicsVQA），视觉检索（ColPali）比文本密集检索（BM25 + ColBERT-v2）好 10-15%。原因：

- 解析器无法完美捕捉的表格：视觉 patch 能捕捉到表格（行 x 列）的全局结构。
- 图表：数值中轴的值很难从字符中解析出来。视觉 patch 直接将其捕捉为像素值。
- 多模态：页眉/页脚/图像的布局结构被保留在视觉 patch 中。

### ColPali 的训练

ColPali 在 ViDoRe（视觉文档检索）基准上进行训练。ViDoRe 包含来自 DocVQA、InfographicsVQA 等的查询-检索对。它在 Pali 权重的基础上进行微调。

训练损失：对比损失（InfoNCE），其中查询 anchor 和正确的文档页面是正对，批内其他文档是负对。

训练中一个关键的复杂性是：视觉编码器输出多种多 patch 的嵌入。ColPali 使用后期交互，这意味着批内对比是嵌入空间的直接比较。

### 与文本检索的比较

| 方法 | 编码器 | 表示 | 召回率（@100，DocVQA） |
|------|--------|------|------------------------|
| BM25 | 词袋 | 稀疏向量 | 54.9 |
| ColBERT-v2 | BERT | 每 token 嵌入向量 | 71.9 |
| SPLADE | BERT | 稀疏学习 | 62.4 |
| ColPali | Pali（ViT + LLM） | 每 patch 嵌入向量 | 83.3 |

结果不言自明。视觉表示对于文档检索至关重要。

### ColPali 在 RAG 中的应用

一旦页面被 ColPali 检索到，它们需要被送入生成器 VLM。这可以通过以下两种方式完成：

- 页面图像被送入多模态 LLM（如 LLaVA-NeXT、Idefics2）。生成的 LLM 直接根据图像回答。这是最有效的设置。
- 或者使用解析器提取文本并将其发送到纯文本 LLM（次优：又重新引入了解析瓶颈）。

对于检索增强生成（RAG），正确的流水线是页面图像 → ColPali 检索器 → 多模态 LLM 生成器。该流水线完全保持原生视觉状态。

## 使用它

`code/main.py` 构建了一个在合成数据集上的 ColPali 玩具检索器：

- 合成文档是带有随机单个单词的 4x4 像素网格（一个"页面"由 4x4=16 个视觉 token 组成）。
- 查询是单词（"dog"、"cat"、"table"）。
- ColPali 编码器：ViT（4 层，64 维隐藏 → 16 个视觉 token，将每个 patch 投影到一个 16 维的后期交互嵌入）和文本编码器（类似 BERT 的嵌入，将每个 token 编码为 16 维向量）。
- 后期交互：MaxSim（带有最大 token 求和）：S(q, d) = sum_{q_token} max_{patch} cos(E_q, E_d_patch)。
- 在合成的 40 个文档-20 个查询对上训练。

该检索器正确识别哪些文档与哪些查询匹配。

## 交付物

本课程产出 `outputs/skill-vision-document-retrieval-architect.md`。给定一个文档检索产品（发票、合同、研究报告），在 ColPali（视觉检索）和密集文本检索（ColBERT-v2）之间做出选择，并附上召回率比较和延迟估算。

## 练习

1. ColPali 在 ViDoRe 上针对 DocVQA 数据取得了 83.3 的召回率（@100）。ColBERT-v2 的召回率为 71.9。哪些特定的查询-文档对可以从视觉检索中受益？列举三个。

2. ColPali 使用后期交互相似度，其复杂度为 O(|查询| × |文档_patches|)。给定 32 个查询 token 和 1024 个文档 patch，计算每个文档的相似度成本。与余弦相似度（O(|查询_嵌入|)）进行对比。

3. 在 ColPali 中，使用对比损失（InfoNCE）在批内负例上进行训练。批次大小从 32 增加到 256 时，估计召回率变化。

4. 使用 ColPali 作为检索器的 RAG 流水线：如果生成器无法原生访问图像（只使用文本），你将如何桥接 ColPali 的输出？

5. 阅读 ColPali 论文第 3.1 节。描述 Pali 编码器架构和训练目标。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 后期交互 | "ColBERT 风格" | 在最终表示上检索，而不是在早期融合的交叉编码器上；保留查询和文档的独立嵌入 |
| MaxSim | "最大 token 相似度" | 对每个查询 token 取文档中最相似的 patch，并求和 |
| ViDoRe | "视觉文档检索" | ColPali 的评估基准；包含来自 DocVQA、InfographicsVQA 等的检索对 |
| 视觉 patch token | "文档 patch" | 将文档图像划分成 patch 并由视觉编码器编码的视觉 token |
| 端到端视觉检索 | "无解析" | 绕过文本解析和嵌入，直接根据查询匹配文档图像 |
| Pali | "视觉模型" | Google 的文档理解 VLM；ColPali 从 Pali 初始化权重 |

## 延伸阅读

- [Faysse 等人 — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (SIGIR 2020)](https://arxiv.org/abs/2004.12832)
- [ColPali 团队 — Vidore / ColPali 仓库](https://github.com/illuin-tech/colpali)
- [Google — Pali (arXiv:2209.06794)](https://arxiv.org/abs/2209.06794)
- [Formal 等人 — ColPali v2 (arXiv:2410.16263)](https://arxiv.org/abs/2410.16263)
