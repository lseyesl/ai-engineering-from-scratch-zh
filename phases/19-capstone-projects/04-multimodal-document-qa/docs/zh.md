# 顶点项目 04——多模态文档问答（视觉优先 PDF、表格、图表）

> 2026 年的文档问答前沿已从 OCR 后文本转向视觉优先的后期交互。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面视为图像，用多向量后期交互嵌入，让查询直接关注 patches。在金融 10-K 报告、科学论文和手写笔记上，这种模式以较大优势击败了 OCR 优先方法。端到端构建处理 10k 页面的管道，并发布与 OCR 后文本的并排比较。

**类型:** Capstone
**语言:** Python（管道）、TypeScript（查看器 UI）
**前置要求:** Phase 4（计算机视觉）、Phase 5（NLP）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）
**涉及阶段:** P4 · P5 · P7 · P11 · P12 · P17
**时间:** 30 小时

## 问题

企业有大量 PDF 被 OCR 管道破坏：扫描的 10-K 报告带有旋转的表格、密集公式的科学论文、只有作为图像才有意义的图表、手写注释。将这些视为文本优先意味着丢失一半的信号。2026 年的答案是原始页面图像的后期交互多向量检索。ColPali（Illuin Tech）引入了它；ColQwen2.5-v0.2 和 ColQwen3-omni 推动了准确性。在 ViDoRe v3 上，视觉优先检索的分数以有意义的差距高于 OCR 后文本——而且在图表、表格和手写上的差距更大。

权衡点是存储和延迟。ColQwen 嵌入是每页约 2048 个 patch 向量，而不是单个 1024 维向量。原始存储膨胀。DocPruner（2026）在不造成显著准确性损失的情况下实现 50% 的剪枝。你将索引 10k 页面，测量 ViDoRe v3 nDCG@5，在 2 秒内提供答案，并直接与 OCR 后文本基线进行比较。

## 概念

后期交互意味着每个查询 token 对每个 patch token 进行评分，每个查询 token 的最大分数汇总。你在不需要单一池化向量的情况下获得细粒度匹配。多向量索引（Vespa、Qdrant 多向量或 AstraDB）存储每个 patch 的嵌入，并在检索时运行 MaxSim。

回答器是一个视觉语言模型，它接受查询加上检索到的 top-k 页面图像，并输出带证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于公式和科学符号，可选的 OCR 后备方案（Nougat、dots.ocr）作为额外文本通道接入。

评估是一个二维矩阵。一个轴：内容类型（纯文本段落、密集表格、柱状/折线图、手写笔记、公式）。另一个轴：检索方法（视觉优先后期交互 vs OCR 后文本 vs 混合）。每个单元格获得 nDCG@5 和答案准确性。报告就是交付物。

## 架构

```
PDFs -> 页面渲染器 (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (每页多向量，~2048 patches)
           |
           +------> DocPruner 50% 压缩
           |
           v
   多向量索引 (Vespa 或 Qdrant multi-vector)
           |
query ----+----> 检索 top-k 页面 (MaxSim)
           |
           v
  VLM 回答器: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    输入: query + top-k 页面图像 + 可选 OCR 文本
           |
           v
   答案 + 引用的页码 + 证据区域
           |
           v
   Streamlit / Next.js 查看器: 源页面上的高亮框
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，纵向标准化
- 后期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni
- 索引：支持多向量字段的 Vespa，或 Qdrant 多向量，或带 MaxSim 的 AstraDB
- 剪枝：DocPruner 2026 策略（保留高方差 patches，50% 压缩，准确性损失 < 0.5%）
- OCR 后备方案（公式/密集表格）：dots.ocr 或 Nougat
- VLM 回答器：自托管 Qwen3-VL-30B 或托管 Gemini 2.5 Pro；InternVL3 作为备用
- 评估：ViDoRe v3 基准、M3DocVQA 用于多页推理
- 查看器 UI：带画布叠加的 Next.js 15，用于证据区域

## 构建它

1. **摄取。** 遍历 10k 个 PDF 页面的语料库，涵盖 10-K 报告、科学论文和扫描文档。将每页渲染为 1536x2048 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 对每页图像运行 ColQwen2.5-v0.2。输出形状为约 2048 个 patch 嵌入，维度 128。应用 DocPruner 保留信号最强的一半。写入 Vespa 多向量字段或 Qdrant 多向量。

3. **查询。** 对每个传入查询，用查询塔（token 级嵌入）嵌入。对索引运行 MaxSim：对每个查询 token，取页面 patch 嵌入上的最大点积，求和。返回 top-k 页面。

4. **合成。** 调用 Qwen3-VL-30B，传入查询和 top-5 页面图像。提示："仅使用提供的页面作答。按 (doc_id, page) 引用每个声明，并指明区域（图形、表格、段落）。"

5. **证据区域。** 后处理答案以提取引用的区域。如果 VLM 输出边界框（Qwen3-VL 支持），在查看器中将其渲染为叠加层。

6. **OCR 后备方案。** 对于被识别为公式密集的页面（基于图像方差的启发式），运行 Nougat 或 dots.ocr，并将 OCR 文本作为额外通道与图像一起传递。

7. **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页 QA 准确性）。还在相同语料库上运行 OCR 后文本管道，使用相同的合成器。生成一个内容类型 × 方法的矩阵。

8. **UI。** 先做 Streamlit 原型；Next.js 15 生产查看器，逐页显示证据区域叠加。

## 使用它

```
$ doc-qa ask "2024 年 EMEA 分部的营业利润率变化是多少？"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, 引用 (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA 营业利润率从 18.2% 降至 16.8%，下降 140 个基点。
  引用: 10-K-2024.pdf p.88 (表 4, 分部营业利润率)
         10-K-2024.pdf p.92 (MD&A, 经营业绩)
[viewer]     以高亮边界框叠加在 p.88 表 4 上打开
```

## 交付物

`outputs/skill-doc-qa.md` 描述了交付物：一个视觉优先的多模态文档问答系统，针对特定语料库调优，并在 ViDoRe v3 上与 OCR 后文本基线进行评估。

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确性 | 基准数字 vs OCR 文本基线和已发布排行榜 |
| 20 | 证据区域定位 | 引用的区域中实际包含答案片段的比例 |
| 20 | 存储和延迟工程 | DocPruner 压缩比、索引 p95、答案 p95 |
| 15 | 源检查 UX | 查看器清晰度、叠加保真度、并排比较工具 |
| **100** | | |

## 练习

1. 在同一语料库上测量 ColQwen2.5-v0.2 vs ColQwen3-omni。一个模型正确而另一个错过的页面是哪些？向索引添加"内容类"标签以按类型路由。

2. 激进地剪枝嵌入（75%，90%）。找到压缩断崖：ViDoRe nDCG@5 下降到低于 OCR 基线的点。

3. 构建混合方法：并行运行 OCR 后文本和 ColQwen，用 RRF 融合，用交叉编码器重排序。混合方法能单独击败任何一种吗？它在哪些方面帮助最大？

4. 将 Qwen3-VL-30B 替换为较小的 VLM（Qwen2.5-VL-7B）。测量准确性随美元成本的曲线。

5. 添加手写笔记支持。渲染手写语料库，用 ColQwen 嵌入，测量检索。与手写 OCR 管道进行比较。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Late interaction | "ColPali 风格检索" | 查询 token 独立对页面 patches 评分；MaxSim 聚合 |
| Multi-vector | "每 patch 嵌入" | 每个文档有许多向量，而非一个池化向量 |
| MaxSim | "后期交互评分" | 对每个查询 token，取文档向量上的最大相似度；求和 |
| DocPruner | "Patch 压缩" | 2026 年剪枝，保留 50% patches，准确性损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026 年测量视觉文档检索的标准 |
| Evidence region | "引用的边界框" | 源页面上定位答案片段的 bbox |
| OCR fallback | "公式通道" | 与视觉一起用于公式或表格密集页面的文本管道 |

## 延伸阅读

- [ColPali (Illuin Tech) 仓库](https://github.com/illuin-tech/colpali)——参考后期交互文档检索
- [ColPali 论文 (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)——基础方法论文
- [ColQwen 家族 (Hugging Face)](https://huggingface.co/vidore)——生产就绪的检查点
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952)——多页多模态 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html)——参考服务栈
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors)——备选索引
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html)——备选托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat)——支持公式的 OCR 后备方案
