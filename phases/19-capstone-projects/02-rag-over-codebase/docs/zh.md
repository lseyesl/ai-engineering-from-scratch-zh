# 顶点项目 02——基于代码库的 RAG（跨仓库语义搜索）

> 2026 年，每个正经的工程组织都运行内部代码搜索，它理解含义而不仅仅是字符串。Sourcegraph Amp、Cursor 的代码库答案、Augment 的企业图、Aider 的 repomap、Pinterest 的内部 MCP——形态相同。摄取多个仓库，用 tree-sitter 解析，在函数和类级别分块嵌入，混合搜索，重排序，带引用给出答案。这个顶点项目要求你构建一个能处理 10 个仓库中 200 万行代码的系统，并在每次 git 推送后存活增量重新索引。

**类型:** Capstone
**语言:** Python（摄取）、TypeScript（API + UI）
**前置要求:** Phase 5（NLP 基础）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 17（基础设施）
**涉及阶段:** P5 · P7 · P11 · P13 · P17
**时间:** 30 小时

## 问题

到 2026 年，每个前沿编码智能体都附带代码库检索层，因为仅靠上下文窗口无法解决跨仓库问题。Claude 的 1M token 上下文有帮助；但它并没有消除对分级检索的需求。对原始块进行简单余弦搜索会在生成代码、单仓库重复以及很少导入的符号长尾上毒化结果。生产答案是混合（稠密 + BM25）搜索，基于 AST 感知的块，配合重排序器，后端有符号引用图支持。

你通过索引一个真实的项目群（不是一个教程仓库）并测量 MRR@10、引用忠实度和增量新鲜度来学习这一点。故障模式是基础设施层面的：一个 10 万文件的单仓库、一个触及半数文件的推送、一个需要跨越四个仓库才能正确回答的查询。

## 概念

AST 感知的摄取管道用 tree-sitter 解析每个文件，提取函数和类节点，并在节点边界处而不是固定 token 窗口处分块。每个块有三种表示：稠密嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 项和简短的自然语言摘要。摘要增加了第三种可检索的模态——用户问"X 是如何授权的"，摘要提到"authz"，即使代码中只有 `check_permission`。

检索是混合式的。查询同时触发稠密和 BM25 搜索，合并 top-k，并将并集交给交叉编码器重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表交给长上下文合成器（带提示缓存的 Claude Sonnet 4.7，或自托管的 Llama 3.3 70B），指令要求对每个声明引用文件和行范围。没有引用的答案会被后置过滤器拒绝。

增量新鲜度是基础设施问题。Git 推送触发差异：哪些文件更改了，哪些符号更改了。只有受影响的块重新嵌入。受影响的跨文件符号边（导入、方法调用）被重新计算。索引在不重新处理每次提交的 200 万行代码的情况下保持一致。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：支持 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）的 tree-sitter
- 稠密嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 作为备用
- 稀疏索引：Tantivy（Rust）配合 BM25F，在符号名与体之间按字段加权
- 向量数据库：Qdrant 1.12 混合搜索，或 pgvector + pgvectorscale（适用于 5000 万向量以下的团队）
- 块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，带提示缓存
- 重排序器：Cohere rerank-3 或自托管 bge-reranker-v2-gemma-2b
- 编排：摄取用 LlamaIndex Workflows，查询智能体用 LangGraph
- 合成器：带提示缓存的 Claude Sonnet 4.7（1M 上下文）
- 符号图：Neo4j（托管）或 kuzu（嵌入式），用于导入和调用边
- 可观测性：每个检索+合成步骤的 Langfuse spans

## 构建它

1. **摄取遍历器。** 遍历 git 历史，每个推送钩子触发。收集更改的文件。对每个文件，用 tree-sitter 解析，提取函数和类节点及其完整源码跨度。发出块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **块摘要器。** 将块分批送入 Haiku 4.5 调用，系统前言使用提示缓存。提示："用一句话总结这个函数，说明其公共约定和副作用。"将摘要与块一起存储。

3. **嵌入池。** 两个并行队列：稠密（Voyage-code-3 批量 128）和摘要（同一模型，但针对摘要字符串）。将向量写入 Qdrant，附带负载 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名权重 4、符号体权重 1、摘要权重 2。支持"查找名为 X 的函数"查询以及"查找执行 X 操作的函数"查询。

5. **符号图。** 对每个块，记录边：导入（此文件使用仓库 Z 中的符号 Y）、调用（此函数调用类 C 上的方法 M）、继承。存储在 kuzu 中。在查询时用于跨仓库边界扩展检索。

6. **查询智能体。** 包含三个节点的 LangGraph。`retrieve` 并行触发稠密搜索和 BM25 搜索，按 (repo, path, symbol) 去重。`rerank` 对 top-50 运行交叉编码器并保留 top-10。`synth` 调用 Claude Sonnet 4.7，将重排序后的块放入上下文，缓存系统提示，要求提供 file:line 引用。

7. **引用强制。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的声明被标记为重新询问或丢弃。仅返回带引用的答案给用户。

8. **增量重新索引。** 在每个 webhook 上，计算符号级别的差异。仅重新嵌入文本更改的块。重新计算导入更改的块的符号边。衡量：对于 200 万 LOC 的项目群，50 个文件的推送在 60 秒内完成重新索引。

9. **评估。** 标记 100 个跨仓库问题，附带 gold 文件:行答案。测量 MRR@10、nDCG@10、引用忠实度（具有可验证锚点的声明比例）和 p50/p99 延迟。

## 使用它

```
$ code-rag ask "S3 multipart abort 是如何接入我们的重试预算的？"
[retrieve]  12 chunks dense + 7 chunks bm25, 去重后 16 个唯一结果
[rerank]    保留 top-5 (cohere rerank-3)
[synth]     claude-sonnet-4.7, 缓存命中率 68%, 2.1s
answer:
  Multipart abort 由 `AbortMultipartOnFail` 在
  services/uploader/retry.go:122-148 中触发，它会减少
  config/budgets.yaml:34-51 中定义的每桶重试预算 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付物

交付技能文件 `outputs/skill-codebase-rag.md`。给定一个代码库群，它架设起摄取管道、混合索引和查询智能体，并为任何跨仓库问题返回带引用的答案。评分标准：

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | 检索质量 | 在 100 个问题的留出集上的 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度 | 具有可验证 file:line 锚点的答案声明比例 |
| 20 | 延迟和规模 | 在索引语料库规模上 10k QPS 时的 p95 查询延迟 |
| 20 | 增量索引正确性 | 从 git 推送到可搜索的时间（针对 50 个文件的提交）|
| 15 | UX 和答案格式 | 引用可点击性、片段预览、后续问题支持 |

## 练习

1. 将 Voyage-code-3 替换为自托管的 nomic-embed-code。测量 MRR@10 差异。报告启用重排序后差距是否缩小。

2. 将 20% 的生成代码（LLM 制作的样板）注入语料库并重新评估。观察检索毒化。向负载中添加"generated"标志并对这些命中降低权重。

3. 在你的语料库规模上基准测试 Qdrant 混合搜索 vs pgvector + pgvectorscale。报告批量大小为 1 时的 p99。

4. 添加基于采样的漂移检查：每周重新运行 100 个问题的评估。MRR@10 下降超过 5% 时发出警报。

5. 扩展到跨语言符号解析：一个调用 Go 服务（通过 gRPC）的 Python 函数。使用符号图链接它们。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| AST-aware chunking | "函数级分割" | 在 tree-sitter 节点边界处而不是固定 token 窗口处切割代码 |
| Hybrid search | "稠密 + 稀疏" | 并行运行 BM25 和向量搜索，合并 top-k，重排序 |
| Cross-encoder rerank | "第二阶段排名" | 同时评分每个（查询，候选）对的模型，比余弦更准确 |
| Prompt caching | "缓存的系统提示" | 2026 年 Claude/OpenAI 功能，折扣重复前缀 token 高达 90% |
| Symbol graph | "代码图" | 跨文件和仓库的导入、调用、继承边 |
| Citation faithfulness | "有根据的答案率" | 用户可以通过点击锚点并读取引用的跨度来验证的声明比例 |
| Incremental re-index | "推送到搜索的时间" | 从 git 推送到更改的符号可查询的墙上时钟时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com)——生产级跨仓库代码智能
- [Sourcegraph Cody RAG 架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase)——此顶点项目的参考深度解读
- [Aider repo-map](https://aider.chat/docs/repomap.html)——tree-sitter 排名的仓库视图
- [Augment Code 企业图](https://www.augmentcode.com)——商业符号图 RAG
- [Qdrant 混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/)——参考实现
- [Voyage AI 代码嵌入](https://docs.voyageai.com/docs/embeddings)——Voyage-code-3 详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank)——交叉编码器参考
- [Pinterest MCP 内部搜索](https://medium.com/pinterest-engineering)——内部平台参考
