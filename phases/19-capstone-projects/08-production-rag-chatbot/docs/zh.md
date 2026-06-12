# 顶点项目 08——面向受监管领域的生产 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都运行相同的生产形态。用 docling 或 Unstructured 和 ColPali 摄取视觉内容。混合搜索。用 bge-reranker-v2-gemma 重排序。用使用提示缓存的 Claude Sonnet 4.7 合成，命中率 60-80%。用 Llama Guard 4 和 NeMo Guardrails 守卫。用 Langfuse 和 Phoenix 监控。用 RAGAS 在 200 个问题的黄金集上评分。在受监管领域（法律、临床、保险）构建一个，顶点项目就是通过黄金集、红队测试和漂移仪表板。

**类型:** Capstone
**语言:** Python（管道 + API）、TypeScript（聊天 UI）
**前置要求:** Phase 5（NLP）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段:** P5 · P7 · P11 · P12 · P17 · P18
**时间:** 30 小时

## 问题

受监管领域的 RAG（法律合同、临床试验方案、保险政策）是 2026 年部署最多的生产形态，因为 ROI 显而易见，风险具体而明确。Harvey（Allen & Overy）为法律领域构建了它。Mendable 提供开发者文档版本。Glean 覆盖企业搜索。模式是：高保真摄取，混合检索加重排序，带引用强制和提示缓存的合成，多层安全防护，以及连续漂移监控。

困难的部分不是模型。它们是管辖感知的合规性（HIPAA、GDPR、SOC2）、引用级别的可审计性、成本控制（提示缓存命中率高时提供 60-90% 折扣）、通过 RAGAS 忠实度进行的幻觉检测，以及当源文档更新而索引未跟上时的漂移检测。这个顶点项目要求你交付所有这些内容，附带 200 个问题的黄金集和红队套件。

## 概念

管道有两个方面。**Ingestion**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的文档；块获得摘要、标签和基于角色的访问标签。向量进入 pgvector + pgvectorscale（低于 5000 万向量）或 Qdrant Cloud；稀疏 BM25 并行运行。**Conversation**：LangGraph 处理记忆和多轮对话；每个查询运行混合检索，用 bge-reranker-v2-gemma-2b 重排序，用 Claude Sonnet 4.7（提示缓存）合成，输出通过 Llama Guard 4 和 NeMo Guardrails，并输出带引用锚点的响应。

评估栈有四层。**Golden set**（200 个带引用的标记 Q/A）用于正确性。**Red team**（越狱、PII 提取尝试、领域外问题）用于安全性。**RAGAS** 用于每轮自动的忠实度/答案相关性/上下文精度。**Drift dashboard**（Arize Phoenix）每周监控检索质量和幻觉分数。

提示缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示 + 检索到的上下文。在 60-80% 命中率时，每查询成本下降 3-5 倍。管道必须设计为稳定的前缀（系统提示 + 重排序上下文在前）以实现高缓存命中率。

## 架构

```
documents (合同、协议、政策)
      |
      v
docling / Unstructured 解析 + ColPali 用于视觉内容
      |
      v
块 + 摘要 + 角色标签 + 管辖标签
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph 对话智能体
   +--- retrieve (混合)
   +--- 按 role + jurisdiction 过滤
   +--- rerank (bge-reranker-v2-gemma-2b 或 Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, 提示缓存)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio 输出 PII 擦洗)
   +--- cite + return
      |
      v
评估:
  RAGAS faithfulness / answer_relevance / context_precision (在线)
  Langfuse 注释队列 (采样)
  Arize Phoenix 漂移 (每周)
  红队套件 (发布前)
```

## 技术栈

- 摄取：Unstructured.io 或 docling 用于结构化文档；ColPali 用于视觉丰富的 PDF
- 向量数据库：低于 5000 万向量时使用 pgvector + pgvectorscale；否则使用 Qdrant Cloud
- 稀疏：Tantivy BM25，带字段权重
- 编排：LlamaIndex Workflows（摄取）+ LangGraph（对话）
- 重排序器：自托管 bge-reranker-v2-gemma-2b 或托管 Voyage rerank-2
- LLM：带提示缓存的 Claude Sonnet 4.7；备用 Llama 3.3 70B 自托管
- 评估：RAGAS 0.2 在线，DeepEval 用于幻觉和越狱套件
- 可观测性：带注释队列的自托管 Langfuse；Arize Phoenix 用于漂移
- 防护：Llama Guard 4 输入/输出分类器、NeMo Guardrails v0.12 策略、Presidio PII 擦洗
- 合规性：块上基于角色的访问标签；用于 GDPR/HIPAA 的管辖标签

```figure
canary-rollout
```

## 构建它

1. **摄取。** 用 Unstructured 或 docling 解析你的语料库（严肃构建需要 1000-10000 个文档）。对于扫描/视觉密集页面，通过 ColPali 路由。生成带摘要、角色标签、管辖标签的块。

2. **索引。** 稠密嵌入（Voyage-3 或 Nomic-embed-v2）进入 pgvector + pgvectorscale。通过 Tantivy 的 BM25 侧索引。角色和管辖过滤器作为负载。

3. **混合检索。** 先按角色+管辖过滤；然后并行稠密 + BM25；用倒数排名融合合并；top-20 给重排序器；top-5 给合成器。

4. **带提示缓存的合成。** 缓存头中的系统提示 + 静态策略；重排序上下文作为缓存扩展；用户问题作为未缓存后缀。目标稳态 60-80% 缓存命中率。

5. **防护。** 输入上的 Llama Guard 4；NeMo Guardrails 栏杆阻止领域外问题或策略禁止的主题；Presidio 擦洗输出中的意外 PII；引用强制后置过滤器。

6. **黄金集。** 200 个由领域专家标记的 Q/A 对，附带（答案、引用）。对智能体进行精确引用匹配、答案正确性、忠实度（RAGAS）评分。

7. **红队。** 50 个对抗性提示：越狱（PAIR、TAP）、PII 泄露尝试、领域外、跨管辖泄露。以通过/失败和严重性评分。

8. **漂移仪表板。** Arize Phoenix 每周跟踪检索质量（nDCG、引用忠实度）。下降 5% 时发出警报。

9. **成本报告。** Langfuse：提示缓存命中率、每查询 token、按阶段划分的 $/query 明细。

## 使用它

```
$ chat --role=analyst --jurisdiction=GDPR
> 根据我们的合同，欧盟用户配置文件的数据留存义务是什么？
[retrieve]  混合 top-20 过滤到 GDPR + analyst-role
[rerank]    保留 top-5
[synth]     claude-sonnet-4.7, 缓存命中 74%, 0.8s
answer:
  合同（第 12.4 节，主服务协议，日期 2024-03-11）
  要求根据 GDPR 第 17 条，在终止后 30 天内删除欧盟用户配置文件。
  DPA 修正案（DPA-v2.1，第 5 节）将此期限缩短为 14 天（针对"受限"类别数据）。
  引用: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付物

`outputs/skill-production-rag.md` 描述了交付物。一个部署的受监管领域聊天机器人，带合规标签，通过评分标准，通过实时漂移监控进行观察。

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | RAGAS 忠实度 + 答案相关性 | 黄金集（200 Q/A）上的在线分数 |
| 20 | 引用正确性 | 具有可验证源锚点的答案比例 |
| 20 | 防护覆盖率 | Llama Guard 4 通过率 + 越狱套件结果 |
| 20 | 成本/延迟工程 | 提示缓存命中率、p95 延迟、$/query |
| 15 | 漂移监控仪表板 | Phoenix 实时仪表板，带每周检索质量趋势 |

## 练习

1. 在另一个管辖领域下构建第二个语料库切片（例如，与 GDPR 并行的 HIPAA）。在 20 个问题的跨管辖探测上演示角色+管辖过滤防止交叉泄露。

2. 测量一周生产流量中的提示缓存命中率。识别哪些查询破坏了缓存前缀。重新结构化。

3. 添加带 10k token 摘要缓冲区的多轮记忆。测量随着对话增长忠实度是否下降。

4. 将 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量 $/query 和忠实度差异。

5. 添加"不确定"模式：如果前几个重排序分数低于阈值，智能体说"我没有可靠的引用"而不是回答。测量虚假信心的减少。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Prompt caching | "缓存的系统 + 上下文" | Claude/OpenAI 功能：命中的缓存前缀 token 折扣 60-90% |
| RAGAS | "RAG 评估器" | 忠实度、答案相关性、上下文精度的自动评分 |
| Golden set | "标记的评估" | 200+ 个专家标记的 Q/A 及引用；地面真相 |
| Jurisdiction tag | "合规标签" | 附加到块的 GDPR/HIPAA/SOC2 范围；由检索过滤器执行 |
| Citation faithfulness | "有根据的答案率" | 由可检索源片段支持的声明比例 |
| Drift | "检索质量衰减" | nDCG 或引用分数的每周变化；警报阈值 5% |
| Red team | "对抗性评估" | 发布前越狱、PII 提取、领域外探测 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai)——参考法律生产栈
- [Glean 企业搜索](https://www.glean.com)——企业规模 RAG 参考
- [Mendable 文档](https://mendable.ai)——开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/)——托管摄取
- [Anthropic 提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)——成本杠杆参考
- [RAGAS 0.2 文档](https://docs.ragas.io/)——权威 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)——参考漂移可观测性
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/)——2026 安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/)——策略栏杆框架
