# 提示缓存与语义缓存经济学

> **定价快照日期：2026-04。以下数字反映本课程发布时各厂商的价格表；在使用前请对照所附文档进行验证。**
>
> 缓存发生在两个层面。L2（提供商级别）提示/前缀缓存复用重复前缀的注意力 KV——Anthropic 的提示缓存文档声称在长提示上可降低高达 90% 的成本和 85% 的延迟；对于 Claude 3.5 Sonnet，缓存读取为 $0.30/M，而新鲜读取为 $3.00/M，TTL 为 5 分钟，1 小时 TTL 选项的写入溢价为 2 倍（docs.anthropic.com，2026-04）。OpenAI 的提示缓存对 ≥1024 token 的提示自动生效，缓存输入价格约为新鲜输入的 10%（platform.openai.com，2026-04）；各模型的确切缓存价格取决于实时价格表。L1（应用级别）语义缓存在嵌入相似性命中时完全跳过 LLM。厂商声称的"95% 准确率"指的是匹配正确性，而非命中率——报告的生产命中率范围从 10%（开放式聊天）到 70%（结构化 FAQ）；没有厂商发布官方基线，因此将这些视为社区遥测数据而非保证。生产陷阱：并行化会破坏缓存（N 个并行请求在第一次缓存写入之前发出，可能使成本膨胀数倍），而前缀中的动态内容会完全阻止缓存命中。ProjectDiscovery 报告通过将动态文本移出可缓存前缀，命中率从 7% 提升到 74%（2025-11）。

**Type:** Learn
**Languages:** Python（stdlib，双层缓存模拟器）
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)、Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 分钟

## 学习目标

- 区分 L2 提示/前缀缓存（提供商端 KV 复用）和 L1 语义缓存（相似提示时绕过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记和两个 TTL 选项（5 分钟 vs 1 小时）及其价格乘数。
- 根据命中率、提示/响应混合和 token 价格计算预期的月度节省。
- 说出使账单膨胀 5-10 倍的并行化反模式和使命中率崩溃的动态内容反模式。

## 问题

你在 RAG 服务中添加了提示缓存。账单保持不变。你测量了命中率；只有 7%。你的提示看起来是静态的，但实际并非如此——系统提示包含精确到分钟的当前时间、请求 ID 以及为了多样性而随机排序的示例。每个请求都写入新的缓存条目，读取为零。

另外，你的代理每个用户问题并行发起十个工具调用。所有十个调用在第一次缓存写入完成之前就到达了提供商。十次写入，零次读取。你的账单是"使用缓存"预期成本的 5-10 倍。

缓存是一种协议，而不是一个开关。两个层面，两种不同的故障模式。

## 概念

### L2——提供商提示/前缀缓存

提供商存储可缓存前缀的注意力 KV，并在下一个匹配该前缀的请求时复用。你支付一次写入成本，读取几乎免费。

**Anthropic（Claude 3.5 / 3.7 / 4 系列）**：请求中的显式 `cache_control` 标记。你标记哪些块是可缓存的。TTL：5 分钟（写入成本为基础价格的 1.25 倍）或 1 小时（写入成本为基础价格的 2 倍）。缓存读取：Claude 3.5 Sonnet 上 $0.30/M vs $3.00/M 新鲜读取——便宜 10 倍（docs.anthropic.com，截至 2026-04）。不同模型（Opus/Haiku 单独定价）费率不同；始终交叉核对实时定价页面。

**OpenAI**：对 ≥1024 token 的提示自动缓存（platform.openai.com，2026-04）。无需显式标记。在当前 gpt-4o/gpt-5 价格表上，缓存输入约为新鲜输入的 10%。文档和发布说明均未发布官方命中率基线；社区报告在精心设计提示的情况下约为 30-60%。通过监控 `usage.cached_tokens` 自行测量。

**Google（Gemini）**：通过显式 API 进行上下文缓存；1M token 上下文意味着缓存的收益更大。

**自托管（vLLM、SGLang）**：Phase 17 · 06 介绍了 RadixAttention——在你自己的计算资源上实现相同的模式。

### L1——应用级语义缓存

在调用 LLM 之前，先哈希提示、嵌入，然后查找类似的缓存请求（余弦相似度高于阈值，通常为 0.95+）。命中时返回缓存的响应。未命中时调用 LLM 并缓存结果。

开源：Redis Vector Similarity、GPTCache、Qdrant。商业：Portkey Cache、Helicone Cache。

厂商的准确率声称指的是返回的缓存响应在语义上的恰当程度——而非命中率。生产命中率：

- 开放式聊天：10-15%。
- 结构化 FAQ/支持：40-70%。
- 代码问题：20-30%（微小变体破坏命中）。
- 重复提示的语音代理：50-80%（语音归一化后为固定集）。

### 并行化反模式

你的代理并行进行 10 个工具调用。所有 10 个都有相同的 4K token 系统提示。Anthropic 缓存写入是按请求的；第一次缓存写入在提供商看到提示后约 300 ms 完成。请求 2-10 在同一毫秒窗口内到达，每个都看到缓存未命中。你支付 10 次写入溢价，0 次读取折扣。

修复方法：使用串行优先批处理——先单独发出请求 1，然后在 1 的缓存填充后再发出 2-10。这会给第一个工具调用增加 300 ms；但节省 5-10 倍的账单。

### 动态内容反模式

你的系统提示看起来像：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都是唯一的。每个请求都写入。零命中。

修复方法：将所有真正静态的内容移到可缓存前缀；在缓存边界后追加动态内容：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 以此方式将缓存命中率从 7% 提升到 74%，并公开了其分析过程。

### 堆叠批处理 + 缓存用于夜间工作负载

批处理 API（Phase 17 · 15）在 24 小时周转下提供 50% 折扣。在此基础上叠加缓存输入可再获得约 10 倍收益。通过堆叠，夜间分类、标注和报告生成工作负载的成本可降至同步非缓存方式的约 10%。

### 你应该记住的数字

定价数据采集于 2026-04，来自所附厂商文档，每隔几个月就会变化——在依赖它们之前请重新核对。

- Anthropic 缓存读取：Claude 3.5 Sonnet 上 $0.30/M，约为新鲜输入的 10 分之一（docs.anthropic.com）。
- Anthropic 缓存写入溢价：1.25 倍（5 分钟 TTL）或 2 倍（1 小时 TTL）。
- OpenAI 自动缓存：适用于 ≥1024 token 的提示；缓存输入在当前价格表上约为新鲜输入的 10%（platform.openai.com）。
- 语义缓存命中率（社区报告）：开放式聊天约 10%；结构化 FAQ 高达约 70%。非厂商文档化的基线。
- ProjectDiscovery：将动态内容移出前缀后，命中率从 7% 提升到 74%（项目博客，2025-11）。
- 并行化反模式：当 N 个并行请求错过第一次缓存写入时，典型报告显示账单膨胀 5-10 倍。

## 使用它

`code/main.py` 模拟混合工作负载下的 L1 + L2 缓存。报告命中率、账单，并显示并行化惩罚。

## 交付物

本课程产出 `outputs/skill-cache-auditor.md`。根据提示模板和流量审计可缓存性，并建议重构方案。

## 练习

1. 运行 `code/main.py`。切换并行化标志。账单变化多少？
2. 你的系统提示中有一个日期。将其移出。展示前后命中率的数学计算。
3. 根据你的请求到达率，计算 1 小时 TTL（2 倍写入）与 5 分钟 TTL（1.25 倍写入）的盈亏平衡点。
4. 语义缓存阈值为 0.95 时命中率为 20%。阈值为 0.85 时命中率为 50%，但你会看到不正确的缓存响应。选择合适的阈值并说明理由。
5. 你每个用户问题批处理 10 个并行子查询。在不增加端到端延迟的情况下，改写以提高缓存友好性。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| L2 prompt cache | "前缀缓存" | 提供商存储重复前缀的 KV |
| `cache_control` | "Anthropic 缓存标记" | 标记可缓存块的显式属性 |
| Cache write premium | "写入税" | 首次未命中到缓存的额外成本（1.25 倍或 2 倍） |
| L1 semantic cache | "嵌入缓存" | 调用 LLM 前的应用级哈希和嵌入查找 |
| GPTCache | "LLM 缓存库" | 流行的开源 L1 缓存库 |
| Cache hit rate | "命中/总数" | 从缓存服务的请求比例 |
| Parallelization anti-pattern | "N 写陷阱" | N 个并行请求 N 次缓存未命中 |
| Dynamic content trap | "提示中的时间陷阱" | 前缀中的动态字节破坏命中率 |
| RadixAttention | "副本内缓存" | SGLang 的前缀缓存实现 |

## 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)——官方 `cache_control` 语义和 TTL。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)——自动缓存行为和资格条件。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
