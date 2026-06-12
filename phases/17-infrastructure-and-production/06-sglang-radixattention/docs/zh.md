# SGLang 与面向前缀密集型工作负载的 RadixAttention

> SGLang 将 KV 缓存视为存储在基树（radix tree）中的一等可复用资源。vLLM 按 FCFS（先到先服务）调度请求，而 SGLang 的缓存感知调度器优先处理具有更长共享前缀的请求——实质上是一种深度优先的基树遍历，使热分支保持在 HBM 中。在 Llama 3.1 8B 上使用 ShareGPT 风格的 1K 提示时，SGLang 达到约 16,200 tok/s，而 vLLM 约 12,500，优势约 29%。在前缀密集的 RAG 工作负载上，优势可达 6.4 倍。在语音克隆类工作负载上，缓存命中率超过 86%。到 2026 年已部署在 400,000+ GPU 上，覆盖 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS。问题在于：当前缀排序不一致时，6.4 倍的数字会消失——排序是工程师的杠杆。

**Type:** Learn
**Languages:** Python（stdlib，简易基树缓存 + 缓存感知调度器）
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)、Phase 14 (Agentic RAG)
**Time:** ~75 分钟

## 学习目标

- 绘制 RadixAttention 的示意图：前缀如何存储在基树中，KV 块如何在根植于同一分支的序列间共享。
- 解释缓存感知调度，以及为什么 FCFS 对前缀密集型流量是不正确的。
- 根据前缀缓存命中率和提示长度分布，计算工作负载的预期加速比。
- 说出使 6.4 倍数字变为现实与失去优势的提示排序规范。

## 问题

经典推理服务将每个请求的提示视为不透明数据。即使 5,000 个 RAG 请求都以相同的 2,000 token 系统提示加上相同的检索前导词开头，vLLM 仍然会预填充这 2,000 token 前缀 5,000 次。GPU 一遍又一遍地做同样的工作。

观察：在代理（agent）和 RAG 工作负载中，提示几乎总是共享长前缀。系统提示、工具 schema、少样本示例、检索头、对话历史——所有这些都在请求间重复。如果你存储了一次该前缀的 KV 缓存并复用它，就不需要再次预填充。

RadixAttention 正是这样做的。token 被索引到一个基树中；每个节点拥有其从根到该节点路径上 token 序列的 KV 块。一个新请求遍历这棵树：任何 token 匹配的节点都复用该节点的 KV 块。预填充成本变得与"新"后缀成正比，而非整个提示。

挑战在于调度。如果两个请求共享一个 2,000 token 前缀，而第三个请求只共享同一前缀的 200 token，你希望一起服务这两个长共享请求，以便长前缀保持在 HBM 中。FCFS 则相反——它服务最先到达的请求，可能会在下一个长前缀请求到达之前逐出热分支。

## 概念

### 作为 KV 索引的基树

基树（紧凑字典树）存储 token 序列。每个节点拥有一个 token 范围以及为该范围计算的 KV 块。子节点扩展一个或多个 token 的序列。

```
root
  |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
       |- "Context: <doc A>..."        (500 tokens, 31 blocks)
            |- "Question: Alice..."    (80 tokens, 5 blocks)
            |- "Question: Bob..."      (95 tokens, 6 blocks)
       |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

一个新请求到来，包含系统提示 + "Context: <doc A>" + "Question: Carol"。调度器遍历：系统前缀匹配（复用 124 块），doc-A 分支匹配（复用 31 块），然后只为新内容 "Question: Carol" 分配新块（4 块）。预填充成本：4 块新 token。没有基树时：160 块。预填充节省约 40 倍。

### 缓存感知调度

如果缓存频繁抖动，基于基树的复用就毫无意义。两个关键策略：

1. **深度优先派发**。从队列中选择下一个请求时，优先选择根植于当前运行集合同一分支的请求。这使热分支保持在驻留状态。
2. **分支级 LRU，而非块级 LRU**。逐出整个分支（从使用最少的叶子开始），而非单个块，使缓存形状与基树形状匹配。

FCFS 违反了这两条。一个共享 2,000 token 的请求排在共享 50 token 的请求后面，然后 2,000 token 分支被逐出以容纳 50 token 的请求。

### 你应该记住的基准测试数字

- Llama 3.1 8B, H100, ShareGPT 1K 提示：SGLang 约 16,200 tok/s vs vLLM 约 12,500（约 29% 优势）。
- 前缀密集的 RAG（相同系统 + 相同文档，不同问题）：SGLang 上最高 6.4 倍。
- 语音克隆工作负载：86.4% 前缀缓存命中率。
- SGLang 客户的生产命中率：50-99%，取决于提示规范。
- 到 2026 年已部署在 400,000+ GPU 上。

### 排序陷阱

6.4 倍的数字依赖于一致的提示模板排序。如果你的客户端在某些请求中将提示构造为 `[system, tools, context, history, question]`，而在其他请求中构造为 `[system, context, tools, history, question]`，基树就无法找到共享前缀。对人类来说看起来是共享前缀的内容，对基树来说是两条不同的序列。

工程师的杠杆：你的提示模板就是一个缓存键。固定顺序。将所有不可变内容（系统提示、工具、schema）放在前面。检索上下文放在中间。用户问题放在最后。不要将动态内容交织到前缀中。

研究中的一个真实案例：将动态内容移出可缓存前缀，一次更改就将一个部署的缓存命中率从 7% 提升到 74%。

### RadixAttention 的赢与输

赢：
- RAG（相同检索前导词，不同问题）。
- 代理（相同工具 schema，不同查询）。
- 带有长系统提示的聊天。
- 具有重复前导词的语音/视觉工作负载。

输（回到 vLLM 级吞吐量）：
- 具有唯一提示的单次生成（代码补全、无系统提示的开放式聊天）。
- 每个请求都在前缀中交织唯一内容的动态提示。

### 为什么这是调度器问题，而非仅仅是内核问题

你可以将 KV 复用实现为一个内核技巧。SGLang 的洞见在于：只有当调度器使热分支保持驻留时，复用才有价值。在混合负载下，一个朴素的"如果可用就复用"策略会导致缓存抖动。基于基树索引的调度器才是将内核技巧转化为 29% 生产优势的关键。

### 与 vLLM 的关系

这两个系统并非严格意义上的竞争对手。到 2026 年，vLLM 添加了前缀缓存（`--enable-prefix-caching`）和一个缓存感知路由器（vLLM Router in Rust）。差距缩小了但并未完全消失——SGLang 的整个栈是基树优先的；vLLM 是后来嫁接的。对于以前缀复用为主的工作负载，SGLang 仍是默认选择。对于没有强前缀模式的通用推理服务，vLLM 仍然与之相当或更好。

```figure
roofline
```

## 使用它

`code/main.py` 实现了一个简易基树 KV 缓存加一个支持两种策略的调度器：FCFS 和缓存感知。在相同工作负载上运行两者，报告前缀缓存命中率和吞吐量差异。然后运行一个"乱序"工作负载来展示 6.4 倍的崩溃。

## 交付物

本课程产出 `outputs/skill-radix-scheduler-advisor.md`。给定工作负载描述（提示模板形状、检索模式、并发租户数），它生成一个提示排序规范和是否采用 SGLang 的建议。

## 练习

1. 运行 `code/main.py`。在相同工作负载上比较 FCFS 和缓存感知。差异来自哪里——预填充节省、解码节省还是队列延迟？
2. 修改工作负载，使提示随机排列 `[system, tools, context]`。重新运行。命中率发生了什么变化？为什么？
3. 计算在 Llama 3.1 8B 上将 2,000 token 系统提示作为一个基树分支驻留所需的 HBM 成本。与没有前缀复用的 16 序列批次成本进行比较。
4. 阅读 SGLang RadixAttention 论文。用三句话解释为什么树形 LRU 逐出在前缀密集型负载下优于块形 LRU 逐出。
5. 某个客户报告只有 8% 的缓存命中率。说出三个可能的原因以及你为每个原因运行的诊断。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| RadixAttention | "那个 SGLang 的东西" | KV 缓存被索引为基树，使共享前缀可以复用块 |
| Radix tree | "紧凑字典树" | 每个节点拥有一个 token 范围及其 KV 块的树 |
| Cache-aware scheduler | "热分支优先" | 优先选择与驻留分支共享的请求的调度器 |
| Prefix-cache hit rate | "你的提示中有多少是免费的" | 从复用的 KV 块服务的提示 token 比例 |
| FCFS | "先到先服务" | 破坏前缀局部性的默认调度方式 |
| Branch-level LRU | "逐出叶子" | 与基树形状匹配的逐出策略 |
| Prompt template ordering | "缓存键" | 提示的组件顺序决定了树可以共享什么 |
| System prompt pinning | "驻留前缀" | 保持不可变系统提示部分驻留，避免逐出抖动 |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang)——源代码和文档。
- [SGLang documentation](https://sgl-project.github.io/)——RadixAttention 和调度详情。
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104)——设计参考。
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/)——基准测试数字和调度器原理。
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html)——vLLM 自己的类基树实现，供比较。
