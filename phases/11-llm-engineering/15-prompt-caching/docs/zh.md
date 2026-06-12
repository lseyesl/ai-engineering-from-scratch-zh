# 提示缓存与上下文缓存

> 你的系统提示是 4,000 个 token。你的 RAG 上下文是 20,000 个 token。你每次请求都发送两者。你每次也都为它们付费。提示缓存让提供商在它们那边保持该前缀温热，并在重用时按正常费率的 10% 收费。正确使用时，它将推理成本降低 50-90%，并将首 token 延迟降低 40-85%。

**类型：** Build
**语言：** Python
**前置要求：** 阶段 11 · 01（提示工程），阶段 11 · 05（上下文工程），阶段 11 · 11（缓存与成本）
**预计时间：** ~60 分钟

## 问题

一个编码代理在对话的每一轮向 Claude 发送相同的 15,000 token 系统提示。二十轮，按 $3/M 输入 token 计算，仅仅是输入成本就是 $0.90——还没算用户的任何实际消息。乘以 10,000 个日常对话，账单就达到 $9,000/天，只为了从不改变的文本。

你不能缩小提示而不损害质量。你不能避免发送它——模型每轮都需要它。唯一的办法是停止为提供商已经看过的前缀支付全价。

这个办法就是提示缓存。Anthropic 在 2024 年 8 月发布了它（2025 年推出了 1 小时扩展 TTL 变体），OpenAI 在同年晚些时候将其自动化，Google 随 Gemini 1.5 推出了显式上下文缓存，现在三者都在其前沿模型上将其作为一级功能提供。

## 概念

![提示缓存：一次写入，廉价读取](../assets/prompt-caching.svg)

**机制。** 当请求的前缀与最近请求的前缀匹配时，提供商从之前运行中提供 KV 缓存，而不是重新编码 token。第一次你支付少量写入溢价，之后每次你获得大量读取折扣。

**2026 年的三种提供商风格。**

| 提供商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 内容块上的显式 `cache_control` 标记 | 输入 90% 折扣 | 25% 附加费 | 5 分钟（可延长至 1 小时） | 1,024 token（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入 50% 折扣 | 无 | 最多 1 小时（尽力而为） | 1,024 token |
| Google (Gemini) | 显式 `CachedContent` API | 按存储计费；读取约按正常费率的 25% | 每 token·小时存储费 | 用户设置（默认 1 小时） | 4,096 token（Flash），32,768（Pro） |

**不变规则。** 三者都只缓存前缀。如果任何 token 在请求之间不同，第一个不同 token 之后的一切都是未命中。将*稳定*部分放在顶部，*可变*部分放在底部。

### 缓存友好的布局

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

违反此顺序——将用户消息放在系统提示之上，在少样本之间穿插动态检索结果——缓存永远不会命中。

### 盈亏平衡计算

Anthropic 的 25% 写入溢价意味着一个缓存块必须至少被读取两次才能净节省成本。1 次写入 + 1 次读取平均每次请求 0.675x 成本（节省 32%）；1 次写入 + 10 次读取平均 0.205x（节省 80%）。经验法则：缓存任何你预期在 TTL 内至少重用 3 次的内容。

## 构建它

### 步骤 1：使用显式标记的 Anthropic 提示缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```


`cache_control` 标记告诉 Anthropic 将块存储 5 分钟。在该窗口内重用命中；过期后重写。

**响应使用量字段：**
```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```


在 CI 中检查这两个字段——如果 `cache_read_input_tokens` 在请求之间保持为零，你的缓存键在漂移。

### 步骤 2：一小时扩展 TTL

对于长时间运行的批处理作业，5 分钟默认值在作业之间过期。设置 `ttl`：
```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 花费 2 倍写入溢价（比基线多 50% 而不是 25%），但在任何重用前缀超过 5 次的批处理中迅速回本。

### 步骤 3：OpenAI 自动缓存

OpenAI 不给你任何可配置的东西。任何超过 1,024 token 且与最近请求匹配的前缀会自动获得 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```


同样的缓存友好布局规则适用。两件事会破坏 OpenAI 的缓存但不会破坏 Anthropic 的：更改 `user` 字段（用作缓存键组成部分）和重新排序工具。

### 步骤 4：Gemini 显式上下文缓存

Gemini 将缓存视为你创建和命名的一级对象：
```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```


Gemini 按每 token·小时收取存储费，只要缓存存在，读取按约正常输入费率的 25%。当你跨多天跨会话重用同一个巨大提示时，这是正确的形态。

### 步骤 5：测量生产中的命中率

见 `code/main.py` 获取一个模拟的三提供商会计器，跟踪写入/读取/未命中计数并计算每 1K 请求的混合成本。在目标命中率上门控部署——大多数生产 Anthropic 设置在预热后应看到 >80% 的读取比例。

## 2026 年仍然存在的陷阱

- **顶部的动态时间戳。** 系统提示顶部的 `"Current time: 2026-04-22 15:30:02"`。每个请求都未命中。将时间戳移到缓存断点下方。
- **工具重新排序。** 以稳定顺序序列化工具——部署之间的字典重排会破坏每次命中。
- **自由文本近重复。** "You are helpful." vs "You are a helpful assistant."——一个字节的差异 = 完全未命中。
- **块太小。** Anthropic 强制执行 1,024 token 下限（Haiku 为 2,048）。更小的块静默地不被缓存。
- **盲目的成本仪表板。** 将"输入 token"拆分为已缓存和未缓存。否则流量下降看起来像缓存胜利。

## 使用它

2026 年缓存栈：

| 场景 | 选择 |
|-----------|------|
| 具有稳定 10k+ 系统提示的代理，多轮 | Anthropic `cache_control` 带 5 分钟 TTL |
| 重用前缀超过 30 分钟的批处理作业 | Anthropic 带 `ttl: "1h"` |
| GPT-5 上的无服务器端点，无自定义基础设施 | OpenAI 自动（只需使你的前缀稳定且长） |
| 多天重用巨大代码/文档语料库 | Gemini 显式 `CachedContent` |
| 跨提供商回退 | 保持可缓存前缀布局在提供商之间相同，以便任何命中都能工作 |

与语义缓存（阶段 11 · 11）结合用于用户消息层：提示缓存处理*token 相同*的重用，语义缓存处理*含义相同*的重用。

## 交付物

保存 `outputs/skill-prompt-caching-planner.md`：
```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 对一个针对 Claude 的 10 轮对话，系统提示为 5,000 token。先不带 `cache_control` 运行一次，然后带。报告每次的输入 token 账单。
2. **中等。** 编写一个测试工具，给定提示模板和请求日志，计算每个提供商的预期命中率和美元节省（Anthropic 5m, Anthropic 1h, OpenAI 自动, Gemini 显式）。
3. **困难。** 构建一个布局优化器：给定提示和标记为 `stable=True/False` 的字段列表，重写提示以在最大缓存友好位置放置单个缓存断点，而不丢失信息。在真实的 Anthropic 端点上验证。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 提示缓存 | "使长提示变便宜" | 重用提供商端 KV 缓存以匹配前缀；重复输入 token 享受 50-90% 折扣。 |
| `cache_control` | "Anthropic 标记" | 内容块属性，声明"到此为止的所有内容都是可缓存的"；`{"type": "ephemeral"}`。 |
| 缓存写入 | "支付溢价" | 填充缓存的第一个请求；在 Anthropic 上按 ~1.25x 输入费率收费，在 OpenAI 上免费。 |
| 缓存读取 | "折扣" | 匹配前缀的后续请求；按 10%（Anthropic）、50%（OpenAI）、~25%（Gemini）收费。 |
| TTL | "存活时间" | 缓存保持温热的时间；Anthropic 5m 默认（可延长至 1h），OpenAI 尽力而为至 1h，Gemini 用户设置。 |
| 扩展 TTL | "1 小时 Anthropic 缓存" | `{"type": "ephemeral", "ttl": "1h"}`；2 倍写入溢价，但对批处理重用来说值得。 |
| 前缀匹配 | "为什么我的缓存未命中" | 缓存仅当从开始到断点的每个 token 都逐字节相同时才命中。 |
| 上下文缓存 (Gemini) | "显式那个" | Google 的命名、按存储计费的缓存对象；最适合多天重用大型语料库。 |

## 延伸阅读

- [Anthropic——提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)——`cache_control`、1h TTL、盈亏平衡表。
- [OpenAI——提示缓存](https://platform.openai.com/docs/guides/prompt-caching)——自动前缀匹配。
- [Google——上下文缓存](https://ai.google.dev/gemini-api/docs/caching)——`CachedContent` API 和存储定价。
- [Anthropic 工程——用于长上下文工作负载的提示缓存](https://www.anthropic.com/news/prompt-caching)——原始发布文章，带有延迟数据。
- 阶段 11 · 05（上下文工程）——如何分割提示以便缓存生效。
- 阶段 11 · 11（缓存与成本）——将提示缓存与用户消息的语义缓存配对。
- [Pope 等人, "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102)——提示缓存暴露给用户的 KV 缓存内存模型；解释了为什么缓存前缀比重新计算便宜约 10 倍。
- [Agrawal 等人, "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369)——预填充是提示缓存加速的阶段；本文解释了为什么缓存命中时 TTFT 急剧下降而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — prompt caching sits alongside speculative decoding, Flash Attention, and MQA/GQA as levers that bend the inference cost curve; read this for the other three.
