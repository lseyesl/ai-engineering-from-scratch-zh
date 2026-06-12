# 行动预算、迭代上限与成本调控器 (Action Budgets, Iteration Caps, and Cost Governors)

> 一个中型电商智能体的月度 LLM 成本在团队启用"订单追踪"技能后从 1,200 美元跃升至 4,800 美元。这不是定价 bug。这是一个智能体找到了新循环并在其中持续消费。微软的 Agent Governance Toolkit（2026 年 4 月 2 日）将针对此类问题的防御措施系统化：每次请求的 `max_tokens`、每个任务的 token 和美元预算、每天/每月的上限、迭代上限、分层模型路由、提示缓存、上下文窗口化、对昂贵操作的 HITL 检查点、预算超限时的终止开关。Anthropic 的 Claude Code Agent SDK 以不同名称提供了相同的原语。财务速度限制（例如，10 分钟内消费超过 50 美元则切断访问）比月度上限能更快捕获循环。

**Type:** 学习 (Learn)
**Languages:** Python（stdlib，分层成本调控器模拟器）
**Prerequisites:** Phase 15 · 10（权限模式），Phase 15 · 12（持久化执行）
**Time:** ~60 分钟

## 问题 (The Problem)

自主智能体 (Autonomous agents) 每次交互都在花真金白银。聊天机器人的糟糕输出是一次糟糕回复；智能体的糟糕循环是一张账单。业界对此类故障模式的术语是"拒绝钱包" (Denial of Wallet) —— 智能体持续推理、持续调用工具、持续计费，而没有任何机制阻止它，因为根本没有设计这样的机制。

解决方案不是一个单一数字。它是一个在不同时间尺度和粒度上的限制堆栈：每次请求、每个任务、每小时、每天、每月。一个设计良好的堆栈能在几分钟内捕获失控循环，几小时内捕获缓慢泄漏，一天内捕获糟糕发布。当智能体是长周期且自主运行时，同一个堆栈始终维持预算。

这是一堂工程课：数学很简单，但团队失败的地方在于纪律。下面列出的所有限制均来自 Microsoft Agent Governance Toolkit 或 Anthropic Claude Code Agent SDK 文档。

## 概念 (The Concept)

### 成本调控器堆栈 (The cost-governor stack)

1. **每次请求的 `max_tokens`。** 简单。防止任何单次调用产生无界补全。
2. **每个任务的 token 预算。** 在整个运行过程中，不超过 N 个 token。达到上限时硬停止。
3. **每个任务的美元预算。** 与 token 相同，但以货币计。Claude Code 中的 `max_budget_usd`。
4. **每个工具调用上限。** 不超过 N 次 `WebFetch` 调用、N 次 `shell_exec` 调用等。
5. **迭代上限 (`max_turns`)。** 智能体循环的总迭代次数；防止无限推理循环。
6. **每分钟/每小时/每天/每月上限。** 滚动窗口。在不同时间尺度上捕获泄漏。
7. **财务速度限制 (Financial velocity limit)。** 例如，"如果 10 分钟内消费超过 50 美元，切断访问。" 在月度上限触发之前捕获基于循环的燃烧。
8. **分层模型路由 (Tiered model routing)。** 默认使用较小模型；仅在分类器判断任务需要时才升级到较大模型。
9. **提示缓存 (Prompt caching)。** 系统提示和稳定上下文存储在提供者缓存中；重新发送的 token 成本接近于零。
10. **上下文窗口化 (Context windowing)。** 压缩/总结以将活动上下文保持在阈值以下；直接降低 token 成本。
11. **对昂贵操作的 HITL 检查点。** 在执行已知昂贵的操作（长时间工具调用、大文件下载、昂贵的模型升级）之前，需要人工确认。
12. **预算超限时的终止开关 (Kill switch on budget breach)。** 任何上限触发时会话中止。上限被记录；需要单独的重新启用路径。

### 为什么是堆栈，而不是单一上限 (Why the stack, not one cap)

单一的月度上限只有在钱包空了之后才能捕获失控智能体。单一的每次请求上限在会话级别什么也捕获不到。不同的故障模式需要不同的时间尺度：

- **失控循环 (Runaway loop)**（智能体陷入 5 秒重试循环）：由速度限制捕获。
- **缓慢泄漏 (Slow leak)**（智能体每个任务执行约 2 倍预期工作）：由每日上限捕获。
- **糟糕发布 (Bad release)**（新版本使用 5 倍 token）：由每周/每月上限捕获。
- **合法激增 (Legitimate surge)**（真实需求，不是 bug）：由小时/日上限捕获，并有清晰日志。

### Claude Code 的预算表面 (Claude Code's budget surface)

Claude Code Agent SDK 暴露了（公开文档）：

- `max_turns` —— 迭代上限。
- `max_budget_usd` —— 美元上限；超限时会话中止。
- `allowed_tools` / `disallowed_tools` —— 工具允许列表和拒绝列表。
- 工具使用前的钩子点，用于自定义成本核算。

结合权限模式阶梯（第 10 课）。没有 `max_budget_usd` 的 `autoMode` 会话是无治理的自主权。Anthropic 明确将 Auto Mode 定位为需要预算控制；分类器与成本是正交的。

### EU AI 法案、OWASP Agentic Top 10

Microsoft 的 Agent Governance Toolkit 涵盖了 OWASP Agentic Top 10 和 EU AI 法案第 14 条（人类监督）的要求。在欧盟的生产环境中，日志记录和上限执行不是可选项。

### 观察到的 $1,200 → $4,800 案例 (The observed $1,200 → $4,800 case)

Microsoft 文档中的真实案例：一个电商智能体在添加新工具后月度成本翻了三倍。该工具允许智能体在每次会话中轮询订单状态。没有循环检测。没有每个工具的上限。没有周环比增长的告警。解决方案是每个工具的上限加上每日增长告警。这是一个模板：每个新的工具表面都是一个潜在的新循环；每个新工具都需要自己的上限和自己的告警。

## 使用它 (Use It)

`code/main.py` 模拟了有无分层成本调控器堆栈的智能体运行。模拟的智能体在若干轮后漂移进入轮询循环；分层堆栈在速度窗口内捕获它，而单一的月度上限要等到数天后才会触发。

## 交付物 (Ship It)

`outputs/skill-agent-budget-audit.md` 审计一个拟议的智能体部署的成本调控器堆栈，并标记缺失的层级。

## 练习 (Exercises)

1. 运行 `code/main.py`。确认在轮询循环轨迹上速度限制先于迭代上限触发。现在禁用速度限制，测量智能体在迭代上限捕获它之前"花费"了多少。

2. 为浏览器智能体（第 11 课）设计一套每个工具的上限。哪个工具需要最严格的上限？哪个工具可以无限制运行而没有风险？

3. 阅读 Microsoft Agent Governance Toolkit 文档。列出该工具包命名的每种上限类型。将每种映射到一种故障模式（失控循环、缓慢泄漏、糟糕发布、激增）。

4. 为一个实际任务（例如，"在仓库中分类 50 个 issue"）定价一次无人值守的夜间运行。将 `max_budget_usd` 设为你点估计的 2 倍。证明 2 倍的合理性。

5. Claude Code 的 `max_budget_usd` 在会话聚合成本上触发。设计一个你会在外部强制执行的补充速度限制。什么触发切断，重新启用看起来什么样？

## 关键术语 (Key Terms)

| Term | What people say | What it actually means |
|---|---|---|
| Denial of Wallet | "失控账单" | 智能体循环产生消费，没有上限阻止它 |
| max_tokens | "每次请求上限" | 单次补全大小的上限 |
| max_turns | "迭代上限" | 会话中智能体循环迭代次数的上限 |
| max_budget_usd | "美元终止开关" | 会话成本上限；超限时中止 |
| Velocity limit | "速率上限" | 短时间窗口内的消费限制（例如，50 美元/10 分钟） |
| Tiered routing | "小模型优先" | 默认使用廉价模型；仅在分类器认为必要时升级 |
| Prompt caching | "缓存的系统提示" | 提供者端缓存将重新发送的 token 成本降至接近零 |
| HITL checkpoint | "人工审批关卡" | 在昂贵操作前需要人工确认 |

## 延伸阅读 (Further Reading)

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) —— `max_turns`、`max_budget_usd`、工具允许列表。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) —— 成本调控器检查点。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) —— 提供者端成本控制。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) —— 缓存机制。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) —— 长周期智能体的成本概况。