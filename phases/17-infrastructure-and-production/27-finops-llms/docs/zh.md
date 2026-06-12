# LLM 的 FinOps——单位经济学与多租户归因

> 传统的 FinOps 在 LLM 支出上失效。成本是 token 交易，而非资源运行时间。标签无法映射——API 调用是一笔交易，而非一项资产。工程决策（提示设计、上下文窗口、输出长度）就是财务决策。2026 年手册有三个归因维度需要从第一天开始仪表化：按用户（`user_id`）用于席位定价和扩展、按任务（`task_id` + `route`）用于产品面成本和优先级排序、按租户（`tenant_id`）用于单位经济学和续约。四个 token 层——提示、工具、记忆、响应——单一桶隐藏了真实支出。多租户产品的执行阶梯：每租户速率限制（预期峰值的 2-3 倍，清晰的 429 + retry-after）；每日支出上限（合同上限的 1.5-3 倍；触发速率收紧 + 告警）；支出 z-score > 4 时启用终止开关（自动暂停 + 呼叫值班人员）。归因模式：标记加聚合、遥测连接器（trace-ID → 计费；精度最高）、采样加外推、基于模型的分配、事件溯源、实时流式。单位指标：每次解决查询的成本、每个生成产物的成本——而非 $/M token。追溯标记总是会遗漏；在请求创建时进行仪表化。

**Type:** Learn
**Languages:** Python（stdlib，带终止开关的简易成本归因模拟器）
**Prerequisites:** Phase 17 · 13 (Observability)、Phase 17 · 14 (Caching)
**Time:** ~60 分钟

## 学习目标

- 解释为什么传统 FinOps（标签 + 层级）在 LLM 支出上失效，并说出三个新的归因维度。
- 列举四个 token 层（提示、工具、记忆、响应）以及为什么单一桶计费会隐藏成本。
- 为多租户产品设计执行阶梯（速率限制 → 支出上限 → 终止开关）。
- 选择单位指标（每次解决查询/每个产物的成本）而非 $/M token。

## 问题

你的账单显示 $40,000。你不知道：
- 哪个租户花的。
- 哪个产品功能驱动的。
- 是否有任何单个用户滥用。
- 是提示膨胀、工具调用还是记忆放大造成的。

提供商侧的标记加聚合适用于云资源（EC2、S3），其中标签传播到账单行项目。LLM API 调用不会自动标记——你必须在调用点打上用户/任务/租户标记并传递下去。追溯归因总是会遗漏边缘情况。

## 概念

### 三个归因维度

**按用户**（`user_id`）：谁花费了什么。驱动席位定价、扩展对话、识别高价值用户。

**按任务**（`task_id` + `route`）：哪个产品面花费了什么。驱动功能优先级排序、终止昂贵功能的决策。

**按租户**（`tenant_id`）：哪个客户是盈利的。驱动单位经济学、续约定价、层级阈值。

从第一天起在调用点对三者进行仪表化。追溯总是更差。

### 四个 token 层

| 层 | 示例 | 占总量的典型比例 |
|-------|---------|---------------------|
| 提示 | 系统 + 用户输入 | 40-60% |
| 工具 | 工具调用结果反馈 | 20-40%（代理工作负载）|
| 记忆 | 之前的对话/检索到的文档 | 10-30% |
| 响应 | 模型输出 | 10-30% |

将四者放在同一个桶中会使优化变得盲目。在你的归因 schema 中分开它们。

### 执行阶梯

1. **每租户速率限制**。预期峰值的 2-3 倍。返回 429 并带 `Retry-After`。租户感到摩擦；无意外账单。

2. **每租户每日支出上限**。合同上限的 1.5-3 倍。触发：收紧速率限制 + 通知客户成功团队。

3. **终止开关**，当支出 z-score > 4（相对于租户基线）。自动暂停租户；呼叫值班人员；升级到运维 + 客户成功。

### 归因模式

- **标记加聚合**：打上元数据头部；稍后聚合。简单；粗略。
- **遥测连接器**：通过 trace ID 将追踪连接到计费。精度最高。成熟团队的做法。
- **采样加外推**：采样 5-10%，乘以。对于粗略成本有效；遗漏尾部。
- **基于模型的分配**：回归推断成本驱动因素。用于没有标签的遗留数据。
- **事件溯源**：成本作为流中的事件（Kafka / Kinesis）。实时。
- **实时流式**：仪表板亚秒级更新。

### 每 X 成本是单位指标

$/M token 是供应商用语。产品指标：

- 每次解决支持工单的成本。
- 每篇生成文章的成本。
- 每次成功代理任务的成本。
- 每用户会话分钟的成本。

将成本与产品成果挂钩。否则优化就没有锚点。

### 成本归因追踪形状

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每次调用都发出。存储在数据湖中。按维度聚合。Phase 17 · 13 的可观测性栈就是存放此数据的地方。

### 复合节省栈

栈：缓存 + 批处理 + 路由 + 网关。使用全部四个：
- L2 缓存（Phase 17 · 14）：输入约便宜 10 倍。
- 批处理（Phase 17 · 15）：5 折。
- 路由到廉价模型（Phase 17 · 16）：成本降低 60%。
- 网关效率（Phase 17 · 19）：冗余 + 重试。

最佳情况堆叠：约基线的 5-10%。大多数团队使用 2-3 个杠杆；很少堆叠全部四个。

### 你应该记住的数字

- 归因维度：按用户、按任务、按租户。
- 四个 token 层：提示、工具、记忆、响应。
- 终止开关：支出 z-score > 4。
- 单位指标：每次解决查询的成本，而非 $/M token。
- 堆叠优化：可能达到基线的约 5-10%。

## 使用它

`code/main.py` 模拟一个多租户 LLM 服务，具有三级执行阶梯。注入一个滥用租户并演示终止开关触发。

## 交付物

本课程产出 `outputs/skill-finops-plan.md`。根据产品和规模，设计归因 schema 和执行阶梯。

## 练习

1. 运行 `code/main.py`。终止开关在什么 z-score 时触发？你如何选择阈值？
2. 设计一个每租户、每任务成本仪表板。你首先构建哪 5 个视图？
3. 你最大的租户单位经济学为负。提出三种按客户影响排序的干预措施。
4. 计算支持产品的每次解决工单成本：每工单 300 万 token，约 800 工单/天，GPT-5 缓存费率。
5. 论证追溯标记是否永远有效。在什么时候可以接受？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Per-user attribution | "用户级成本" | 每次调用都打上 `user_id` |
| Per-task attribution | "功能成本" | `task_id` + `route` 标识产品面 |
| Per-tenant attribution | "客户成本" | `tenant_id`；驱动单位经济学 |
| Four token layers | "成本层" | 提示 + 工具 + 记忆 + 响应 |
| Rate limit | "429 防护" | 在网关执行的每租户上限 |
| Daily spend cap | "每日上限" | 租户范围的预算，带告警 |
| Kill switch | "自动暂停" | 支出 z-score > 4 触发自动暂停 |
| Cost per resolved | "产品单位指标" | 成本与产品成果挂钩，而非 token |
| Telemetry joiner | "追踪到计费" | 精度最高的归因模式 |
| Stacked optimization | "缓存+批处理+路由+网关" | 复合节省至基线的约 5-10% |

## 延伸阅读

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
