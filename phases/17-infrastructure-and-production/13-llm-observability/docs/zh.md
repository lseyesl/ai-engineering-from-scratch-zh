# LLM 可观测性栈选型

> 2026 年的可观测性市场分为两类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估（evals）、提示管理、会话回放捆绑在一起。网关/仪表化工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 核心基于 MIT 许可，具有强大的开源优势（免费云服务 50K 事件/月）。Phoenix 是 OpenTelemetry 原生工具，采用 Elastic License 2.0——擅长漂移/RAG 可视化，但非持久化生产后端。Arize AX 使用零拷贝 Iceberg/Parquet 集成，声称比单体可观测性便宜 100 倍。LangSmith 在 LangChain/LangGraph 领域领先，$39/用户/月，仅企业版支持自托管。Helicone 基于代理，15-30 分钟即可完成设置，免费 100K 请求/月，但代理追踪深度不足。常见生产模式：Gateway（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），通过 OpenTelemetry 粘合。

**Type:** Learn
**Languages:** Python（stdlib，简易追踪采样模拟器）
**Prerequisites:** Phase 17 · 08 (Inference Metrics)、Phase 14 (Agent Engineering)
**Time:** ~60 分钟

## 学习目标

- 区分开发平台（捆绑：评估 + 提示 + 会话）与网关/遥测工具（仅追踪 + 指标）。
- 映射六大工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）的许可、定价和最佳用例。
- 解释 OpenTelemetry 粘合模式——如何将网关工具与独立的评估平台结合使用。
- 说出 2026 年的成本差异化因素（Arize AX 的零拷贝方法与单体数据摄入）及其约 100 倍的乘数。

## 问题

你发布了一个 LLM 功能。它能工作。但你对提示失败、工具循环、延迟回归、成本飙升或提示缓存命中率毫无可见性。你在谷歌上搜索"LLM 可观测性"，得到八个工具，它们都以三种不同的价格水平声称解决了同样的问题。

它们解决的不是同一个问题。LangSmith 回答"这个 LangGraph 运行为什么失败？"Phoenix 回答"我的 RAG 流水线是否在漂移？"Helicone 回答"哪个应用在烧 token？"Langfuse 回答"我能自托管整个方案吗？"不同的工具，不同的受众。

选型涉及四个轴：技术栈（LangChain？原生 SDK？多供应商？）、许可容忍度（仅 MIT？Elastic 可接受？商业版也行？）、预算（免费层？$100/月？$1000/月？）和自托管（必须？有更好？绝不需要？）。

## 概念

### 两类工具

**开发平台**将可观测性与评估、提示管理、数据集版本控制、会话回放捆绑在一起。你运行实验，查看哪个提示有效，用旧胜出者对新提示进行数据集回归。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具**对推理调用进行仪表化——提示、响应、token、延迟、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。功能极简。可以通过 OpenTelemetry 与独立的评估工具结合使用。

### Langfuse——开源平衡

- 核心代码 Apache / MIT 许可；通过 Docker 自托管。
- 云免费层：50K 事件/月。付费：$29/月（团队版）。
- 评估、提示管理、追踪、数据集。合理覆盖了所有四项开发平台功能。
- 最佳场景：你需要 LangSmith 级功能，但必须自托管或使用开源许可。

### Phoenix（Arize）——遥测优先，OpenTelemetry 原生

- Elastic License 2.0；自托管简便。
- 在 RAG 和漂移可视化方面表现出色。嵌入空间散点图作为一等公民提供。
- 并非设计为持久化生产后端——主要用于开发阶段的可观测性。
- 最佳场景：RAG 流水线开发、漂移调试，与独立的网关配合用于生产。

### Arize AX——规模方案

- 商业产品。通过 Iceberg/Parquet 实现零拷贝数据湖集成。
- 声称在大规模下比单体可观测性（Datadog 类）便宜约 100 倍。原理：你将追踪数据存储在 S3 上自己的 Parquet 中；Arize 直接读取。
- 最佳场景：每天 >1000 万条追踪，已有数据湖，需要 LLM 专属仪表板而无需 Datadog 的定价。

### LangSmith——LangChain/LangGraph 优先

- 商业产品，$39/用户/月。仅企业版支持自托管。
- 在 LangChain 和 LangGraph 栈上最佳。如果你不在这些栈上，吸引力较小。
- 最佳场景：团队已承诺使用 LangChain，愿意付费。

### Helicone——基于代理的最小可行方案

- 15-30 分钟即可完成设置，只需将 `OPENAI_API_BASE` 替换为 Helicone 代理。
- MIT 许可；免费 100K 请求/月，付费 $20/月起。
- 还包括故障转移、缓存、速率限制——同时充当网关。
- 在代理/多步骤追踪方面深度不足。
- 最佳场景：快速上手，单栈应用，需要网关 + 可观测性一体化。

### Opik（Comet）——开源开发平台

- Apache 2.0，完全开源。
- 功能集与 Langfuse 类似，带有 Comet 的血统。
- 最佳场景：ML 团队已在使用 Comet，希望在同一个面板中获得 LLM 可观测性。

### SigNoz——OpenTelemetry 优先的全栈 APM

- Apache 2.0。通过 OpenTelemetry 处理通用 APM 和 LLM。
- 最佳场景：跨服务和 LLM 调用的统一可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年底发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。支持 OTel 的工具可以互操作。新兴的生产模式：

1. 从每个 LLM 调用发出带 GenAI 约定的 OTel。
2. 路由到网关（Helicone / Portkey）用于日常使用。
3. 双发到评估平台（Phoenix / Langfuse）用于回归检测。
4. 归档到数据湖（Iceberg）用于通过 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误的层级进行仪表化

在代理框架内部进行仪表化（例如添加 LangSmith 追踪）会将你与该框架耦合。在 HTTP/OpenAI-SDK 层进行仪表化（通过 OpenLLMetry 或你的网关）是可移植的。

### 采样——你无法保留所有数据

在每天超过 100 万请求时，全量追踪保留的成本可能超过 LLM 调用本身。按规则采样：100% 保留错误，100% 保留高成本，5% 保留成功。始终保留聚合数据；保留原始数据用于长尾分析。

### 你应该记住的数字

- Langfuse 免费云：50K 事件/月。
- LangSmith：$39/用户/月。
- Helicone 免费：100K 请求/月。
- Arize AX 声称：大规模下比单体便宜约 100 倍。
- OpenTelemetry GenAI 约定：2025 年发布，2026 年广泛采用。

## 使用它

`code/main.py` 模拟一天 100 万条追踪数据在不同保留策略（100% 摄入、采样、采样 + 错误）下的情况。报告每种策略下的存储成本和丢失的数据。

## 交付物

本课程产出 `outputs/skill-observability-stack.md`。根据技术栈、规模、预算、许可偏好选择工具。

## 练习

1. 你的团队使用 LangChain，需要开源自托管可观测性。选择 Langfuse 或 Opik 并说明理由。
2. 在每天 500 万条追踪、Datadog 报价 $150K/月的情况下，计算 Arize AX 的盈亏平衡点。
3. 设计一套你的组织指南应强制要求每个 LLM 调用携带的 OpenTelemetry GenAI 属性集。
4. 论证仅使用 Phoenix 是否足够用于生产环境。在什么情况下它不够？
5. Helicone 有 20ms 的代理开销。在 P99 TTFT 为 300 ms 时，这是否可接受？如果 SLA 是 100 ms 呢？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OpenLLMetry | "LLM 的 OTel" | 用于 LLM 的开源 OpenTelemetry 仪表化 |
| GenAI conventions | "OTel 属性" | LLM 调用的标准 OTel 属性名称 |
| LangSmith | "LangChain 可观测性" | 与 LangChain 生态捆绑的商业平台 |
| Langfuse | "开源 LangSmith" | MIT 开源，功能集类似 |
| Phoenix | "Arize 开发工具" | OpenTelemetry 原生开发/评估平台 |
| Arize AX | "规模可观测性" | 商业零拷贝 Iceberg/Parquet 可观测性 |
| Helicone | "代理可观测性" | 收集 LLM 遥测的 HTTP 代理 + 网关功能 |
| Opik | "Comet LLM" | Comet 出品的 Apache 2.0 开源开发平台 |
| Session replay | "追踪回放" | 重放包含工具调用的完整代理会话 |
| Eval | "离线测试" | 在标记数据集上运行候选模型/提示 |

## 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
