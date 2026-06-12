# 顶点项目 11——LLM 可观测性与评估仪表板

> Langfuse 转向了开放核心。Arize Phoenix 发布了 2026 年 GenAI 语义约定映射。Helicone 和 Braintrust 都加倍押注按用户成本归因。Traceloop 的 OpenLLMetry 成为事实上的 SDK 检测标准。生产形态是 ClickHouse 用于跟踪、Postgres 用于元数据、Next.js 用于 UI，以及一小组在采样跟踪上运行的评估任务（DeepEval、RAGAS、LLM-judge）。构建一个自托管的，从至少四个 SDK 家族摄取，并演示在五分钟内捕获注入的回归。

**类型:** Capstone
**语言:** TypeScript (UI)、Python / TypeScript（摄取 + 评估）、SQL (ClickHouse)
**前置要求:** Phase 11（LLM 工程）、Phase 13（工具）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段:** P11 · P13 · P17 · P18
**时间:** 25 小时

## 问题

2026 年每个运行生产流量的 AI 团队都在模型旁边维护一个可观测平面。成本归因。幻觉检测。漂移监控。越狱信号。SLO 仪表板。PII 泄露告警。开源参考——Langfuse、Phoenix、OpenLLMetry——都收敛到 OpenTelemetry GenAI 语义约定作为摄取模式。你现在可以用一个 SDK 检测 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM，并发送兼容的 spans。

你将构建一个自托管的仪表板，从至少四个 SDK 家族摄取，在采样跟踪上运行一小部分评估任务，检测漂移并发出告警。测量标准：给定一个故意注入的回归（开始产生 PII 的提示），仪表板在五分钟内捕获它并发出告警。

## 概念

摄取是 OTLP HTTP。SDK 产生 GenAI-semconv spans：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Spans 落在 ClickHouse 中用于列分析；元数据（用户、会话、应用）落在 Postgres 中。

评估作为批处理作业在采样跟踪上运行。DeepEval 评分忠实度、毒性和答案相关性。RAGAS 在跟踪携带检索上下文时评分检索指标。自定义 LLM-judges 运行领域特定检查（PII 泄露、违反策略的响应）。评估运行将评估 spans 写回相同的 ClickHouse，链接到父跟踪。

漂移检测随时间观察嵌入空间分布（提示嵌入上的 PSI 或 KL 散度）加上评估分数趋势。告警通过 Prometheus Alertmanager 然后 Slack / PagerDuty。UI 是带 Recharts 的 Next.js 15。

## 架构

```
生产应用:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (原始事件)
       |
       +---> 评估任务 (DeepEval, RAGAS, LLM-judge)
       |     采样或全跟踪
       |     写回评估 spans
       |
       +---> 漂移检测器 (提示嵌入上的 PSI / KL)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 仪表板 (Recharts)
```

## 技术栈

- 摄取：OpenTelemetry SDK + GenAI 语义约定；OTLP HTTP 传输
- 收集器：带尾采样处理器的 OpenTelemetry 收集器（用于成本控制）
- 存储：ClickHouse 用于 spans，Postgres 用于元数据，S3 用于原始事件归档
- 评估：DeepEval、RAGAS 0.2、Arize Phoenix 评估器包、自定义 LLM-judge
- 漂移：每周池化提示嵌入上的 PSI / KL（sentence-transformers）
- 告警：Prometheus Alertmanager -> Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + server actions
- 开箱即用支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 构建它

1. **收集器配置。** OpenTelemetry 收集器，带 OTLP HTTP 接收器、尾采样器（保留 100% 错误跟踪和 10% 成功跟踪）、以及导出到 ClickHouse 和 S3 的导出器。

2. **ClickHouse 模式。** 表 `spans`，列镜像 GenAI semconv：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，以及用于长负载的 JSON bag。按 user_id 和 app_id 添加二级索引。

3. **SDK 覆盖测试。** 使用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）和 OpenLLMetry 自动检测编写一个小型客户端应用。验证每个产生规范 GenAI spans 并落入 ClickHouse。

4. **评估任务。** 一个定时作业读取过去 15 分钟的采样跟踪并运行 DeepEval 忠实度、毒性和答案相关性。输出是链接到父跟踪的评估 spans。

5. **自定义 LLM-judge。** 一个 PII 泄露评判者：给定一个响应，调用守卫 LLM 评分 PII 泄露的可能性。高分数响应进入分类队列。

6. **漂移检测。** 每周定时任务计算本周池化提示嵌入与过去 4 周基线的 PSI。如果 PSI 高于阈值，发出告警。

7. **仪表板。** Next.js 15，页面：概览（spans/sec、cost/user、p95 延迟）、跟踪（搜索 + 瀑布）、评估（忠实度趋势、毒性）、漂移（PSI 随时间变化）、告警。

8. **告警链。** Prometheus 导出器读取评估分数聚合和延迟百分位数；Alertmanager 路由到 Slack（警告）和 PagerDuty（严重违规）。

9. **回归探测。** 注入一个 bug：被评估的聊天机器人开始 1% 的时间泄露虚假的 SSN。测量 MTTR：从 bug 部署到 Slack 告警。

## 使用它

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  接受 1 个跟踪, 3 个 spans
[clickhouse] 插入 3 个 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      每周 PSI 0.08 (低于 0.2 阈值)
[ui]         服务于 https://obs.example.com
```

## 交付物

`outputs/skill-llm-observability.md` 是交付物。给定一个 LLM 应用，仪表板摄取其跟踪，运行评估，在漂移上发出告警，并在 Next.js 中显示按用户成本细分。

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | 跟踪模式覆盖 | 产生规范 GenAI spans 的 SDK 家族数量（目标：6+）|
| 20 | 评估正确性 | DeepEval / RAGAS 分数与手工标记集的对比 |
| 20 | 仪表板 UX | 注入回归的 MTTR（目标在 5 分钟以下）|
| 20 | 成本/规模 | 在 1k spans/sec 下稳定摄取无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链端到端执行 |

## 练习

1. 为 Haystack 框架添加自定义检测。验证规范 spans 以忠实的 `gen_ai.*` 属性落入 ClickHouse。

2. 在同一跟踪上将 DeepEval 替换为 Phoenix 评估器。测量两个评估引擎之间的分数漂移。

3. 优化漂移检测器：按 app-id 而非全局计算 PSI。显示按应用的漂移轨迹。

4. 添加"用户影响"页面：每用户成本和每用户失败率，带迷你图。

5. 构建一个保留 100% 毒性 > 0.5 的跟踪加上其余 10% 分层样本的尾采样策略。测量引入的采样偏差。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| GenAI semconv | "OTel LLM 属性" | 2025 年用于 LLM span 属性的 OpenTelemetry 规范（system、model、tokens）|
| Tail sampling | "跟踪后采样" | 收集器在跟踪完成后决定保留或丢弃（可以偷看错误）|
| PSI | "总体稳定性指数" | 比较两个分布的漂移指标；> 0.2 通常表示有意义的漂移 |
| LLM-judge | "评估即模型" | 一个 LLM 根据评分规则对另一个 LLM 输出进行评分（忠实度、毒性、PII）|
| Tail-sampling policy | "保留规则" | 决定哪些跟踪被持久化与丢弃的规则；错误 + 采样率 |
| Eval span | "链接的评估跟踪" | 携带链接到原始 LLM 调用 span 的评估分数的子 span |
| Cost per user | "单位经济学" | 在一个窗口内归属于一个用户 ID 的美元成本；关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse)——参考开放核心可观测性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)——备选参考，强漂移支持
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry)——自动检测 SDK 家族
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——摄取模式
- [Helicone](https://www.helicone.ai)——备选托管可观测性
- [Braintrust](https://www.braintrust.dev)——备选评估优先平台
- [ClickHouse 文档](https://clickhouse.com/docs)——列式 span 存储
- [DeepEval](https://github.com/confident-ai/deepeval)——评估器库
