# LLM API 负载测试 — 为什么 k6 和 Locust 会撒谎

> 传统的负载测试工具并非为流式响应、可变输出长度、Token 级指标或 GPU 饱和而设计。两个陷阱会坑害大多数团队。GIL 陷阱：Locust 的 Token 级测量在 Python GIL 下运行分词，在高并发下与请求生成竞争；分词积压会夸大报告的 Token 间延迟 — 瓶颈在客户端，而非服务器。提示均匀性陷阱：循环中使用相同提示只测试了 Token 分布上的一个点；真实流量具有可变长度和多样的前缀匹配。LLMPerf 通过 `--mean-input-tokens` + `--stddev-input-tokens` 解决了这个问题。2026 年工具映射：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 Token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）** — 支持流式、Kubernetes 原生分布式（通过 TestRun/PrivateLoadZone CRD），最适合 CI/CD 门控；Vegeta 用于 Go 恒定速率饱和；Locust 2.43.3 仅配合 LLM-Locust 扩展用于流式。负载模式：稳态、斜坡、突发（自动扩缩测试）、浸泡（内存泄漏）。

**类型：** Build
**语言：** Python（stdlib，玩具级真实提示生成器 + 延迟收集器）
**前置要求：** Phase 17 · 08（推理指标），Phase 17 · 03（GPU 自动扩缩）
**时间：** ~75 分钟

## 学习目标

- 解释使通用负载测试工具对 LLM API 撒谎的两个反模式（GIL 陷阱、提示均匀性陷阱）。
- 根据目的选择工具：LLMPerf（基准测试）、k6 + 流式扩展（CI 门控）、guidellm（大规模合成）、GenAI-Perf（NVIDIA 参考）。
- 设计四种负载模式（稳态、斜坡、突发、浸泡）并指出每种模式能捕获的故障模式。
- 使用输入 Token 的均值 + 标准差而非固定长度构建真实的提示分布。

## 问题

你用 k6 测试了 LLM 端点，500 个并发用户。它撑住了。你上线了。在生产环境中，实际只有 200 个用户，服务却崩溃了 — P99 TTFT 爆炸，GPU 打满。

发生了两件事。首先，k6 发送了 500 个相同的提示 — 你的请求合并和前缀缓存让你看起来在处理 500 个并发解码，实际上只处理了一个。其次，k6 不会像人眼体验那样追踪流式响应上的 Token 间延迟；它看到的是一个 HTTP 连接，而不是 500 个以不同间隔到达的 Token。

LLM 的负载测试是一门独立的学科。

## 概念

### GIL 陷阱（Locust）

Locust 使用 Python，在客户端 GIL 下运行分词。在高并发下，分词器在请求生成后面排队。报告的 Token 间延迟包含了客户端分词积压。你以为服务器慢，其实是测试工具的问题。

修复：LLM-Locust 扩展将分词移到独立进程中，或使用编译语言工具（k6、使用 tokenizers.rs 的 LLMPerf）。

### 提示均匀性陷阱

所有已知的负载测试工具都允许你配置一个提示。在 10,000 次迭代的循环测试中，每次都发送完全相同的提示。服务器每次都看到相同的前缀 — 前缀缓存命中率接近 100%，吞吐量看起来很好。

修复：从提示分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150` — 不同的长度，不同的内容。

### 四种负载模式

1. **稳态（Steady-state）** — 恒定 RPS 持续 30-60 分钟。捕获：基线性能回归。
2. **斜坡（Ramp）** — 在 15 分钟内将 RPS 从 0 线性增加到目标值。捕获：容量拐点、预热异常。
3. **突发（Spike）** — 突然 3-10 倍 RPS 持续 2 分钟然后恢复。捕获：自动扩缩延迟、队列饱和、冷启动影响。
4. **浸泡（Soak）** — 稳态持续 4-8 小时。捕获：内存泄漏、连接池漂移、可观测性溢出。

### 2026 年工具映射

**LLMPerf**（Anyscale）— Python 但使用 Rust 后端分词。均值/标准差提示。支持流式。性能运行的最佳默认选择。

**NVIDIA GenAI-Perf** — NVIDIA 的参考工具。使用 Triton 客户端；全面的指标覆盖。注意其 ITL 不包括 TTFT；LLMPerf 包括。两个工具对同一服务器会产生不同的 TPOT。

**LLM-Locust**（TrueFoundry）— 修复 GIL 陷阱的 Locust 扩展。熟悉的 Locust DSL + 流式指标。

**guidellm** — 大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go，编译语言，无 GIL）增加了流式感知指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 实现 Kubernetes 原生分布式测试。
- 最适合 CI/CD 门控和 SLA 测试。

**Vegeta** — Go 语言，比 k6 更简单。恒定速率 HTTP 饱和。不支持 LLM，但适合网关/速率限制测试。

**Locust 2.43.3 原生** — 对 LLM 存在 GIL 陷阱。仅配合 LLM-Locust 扩展使用。

### CI 中的 SLA 门控

在 PR 上运行 k6：

- 在基线 RPS 下进行 30-50 次迭代。
- 门控：P50/P95 TTFT，5xx < 5%，TPOT 低于阈值。
- 超出则中断构建。

### 真实的提示分布

从真实流量样本（如果有）或已发布的分布（例如，ShareGPT 提示用于聊天，HumanEval 用于代码）构建。将均值 + 标准差输入 LLMPerf。无论如何都要避免单提示循环。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：流式感知指标。
- 典型 LLMPerf 运行：在并发 X 下 100-1000 个请求。
- 典型 CI 门控：每个 PR 30-50 次迭代。
- 四种模式：稳态、斜坡、突发、浸泡。

## 使用它

`code/main.py` 模拟了一个具有真实提示分布的负载测试，测量有效 TPOT，并演示了均匀提示陷阱。

## 交付物

本课程产出 `outputs/skill-load-test-plan.md`。根据工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。比较均匀分布与真实分布 — 差距在哪里？
2. 编写 CI 门控的 k6 脚本：TTFT P95 < 800 ms，100 并发，运行时间 5 分钟。
3. 你的浸泡测试显示内存以 50 MB/小时增长。列出三个原因以及区分它们的检测手段。
4. 突发测试从 10 RPS 到 100 RPS。如果 Karpenter + vLLM 生产栈已就位（Phase 17 · 03 + 18），预期的恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 在同一服务器上报告 TPOT=11ms。解释原因。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| LLMPerf | "LLM 测试工具" | Anyscale 基准测试工具，支持流式 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 参考测试工具 |
| LLM-Locust | "LLM 版 Locust" | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | "合成基准" | 大规模合成测试工具 |
| k6 Operator | "K8s k6" | 基于 CRD 的分布式 k6 |
| GIL 陷阱 | "Python 客户端开销" | 分词积压夸大了报告的延迟 |
| 提示均匀性陷阱 | "单提示谎言" | 循环中使用相同提示命中缓存，夸大吞吐量 |
| 稳态 | "恒定负载" | 持续 N 分钟的平坦 RPS |
| 斜坡 | "线性上升" | 在持续时间内从 0 到目标值 |
| 突发 | "爆发测试" | 突然倍增然后恢复 |
| 浸泡 | "长时间测试" | 持续数小时以检测泄漏 |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)