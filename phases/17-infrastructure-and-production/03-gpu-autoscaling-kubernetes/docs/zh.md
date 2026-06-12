# Kubernetes 上的 GPU 自动扩缩容 — Karpenter、KAI Scheduler、Gang Scheduling

> 三层，而非一层。Karpenter 动态配置节点（一分钟内，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 gang scheduling、拓扑感知和层级队列 — 它防止了 7-of-8 部分分配陷阱，即七个节点等待并空转消耗资源，只差一个 GPU。应用级自动扩缩器（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于推理特定信号进行扩缩 — 队列深度、KV 缓存利用率 — 而非 CPU/DCGM 占空比。经典的 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是一个占空比度量：100% 可能代表 10 个请求或 100 个。vLLM 预分配 KV 缓存内存，因此内存永远不会触发缩容。本课教你组合这三层，并避免默认的 Karpenter `WhenEmptyOrUnderutilized` 策略在推理中途终止运行中的 GPU 任务。

**Type:** Learn
**Languages:** Python (stdlib, toy queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## Learning Objectives

- 绘制三层自动扩缩容架构图（节点配置、gang scheduling、应用级）并说出每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 是 vLLM 错误的 HPA 信号，并说出两个替代信号（队列深度、KV 缓存利用率）。
- 描述 gang scheduling 以及 KAI Scheduler 防止的部分分配失败模式（7 个 GPU 中有 8 个空闲）。
- 说出会终止运行中 GPU 任务的 Karpenter 整合策略 (`WhenEmptyOrUnderutilized`) 并说明 2026 年的安全替代方案。

## 问题

你的团队在 Kubernetes 上部署了 LLM 服务。你使用 `DCGM_FI_DEV_GPU_UTIL` 作为信号设置了 HPA。服务在工作时间保持在 100% 利用率。HPA 从不扩容 — 它已经认为你满了。你手动添加了一个副本；TTFT 下降。HPA 仍然不扩容。信号在欺骗你。

另外，你使用 Cluster Autoscaler 管理节点。凌晨 2 点来了一个 100 万 token 的提示；集群花了 3 分钟配置节点，请求超时。

再另外，你部署了一个需要跨 2 个节点共 8 个 GPU 的 70B 模型。集群有 7 个 GPU 空闲，1 个分散在 3 个节点上。Cluster Autoscaler 为那 1 个缺失的 GPU 配置了一个节点。七个节点等待了 4 分钟，空转消耗资源，而 Kubernetes 在启动最后一个 GPU。

三层，三种不同的失败模式。2026 年的 GPU 感知自动扩缩容不是"开启 HPA"。而是组合节点配置、gang scheduling 和应用信号自动扩缩容。

## 概念

### 第一层 — 节点配置 (Karpenter)

Karpenter 监视待处理的 Pod，并在约 45-60 秒内配置节点（Cluster Autoscaler 通常需要 90-120 秒来配置 GPU 节点）。它根据 `NodePool` 约束动态选择实例类型 — 如果你的 Pod 需要 8 个 H100 且集群没有匹配的节点，Karpenter 直接配置一个，而不是扩展现有组。

**整合陷阱**：Karpenter 的默认 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池是危险的。它会终止运行中的 GPU 节点，将 Pod 迁移到更便宜的合适实例。对于推理工作负载，这意味着驱逐运行中的请求并在新节点上重新加载 70B 模型。损失是数分钟的容量加上请求失败。

GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

让 Karpenter 在一小时后整合真正空的节点，但从不驱逐运行中的任务。

### 第二层 — Gang Scheduling (KAI Scheduler)

KAI Scheduler（项目名"Karp"后改名）处理默认 kube-scheduler 做不到的事情：

**Gang scheduling** — 全有或全无调度。一个需要 8 个 GPU 的分布式推理 Pod，要么全部 8 个一起启动，要么都不启动。没有这个，就会出现部分分配陷阱：7 个 Pod 中的 8 个启动，无限等待，空转消耗资源。

**拓扑感知** — 知道哪些 GPU 共享 NVLink，哪些在同一机架上，哪些之间有 InfiniBand。相应地放置 Pod。DeepSeek-V3 67B 的张量并行工作负载必须停留在一个 NVLink 域内；KAI Scheduler 尊重这一点。

**层级队列** — 多个团队竞争同一个 GPU 池，带有优先级和配额。团队 A 的生产峰值只有在优先级规则允许时才会被团队 B 的训练任务抢占。

KAI 作为辅助调度器与 kube-scheduler 一起部署；你通过注解让工作负载使用它。Ray 和 vLLM 生产栈都集成了它。

### 第三层 — 应用级信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是一个占空比度量 — 它测量 GPU 在每个采样间隔是否在做工作。100% 利用率可能意味着 10 个并发请求或 100 个；GPU 无论如何都在忙。基于占空比扩缩容是盲目扩缩。

更糟的是，vLLM 和类似引擎预分配 KV 缓存内存（最高 `--gpu-memory-utilization`）。即使只有一个请求，内存使用也保持在 90% 左右。基于内存的 HPA 永远不会缩容。

**2026 年替代信号**：

- 队列深度（等待预填充的请求数）。
- KV 缓存利用率（分配给活动序列的块比例）。
- 每副本 P99 TTFT（你的 SLA 信号）。
- Goodput（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 使用这些信号并扩缩副本。它们完全取代了 LLM 服务的 HPA。

### 何时使用什么

| 扩缩决策 | 工具 |
|----------------|------|
| 添加/移除节点 | Karpenter |
| 调度多 GPU 任务 | KAI Scheduler |
| 添加/移除副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级 | KAI Scheduler 队列 |

### 分离式预填充/解码使一切更复杂

如果你运行分离式预填充/解码（Phase 17 · 17），你有两个 Pod 类别，具有不同的扩缩触发条件：预填充 Pod 基于队列深度扩缩，解码 Pod 基于 KV 缓存压力扩缩。llm-d 将这些暴露为独立的 `Services`，每个角色有自己的 HPA。不要试图在两者前面放一个单一的 HPA。

### 冷启动在这里也很重要

冷启动缓解（Phase 17 · 10）是节点配置时间变得用户可见的地方。Karpenter 的 45-60 秒预热加上 20GB 模型加载加上引擎初始化意味着从零开始的请求需要 2-5 分钟。为 SLO 关键路径保持一个热池 (`min_workers=1`)，或在应用层使用 Modal 风格的检查点。

### 你应该记住的数字

- Karpenter 节点配置：约 45-60 秒 vs Cluster Autoscaler 约 90-120 秒（GPU 节点）。
- KAI Scheduler 防止部分分配浪费 — 7-of-8 陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：有缺陷；使用队列深度或 KV 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：终止运行中的 GPU 任务。推理使用 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## Use It

`code/main.py` 在突发 GPU 工作负载上模拟三层自动扩缩器。比较朴素 HPA（占空比）、队列深度 HPA 和 KAI gang 调度扩缩。报告未满足的请求、空闲 GPU 分钟数和综合评分。

## Ship It

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形态和 SLO，设计一个三层自动扩缩容计划。

## Exercises

1. 运行 `code/main.py`。在突发工作负载下，朴素占空比 HPA 比队列深度 HPA 多丢弃多少请求？差异来自哪里？
2. 为在 H100 SXM5 上服务 Llama 3.3 70B FP8 的集群设计一个 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter` 以及一个将非 GPU 工作负载排除在这些节点之外的污点。
3. 你的团队报告部署卡在 Pending 状态，原因是"GPU 可用但 Pod 无法调度"。诊断 — 这是 Karpenter、kube-scheduler 还是 KAI Scheduler 的问题？哪些指标可以确认？
4. 为分离式预填充 Pod 选择一个扩缩信号，为解码 Pod 选择另一个不同的信号。证明两者的合理性。
5. 计算 `WhenEmptyOrUnderutilized` 整合陷阱在一个 24x7 生产服务上的成本，该服务平均每天发生 60 次请求丢弃事件，P99 TTFT > 10s。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Karpenter | "节点配置器" | Kubernetes 节点自动扩缩器；亚分钟级配置 |
| Cluster Autoscaler | "旧版扩缩器" | Kubernetes 节点自动扩缩器前身；更慢，基于组 |
| KAI Scheduler | "GPU 调度器" | 用于 gang + 拓扑 + 队列的辅助调度器 |
| Gang scheduling | "全有或全无" | 原子性调度 N 个 Pod，否则全部推迟 |
| Topology awareness | "机架感知" | 基于 NVLink/IB/机架放置 Pod |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 利用率" | 占空比度量；不是 LLM 的扩缩信号 |
| Queue depth | "等待请求" | 预填充绑定扩缩的正确 HPA 信号 |
| KV cache utilization | "内存压力" | 解码绑定扩缩的正确 HPA 信号 |
| Consolidation | "Karpenter 整合" | 终止节点以迁移到更便宜的实例类型 |
| `WhenEmpty + 1h` | "安全整合" | 不驱逐运行中 GPU 任务的策略 |

## Further Reading

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置示例。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — 整合策略语义和 GPU 安全默认值。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩缩信号。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管 Kubernetes 特定指南。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 设计。