# 无服务器 LLM 的冷启动缓解

> 一个 20 GB 的模型镜像从冷启动到提供服务需要 5-10 分钟（7B）到 20 分钟以上（70B）。在真正的无服务器世界中，这不是预热 — 这是宕机。缓解措施在五个层面运作：预填充节点镜像（AWS 上的 Bottlerocket、双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer、vLLM 原生支持）、GPU 内存快照（Modal 检查点，重启速度提升最多 10 倍）、温池（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM 流水线，延迟降低 10-200 倍），以及传输输入 token（KB）而非 KV 缓存（GB）的实时迁移。Modal 发布 2-4 秒冷启动作为下限；Baseten 默认 5-10 秒，预热后可低于 1 秒。本课程教你如何测量、预算和堆叠这五个层面。

**类型：** Learn
**语言：** Python（stdlib，玩具级冷启动路径模拟器）
**前置知识：** Phase 17 · 02（推理平台经济学），Phase 17 · 03（GPU 自动扩缩容）
**时间：** ~60 分钟

## 学习目标

- 列举冷启动缓解的五个层面，并为每个层面说出一个工具或模式。
- 对于一个 70B 模型，将总冷启动时间计算为（节点配置）+（权重下载）+（权重加载到 HBM）+（引擎初始化）的总和。
- 解释为什么实时迁移传输输入 token（KB）而非 KV 缓存（GB），以及代价是什么（重新计算）。
- 说出温池的权衡（为空闲 GPU 付费或接受冷启动尾部延迟），以及 `min_workers > 0` 成为强制要求的 SLA 阈值。

## 问题

你的无服务器 LLM 端点夜间缩容到零。早上 8 点流量激增。第一个请求需要等待：

1. Karpenter 配置一个 GPU 节点：45-60 秒。
2. 容器拉取一个 30 GB 的镜像（含权重）：120-300 秒。
3. 引擎将权重加载到 HBM：45-120 秒，取决于模型大小和存储速度。
4. vLLM 或 TRT-LLM 初始化 CUDA 图、KV 缓存池、分词器：10-30 秒。

总计：220-510 秒（大约 3-8 分钟）才能返回第一个 token。你的 SLA 是 2 秒。你部署了一个温池（`min_workers=1`），问题似乎消失了 — 但现在你 24x7 为一个空闲 GPU 付费。如果你的服务有 5 个产品，每个有一个温副本，那就是 5 × 24 × 30 = 3,600 GPU 小时/月，无论是否有用户调用。

冷启动缓解就是在保持无服务器经济性的同时，逼近始终在线（always-on）的延迟。

## 概念

### 第一层 — 预填充节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构将操作系统与数据分离。用预拉取的容器镜像快照数据卷；在 `EC2NodeClass` 中引用快照 ID。新节点启动时权重已在本地 NVMe 上 — 步骤 2 和部分步骤 3 消失。原生与 Karpenter 配合使用。对于大型模型，典型节省：每次冷启动 2-4 分钟。

GCP 上的等效方案：预烘焙容器层的自定义 VM 镜像。Azure 上：相同模式的托管磁盘快照。

### 第二层 — 模型流式加载（Run:ai Model Streamer）

不是在回答第一个请求之前加载完整文件，而是逐层将权重流式传输到 GPU 内存，并在第一个 transformer 块驻留后立即开始处理。NVIDIA Run:ai Model Streamer 在 2026 年的 vLLM 中原生提供。支持 S3、GCS 和本地 NVMe。通过将 I/O 与计算设置重叠，大型模型的权重加载时间大约减半。

### 第三层 — GPU 内存快照（Modal）

Modal 在首次加载后对 GPU 状态（权重、CUDA 图、KV 缓存区域）进行快照。后续重启直接反序列化到 HBM — 比重新初始化快 10 倍。这是最接近"在 2 秒内启动一个热 GPU"的方案。权衡：快照是每个 GPU 拓扑的，所以如果 Karpenter 将你迁移到不同的 SKU，你需要重新创建快照。

### 第四层 — 温池（min_workers=1）

最简单的缓解措施：始终保持一个副本就绪。成本是一个 GPU 的小时费率 24x7。对于小模型来说，这个算术很残酷（你支付 $0.85-$1.50/小时来避免 30 秒的冷启动），但对大模型来说还算友好（支付 $4/小时来避免 5 分钟的冷启动）。温池成为强制要求的 SLA 阈值：通常在 70B+ 模型上 TTFT P99 < 60 秒。

### 第五层 — 分层加载（ServerlessLLM）

ServerlessLLM 将存储视为一个层次结构：NVMe（快速但容量大）、DRAM（中等但分层）、HBM（小但即时）。权重预加载到 DRAM；按需加载到 HBM。论文报告与朴素的磁盘到 HBM 相比，冷加载延迟降低 10-200 倍。生产采用尚在早期，但已存在与 vLLM 的集成。

### 第六层 — 实时迁移（额外模式）

当一个节点不可用时（Spot 实例回收、节点排空），传统模式是冷启动另一个副本并排空请求队列。实时迁移将输入 token（千字节）移动到一个已加载模型的目地节点，并在目地节点上重新计算 KV 缓存。重新计算比通过网络传输 GB 级别的 KV 缓存更便宜。适用于分离式部署。

### 温池的数学

对于一个 P99 TTFT SLA 为 2 秒的服务，问题不是"是否使用温池"，而是"多少个温副本，以及哪些路径使用它们"。

- 高价值交互路径（实时聊天、语音代理）：`min_workers=1-2`。
- 后台批处理路径（夜间分类）：接受缩容到零，5-10 分钟冷启动可容忍。
- 高级用户层：每个租户的 `min_workers` 和专用容量。

### 先测量再优化

70B 模型在新节点上的冷启动分解（示意）：

| 阶段 | 时间 | 缓解措施 |
|------|------|---------|
| 节点配置 | 50 秒 | Bottlerocket + 预填充镜像、温池 |
| 镜像拉取 | 180 秒 | 预填充数据卷（消除） |
| 权重到 HBM | 75 秒 | 模型流式加载（减半）；GPU 快照（消除） |
| 引擎初始化 | 20 秒 | 持久 CUDA 图缓存 |
| 首次前向 | 3 秒 | 最小固有延迟 |
| **冷启动总计** | **328 秒** | |
| **缓解后总计** | **约 15 秒** | 22 倍减少 |

### 你应该记住的数字

- Modal 冷启动：2-4 秒（使用 GPU 快照）。
- Baseten 默认冷启动：5-10 秒；预热后低于 1 秒。
- 原始 70B 冷启动：3-8 分钟。
- Run:ai Model Streamer：约 2 倍权重加载加速。
- ServerlessLLM 分层加载：10-200 倍延迟降低（论文数据）。

## 使用它

`code/main.py` 模拟有无每种缓解措施的冷启动路径。报告总冷启动时间、温池成本，以及温池收回成本所需的盈亏平衡请求率。

## 产出

本课程产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型大小和流量模式，选择要堆叠的缓解措施。

## 练习

1. 运行 `code/main.py`。计算温副本比通过 SLO 违规导致的额外请求丢弃来支付冷启动成本更便宜的盈亏平衡请求率。
2. 你部署一个 13B 模型，P99 TTFT SLA 为 3 秒。选择实现该目标的最小缓解措施堆叠（最少的层数）。
3. Bottlerocket 预填充消除了镜像拉取，但权重仍然从快照加载到 HBM。计算如果快照支持的 NVMe 以 7 GB/s 读取，70B 模型的挂钟时间。
4. 你的无服务器提供商提供 GPU 快照（Modal），你的团队拒绝因为"快照泄露 PII"。论证双方 — 实际风险是什么，缓解措施是什么（临时快照、加密、命名空间隔离）？
5. 设计一个分层温池策略：付费用户、试用用户和批处理工作负载各有多少温副本？展示计算过程。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Cold start | "the big pause" | 从请求到新副本上首 token 的时间 |
| Warm pool | "always-on minimum" | `min_workers >= 1` 以保持至少一个副本就绪 |
| Pre-seeded image | "baked AMI" | 容器权重预驻留的节点镜像 |
| Bottlerocket | "AWS node OS" | AWS 容器优化操作系统，支持双卷快照 |
| Model streamer | "streaming load" | 将权重 I/O 与计算设置重叠 |
| GPU snapshot | "checkpoint to HBM" | 序列化加载后的 GPU 状态；重启时反序列化 |
| Tiered loading | "NVMe + DRAM + HBM" | 存储层级层次结构；按需加载 |
| Live migration | "move tokens" | 传输输入（KB），在目地节点重新计算 KV |
| `min_workers` | "warm replicas" | 无服务器最小保活副本数 |
| Scale-to-zero | "full serverless" | 空闲时无成本；接受完整冷启动代价 |

## 延伸阅读

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal 发布的基准测试和检查点架构。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预填充数据卷快照模式。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 将权重加载与计算设置重叠。
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — 预热操作手册。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 分离式部署的实时迁移。