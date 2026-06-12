# 多区域 LLM 服务与 KV 缓存局部性

> 轮询负载均衡对缓存 LLM 推理是有害的。一个没有落在持有其前缀的节点上的请求需要支付完整的 prefill 成本 — 长提示下 P50 约 800 ms，而缓存命中时约 80 ms。2026 年的生产模式是缓存感知路由器（Rust 编写的 vLLM Router、llm-d router），它消费 KV 缓存事件并基于前缀哈希匹配进行路由。最近的研究（GORGO）将跨区域网络延迟作为路由目标的显式项。商业"跨区域推理"产品（Bedrock 跨区域推理、GKE 多集群网关）将推理视为黑盒 — 它们处理可用性，而非 TTFT。JPMorgan 和 Mayo Clinic 在 2024 年 11 月运行了 us-east-1 故障切换，耗时约 22 分钟。灾难恢复的现实：32% 的 LLM 灾难恢复失败是因为团队备份了权重但忘记了分词器文件或量化配置。

**类型：** Learn
**语言：** Python（stdlib，玩具级前缀缓存感知路由器模拟器）
**前置知识：** Phase 17 · 04（vLLM 服务），Phase 17 · 06（SGLang RadixAttention）
**时间：** ~60 分钟

## 学习目标

- 解释为什么轮询负载均衡会破坏缓存推理，并量化 TTFT 惩罚。
- 绘制缓存感知路由器的架构图：输入（KV 缓存事件）、算法（前缀哈希匹配）、决胜策略（GPU 利用率）。
- 说出 LLM 灾难恢复的 32% 失败驱动因素（缺少分词器文件/量化配置），并列出三文件灾难恢复清单。
- 区分商业跨区域产品（Bedrock CRI、GKE Multi-Cluster Gateway）与 KV 感知路由。

## 问题

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个 ALB 使用轮询策略。生产中的前缀缓存命中率下降到 8%。TTFT P50 翻了三倍。你的 vLLM 日志显示每个请求都在支付完整的 prefill 成本。

轮询对于无状态服务是最优的。LLM 推理本质上是状态化的 — KV 缓存编码了模型看到的一切。盲目路由就是路由到错误的缓存。

另外，你的团队有一个灾难恢复计划。你将模型权重备份到跨区域的 S3。一个区域故障发生；你尝试故障切换；副本拒绝启动。你忘记了 tokenizer.json、量化配置和 RoPE 缩放配置在另一个你没有同步的桶里。

多区域 LLM 服务是一个缓存问题、一个路由问题和一个灾难恢复卫生问题 — 而不是一个负载均衡器问题。

## 概念

### 缓存感知路由

请求到达时带有提示。路由器对前缀进行哈希（比如前 512 个 token）；它询问每个副本"你有这个前缀的缓存吗？"。副本在分配和驱逐块时通过发布/订阅通道发布 KV 缓存事件。路由器选择有匹配的副本，如果没有匹配则回退到基于 GPU 利用率的决胜策略。

**vLLM Router**（Rust，2026 年生产堆栈）：订阅 `kv.cache.block_added` 事件，维护前缀哈希到副本索引的映射，以 O(1) 查找路由。没有匹配时回退到最小队列深度。

**llm-d router**：相同模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（Phase 17 · 06）是副本内的等价方案。跨副本路由严格来说是上游的。

### 数据

2K token 提示的 TTFT P50，Llama 3.3 70B FP8，H100：
- 缓存命中（同一副本，前缀驻留）：约 80 ms。
- 缓存未命中（冷 prefill）：约 800 ms。

10 倍差距。如果你的路由器在副本间达到 60-80% 的前缀缓存命中率，你就在 N 副本容量下逼近了单副本性能。如果只有 10%，你就在逼近朴素扩展。

### 跨区域有一个新的约束 — 网络延迟

区域间 RTT：
- us-east-1 ↔ us-west-2：约 65 ms。
- us-east-1 ↔ eu-west-1：约 75 ms。
- us-east-1 ↔ ap-southeast-1：约 220 ms。

如果路由将一个请求从 us-east-1 发送到 ap-southeast-1 的热前缀，节省的 prefill（800 → 80 ms）被 440 ms 的往返时间所淹没。GORGO（2026 年研究）将其显式化 — 联合最小化 `prefill_time + network_latency`，而不是单独最小化 prefill。通常答案是在区域内部路由，除非是巨大的多 MB 前缀，此时 prefill 占主导地位。

### 商业"跨区域推理"在这里没有帮助

AWS Bedrock 跨区域推理在容量压力期间自动将请求路由到其他区域。它优化可用性，而非 TTFT，并将推理视为黑盒。GKE Multi-Cluster Gateway 也是如此 — 服务级故障切换，没有 KV 缓存感知。

即使使用这些，你仍然需要一个应用层的缓存感知路由器。它们处理"us-east-1 着火了"的情况。缓存感知路由处理 TTFT 的情况。

### 灾难恢复卫生 — 32% 的文件缺失问题

2026 年被广泛引用的统计数据：32% 的 LLM 灾难恢复失败是因为团队备份了权重但忘记了：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ 缩放因子、GPTQ 零点）
- 模型特定配置（RoPE 缩放、注意力掩码、聊天模板）
- 引擎配置（`vllm_config.yaml`、采样默认值、LoRA 适配器清单）

修复方法是三文件最小灾难恢复清单：

1. HF 模型仓库下的所有文件（权重 + 配置 + 分词器）。
2. 引擎特定的服务配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁定）。

另外：每季度进行一次灾难恢复演练。JPMorgan 在 2024 年 11 月的 us-east-1 演练达到 22 分钟恢复时间，仅仅是因为操作手册经过演练。

### 数据驻留是正交的

欧盟客户的 PHI 不能离开欧盟。如果你的缓存感知路由器为了前缀匹配将巴黎发起的请求发送到 us-east-1，无论 TTFT 提升如何，你都违反了 GDPR。在优化缓存之前，先按数据驻留边界对路由器进行分区。

### 你应该记住的数字

- 缓存命中与未命中的 TTFT 差距：约 10 倍（2K 提示下 80 ms 与 800 ms）。
- 美欧区域间 RTT：约 75 ms。
- 灾难恢复失败：32% 缺少分词器/量化配置。
- JPMorgan us-east-1 故障切换 2024 年 11 月：22 分钟（30 分钟 SLA）。

## 使用它

`code/main.py` 模拟三种路由策略（轮询、缓存感知区域、缓存感知全局）在多区域工作负载上的表现。报告缓存命中率、TTFT P50/P99 和跨区域账单。

## 产出

本课程产出 `outputs/skill-multi-region-router.md`。给定区域、数据驻留约束和 SLA，设计一个路由计划。

## 练习

1. 运行 `code/main.py`。在给定 75 ms RTT 的情况下，提示长度达到多少时跨区域路由优于仅本地路由？
2. 你的缓存命中率从 70% 下降到 12%。诊断三个可能的原因以及每个原因可以确认的可观测指标。
3. 为一个在 vLLM 中服务、带有 5 个 LoRA 适配器的 70B AWQ 量化模型设计灾难恢复清单。列出每个文件和配置。
4. 论证 Bedrock 跨区域推理对于有严格 TTFT SLO 的金融科技公司是否"足够"。引用具体行为。
5. 一个巴黎发起的请求匹配了 us-east-1 中的一个前缀。你会路由它吗？写出策略。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Cache-aware routing | "smart LB" | 基于前缀哈希匹配路由到持有 KV 缓存的副本 |
| KV-cache events | "cache pub-sub" | 副本发布块添加/驱逐事件；路由器建立索引 |
| Prefix hash | "cache key" | 前 N 个 token 的哈希，用作路由器查找键 |
| GORGO | "cross-region routing research" | arXiv 2602.11688；网络延迟作为显式项 |
| Cross-region inference | "Bedrock CRI" | AWS 产品；可用性故障切换，非 TTFT 感知 |
| DR manifest | "the backup list" | 恢复所需的每个文件 — 不仅仅是权重 |
| Data residency | "GDPR boundary" | 法律约束，限制哪些区域可以处理用户数据 |
| RTT | "round-trip time" | 网络延迟；美欧 75 ms，美亚太 220 ms |
| LLM-aware LB | "cache-hit LB" | 缓存感知路由器作为一个产品类别 |

## 延伸阅读

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 带网络延迟项的跨区域 KV 缓存复用。
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 可用性故障切换文档。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — 缓存感知路由器源码。