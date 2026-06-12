# 分离式预填充/解码（Disaggregated Prefill/Decode）—— NVIDIA Dynamo 与 llm-d

> 预填充（prefill）是计算密集型；解码（decode）是内存密集型。在同一 GPU 上运行两者会浪费一种资源。分离式架构将它们拆分到独立的资源池，并通过 NIXL（RDMA/InfiniBand 或 TCP 回退）在两者之间传输 KV 缓存。NVIDIA Dynamo（GTC 2025 发布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler + SLA Planner 自动匹配预填充:解码比例以满足 SLO。NVIDIA 发布的吞吐量提升大致如下——developer.nvidia.com（2025-06）显示 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上在中延迟场景下约有 6 倍改进，而 Dynamo 产品页面（developer.nvidia.com，无日期）宣称在 GB300 NVL72 + Dynamo 上 MoE 吞吐量相比 Hopper 提升高达 50 倍。"30 倍"这个数字是社区对全栈 Blackwell + Dynamo + DeepSeek-R1 报告的汇总；我们未找到确切声明 30 倍的单一主要来源，因此应将其视为方向性说法。llm-d（Red Hat + AWS）是 Kubernetes 原生的：预填充/解码/路由器作为独立 Service，配备按角色 HPA。llm-d 0.5 增加了分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩至零。经济学：多个客户披露的内部汇总表明，在恒定 SLA 下，从共置服务切换到 Dynamo 分离式架构后，$2M 级别的推理支出可节省 30-40%（即 $600-800K/年）；具体的 $2M→$600-800K 数字是内部综合数据，而非单一已发表的案例研究——请将其用作数量级参考，而非可引用的来源。短提示（<512 token，短输出）不值得承担传输成本。

**类型：** 学习
**语言：** Python（stdlib，玩具级分离式 vs 共置模拟器）
**前置要求：** Phase 17 · 04（vLLM 服务内部原理），Phase 17 · 08（推理指标）
**时间：** ~75 分钟

## 学习目标

- 解释为什么预填充和解码有不同的最优 GPU 分配，并量化共置下的浪费。
- 绘制分离式架构图：预填充池、解码池、通过 NIXL 的 KV 传输、路由器。
- 指出分离式架构何时不划算（短提示、短输出）。
- 区分 NVIDIA Dynamo（栈上协调器）与 llm-d（Kubernetes 原生），并将每个匹配到相应的运维上下文。

## 问题

你在 8 块 H100 上运行 Llama 3.3 70B。在混合工作负载下（长提示 + 短输出），GPU 在解码期间空闲，因为大部分计算已花在预填充上。在不同的工作负载下（短提示 + 长输出），情况相反。共置预填充 + 解码意味着你为两者都过度配置了。

预算影响：20-40% 的 GPU 时间浪费在错误的资源上。你购买 H100 算力来运行内存密集型的解码，或者购买 H100 HBM 带宽来运行计算密集型的预填充。两者都是昂贵的浪费。

分离式架构将预填充和解码拆分到独立的资源池，每个池针对各自的瓶颈进行优化。KV 缓存通过高带宽互连从预填充池传输到解码池。

## 概念

### 为什么瓶颈不同

**预填充**——在一次前向传播中运行整个输入提示的 transformer。矩阵乘法占主导；计算密集型。H100 FP8 提供约 2000 TFLOPS 的有效吞吐量。批处理效率高——一次前向传播处理许多 token。

**解码**——一次生成一个 token，每次迭代读取全部权重。内存带宽密集型。HBM3 提供约 3 TB/s。批处理效率仅在高度并发时良好——权重读取在批次中摊销。

共置：你购买为两者优化的 GPU。H100 两者都擅长，但无论哪种用法成本都一样。在规模下，你希望预填充池使用 H100/计算密集型；解码池使用 H200/内存密集型，或使用激进的量化。

### 架构

```
             ┌──────────────┐
   Request → │    Router    │ ───────────────────────┐
             └──────┬───────┘                        │
                    │                                │
                    ▼ (prompt only)                  │
             ┌──────────────┐    KV cache    ┌───────▼──────┐
             │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
             │  (compute)   │                │  (memory)    │
             └──────────────┘                └──────┬───────┘
                                                    │ tokens
                                                    ▼
                                                  Client
```

NIXL 是 NVIDIA 的节点间传输。可用时使用 RDMA/InfiniBand，否则使用 TCP 回退。传输延迟是真实存在的——对于 70B FP8 上 4K token 提示的 KV 缓存，通常为 20-80 ms。这就是为什么短提示不值得分离式架构：传输开销超过了节省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：
- 作为编排器位于 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 测量工作负载，SLA Planner 自动配置预填充:解码比例。
- Rust 核心，Python 可扩展。
- 吞吐量提升：NVIDIA 报告 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上中延迟场景下提升 6 倍（developer.nvidia.com，2025-06）；社区关于全栈 Blackwell + Dynamo + DeepSeek-R1 "高达 30 倍"的报告缺乏单一主要来源，应视为方向性说法。
- GB300 NVL72 + Dynamo：MoE 吞吐量相比 Hopper 提升高达 50 倍（Dynamo 产品页面，developer.nvidia.com，无日期）。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- 预填充/解码/路由器作为独立的 Kubernetes Service。
- 按角色 HPA，使用队列深度（预填充）/ KV 利用率（解码）信号。
- `topologyConstraint packDomain: rack` 将预填充+解码集群打包到同一机架，以实现高带宽 KV 传输。
- llm-d 0.5（2026）：分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩至零。

如果你想要托管的栈上编排器，使用 Dynamo。如果你想要 Kubernetes 原生原语且已投入 CNCF 生态系统，使用 llm-d。

### 经济学

内部综合数据（非单一已发表案例研究——数量级参考）：

- 共置服务上 $2M/年的推理支出。
- 切换到使用 Dynamo 的分离式架构。
- 相同请求量，相同 P99 延迟 SLA。
- 报告节省：$600K-$800K/年（30-40% 减少）。
- 无需新硬件。

我们综合了多个客户披露而非单一可引用案例研究得出此数字；最接近的已发表数据点是 Baseten 使用 Dynamo KV 路由实现 2 倍更快的 TTFT / 61% 更高吞吐量（baseten.co，2025-10），以及 VAST + CoreWeave 在 40-60% KV 命中率下预测 60-130% 更多 token/$（vastdata.com，2025-12）。节省来自每个池的合理规模调整；预填充密集型工作负载（8K+ 前缀的 RAG）比均衡工作负载受益更多。

### 何时不应分离

- 提示 < 512 token 且输出 < 200 token：传输开销超过收益。
- 小集群（< 4 块 GPU）：池多样性不足。
- 团队无法运维两个具有按角色扩缩的 GPU 池：Dynamo 有帮助但并非微不足道。
- 没有 RDMA 网络：TCP 传输开销更大。

### 路由器与 Phase 17 · 11 的集成

分离式路由器是 KV 缓存感知的（Phase 17 · 11）。请求落在持有其前缀的解码池上——如果没有匹配，则走预填充 → 解码流程。命中率和分离式架构相互叠加——缓存感知路由器决定了是否需要新的预填充。

### MoE 在 Blackwell 上才是真正的数字所在

GB300 NVL72 + Dynamo 显示 MoE 吞吐量相比 Hopper 基线提升 50 倍。MoE 专家路由在预填充上是计算密集型，但在解码上是内存密集型（专家缓存），因此分离式架构是双重收益。2026 年前沿模型服务以 MoE 为主（DeepSeek-V3、未来的 GPT-5 变体）。

### 你应该记住的数字

基准测试数字会变化——NVIDIA 和推理栈每个季度都会发布更新的结果。引用前请重新核对。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo 上：中延迟场景下吞吐量相比基线提升约 6 倍（developer.nvidia.com，2025-06）；社区关于全栈 Blackwell + Dynamo "高达 30 倍"的说法是方向性汇总，无单一主要来源。
- GB300 NVL72 + Dynamo：MoE 吞吐量相比 Hopper 提升高达 50 倍（developer.nvidia.com，无日期）。
- 节省参考（内部综合数据，非单一案例研究）：$2M 年支出中节省 $600-800K/年，恒定 SLA。
- 分离式阈值：提示 >512 token + 输出 >200 token。
- 通过 NIXL 的 KV 传输：70B FP8 上 4K 提示的 KV 为 20-80 ms。

## 使用它

`code/main.py` 模拟共置 vs 分离式服务。报告吞吐量、每请求成本以及提示长度交叉点。

## 交付物

本课程产出 `outputs/skill-disaggregation-decider.md`。根据工作负载和集群，决定是否分离。

## 练习

1. 运行 `code/main.py`。在什么提示长度下，分离式优于共置？
2. 为一个 P99 前缀长度 8K、输出 300 的 RAG 服务设计预填充池和解码池。
3. Dynamo vs llm-d：为一个纯 Kubernetes 且无 Python 运行时偏好的团队选择一个。
4. 计算 KV 传输成本：70B FP8 上 4K 预填充 = 约 500 MB KV。RDMA 100 GB/s 时，传输 = 5 ms。TCP 10 GB/s 时 = 50 ms。哪个对你的 SLA 重要？
5. MoE 专家路由改变了 KV 访问模式。分离式架构如何处理每个 token 激活不同专家的 MoE？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| Disaggregated serving | "分离预填充/解码" | 每个阶段使用独立的 GPU 池 |
| NIXL | "NVIDIA 传输" | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | "编排器" | vLLM/SGLang/TRT-LLM 的栈上协调器 |
| llm-d | "Kubernetes 原生" | Red Hat + AWS K8s 分离式栈 |
| Planner Profiler | "Dynamo 自动配置" | 测量工作负载，配置池比例 |
| SLA Planner | "Dynamo 策略" | 自动匹配预填充:解码以满足 SLO |
| `packDomain: rack` | "llm-d 拓扑" | 将预填充+解码打包到同一机架以实现快速 KV |
| UCCL | "统一集合" | llm-d 0.5 网络层，支持缩至零 |
| MoE expert routing | "每个 token 的专家" | DeepSeek-V3 模式；分离式架构有帮助 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)