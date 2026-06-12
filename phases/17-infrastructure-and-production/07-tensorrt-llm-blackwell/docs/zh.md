# Blackwell 上使用 FP8 和 NVFP4 的 TensorRT-LLM

> TensorRT-LLM 是 NVIDIA 专属的，但它在 Blackwell 上取胜。在 GB200 NVL72 上使用 Dynamo 编排，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 测得 120B 模型每百万 token 成本为 $0.012，而 H100 + vLLM 为 $0.09/M——7 倍的经济差距。该栈由三种浮点格式叠加而成：FP8 对 KV 缓存和注意力核仍至关重要，因为它具备它们所需的动态范围；NVFP4（4 位微缩放）处理权重和激活；多 token 预测（MTP）和分离式预填充/解码又叠加了 2-3 倍。Day-0 模型支持直接加载 FP4 权重，无需训练后转换。对于 2026 年的工程团队来说，问题在于：TRT-LLM 是一个封闭的 NVIDIA 栈，因此采用它是以可移植性换取吞吐量。在承诺之前，请根据你的模型和硬件组合来计算一下经济账。

**Type:** Learn
**Languages:** Python（stdlib，简易 FP8/NVFP4 内存和成本计算器）
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)、Phase 10 · 13 (Quantization)
**Time:** ~75 分钟

## 学习目标

- 解释为什么即使权重使用 NVFP4，KV 缓存和注意力仍需要 FP8。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并推理节省来源。
- 说出 TRT-LLM 利用的 Blackwell 特定特性（day-0 FP4、MTP、分离式服务、all-to-all 原语）。
- 判断 TRT-LLM 的 NVIDIA 锁定何时值得 Hopper 上 vLLM 的 7 倍成本差距。

## 问题

2026 年推理经济学的边界是"每美元多少 token"。答案取决于四个堆叠的选择：硬件代际（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、推理引擎（vLLM vs SGLang vs TRT-LLM）和编排（普通 vs 分离式 vs Dynamo）。

在 Hopper 上使用 vLLM，120B MoE 的运行成本约 $0.09 每百万 token。在 Blackwell 上使用 TRT-LLM + Dynamo，同样的模型运行成本约 $0.012——便宜 7 倍。部分差距来自硬件（Blackwell 每 GPU LLM 吞吐量是 Hopper 的 11-15 倍）。部分来自栈：FP4 权重、MTP 草稿、分离式预填充/解码，以及用于 MoE 专家通信的 NVLink 5 all-to-all。

你无法在 NVIDIA 栈之外复制这一成果。这就是权衡——用可移植性换取经济性。理解哪些栈选择贡献了多大份额的差距，是本课程的重点。

## 概念

### 为什么 FP8 仍是 KV 缓存的下限

2026 年的一个常见错误：认为 NVFP4 适用于所有地方。事实并非如此。KV 缓存需要 FP8（8 位浮点数），因为它存储的注意力和键值覆盖了广泛的动态范围。将 KV 量化到 FP4 会导致灾难性的精度损失——分布的尾部消失，注意力分数崩溃。FP8 的指数位为 KV 缓存提供了所需的动态范围。

NVFP4（2025-2026）适用于权重和激活。微缩放（microscaling）：每个权重块有自己的缩放因子，因此小块可以在不损失每张量缩放的情况下覆盖不同的动态范围。对于激活，FP4 也能胜任，因为激活在层内是小范围的。

典型的 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。
- 激活：NVFP4。
- KV 缓存：FP8。
- 注意力累加器：FP32（softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 特定原语

- **Day-0 FP4 权重**：模型提供商直接发布 FP4 权重；TRT-LLM 无需训练后转换即可加载。FP4 不需要 AWQ/GPTQ 步骤。
- **多 token 预测（MTP）**：与 EAGLE（Phase 17 · 05）思路相同，但集成到 TRT-LLM 构建中。
- **分离式服务**：预填充和解码在不同的 GPU 池上，KV 缓存通过 NVLink 或 InfiniBand 传输。与 Dynamo（Phase 17 · 20）思路相同。
- **All-to-all 通信原语**：NVLink 5 将 MoE 专家通信延迟降低了 3 倍（相比 Hopper）。TRT-LLM 的 MoE 内核针对此进行了调优。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 上硬件加速的缩放因子处理。

### 你应该记住的数字

- HGX B200 在 GPT-OSS-120B 上通过 TRT-LLM 为 $0.02/M token。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）为 $0.012/M token。
- H100 + vLLM 在可比工作负载上约 $0.09/M token。
- 三个月内 TRT-LLM 更新带来的 2.8 倍吞吐量提升（2026 年）。
- Blackwell 相比 Hopper 每 GPU LLM 吞吐量提升 11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在每个提交任务中占据主导。

### FP4 在质量上的实际成本

NVFP4 是激进的。在推理密集型工作负载（思维链、数学、带长上下文的代码生成）上，FP4 权重可观察到性能下降。逐块校准可以缓解但无法消除。部署推理模型的团队通常使用 FP8 权重 + FP4 激活作为折衷方案，或者坚持使用 H200 全程 FP8。

规则：在承诺使用 NVFP4 权重之前，始终在你的评估集上验证任务质量。

### 为什么这是一个 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 闭源内核。模型需要针对特定的 GPU SKU 编译。不支持 AMD、Intel 或 ARM。如果你的基础设施策略是多供应商，TRT-LLM 对于 TRT-LLM 服务层来说是不可行的——你仍然可以在混合硬件上使用 vLLM 提供服务。如果你是纯 NVIDIA，7 倍差距足以弥补锁定的代价。

### 2026 年实践配方

对于年推理账单超过 $1 亿的情况，在 Hopper + vLLM 上运行意味着损失 7-10 倍的效率。将成本主导的工作负载迁移到 Blackwell + TRT-LLM + Dynamo。将实验层保留在 H100 + vLLM 上以保持模型迭代速度。在每个 NVFP4 转换模型进入生产前验证质量。

### 分离式服务的额外收益

TRT-LLM 的分离式服务（独立的预填充和解码池）在 Phase 17 · 20 中有深入介绍。在 Blackwell 上，乘数效应叠加：FP4 权重 × MTP 加速 × 分离式部署 × 缓存感知路由。7 倍的数字假设了这一完整栈。

```figure
pipeline-parallel
```

## 使用它

`code/main.py` 计算模型在三种栈上的 HBM 占用、解码吞吐量（内存受限区间）和 $/M-token：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它以看到复合效应以及每个变化贡献的差距份额。

## 交付物

本课程产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型大小和年 token 量，判断 Blackwell + TRT-LLM 栈是否值得 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。对于一个活跃参数占 30% 的 120B MoE，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 下的内存带宽受限解码吞吐量。最大的跃升来自哪里？
2. 某客户每年在 H100 + vLLM 上花费 $2M。考虑到 7 倍的经济差距，他们需要购买多少 Blackwell GPU 才能在 12 个月内摊销迁移到 TRT-LLM 的成本？
3. 在 NVFP4 权重转换后，你发现 MATH 上的准确率下降了 3 个点。说出两种恢复路径：一种质量优先（保留 FP8 权重），一种成本优先（使用领域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务的 Blackwell 对 Hopper 差距最小，为什么？
5. 计算 405B 模型在 NVFP4 权重 + FP8 KV 缓存 + 128k 上下文下所需的 HBM。单个 GB200 NVL72 节点能容纳吗？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| FP8 | "8 位浮点" | 8 位浮点数；由于动态范围需求，用于 KV 缓存和注意力 |
| NVFP4 | "4 位微缩放" | NVIDIA 的 4 位微缩放 FP 格式；Blackwell 上的权重和激活 |
| MXFP8 | "MX 八位" | 微缩放 FP8 变体；Blackwell Tensor Core 上硬件加速 |
| Day-0 FP4 | "直接发布 FP4 权重" | 模型提供商直接发布 FP4 权重；无需训练后转换步骤 |
| MTP | "多 token 预测" | TRT-LLM 集成的投机解码草稿（Phase 17 · 05） |
| Disaggregated serving | "分离式预填充/解码" | 预填充和解码在不同 GPU 池上；KV 通过 NVLink/IB 传输 |
| All-to-all | "MoE 专家通信" | 将 token 路由到专家 GPU 的通信模式；NVLink 5 降低 3 倍 |
| InferenceX | "SemiAnalysis 推理基准" | 2026 年行业接受的每 token 成本基准 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/)——2026 年 4 月 MLPerf 结果。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/)——NVLink 5 all-to-all 和 MoE 内核。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html)——官方引擎文档。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)——TRT-LLM 之上的分离式编排。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)——发布 Blackwell 数字的基准测试套件。
