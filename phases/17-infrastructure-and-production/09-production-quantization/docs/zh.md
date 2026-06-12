# 生产量化 — AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> 量化格式不是通用选择 — 它是硬件、服务引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 主导 CPU 和边缘场景，通过 llama.cpp 和 Ollama 交付。GPTQ 在 vLLM 中胜出，当你需要在同一基座上使用多 LoRA 时。AWQ 配合 Marlin-AWQ 内核在 INT4 下为 7B 类模型提供约 741 tok/s 的吞吐量和最佳的 Pass@1 — 这是 2026 年数据中心生产的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上保持中间地带 — 接近无损且广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）是激进的方案，需要逐块验证。两个陷阱会困扰团队：校准数据集必须匹配部署领域，KV 缓存与权重量化是分开的 — AWQ 课程中"我的模型现在只有 4 GB"的说法忘记了生产批处理规模下 10-30 GB 的 KV 缓存。

**类型：** Learn
**语言：** Python（stdlib，玩具级跨格式内存和吞吐量比较工具）
**前置知识：** Phase 10 · 13（量化基础），Phase 17 · 04（vLLM 服务内部原理）
**时间：** ~75 分钟

## 学习目标

- 说出 2026 年六种生产量化格式及其最佳应用场景。
- 根据硬件（CPU 与 GPU、Hopper 与 Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（常规聊天、推理、多 LoRA）选择格式。
- 计算所选格式节省的权重内存以及未触及的 KV 缓存大小。
- 说出导致量化模型在领域流量上性能下降的校准数据集陷阱。

## 问题

量化减少了内存和 HBM 带宽，这正是解码所需要的。一个 FP16 的 70B 模型是 140 GB 的权重。将权重量化到 INT4（AWQ 或 GPTQ）后，模型变成 35 GB — 可以放入一个 H100 并留出 KV 缓存的空间，这很重要，因为在 128 个并发序列和 2k 上下文下，仅 KV 缓存就需要 20-30 GB。

但量化不是免费的。激进的量化会降低质量，尤其是在推理密集型任务上。不同的格式与不同的引擎配合使用。不同的硬件原生支持不同的精度。2026 年的格式动物园是真实存在的，你不能照搬别人的选择 — 你必须根据自己的技术栈来选择。

## 概念

### 六种格式

| 格式 | 比特数 | 最佳场景 | 引擎 |
|------|--------|---------|------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM、TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM（Marlin-AWQ）、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### GGUF — CPU/边缘默认选择

GGUF 是一种文件格式，而非量化方案本身 — 它将 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包在一个容器中。Q4_K_M 和 Q5_K_M 是生产默认值 — 在 4-5 比特下接近 BF16 质量。当部署目标是 CPU 或边缘时，这是最佳选择，因为 llama.cpp 是目前最快的 CPU 推理引擎。

在 vLLM 中的吞吐量损失：7B 模型约 93 tok/s — 该格式未针对 GPU 内核优化。当部署目标是 CPU/边缘时使用 GGUF。否则不要使用。

### GPTQ — vLLM 中的多 LoRA

GPTQ 是一种带有校准过程的训练后量化算法。Marlin 内核使其在 GPU 上快速运行（比非 Marlin GPTQ 快 2.6 倍）。7B 模型约 712 tok/s。

独特的优势：GPTQ-Int4 在 vLLM 中支持 LoRA 适配器。如果你正在服务一个基座模型加上 10-50 个微调变体（每个作为 LoRA），GPTQ 是你的选择。截至 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ — 数据中心 GPU 默认选择

激活感知权重量化（Activation-aware Weight Quantization）。在量化过程中保护约 1% 最重要的权重。Marlin-AWQ 内核：比朴素实现快 10.9 倍。7B 模型约 741 tok/s，INT4 格式中最佳的 Pass@1。

除非你需要多 LoRA（使用 GPTQ）或激进的 Blackwell FP4（使用 NVFP4），否则新的 GPU 服务选择 AWQ。

### FP8 — 可靠的中间选择

8 位浮点数。接近无损。广泛支持。Hopper Tensor Cores 原生加速 FP8。Blackwell 继承支持。当质量不可妥协时（推理、医疗、代码生成），FP8 是 2026 年安全的默认选择。内存节省是 INT4 的一半，但质量风险远低于 INT4。

### MXFP4 / NVFP4 — Blackwell 激进方案

微缩放 FP4（Microscaling FP4）。每个权重块有自己的缩放因子。激进但在 Blackwell Tensor Cores 上由硬件加速。相比 FP8，每个 token 的字节数减半 — 这是 Phase 17 · 07 中的经济优势。

注意事项：
- 尚不支持 LoRA（2026 年初）。
- 在推理密集型工作负载上可见质量下降。
- 需要针对每个模型在你的评估集上验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集 — 通常是 C4 或 WikiText。对于领域模型（代码、医疗、法律），在通用网页文本上校准会让算法在保护哪些权重上做出错误决策。HumanEval 上的 Pass@1 可能下降几个百分点。

修复方法：在领域内数据上校准。几百个领域样本通常就足够了。在部署前在评估集上测试。

### KV 缓存陷阱

AWQ 将权重缩小到 4 比特。KV 缓存是独立的，保持 FP16/FP8。对于一个使用 AWQ 的 70B 模型：

- 权重：约 35 GB（从 140 GB 的 INT4）。
- 128 并发 × 2k 上下文下的 KV 缓存：约 20 GB。
- 激活值：约 5 GB。
- 总计：约 60 GB — 可以放入 H100 80GB。

天真地认为"我将模型量化到 4 GB"忘记了另外 30-50 GB。要整体规划 HBM 预算。

另外，KV 缓存量化（FP8 KV 或 INT8 KV）是一个不同的选择，有其自身的权衡 — 它直接影响注意力精度，不是免费的胜利。

### AWQ INT4 对推理有风险

思维链、数学、长上下文的代码生成 — 这些任务在激进量化下明显受损。AWQ INT4 在 MATH 上损失约 3-5 个百分点。对于推理密集型工作负载，使用 FP8 或 BF16；接受内存成本。

### 2026 年选择指南

- CPU/边缘服务：GGUF Q4_K_M。搞定。
- GPU 服务，常规聊天，无 LoRA：AWQ。
- GPU 服务，多 LoRA：GPTQ 配合 Marlin。
- 推理工作负载：FP8。
- Blackwell 数据中心，已验证质量：NVFP4 + FP8 KV。
- 不确定：在每个候选格式上运行 1,000 样本评估。

```figure
gpu-memory-breakdown
```

## 使用它

`code/main.py` 计算六种格式在不同模型大小下的内存占用（权重 + KV + 激活值）和相对吞吐量。展示 KV 缓存何时占主导地位，权重压缩何时有回报，以及 FP8 何时是安全选择。

## 产出

本课程产出 `outputs/skill-quantization-picker.md`。给定硬件、模型大小、工作负载类型和质量容忍度，选择一种格式并生成校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于一个 70B 模型，128 并发，2k 上下文，计算每种格式的总 HBM 需求。哪种格式可以放入一个 H100 80GB？
2. 你有一个 7B 编码模型。选择一种格式并说明理由。如果你对质量容忍度的判断有误，恢复路径是什么？
3. 计算为医疗领域模型校准 AWQ 所需的校准数据集大小。为什么更多数据不一定更好？
4. 阅读 Marlin-AWQ 内核论文或发布说明。用三句话解释为什么 AWQ 在 7B 上达到 741 tok/s 而原始 GPTQ 约 712 tok/s。
5. 什么时候将 AWQ 权重与 FP8 KV 缓存结合使用有意义，而什么时候保持 KV 为 BF16？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| GGUF | "llama.cpp format" | 打包 K-quant 变体的文件格式；CPU/边缘默认 |
| Q4_K_M | "Q4 K M" | 4 位 K-quant medium；生产 GGUF 默认值 |
| GPTQ | "gee pee tee q" | 带校准的训练后 INT4；vLLM 中支持 LoRA |
| AWQ | "a w q" | 激活感知 INT4；Marlin 内核；INT4 下最佳 Pass@1 |
| Marlin kernels | "fast INT4 kernels" | Hopper 上 INT4 的自定义 CUDA 内核；10 倍加速 |
| FP8 | "eight-bit float" | Hopper/Ada/Blackwell 上的安全精度默认值 |
| MXFP4 / NVFP4 | "microscaling four" | Blackwell 4 位 FP，带逐块缩放因子 |
| Calibration dataset | "cal data" | 用于选择量化参数的输入文本；必须匹配领域 |
| KV cache quantization | "KV INT8" | 与权重分开的选择；影响注意力精度 |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 对比基准测试。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 按格式分类的吞吐量数据。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 逐格式选择指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式和标志。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — 原始 AWQ 公式。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — 原始 GPTQ 公式。