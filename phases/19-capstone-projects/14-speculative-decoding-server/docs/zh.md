# 顶点项目 14——推测解码推理服务器

> vLLM 0.7 中的 EAGLE-3 在实际流量上提供 2.5-3 倍的吞吐量。P-EAGLE（AWS 2026）进一步推进了并行推测。SGLang 的 SpecForge 大规模训练草稿头。Red Hat 的 Speculators 中心为常见开源模型发布了对齐的草稿。TensorRT-LLM 使推测解码在 NVIDIA 上成为一流功能。2026 年的生产服务栈是带 EAGLE 族草稿的 vLLM 或 SGLang、FP8 或 INT4 量化、以及基于队列等待的 HPA。这个顶点项目是以 2.5 倍以上基线吞吐量服务两个开源模型，并附带完整的尾延迟报告。

**类型:** Capstone
**语言:** Python（服务）、C++ / CUDA（内核检查）、YAML（配置）
**前置要求:** Phase 3（深度学习）、Phase 7（Transformer）、Phase 10（LLM 从零开始）、Phase 17（基础设施）
**涉及阶段:** P3 · P7 · P10 · P17
**时间:** 30 小时

## 问题

推测解码在 2026 年成为商品。EAGLE-3 草稿头在目标模型的隐藏状态上训练，并预测前 N 个 token；目标模型一次验证。60-80% 的接受率转化为 2-3 倍的端到端吞吐量。vLLM 0.7 原生集成这一点。SGLang + SpecForge 为你提供训练管道。Red Hat 的 Speculators 为 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 发布对齐的草稿。

技艺在于服务操作，而非模型。接受率随流量分布（ShareGPT vs 代码 vs 领域数据）漂移。拒绝状态下的尾延迟比无推测更差——你必须在多个批量大小时报告 p99，而不仅仅是稳态 tokens/sec。每百万 token 成本与 Anthropic/OpenAI API 的对比是可信度杠杆。

## 概念

推测解码有两层。一个**草稿**模型（EAGLE-3 头、ngram 或更小的目标对齐模型）每步提出 k 个候选 token。**目标**模型一次验证所有 k 个；任何接受的前缀替换贪心路径。接受率取决于草稿-目标对齐和输入分布。

EAGLE-3 在大多数流量上击败 ngram 草稿。P-EAGLE 为更深的草稿树运行并行推测。权衡：拒绝时的 P99 延迟更高，因为验证扫描更大。服务配置必须报告按批量大小分桶的延迟以揭示这一点。

部署是 Kubernetes。vLLM 0.7 每个 GPU 或张量并行分片运行一个副本。HPA 基于队列等待而非 CPU 自动扩展。FP8（Marlin）和 INT4（AWQ）量化将 GPU 内存保持在 H100/H200 范围内。端到端报告是吞吐量、接受率、批量 1/8/32 时的 p50/p99 和每百万 token 成本。

## 架构

```
请求入口
    |
    v
vLLM 服务器 (0.7) 或 SGLang (0.4)
    |
    +-- 草稿: EAGLE-3 heads | P-EAGLE 并行 | ngram 后备
    +-- 目标: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     量化 FP8-Marlin 或 INT4-AWQ
    |
    v
验证扫描: 通过目标批量处理 k 个草稿 token
    |
    v (接受前缀; 为被拒绝的后缀重新采样)
    v
token 流回客户端
    |
    v
Prometheus 指标: 吞吐量, 接受率, 队列等待, 延迟 p50/p99
    |
    v
基于队列等待指标的 HPA
```

## 技术栈

- 服务：vLLM 0.7 或 SGLang 0.4
- 推测方法：EAGLE-3 草稿头、P-EAGLE 并行推测、ngram 后备
- 草稿训练：SpecForge（SGLang）或 Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8（Marlin）、INT4 AWQ
- 部署：Kubernetes + NVIDIA 设备插件；基于队列等待指标的 HPA
- 评估：ShareGPT、MT-Bench-v2、GSM8K、HumanEval 用于领域分布接受率测量
- 参考：TensorRT-LLM 推测解码用于供应商基线

## 构建它

1. **目标模型准备。** 选择 Llama 3.3 70B。通过 Marlin 量化到 FP8。在 1xH100（或 2x 张量并行）上以 vLLM 0.7 部署。

2. **草稿源。** 从 Red Hat Speculators 拉取对齐的 EAGLE-3 草稿头（或通过 SpecForge 训练一个）。加载到 vLLM 的推测解码配置中。

3. **基线数字。** 推测前：批量 1/8/32 下的 tokens/s、p50/p99 延迟、GPU 利用率。发布。

4. **启用 EAGLE-3。** 翻转配置；重新运行相同的基准。报告加速比、接受率、p99 尾延迟差异。

5. **P-EAGLE。** 启用并行推测；测量更深的草稿树 vs 串行 EAGLE-3。报告 P-EAGLE 在何时有帮助与有害。

6. **领域流量。** 通过相同的服务器运行 ShareGPT vs HumanEval vs 领域特定流量。测量每个分布的接受率。识别草稿何时漂移。

7. **第二个目标模型。** 在 Qwen3-Coder-30B MoE 上运行相同的管道。草稿更棘手（MoE 路由噪声）。报告。

8. **K8s HPA。** 在 K8s 下部署，HPA 跟踪 `queue_wait_ms`。演示负载增加三倍时的扩展。

9. **成本比较。** 在相同的评估上计算每百万 token 成本 vs Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4。发布。

## 使用它

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 激活
[decode]    bs=8, 每步接受 token 数=3.2, 接受率=0.76
[latency]   首 token 42ms, 完整响应 980ms (620 tokens)
[cost]      在持续吞吐量下每百万输出 token 成本 $0.34
```

## 交付物

`outputs/skill-inference-server.md` 描述了交付物。一个经过测量的带推测解码的服务栈、完整的基准报告和 K8s 部署。

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | 与基线相比的测量加速比 | 在两个模型上，匹配质量下吞吐量 2.5 倍以上 |
| 20 | 实际流量上的接受率 | 按分布的接受率报告 |
| 20 | P99 尾延迟纪律 | 有和无推测时批量 1/8/32 下的 p99 |
| 20 | 运维 | K8s 部署、基于队列等待的 HPA、平滑发布 |
| 15 | 报告和方法论 | 清晰解释什么变了以及为什么 |

## 练习

1. 测量草稿落后目标一个版本时的接受率衰减（例如，Llama 3.3 -> 3.4 漂移）。构建监控告警。

2. 实现 ngram 后备：如果 EAGLE-3 接受率低于阈值，切换到 ngram 草稿。报告可靠性改进。

3. 运行受控 MoE 实验：相同的 Qwen3-Coder-30B，有和无注入路由噪声。测量草稿接受敏感性。

4. 扩展到 H200（141 GB）。报告每个副本获得的模型大小余量，以及是否可以为未量化的 Llama 3.3 70B 服务。

5. 在相同的 H100 硬件上基准测试 TensorRT-LLM 推测解码。报告它在哪些方面优于 vLLM。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Draft model | "推测器" | 提出 N 个 token 供目标验证的小模型 |
| EAGLE-3 | "2026 草稿架构" | 在目标隐藏状态上训练的草稿头；约 75% 接受率 |
| P-EAGLE | "并行推测" | 一次目标扫描验证的草稿分支树 |
| Acceptance rate | "命中率" | 被接受且无需重新采样的草稿 token 比例 |
| Quantization | "FP8 / INT4" | 更低精度权重，以在 GPU 内存中容纳更多模型 |
| Queue wait | "HPA 指标" | 推理开始前请求在待处理队列中等待的时间 |
| Speculators hub | "对齐的草稿" | Red Hat Neural Magic 的常见开源模型 EAGLE 草稿中心 |

## 延伸阅读

- [vLLM EAGLE 和 P-EAGLE 文档](https://docs.vllm.ai)——参考服务栈
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/)——并行推测解码论文 + 集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge)——草稿头训练管道
- [Red Hat Speculators](https://github.com/neuralmagic/speculators)——对齐的草稿中心
- [TensorRT-LLM 推测解码](https://nvidia.github.io/TensorRT-LLM/)——供应商替代方案
- [Fireworks.ai 服务架构](https://fireworks.ai/blog)——商业参考
- [EAGLE-3 论文 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840)——方法论文
- [vLLM 仓库](https://github.com/vllm-project/vllm)——代码和基准
