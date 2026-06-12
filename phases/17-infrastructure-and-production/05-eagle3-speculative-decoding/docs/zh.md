# 生产环境中的 EAGLE-3 投机解码

> 投机解码（Speculative decoding）将一个快速的草稿模型与目标模型配对。草稿模型提出 K 个 token；目标模型在单个前向传播中验证；被接受的 token 是免费的。到 2026 年，EAGLE-3 是生产级变体——它在目标模型的隐藏状态（而非原始 token）上训练草稿头，将通用聊天场景下的接受率 alpha 推至 0.6-0.8 区间。正确的问题不是"草稿模型多快"，而是"在我的流量上 alpha 是多少？"如果 alpha 降至约 0.55 以下，在高并发下投机解码将净负收益，因为每个被拒绝的草稿都需要第二次目标前向传播。本课程教你先测量 alpha，再决定是否启用。

**Type:** Learn
**Languages:** Python（stdlib，简易接受率模拟器）
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)、Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 分钟

## 学习目标

- 说出投机解码的三代演变，并解释 EAGLE-3 相比 EAGLE-2 和经典草稿模型改变了什么。
- 定义接受率 alpha，从 alpha 和 K（草稿长度）计算预期加速比，并为目标并发度找出盈亏平衡 alpha。
- 解释为什么投机解码在 vLLM 2026 中是可选的（非默认），以及为什么在不测量 alpha 的情况下开启它是一个生产反模式。
- 编写测量计划：使用哪个基准测试、哪个提示分布、哪个并发点、以哪个指标作为门控。

## 问题

解码受内存带宽限制。在运行 Llama 3.3 70B FP8 的 H100 上，每个解码 token 读取约 140 GB/s 的权重并产生一个 token。解码期间 GPU 计算几乎空闲——瓶颈是 HBM 带宽，而非矩阵乘法吞吐量。

投机解码利用了这一差距。用一个廉价的草稿模型生成 K 个候选 token，然后让目标模型在单个前向传播中验证所有 K 个 token。每个被验证的 token 实际上是免费的（均摊到目标模型无论如何都要做的 K-batch 前向传播中）。

经典的草稿模型方法使用同一系列的小模型（Llama 3.2 1B 为 Llama 3.3 70B 起草）。它有效，但接受率一般——小模型分布与目标模型存在偏差。EAGLE、然后是 EAGLE-2、再然后是 EAGLE-3 直接在目标模型的内部状态上训练一个轻量级草稿头，使草稿的分布更紧密地追踪目标模型。这就是为什么 alpha 从草稿模型的 0.4 提升到 EAGLE-3 的 0.6-0.8。

问题在于：EAGLE-3 在 vLLM 2026 中是可选的。必须显式设置 `speculative_config`。不设置就没有加速。那些在不测量真实流量 alpha 的情况下就开启它的团队，通常会发现尾延迟变得更差，而非更好。

## 概念

### 投机解码实际带来的收益

没有投机解码时，每个 token 的成本是一次目标前向传播。使用投机解码，在草稿长度 K 和接受率 alpha 下，每个目标前向传播的期望 token 数为 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证的开销。对于 K=5, alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。实际数字集中在 2-3x 左右，因为生产流量下的 alpha 很少那么高，且 epsilon 在大批次下会增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 不会消失——它们会迫使目标模型为第一个被拒绝的 token 进行第二次前向传播。在 alpha 降至 0.4 的工作负载上，你需要支付草稿开销加验证加重试。在高并发下（如 256 并发），解码批次已经足够大，使得"单独目标"与"带验证的目标"之间的内存带宽差距缩小。在大多数 2026 硬件上，alpha 低于 0.55 时，投机解码净负收益。

Alpha 随工作负载变化。在 ShareGPT 风格的通用聊天上，基于 ShareGPT 训练的 EAGLE-3 达到 0.6-0.8。在领域特定流量（代码、医疗、法律）上，基于通用数据训练的草稿头降至 0.4-0.6。训练领域特定的草稿头可以恢复 alpha——与目标微调相比，这是一个轻量、快速的训练任务。

### EAGLE 世代一览

- **经典草稿模型**：同一系列的小模型。Alpha 0.3-0.5。基础设施简单——加载两个模型，每个目标前向传播运行 K 次草稿前向。
- **EAGLE-1（2024）**：在目标隐藏状态（最后一层）上训练的单个草稿头。Alpha 约 0.5-0.6。在目标之上参数开销很小。
- **EAGLE-2（2025）**：自适应草稿长度和基于树的草稿（在一次目标传播中验证多个分支）。Alpha 约 0.6-0.7。草稿调度器更复杂。
- **EAGLE-3（2025-2026）**：在多个目标层（不仅是最后一层）上训练的草稿头，对齐更好。通用聊天上 Alpha 约 0.6-0.8。

### 2026 年生产配方

1. 先部署纯目标模型。在目标并发度下测量基线 TTFT、ITL、吞吐量。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 草稿。重新运行基准测试。
3. 记录接受率 alpha。vLLM V1 将其报告为 `spec_decode_metrics.accepted_tokens_per_request`。除以请求的草稿长度得到 alpha。
4. 如果在生产流量分布上 alpha < 0.55，禁用投机解码或训练领域特定的 EAGLE-3 草稿。
5. 在生产并发度下重新运行。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部

平均 ITL 在投机解码下下降。但如果不做调优，P99 可能会变差。被拒绝的草稿触发了一个两遍序列（草稿 + 验证失败 + 重试）。在满批次下，这两遍传播是串行的。关注 P99 ITL，而非 P50。

### EAGLE-3 已在部署的场景

Google 在 2025 年将投机解码部署到 AI Overviews 中（相同质量，更快响应）。vLLM V1 将 `speculative_config` 作为文档化的接口提供；V1 中的 N-gram GPU 投机解码是与分块预填充（chunked prefill）兼容的变体。SGLang 推荐 EAGLE-3 作为前缀密集型工作负载的草稿路径。

### 一行盈亏平衡数学

期望加速比：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 解出 alpha：`alpha_breakeven = verify_overhead / K`。对于典型的 verify_overhead 约 0.15 和 K=5：`alpha_breakeven = 0.03`。但这是原始解码数学。在高并发下验证开销增加，且解码批次已经跨序列均摊了内存读取，因此实际 alpha_breakeven 在实践中升至约 0.45-0.55。

### 何时不使用投机解码

- Batch-1 离线生成，延迟不重要。使用纯目标模型。
- 非常短的输出（少于 50 token）。草稿开销和验证成本占主导。
- 没有领域训练草稿头的专业领域。Alpha 太低。
- vLLM v0.18.0 加草稿模型投机解码加 `--enable-chunked-prefill`。此组合无法编译。文档化的例外是 V1 中的 N-gram GPU 投机解码。

## 使用它

`code/main.py` 模拟了在多种 alpha 值和草稿长度 K 下，有和没有投机解码的解码循环。它打印盈亏平衡 alpha、实测加速比和尾部行为。在多个 (alpha, K) 组合上运行它，可以准确看到投机解码在何时停止产生收益。

## 交付物

本课程产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发度目标，它生成一个分阶段的 EAGLE-3 部署计划——基准测试基线、启用配置、测量 alpha、以 alpha >= 0.55 为门控、监控 P99 ITL。

## 练习

1. 运行 `code/main.py`。在 K=5 时，需要多少 alpha 才能实现 2 倍加速？3 倍加速？这对 verify_overhead 的敏感度如何？
2. 假设生产流量 70% 是通用聊天，30% 是代码。通用聊天在基于 ShareGPT 训练的 EAGLE-3 上 alpha 为 0.7；代码的 alpha 为 0.4。混合 alpha 是多少？投机解码是否净正收益？
3. 阅读 vLLM `speculative_config` 文档。说出三种模式（草稿模型、EAGLE、N-gram）以及哪一种与分块预填充兼容。
4. 启用 EAGLE-3 后，你发现平均 ITL 下降了 25%，但 P99 ITL 上升了 15%。诊断并提出缓解方案。
5. 计算 Llama 3.3 70B 的 EAGLE-3 草稿头的内存成本。与运行 Llama 3.2 1B 作为经典草稿相比如何？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Speculative decoding | "草稿加验证" | 用廉价模型提出 K 个 token，在单次目标前向中验证全部 K 个 |
| Acceptance rate alpha | "投机接受率" | 被目标模型接受的草稿 token 比例；唯一重要的指标 |
| Draft length K | "投机 K" | 草稿每次目标前向提出的 token 数；典型值 4-8 |
| Verify overhead epsilon | "投机开销" | 验证加重试相比纯目标前向的额外成本；随批次增长 |
| EAGLE-3 | "最新 EAGLE" | 2025-2026 变体；在多个目标层上训练草稿头；通用聊天 alpha 0.6-0.8 |
| `speculative_config` | "vLLM 投机配置" | vLLM V1 中的显式可选配置；无默认值意味着无加速 |
| N-gram spec decode | "N-gram 草稿" | 使用提示中的 N-gram 查找进行 GPU 端草稿；与分块预填充兼容 |
| Break-even alpha | "无收益 alpha" | 投机解码产生零加速时的 alpha；关注生产并发度下的该值 |
| Rejected-draft two-pass | "重试成本" | 草稿拒绝时的两次目标前向传播；推高了 P99 尾部 |

## 延伸阅读

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/)——关于 `speculative_config` 和 V1 中分块预填充兼容性的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/)——精确的字段集。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077)——原始 EAGLE 草稿头公式。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858)——自适应草稿和树。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html)——带投机解码的高效 LLM 系统。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding)——生产部署清单。
