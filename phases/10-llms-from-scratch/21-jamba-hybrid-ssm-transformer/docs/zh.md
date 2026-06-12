# Jamba — 混合 SSM-Transformer

> 状态空间模型（SSM）和 Transformer 追求的目标不同。Transformer 以二次代价换取注意力带来的质量；SSM 通过递推实现线性时间推理和常数内存，但质量落后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）将两者放入同一个模型：每 7 个 Mamba 层搭配 1 个 Transformer 层，每隔一个块应用 MoE，以及可装入单张 80GB GPU 的 256k 上下文窗口。Mamba-3（ICLR 2026）通过复数值状态空间和 MIMO 投影强化了 SSM 侧。本课程从头到尾解读这两种架构，并解释为什么混合方案在纯 SSM 和纯 Transformer 长上下文尝试都失败的情况下，历经三年规模扩展仍然存活。

**Type:** 学习
**Languages:** Python（标准库，层混合计算器）
**Prerequisites:** 第 10 阶段 · 第 14 课（开放模型架构），第 10 阶段 · 第 17 课（原生稀疏注意力）
**Time:** ~60 分钟

## 学习目标

- 解释 Jamba 块中的三种原语——Transformer 层、Mamba 层、MoE——以及 1:7:even 交错方案。
- 阐述 SSM 递推的高层形式，以及为什么它能实现常数内存推理。
- 计算 Jamba 模型在 256k 上下文下的 KV 缓存占用，并与纯 Transformer 模型所需内存进行比较。
- 说出 Mamba-3 的三项创新（指数梯形离散化、复数值状态更新、MIMO）以及每项所针对的问题。

## 问题所在

注意力机制的计算量相对于序列长度是二次的。状态空间模型是线性的。这一差异会累积放大：在 256k token 时，Transformer 的注意力映射每个头有 65B 个条目；而 SSM 的递推状态无论序列多长都是固定大小。

纯 SSM 模型（Mamba、Mamba-2）在小规模上能匹配 Transformer 的困惑度，但在状态跟踪任务上落后，且在某些上下文内检索类别上失败。直觉是：SSM 将历史压缩为固定状态，当历史很长时，信息会泄漏。注意力精确记住一切，但付出二次代价。

显而易见的修复：两者都用。在需要精确回忆的地方放置 Transformer 层，其他地方使用 SSM 层，调整比例。Jamba 是第一个以生产级规模交付此混合方案的模型（52B 总参数，12B 活跃参数，256k 上下文，单张 80GB GPU）。Jamba 1.5 将家族扩展到 398B 总参数 / 94B 活跃参数。Mamba-3（ICLR 2026）是目前最佳的纯 SSM 基线，混合模型可以围绕它重建。

本课程阅读这三篇论文，并产出"选择正确比例"的心智模型。

## 核心概念

### 一页纸讲清 SSM

状态空间模型通过固定大小的状态 `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

在每一步，状态通过线性动力学 `A` 演化，接收输入 `B x_t`，并发出输出 `C h_t`。`A, B, C` 可以学习。注意关键性质：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要任何更早的 `x`。内存是常数。推理复杂度是每 token O(1)。

建模质量的关键在于 `A` 的结构。S4（Gu 2021）使用了高度结构化的矩阵，在训练时可以作为长卷积高效求值。Mamba（Gu, Dao 2023）将固定的 `A, B, C` 替换为数据依赖的版本（"选择性"部分）。Mamba-2（2024）进一步简化了结构。Mamba-3（2026）在特定位置重新增加了复杂度。

关键性质：对于解码器 LLM，SSM 层是注意力层的直接替代品，用固定大小的逐层状态替代不断增长的 KV 缓存。

### Jamba 块

Jamba 块根据两个数字交错层：

- `l`：注意力与 Mamba 的比例。Jamba 使用 `l = 8`，即每 7 个 Mamba 层搭配 1 个 Transformer 层（7 Mamba + 1 Attention = 每组 8 层）。
- `e`：MoE 频率。Jamba 使用 `e = 2`，即每隔一层应用 MoE。

块内的层序列：

```
M M M M M M M A (7 Mamba + 1 Attention)
| M | M | M | M (其中 | 标记应用了 MoE)
```

每个 Jamba 块为 8 层。4 个块深（共 32 层），你得到 28 个 Mamba 层和 4 个 Attention 层。其中 16 层使用 MoE。

### 为什么是 1:7 比例

AI21 进行了消融实验：什么样的注意力与 Mamba 比例能在长上下文评估中给出最佳的每参数困惑度 AND 上下文内回忆？

- 注意力过多（1:1）：质量上升但内存和速度退化。
- 注意力过少（1:15）：内存很好但上下文内检索失败。
- 甜点：1:7 或 1:8。

直觉：Transformer 层处理精确回忆和状态跟踪，Mamba 层处理廉价的批量处理。

### 位置编码

Mamba 层本身具有位置感知（通过递推）。原始基于 Mamba 的混合模型中的注意力层不使用 RoPE——SSM 层提供了位置信息。Jamba 1.5 在注意力层中添加了 RoPE 以实现更长上下文的泛化，这是基于经验长上下文评估的事后改进。

### 内存预算

对于 Jamba-1 形状（32 层：28 Mamba + 4 Attention，隐藏维度 4096，32 个注意力头）：

- KV 缓存（仅注意力层）：`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`（256k BF16 时）。仅 4 个注意力层贡献。
- SSM 状态：每 token 前缀 `28 * hidden * state_size`，但这是每层固定大小，不随序列长度缩放。典型 Mamba 状态为每特征 16，隐藏维度 4096：`28 * 4096 * 16 * 2 = 3.7 MB` 总计。

对比纯 Transformer（32 层，相同隐藏维度，32 头全 MHA）：`2 * 32 * 32 * 128 * 256k * 2 = 128 GB`（256k BF16 时）。KV 缓存减少 8 倍。即使对比大多数 2024 模型使用的 GQA(8) 基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 混合方案在 16 GB 下仍小 2 倍。

这就是 AI21 所说的"单张 80GB GPU 上的 256k 上下文"。全 MHA 纯 Transformer 的 KV 缓存装不下；即使 GQA 基线也没有空间留给权重和激活；Jamba 可以。

### Mamba-3：2026 年的纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 侧引入了三项创新：

1. **指数梯形离散化。** 用更具表达力的递推替代 Mamba-2 中的欧拉方法离散化。在核心递推内部对状态输入应用类卷积操作，而非对 `x_t` 的外部卷积。

2. **复数值状态更新。** 之前的 Mamba 将状态矩阵从复数（S4）简化为实对角（Mamba）再到缩放单位矩阵（Mamba-2）。Mamba-3 重新引入复数值——等价于状态上的数据依赖旋转位置嵌入。这恢复了先前实值简化所损失的状态跟踪能力。

3. **多输入多输出（MIMO）投影。** 不使用逐特征的标量投影，而是使用矩阵值投影。在不增加解码延迟的情况下提升建模能力和推理时硬件利用率。

在 1.5B 参数规模下，Mamba-3 相比 Gated DeltaNet 平均下游准确率提升 0.6 个百分点；MIMO 变体再增加 1.2 个百分点，总计 1.8 个百分点增益。在相同状态大小下，Mamba-3 用一半状态即可匹配 Mamba-2。

Mamba-3 尚未在大规模生产混合模型中部署——但它显然是下一代 Jamba 级模型 SSM 侧的候选者。

### 何时选择混合架构

混合架构在以下情况胜出：

- 上下文足够长，纯 Transformer 的 KV 缓存变得痛苦（64k+）。
- 任务混合了短程结构（适合 SSM）和长程回忆（需要 Transformer）。
- 你想在单 GPU 内存预算上部署，而 Transformer 的 KV 缓存单独就装不下。

混合架构在以下情况失败：

- 上下文很短（16k 以下）。SSM 开销是浪费的；纯 Transformer 就行。
- 任务需要处处到处的注意力（深度推理、多文档交叉引用）。混合架构中注意力层的稀疏性会造成损害。
- 你在扩展到万亿参数的前沿模型。纯 Transformer + MLA + MoE（DeepSeek-V3 风格）目前在能力竞赛中领先。

### 竞争格局

| 模型 | 家族 | 规模 | 独特主张 |
|-------|--------|------|-------------|
| Mamba-2 | 纯 SSM | 3B | 线性时间，常数内存 |
| Jamba | 混合 | 52B/12B | 80GB 上的 256k |
| Jamba 1.5 Large | 混合 | 398B/94B | 企业级长上下文 |
| Mamba-3 | 纯 SSM | 1.5B（论文） | 状态跟踪恢复 |
| DeepSeek-V3 | 纯 Transformer + MoE | 671B/37B | 前沿能力 |

2026 年格局：纯 Transformer MoE 主导前沿，但混合架构占据 256k 以上上下文利基。Mamba-3 的状态跟踪增益可能推动下一代混合比例更低（更多 SSM，更少注意力）。

```figure
swiglu-ffn
```

## 使用它

`code/main.py` 是混合架构的内存计算器。给定 SSM-Transformer 比例和隐藏维度/层数配置，它计算：

- 目标上下文下的 KV 缓存。
- SSM 状态内存。
- 一系列模型形状在上下文 N 下的总内存。

计算器支持：

- 纯 Transformer 基线（KV 缓存随 N 增长）。
- Jamba 风格 1:7 混合。
- 纯 SSM（完全没有 KV 缓存）。

数字直接来自 Jamba-1 和 Jamba-1.5 论文中的已发布形状，对假设变体进行外推。

实际部署的集成考虑：

- 大多数生产推理服务器（vLLM、SGLang）支持 Jamba 和 Mamba。检查具体版本。
- 在 256k 上下文下，Jamba 的内存优势体现在并发请求吞吐量。在相同 VRAM 上，你可以装入比 Transformer 序列更多的 Jamba 序列。
- Mamba-3 作为独立模型尚未投入生产——1.5B 研究预览。

## 交付它

本课程产出 `outputs/skill-hybrid-picker.md`。给定工作负载规格（上下文长度分布、任务组合、内存预算），它在纯 Transformer、Jamba 风格混合和纯 SSM 之间推荐，并附带关于内存和质量权衡的明确推理。

## 练习

1. 运行 `code/main.py` 计算 32 层纯 Transformer（隐藏维度 4096，32 头）和相同形状的 Jamba-1 混合在 256k 上下文下的 KV 缓存。验证 AI21 论文声称的约 8 倍内存减少。

2. 修改计算器以建模 1:3 混合（4 Mamba : 1 Attention）和 1:15 混合（14 Mamba : 1 Attention）。绘制 KV 缓存与比例的关系。在什么比例下 KV 缓存等于 SSM 状态内存？

3. 阅读 Jamba 论文第 3 节（arXiv:2403.19887）。解释为什么 AI21 使用 Mamba-1 而非 Mamba-2，尽管 Mamba-2 更快。提示：混合消融部分记录了这一点。

4. 计算 Jamba 1.5 Large（398B 总参数，94B 活跃参数）中每隔一层 MoE 的参数开销。将活跃比例与 DeepSeek-V3（37B/671B）比较，并解释为什么 Jamba 的架构推动活跃比例更高。

5. 阅读 Mamba-3 论文第 3 节（arXiv:2603.15569）。用三句话解释为什么复数值状态更新等价于数据依赖的旋转位置嵌入。将答案与第 7 阶段 · 第 04 课的 RoPE 推导联系起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 状态空间模型（SSM） | "带固定状态的递推" | 具有学习递推 `h_t = A h_{t-1} + B x_t` 的层；每 token 常数内存 |
| 选择性 SSM | "Mamba 的技巧" | 数据依赖的 A, B, C 参数，在线性时间下赋予模型类似门控的选择性 |
| 注意力与 Mamba 比例 | "有多少注意力层" | 在 Jamba 中，`l = 8` 意味着每 7 个 Mamba 层搭配 1 个注意力层 |
| Jamba 块 | "8 层组" | 一个注意力 + 七个 Mamba + 交替位置上的 MoE |
| SSM 状态 | "隐藏缓冲区" | 替代 Mamba 层 KV 缓存的固定大小逐层状态 |
| 256k 上下文 | "Jamba 的旗舰数字" | Jamba-1 在单张 80GB GPU 上适配的序列长度；纯 Transformer 在该规模下无法做到 |
| Mamba-3 | "2026 纯 SSM" | 当前最佳纯 SSM 架构，具有复数状态 + MIMO；混合模型围绕它重建的基线 |
| MIMO | "多输入多输出" | Mamba-3 创新，使用矩阵值投影替代逐特征标量 |
| 指数梯形离散化 | "Mamba-3 的递推" | 更具表达力的递推，包含 Mamba-2 的欧拉方法离散化 |
| 混合架构 | "混合注意力和 SSM" | 任何交错 Transformer 和 SSM 层的模型；Jamba 是生产原型 |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 原始 Jamba 论文，比例消融，256k 上下文声明
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — 放大版家族，398B/94B 和 12B/52B 公开发布
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba 所基于的选择性 SSM 论文
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — 简化的结构化状态空间后继
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — 复数值状态，MIMO，2026 纯 SSM 前沿
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 论文，SSM 谱系在 LLM 领域的起点
