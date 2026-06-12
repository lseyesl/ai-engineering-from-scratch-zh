# Flamingo 与用于少样本 VLM 的门控交叉注意力

> DeepMind 的 Flamingo（2022 年）在此之前做了两件事。它展示了一个单一的模型可以处理任意交错的图像、视频和文本序列。它还展示了 VLM 可以学会上下文学习——给出一个包含三个示例（图像、描述）对的少样本提示，模型可以对一张新图像生成描述，无需任何梯度步骤。其机制是：门控交叉注意力层，插入在冻结的 LLM 现有层之间，带有一个从零开始的学习 tanh 门控，使得 LLM 的文本能力在初始化时得以保留。本课程讲解 Flamingo 的 Perceiver resampler 和门控交叉注意力架构——Gemini 的交错输入和 Idefics2 的视觉 token 的祖先。

**类型：** 学习
**语言：** Python（标准库，门控交叉注意力 + Perceiver resampler 演示）
**前置要求：** Phase 12 · 03（BLIP-2 Q-Former）
**时间：** ~120 分钟

## 学习目标

- 解释门控交叉注意力如何通过 tanh(gate) = 0 在初始化时保留冻结 LLM 的文本能力。
- 讲解 Perceiver resampler：N 个图像 patch → 通过交叉注意力得到 K 个固定的"潜变量"查询。
- 描述 Flamingo 如何使用因果掩码处理交错的图文序列，同时尊重图像的位置。
- 重现一个少样本多模态提示结构（3 个图像-描述示例，然后一个查询图像）。

## 问题

BLIP-2 将 32 个视觉 token 送入冻结 LLM 的输入层。适用于每个提示一张图像。但如果你想在提示中交错放置*许多*图像和文本，比如"这里是图像 A，描述它；这里是图像 B，描述它；现在这里是图像 C，描述它"，该怎么办？LLM 的自注意力需要在一个流中处理图像 token 和文本 token，而且哪些位置可以关注哪些图像的问题变得棘手。

Flamingo 的答案是：不要改变 LLM 的输入流。在现有的 LLM 块之间插入额外的交叉注意力层。文本 token 仍然像往常一样流经 LLM 的因果自注意力。在每几个 LLM 块之间，文本 token 还通过一个新的门控层交叉关注图像特征。门控（初始化为零）意味着在第零步，新层是空操作——模型的行为完全像预训练的 LLM。随着训练的进行，门控打开，视觉信息开始流动。

Flamingo 回答的第二个问题是：如何处理每个提示中可变数量的图像（0、1 或多个）？一个 Perceiver resampler——一个小型交叉注意力模块，接收任意数量的 patch 并产生固定数量的视觉潜变量 token。无论提示中有多少图像，LLM 交叉注意力层看到的是相同的形状。

## 概念

### 冻结的 LLM

Flamingo 从一个冻结的 Chinchilla 70B LLM 开始。所有 70B 权重保持不变。现有的文本自注意力和 FFN 正常运行。

### Perceiver resampler

对于提示中的每张图像，ViT 产生 N 个 patch token。Perceiver resampler 有 K 个固定的可学习潜变量（Flamingo 使用 K=64）。每个 resampler 块有两个子步骤：

1. 交叉注意力：K 个潜变量关注 N 个 patch token（Q 来自潜变量，K/V 来自 patch）。
2. 潜变量内部的自注意力 + FFN。

经过 6 个 resampler 块后，输出是 K=64 个维度为 1024 的视觉 token，无论 ViT 产生了多少个 patch。一张 224x224 的图像（196 个 patch）和一张 480x480 的图像（900 个 patch）都以 64 个 resampler token 输出。

对于视频，resampler 在时间维度上应用：每帧的 patch 产生 64 个潜变量，一个时间位置编码让模型可以区分 t=0 和 t=N。完整的视频变成 T * 64 个视觉 token。

### 门控交叉注意力

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的门控交叉注意力块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个可学习的标量，初始化为零。
- `tanh(0) = 0`，所以在初始化时门控分支贡献为零。
- 随着 `alpha` 远离零，交叉注意力的贡献平滑增长。
- 残差连接意味着即使门控完全打开，也不会覆盖 LLM 的文本表示；它只是在顶部添加视觉信息。

这是 Flamingo 中最重要的设计选择：视觉条件化是加性的、门控的、并在初始化时为零。第 0 步的 Flamingo 在纯文本输入上是一个完美的 Chinchilla 70B。

### 交错输入的掩码交叉注意力

在像 "<图像 A> 描述 A <图像 B> 描述 B <图像 C> ?" 这样的提示中，每个文本 token 应该只看到序列中在它之前的图像。交叉注意力掩码强制执行：位置 `t` 处的文本 token 只关注图像索引 `i < i_t` 的图像 resampler token，其中 `i_t` 是位置 `t` 之前最近的图像。"只看到前一张图像"或"看到所有前面的图像"都是有效的选择；Flamingo 选择了前者。

### 上下文少样本学习

一个 Flamingo 提示看起来像这样：

```
<图像1> 一张猫的照片。<图像2> 一张狗的照片。<图像3> 一张
```

模型看到补全模式并输出"鸟"（或图像3显示的任何内容）。没有梯度步骤。冻结 LLM 的上下文学习能力通过门控交叉注意力传递——这是论文的核心要点，也是它重要的原因。

### 训练数据

Flamingo 在三个数据集上训练：

1. 多模态大规模网络（M3W）：4300 万个带有交错图像和文本的网页，重构阅读顺序。
2. 图文对（ALIGN + LTIP）：44 亿对。
3. 视频-文本对（VTP）：2700 万个短视频片段。

OBELICS（2023 年）是交错网络语料库的开源复现，Idefics、Idefics2 和大多数开源的"类 Flamingo"模型都在其上训练。

### OpenFlamingo 和 Otter

OpenFlamingo（2023 年）是开源复现。架构相同（Perceiver resampler + 对冻结 LLaMA 或 MPT 的门控交叉注意力）。提供 3B、4B、9B 的 checkpoint。由于基础 LLM 更小、数据更少，质量落后于 Flamingo。

Otter（2023 年）基于 OpenFlamingo，使用 MIMIC-IT（一个多模态指令数据集）进行指令调优，表明门控交叉注意力也适用于指令跟随。

### 后代

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力系列，逐渐简化（Idefics2 放弃了 resampler，改用直接的 patch token 加自适应池化）。
- Flamingo 到 Chameleon 的过渡：到 2024 年，许多团队转向了早期融合（课程 12.11）；Flamingo 风格的门控交叉注意力在需要冻结骨干网的生产环境中仍然存在。
- Gemini 的交错输入：概念上继承了 Flamingo 的交错格式灵活性，尽管具体机制是专有的。

### 与 BLIP-2 的比较

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥梁 | 输入处一个 Q-Former | 每 M 层门控交叉注意力 |
| 视觉 token | 每张图像 32 个 | 每张图像每交叉注意力层 64 个 |
| 冻结 LLM | 是 | 是 |
| 少样本上下文学习 | 弱 | 强——论文的核心 |
| 交错输入 | 无原生支持 | 是，设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错页面 |
| 参数量 | 1.88 亿训练 | 约 100 亿训练（交叉注意力层） |
| 计算资源 | 8 个 A100 几天 | 数千个 TPUv4 数周 |

预算有限时选 BLIP-2 进行单图像 VQA。需要交错、少样本或多图像推理时选 Flamingo/Idefics2。

## 使用它

`code/main.py` 演示了：

1. 一个 Perceiver resampler，在 36 个假 patch token 上使用 8 个可学习潜变量（纯 Python 交叉注意力）。
2. 一个门控交叉注意力步骤，`alpha = 0` → 输出等于输入（LLM 不变），然后 `alpha = 2.0` → 视觉贡献混合进来。
3. 一个交错掩码构建器，为"(图像 1) (文本 1) (图像 2) (文本 2)"序列生成 2D 注意力掩码。

## 交付物

本课程产生 `outputs/skill-gated-bridge-diagnostic.md`。给定一个开源 VLM 的配置（resampler 是/否、交叉注意力频率、门控方案），它识别 Flamingo 血统元素并解释冻结策略。在调试为什么微调降低了文本性能时很有用（答案：门控变得太快太宽）。

## 练习

1. 计算 Flamingo-9B 的视觉参数量：9B LLM + 1.4B 门控交叉注意力层 + 64M resampler。总参数中训练部分占多少比例？

2. 在 PyTorch 中实现门控残差 `y = tanh(alpha) * cross + x`。实验证明，当 `alpha=0` 时，初始化时 `y==x` 精确成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390），了解当每个提示的图像数量不同时，它们如何处理 batch 中的多张图像。描述填充策略。

4. 为什么 Flamingo 的交叉注意力掩码让文本 token 只关注*最近的*前一张图像，而不是所有前面的图像？阅读 Flamingo 论文第 2.4 节并解释权衡。

5. 上下文少样本：为一个新的 Flamingo 变体构建一个包含 4 个"图像 → 主要物体颜色"示例的提示。描述当你将示例数量从 0 变化到 8 时的预期准确率模式。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| Perceiver resampler | "固定潜变量交叉注意力" | 从可变数量的输入 patch 中产生 K 个固定 token 的模块 |
| 门控交叉注意力 | "Tanh 门控桥梁" | 残差层 `y = tanh(alpha)*cross + x`，alpha 可学习，初始化为 0 |
| 交错输入 | "混合序列" | 图像和文本按阅读顺序自由混合的提示格式 |
| 冻结 LLM | "无 LLM 梯度" | 文本 LLM 的权重不更新；只有 resampler + 交叉注意力层训练 |
| 少样本 | "上下文示例" | 在提示中给出几个（图像，答案）pair；模型无需微调即可泛化 |
| OBELICS | "交错网络语料库" | 包含 1.41 亿网页的开源数据集，图像和文本按阅读顺序排列 |
| Chinchilla | "70B 冻结基础" | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| 门控计划 | "alpha 如何移动" | 训练期间交叉注意力门控打开的速度 |
| 交叉注意力频率 | "每 M 层" | 门控交叉注意力块的插入频率；Flamingo 使用 M=4 |
| OpenFlamingo | "开源复现" | MosaicML/LAION 的开源 checkpoint，3-9B；架构与 Flamingo 相同 |

## 延伸阅读

- [Alayrac 等人 — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla 等人 — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现。
- [Laurençon 等人 — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网络语料库。
- [Jaegle 等人 — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li 等人 — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 指令调优的 Flamingo 后代。
- [Laurençon 等人 — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代简化。
