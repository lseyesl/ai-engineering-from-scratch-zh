# 任意分辨率视觉：Patch-n'-Pack 与 NaFlex

> 真实的图像不是 224x224 的方块。收据是 9:16，图表是 16:9，医学扫描可能是 4096x4096，手机截图是 9:19.5。2024 年之前 VLM 的答案——将所有内容缩放到固定方块——丢弃了使 OCR、文档理解和高分辨率场景解析工作的信号。NaViT（Google，2023 年）展示了你可以将可变分辨率的 patch 打包到一个 Transformer batch 中，使用块对角掩码。Qwen2-VL 的 M-RoPE（2024 年）完全抛弃了绝对位置表。LLaVA-NeXT 的 AnyRes 将高分辨率图像平铺成基础 + 子图像。SigLIP 2 的 NaFlex 变体（2025 年）现在是希望用单个 checkpoint 服务所有宽高比的开源 VLM 的默认编码器。本课程从头到尾实现 patch-n'-pack。

**类型：** 构建
**语言：** Python（标准库，patch 打包器 + 块对角掩码）
**前置要求：** Phase 12 · 01（ViT patch），Phase 12 · 05（LLaVA）
**时间：** ~120 分钟

## 学习目标

- 将一批可变分辨率图像的 patch 打包成一个序列，并构建块对角注意力掩码。
- 为给定任务在 AnyRes 平铺（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间做出选择。
- 在不进行缩放的条件下计算 OCR、图表和摄影的 token 预算。
- 说出方形缩放的三种失败模式：文本被压扁、内容被裁剪、填充浪费 token。

## 问题

Transformer 期望一个序列。一个 batch 是一组相同长度的序列堆叠。如果你的图像是 224x224，每次得到 196 个 patch token，不需要填充，搞定。在 224 上训练，在 224 上推理，再也不用考虑分辨率。

但现实世界不配合。文档是竖屏的（8.5x11 英寸，大约 2:3）。图表截图是横屏的（16:9）。收据又高又窄（1:3）。医学成像的分辨率是 2048x2048 或更大。移动设备截图是 1170x2532（0.46:1）。

2024 年之前的三个选项以及为什么每个都失败：

1. 缩放到固定方形（224x224 或 336x336）。挤压会扭曲文本和面部。缩小会破坏图表标签和 OCR 内容。这是 LLaVA-1.5 之前的标准做法。
2. 裁剪到固定宽高比。你丢掉了大部分图像，而且选择裁剪位置本身就是一个视觉问题。
3. 填充到最长边。修复了畸变，但对于竖屏图像浪费了 50% 以上的 token 在填充上。所有填充 token 带来二次注意力成本。

2024-2025 年的答案：让 Transformer 以图像的原生分辨率消费 patch，并解决如何将异构 batch 打包成一个序列而不浪费计算的问题。

## 概念

### NaViT 和 patch-n'-pack

NaViT（Dehghani 等人，2023 年）是一篇证明这可以在规模上工作的论文。想法是机械性的：

1. 对于 batch 中的每张图像，在选定的 patch 大小（比如 14）下计算其原生 patch 网格。
2. 将每张图像的 patch 展平成自己的可变长度序列。
3. 将所有图像的 patch 拼接成 batch 的一个长序列。
4. 构建一个块对角注意力掩码，使图像 A 的 patch 只在图像 A 内部关注。
5. 携带每个 patch 的位置信息（2D RoPE 或分数位置嵌入）。

一批三张图像：336x336（576 个 token）、224x224（256 个 token）和 448x336（768 个 token），变成一个 1600 个 token 的序列，带有一个 1600x1600 的块对角掩码。没有填充。没有浪费的计算。Transformer 处理任意宽高比。

NaViT 还引入了分数 patch 丢弃训练法——在整个 batch 中随机丢弃 50% 的 patch——既能正则化又能加速训练。SigLIP 2 继承了这一点。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实的替代方案。给定一个高分辨率图像和一个固定编码器（CLIP 或 SigLIP 在 336 分辨率下），将图像平铺：

1. 从预定义集合中选择一个网格布局——(1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等——最适合图像的宽高比。
2. 将完整图像平铺到网格中；每个平铺变成 336x336 的裁剪。
3. 同时生成一个缩略图：将整张图像缩放到 336x336 作为全局上下文 token。
4. 将每个平铺通过冻结的 336 编码器编码。拼接平铺 token + 缩略图 token。

对于 2x2 网格加缩略图的 672x672 图像：4 * 576 + 576 = 2880 个视觉 token。昂贵但有效——LLM 同时看到局部细节和全局上下文。

当你的编码器冻结且只支持一种分辨率时，AnyRes 是首选方案。它会使大图像的 token 数量激增（1344x1344 图像在 4x4 网格下是 9216 + 576 ≈ 9800 个 token，填满了大多数 8k LLM 上下文）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了多模态旋转位置嵌入。不同于 NaViT 的分数位置或 AnyRes 的平铺加缩略图，每个 patch 携带一个 3D 位置（时间、高度、宽度）。query/key 旋转处理任意的 H、W 和时间长度。

M-RoPE 无需重新训练就能提供原生动态分辨率。在推理时，你提供任意 HxW 的图像，patch 嵌入器产生 H/14 x W/14 个 token，每个 token 获得其（t=0，r=行，c=列）位置，RoPE 以正确的频率旋转注意力，完成。Qwen2.5-VL 和 Qwen3-VL 延续了这一做法。InternVL3 的 V2PE 是相同的想法，但每个模态使用不同的编码。

与 AnyRes 不同，M-RoPE 的 token 数量是 O(H x W / P^2)，以原生分辨率计算——没有乘性的平铺开销。与 NaViT 不同，它仍然期望每次前向传播只有一张图像。跨分辨率的 batch 处理仍然需要 patch-n'-pack 在其之上。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的原生灵活模式。单个模型在推理时服务多种序列长度（256、729、1024 个 token）。内部在训练时使用 NaViT 风格的 patch-n'-pack，每个 patch 使用绝对分数位置。卖点：一个 checkpoint，根据任务在推理时选择你的 token 预算。

对于语义任务（分类、检索），256 个 token。对于 OCR 或图表理解，1024 个 token。无需重新训练。

### 打包掩码

块对角掩码是大多数实现出错的地方。对于覆盖图像 `i=0..B-1`、长度为 `n_i` 的总长度为 `N_total` 的打包序列，形状为 `(N_total, N_total)` 的掩码 `M` 在两个索引落在同一图像的块内时为 1，否则为 0。你可以从累积长度列表构建它：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff 存在 b 使得 offsets[b] <= i < offsets[b+1] 且 offsets[b] <= j < offsets[b+1]
```

在 PyTorch 中，这可以用 `torch.block_diag` 或显式的 gather 在一行内完成。FlashAttention 的可变长度路径（`cu_seqlens`）完全跳过掩码，使用累积长度张量直接在序列内部关注——对于典型 batch，比密集掩码快约 10 倍。

### Token 预算

根据任务选择你的策略：

- OCR / 文档：1024-4096 个 token。SigLIP 2 NaFlex 在 1024 下，或 AnyRes 3x3 + 缩略图。
- 图表和 UI：384-448 原生分辨率下 729-1024 个 token。Qwen2.5-VL 动态分辨率，带最大像素上限。
- 自然照片：256-576 个 token 就够了。下游 LLM 看到足够的信息。在内容密度高的地方支付 token 成本。
- 视频：空间池化后每帧 64-128 个 token，2-8 FPS。课程 12.17 涵盖这一点。

2026 年的生产规则：为每个任务选择一个每任务最大像素上限，以原生宽高比编码直到该上限，打包 batch，跳过填充。Qwen2.5-VL 提供了 `min_pixels` 和 `max_pixels` 来实现这个旋钮。

## 使用它

`code/main.py` 为一批异构图像实现了 patch-n'-pack，使用整数像素坐标。它：

- 接收一个（H，W）图像大小列表。
- 计算每张图像在 patch 大小 14 下的 patch 序列长度。
- 将它们打包成一个总长度为 `sum(n_i)` 的序列。
- 构建块对角注意力掩码（密集形式，为清晰起见）。
- 比较打包成本与方形缩放和 AnyRes 平铺。
- 打印混合 batch（收据、图表、截图、照片）的 token 预算表。

运行它。得出的数字就是为什么每个 2026 年的开源 VLM 都使用 patch-n'-pack 的原因。

## 交付物

本课程产生 `outputs/skill-resolution-budget-planner.md`。给定一个混合宽高比的工作负载（OCR、图表、照片、视频帧）和总 token 预算，它选择合适的策略（NaFlex、AnyRes、M-RoPE 或固定方形）并输出每个请求的配置。当你在为产品确定 VLM 规模时使用此技能——它可以防止潜在地 10 倍 token 激增破坏延迟预算。

## 练习

1. 一张收据是 600x1500（1:2.5）。在 patch 大小 14 下，原生的分辨率 token 有多少？在方形缩放到 336 后呢？哪个在实践中会丢失更多 OCR 准确性？

2. 为一批四张图像构建块对角掩码，长度分别为 256、576、729、1024。验证注意力矩阵是 2585x2585，并且恰好有 `256^2 + 576^2 + 729^2 + 1024^2` 个非零条目。

3. 对于一张 1792x896 的图像在 patch 14 下，比较：（a）方形缩放到 336 然后编码，（b）AnyRes 2x1 + 缩略图，（c）原生的 M-RoPE。哪个使用最少的 token？哪个保留最多细节？

4. 实现分数 patch 丢弃：给定一个打包序列，均匀随机丢弃 50% 的 token，并相应地更新块对角掩码。测量掩码稀疏度的变化。

5. 阅读 Qwen2-VL 论文（arXiv:2409.12191）的第 3.2 节。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么，以及为什么两个边界都很重要。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| Patch-n'-pack | "NaViT 风格打包" | 将不同图像的可变长度 patch 序列拼接成一个 batch 维度 |
| 块对角掩码 | "打包掩码" | 限制每张图像的 patch 只关注自己、不关注包中相邻图像注意力的掩码 |
| AnyRes | "LLaVA-NeXT 平铺" | 将高分辨率图像分割成固定大小的平铺网格加全局缩略图；用固定编码器编码每个平铺 |
| NaFlex | "SigLIP 2 原生灵活" | 单个 SigLIP 2 checkpoint，推理时无需重训练可服务 256/729/1024 的 token 预算 |
| M-RoPE | "多模态 RoPE" | 3D 旋转位置编码（时间、行、列），无需位置表处理任意 H、W、T |
| cu_seqlens | "FlashAttention 打包" | FlashAttention 可变长度路径使用的累积长度张量，替代密集的块对角掩码 |
| min_pixels / max_pixels | "分辨率边界" | Qwen2.5-VL 的每请求旋钮，限制非常小或非常大的输入上的 token 数量 |
| 视觉 token 预算 | "每张图像多少 token" | 每张图像发出的 patch token 的粗略计数；确定 LLM 的提示预算和注意力成本 |

## 延伸阅读

- [Dehghani 等人 — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang 等人 — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon 等人 — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
