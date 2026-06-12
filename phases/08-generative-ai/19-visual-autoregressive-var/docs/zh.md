# 视觉自回归建模（VAR）：下一尺度预测

> 扩散模型在时间上迭代采样（去噪步骤）。VAR 在尺度上迭代采样——它预测 1x1 token，然后 2x2，然后 4x4，直到最终分辨率，每个尺度以前一个尺度为条件。2024 年的论文表明，VAR 在图像生成上匹配 GPT 风格的缩放定律，并在相同计算预算下击败 DiT。本课构建核心机制。

**类型：** 构建
**语言：** Python（使用 PyTorch）
**前置知识：** 阶段 7 第 3 课（多头注意力），阶段 8 第 6 课（DDPM）
**时间：** ~90 分钟

## 问题

自回归生成主导了语言建模，因为它可以预测地扩展：更多计算、更多参数、更低的困惑度、更好的输出。在 2024 年之前，图像生成有两次主要的 AR 尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE 编码上逐 token）。

两者都受到生成顺序问题的困扰。像素和 token 以 2D 网格排列，但 AR 模型必须以 1D 光栅顺序访问它们。早期的角落像素不知道图像最终会变成什么。生成质量的扩展比 GPT-on-text 差，并且在匹配的计算量下从未达到扩散模型的质量。

VAR 通过改变生成的内容来修复生成顺序问题。不是在空间上一个接一个地预测图像 token，VAR 以递增分辨率预测整个图像。第 1 步：预测 1x1 token（整体图像"摘要"）。第 2 步：预测 2x2 token 网格（粗粒度特征）。第 3 步：预测 4x4 网格。第 K 步：预测最终的 (H/8)x(W/8) 网格。

每个尺度关注所有先前的尺度（在"尺度顺序"中因果地）并在自身尺度内并行处理。顺序问题消失了：尺度 k 的整个图像在一次 Transformer 传递中产生。

## 概念

### VQ-VAE 多尺度分词器

VAR 需要一个**多尺度离散分词器**。对于图像 x，它产生一系列分辨率逐渐升高的 token 网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用相同的码本（典型大小为 4096-16384）。每个尺度的分词不是独立的——它的训练使得每个尺度的残差总和重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一个**残差 VQ** 变体。尺度 k 捕捉尺度 1..k-1 遗漏的内容。解码器接收所有尺度嵌入的和并生成图像。

多尺度 VQ 分词器训练一次（像 VQGAN 一样）然后冻结。所有生成工作由之上的自回归模型完成。

### 下一尺度预测

生成模型是一个 Transformer，它看到来自所有先前尺度的 token 并预测下一尺度的 token。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入同时编码尺度索引和尺度内的空间位置。注意力在尺度顺序上是因果的：尺度 k、位置 (i, j) 的 token 可以关注所有尺度 1..k 的 token，以及尺度 k 本身中在各自尺度内顺序中较早出现的 token（VAR 使用固定位置注意力，没有尺度内因果性——一个尺度内的所有位置并行预测）。

训练损失：在每个尺度 k，给定所有先前尺度的 token 预测 token z_k。在离散 VQ 编码上的交叉熵损失。结构与 GPT 相同，只是"序列"现在按照尺度结构排列。

### 生成

推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于 K = 10 个尺度，生成是 10 次 Transformer 前向传播。每次传播并行产生其整个尺度——尺度内没有逐个 token 的自回归。对于 256x256 图像，这大约是 10 次传递 vs DiT 的 28-50 次。

### 为什么下一尺度胜过下一 token

三个结构性优势：
1. **由粗到细与自然图像统计一致。** 人类视觉感知和图像数据都表现出尺度相关的规律性：低频结构稳定且可预测；高频细节以低频内容为条件。下一尺度预测利用了这一点。
2. **尺度内并行生成。** 与 GPT 风格的 token AR 不同，VAR 在一个步骤中产生一个尺度的所有 token。有效生成长度是对数尺度而非线性。
3. **没有生成顺序偏差。** 尺度 k 的 token 看到尺度 k-1 的全部；没有"左侧"或"上方"偏差迫使早期 token 在后期上下文可用之前做出承诺。

### 缩放定律

Tian 等人展示了 VAR 在 ImageNet 上遵循 FID 的幂律缩放曲线——就像 GPT 对困惑度所做的那样。将参数或计算量加倍可靠地将误差减半。这是第一个像语言模型一样清晰地展示这种缩放行为的图像生成模型。结果是 VAR 尺度的预测变得可以通过计算量预测，而不是根据架构进行经验猜测。

### 与扩散的关系

VAR 和扩散共享相同的数据压缩故事：两者都将生成问题分解为一系列更简单的子问题。

- 扩散：逐渐添加噪声，学习撤销一步。
- VAR：逐渐添加分辨率，学习预测下一个尺度。

它们是问题的不同轴。两者都产生可处理的条件分布。经验上 VAR 在推理时更快（更少的传递，尺度内全部并行），并在类别条件 ImageNet 上匹配或击败 DiT。文本条件 VAR（VARclip、HART）是一个活跃的研究方向。

## 动手实现

在 `code/main.py` 中你将：
1. 在合成"图像"数据（2D 高斯环）上构建一个微型**多尺度 VQ 分词器**。
2. 训练一个**VAR 风格 Transformer** 进行下一尺度 token 预测。
3. 通过调用 Transformer 4 次（4 个尺度）并解码来采样。
4. 验证尺度有序训练使得生成在尺度内并行。

这是一个玩具实现。重点是要看到尺度结构化的注意力掩码和尺度内并行的生成实际工作。

## 产出

本课产生 `outputs/skill-var-tokenizer-designer.md`——一个设计多尺度分词器的技能：尺度数量、尺度比例、码本大小、残差共享、解码器架构。

## 练习

1. **尺度数量消融。** 用 4、6、8、10 个尺度训练 VAR。测量重建质量与自回归传递次数的关系。更多尺度 = 更精细的残差 = 更好的质量但更多的传递。

2. **码本大小。** 用码本大小 512、4096、16384 训练分词器。更大的码本提供更好的重建但更难的预测。找到拐点。

3. **尺度内并行检查。** 对于训练好的 VAR，显式测量注意力模式。在尺度 k 内，模型是否关注跨尺度的位置但不关注尺度内位置？验证掩码实现。

4. **VAR vs DiT 缩放。** 对于相同的 ImageNet 类别条件任务，在匹配的参数预算下训练 VAR 和 DiT（例如 33M、130M、458M）。绘制 FID 与计算量的关系。VAR 应在每个大小上都领先于 DiT——在小规模上复现论文的结果。

5. **文本条件化。** 扩展 VAR 以通过 adaLN 接受文本嵌入（CLIP 池化）作为额外条件输入。这是 HART 的配方。FID 在文本对齐采样上提高了多少？

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|----------------|----------------------|
| VAR | "视觉自回归" | 通过在 VQ 网格金字塔上进行下一尺度预测来生成图像 |
| 下一尺度预测 | "先预测粗的，再预测细的" | 模型以递增分辨率预测 token，以所有先前尺度为条件 |
| 多尺度 VQ 分词器 | "残差 VQ" | 产生 K 个递增分辨率 token 网格的 VQ-VAE，解码器将所有尺度和相加 |
| 尺度 k | "金字塔级别 k" | K 个分辨率级别之一，从 k=1 的 1x1 到 k=K 的 (H/p)x(W/p) |
| 尺度内并行 | "每尺度一次前向传播" | 一个尺度的所有 token 在一次 Transformer 传递中预测，而非自回归 |
| 跨尺度因果 | "尺度有序注意力" | 尺度 k 的 token 可以关注所有尺度 1..k 但不能关注尺度 k+1..K |
| 残差 VQ | "相加式分词" | 每个尺度的 token 编码低尺度留下的残差；解码器将所有尺度和相加 |
| VAR 缩放定律 | "图像 GPT 缩放" | FID 遵循计算量的可预测幂律，像语言模型的困惑度 |
| HART | "混合 VAR + 文本" | 文本条件 VAR 变体，结合了 MaskGIT 风格的迭代解码与 VAR 的尺度结构 |
| 尺度位置嵌入 | "(scale, row, col) 三元组" | 位置编码同时携带尺度索引和尺度内的空间坐标 |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，权威参考
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，扩散比较基线
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 多尺度分词器所扩展的注词器家族
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像分词的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — 文本条件 VAR
