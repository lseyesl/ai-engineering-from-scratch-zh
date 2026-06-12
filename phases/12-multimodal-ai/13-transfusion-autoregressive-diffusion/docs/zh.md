# Transfusion：一个 Transformer 中的自回归文本 + 扩散图像

> Chameleon 和 Emu3 把所有赌注押在了离散 token 上。它们有效，但量化瓶颈是可见的——图像质量低于连续空间扩散模型。Transfusion（Meta，Zhou 等人，2024 年 8 月）采取了相反的赌注：保持图像连续，完全放弃 VQ-VAE，用一个变压器训练两个损失。文本 token 获得 next-token 预测。图像 patch 获得流匹配 / 扩散损失。两个目标优化相同的权重。Stable Diffusion 3（MMDiT）的底层架构是一个近亲。本课程阅读 Transfusion 的论点，构建一个玩具双损失训练器，并追踪让一个 Transformer 同时做两项工作的注意力掩码。

**类型：** 构建
**语言：** Python（标准库，MNIST 规模玩具上的双损失训练器）
**前置要求：** Phase 12 · 11（Chameleon），Phase 8（生成式 AI）
**时间：** ~180 分钟

## 学习目标

- 设计一个在一个骨干网上运行两个损失（文本 token 的 NTP，图像 patch 的扩散 MSE）的 Transformer。
- 解释为什么图像 patch 上的双向注意力加上文本 token 上的因果注意力是正确掩码选择。
- 比较 Transfusion 风格（连续图像，扩散损失）与 Chameleon 风格（离散图像，NTP）在计算、质量和代码复杂性方面的差异。
- 说出 MMDiT 的贡献：每个块中的模态特定权重，残差流处的联合注意力。

## 问题

离散 vs 连续图像 token 的争论比 LLM 更古老。连续表示（原始像素、VAE 潜变量）保留了细节。离散 token（VQ 索引）适合 Transformer 的原生词汇表，但在量化步骤中丢失了细节。

Chameleon / Emu3 选择了离散：一个损失，一个架构，但图像保真度受 tokenizer 质量限制。

扩散模型选择了连续：卓越的图像质量，但模型与 LLM 分离，复杂的噪声调度工程，且与文本生成没有干净的集成。

Transfusion 问：我们可以两者兼得吗？保持图像连续，仍然训练一个模型，使用两个损失缝合到一个梯度步骤中。

## 概念

### 双损失架构

单个仅解码器 Transformer 处理包含以下内容的序列：

- 文本 token（离散，来自 BPE 词汇表）。
- 图像 patch（连续的，16x16 像素块通过线性嵌入投影到隐藏维度——与 ViT 编码器的输入相同）。
- `<image>` 和 `</image>` 标签，标记连续 patch 所在位置。

前向传播运行一次。每个 token 的损失从两个头中选择：

- 对于文本 token：在词汇表 logits 头上的标准交叉熵。
- 对于图像 patch：在连续 patch 上的扩散损失——预测添加到每个 patch 的噪声。

梯度流经共享的 Transformer 体。两个损失同时改进共享权重。

### 注意力掩码：因果文本 + 双向图像

文本 token 必须是因果的——你不能让文本 token 关注未来的文本，否则 teacher forcing 会失效。然而，图像 patch 代表一个快照；它们应该在同一个图像块内彼此双向关注。

掩码：

```
M[i, j] = 1 如果：
  (i 是文本 且 j 是文本 且 j <= i)             # 文本的因果掩码
  或 (i 是图像 且 j 是图像 且 same_image_block(i, j))   # 图像内双向
  或 (i 是文本 且 j 是图像 且 j < i_image_end)    # 文本关注之前的图像
  或 (i 是图像 且 j 是文本 且 j < i_image_start)   # 图像关注之前的文本
```

在训练和推理时作为块三角掩码实现。

### Transformer 内部的扩散损失

扩散损失是标准的：向图像 patch 添加噪声，要求模型预测噪声（或等价的干净 patch）。Transfusion 的版本使用流匹配——预测从噪声到干净的 velocity 场。

训练期间：
1. 对于每个图像 patch x0，采样随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（流匹配的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与来自同一序列的文本 NTP 损失一起反向传播。

推理时，生成是：
- 文本 token：标准自回归采样。
- 图像 patch：基于前面文本 token 条件的扩散采样循环（通常 10-30 步）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等人，2024 年 3 月）与 Transfusion 大约同时发布了 MMDiT（多模态扩散 Transformer）。这两个架构是兄弟姐妹。

MMDiT 的关键区别：

- 每块的模态特定权重。每个 Transformer 块对文本 token 和图像 patch 有分离的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其他一切都是模态特定的。
- 整流流训练。一种特定的流匹配变体，具有已知的采样和比 DDPM 更简单的数学。
- 规模。MMDiT 是 SD3（2B 和 8B 参数变体）的骨干网。Transfusion 的论文扩展到 7B。

两者都收敛到相同的核心思想：一个 Transformer 在文本上运行 NTP，在连续图像表示上运行扩散。

### 为什么这比 Chameleon 风格更好

连续扩散和离散 NTP 在图像生成上的质量差距是可测量的。Transfusion 论文报告：

- 在 7B 参数下，在 FID 上比相同大小的 Chameleon 风格模型好 3-5 个点。
- 不需要 tokenizer 训练——图像编码器更简单（线性投影到隐藏维度，与 ViT 的输入层相同）。
- 推理可以并行化图像 patch 去噪，与自回归图像 token 不同。

缺点：Transfusion 是一个双损失模型，使训练动态更棘手。损失权重需要调优。NTP 和扩散之间的调度不匹配可能导致一个头占主导。

### 下游的发展

Janus-Pro（课程 12.15）通过解耦用于理解和生成的视觉编码器来改进 Transfusion 的想法——SigLIP 用于一个方向，VQ 用于另一个——同时共享 Transformer 体。Show-o（课程 12.14）将扩散换为离散扩散（掩码预测）。统一生成家族在 Transfusion 之后迅速分支。

2026 年生成图像的生产 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎肯定使用这个家族的某种后代。细节是专有的。

## 使用它

`code/main.py` 在一个微小的 MNIST 类问题上构建了一个玩具 Transfusion：

- 文本描述是描述数字（0-9）的短整数序列。
- 图像是 4x4 字节网格。
- 一对共享权重的线性投影作为 Transformer 的替代品；文本上的 NTP 损失，噪声 patch 上的 MSE 损失。
- 训练循环交替两个损失，注意力掩码是显式的。
- 生成在一次前向传播中产生一个文本描述和一个 4x4 图像。

Transformer 是玩具模型。双损失管道、注意力掩码构建和推理循环是真正的产物。

## 交付物

本课程产生 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本 + 图像、文本 + 音频、文本 + 视频），它设计双损失调度（损失权重、掩码形状、共享 vs 模态特定块）并标记实现风险。

## 练习

1. 一个 Transfusion 风格模型训练 70% 文本 token 和 30% 图像 patch。图像扩散损失的数量级大约是文本 NTP 损失的 10 倍。什么损失权重能使它们平衡？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现块三角掩码。将每个条目标记为 0 或 1。

3. MMDiT 具有模态特定的 QKV 权重。与 Transfusion 的完全共享 Transformer 相比，这增加了多少参数开销？在 7B 参数下，值得吗？

4. 生成：给定一个文本提示，模型运行 NTP 得到 50 个 token，然后遇到 `<image>`，然后在 256 个 patch 上运行 20 个去噪步的扩散。总共多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述整流流以及为什么它比 DDPM 在更少的推理步骤中收敛。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 双损失训练 | "NTP + 扩散" | 单个 Transformer 在同一个梯度步骤中同时优化文本 token 上的交叉熵和连续图像 patch 上的 MSE |
| 流匹配 | "整流流" | 预测从噪声到干净数据的 velocity 场的扩散变体；比 DDPM 数学更简单 |
| MMDiT | "多模态 DiT" | Stable Diffusion 3 的架构：联合注意力，模态特定的 MLP 和归一化 |
| 块三角掩码 | "因果文本 + 双向图像" | 注意力掩码在文本上是因果的，但在图像区域内是双向的 |
| 连续图像表示 | "无 VQ" | 图像 patch 作为实值向量，而不是整数码本索引 |
| Velocity 预测 | "v-参数化" | 网络输出是噪声和数据之间的 velocity 场，而不是噪声本身 |

## 延伸阅读

- [Zhou 等人 — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser 等人 — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao 等人 — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie 等人 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
