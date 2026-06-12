# Chameleon 与早期融合纯 Token 多模态模型

> 我们迄今看到的所有 VLM 都保持图像和文本分离。视觉 token 来自视觉编码器，流入投影器，然后在 LLM 内部与文本汇合。视觉和文本词汇表从不重叠。Chameleon（Meta，2024 年 5 月）问道：如果它们重叠呢？训练一个 VQ-VAE，将图像转换为来自共享词汇表的离散 token 序列。现在每个多模态文档是一个序列——文本 token 和图像 token 交错排列，一个单一的自回归损失。副作用：模型可以在一次推理调用中生成混合模态输出——交替输出文本和图像 token。本课程阅读早期融合论题，并从头到尾构建一个玩具版本。

**类型：** 构建
**语言：** Python（标准库，VQ-VAE tokenizer + 交错解码器）
**前置要求：** Phase 12 · 05，Phase 8（生成式 AI）
**时间：** ~180 分钟

## 学习目标

- 解释为什么共享词汇表 + 单一损失会改变模型能做什么。
- 描述 VQ-VAE 如何将图像 token 化为与 Transformer 的 next-token 目标兼容的离散序列。
- 说出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 放置、LayerNorm 排序。
- 比较 Chameleon vs BLIP-2 的 Q-Former 方法，并描述每种方法何时是合适的选择。

## 问题

基于适配器的 VLM（LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两种不同的东西。文本 token 经过 `embed(text_token)`；图像经过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，在中间某处合并。

三个后果：

1. LLM 只能消费图像，不能生成图像。输出只有文本。
2. 混合模态文档（交替段落和图像，如文章）很笨拙——你要么在模型外部解析多模态输入，要么链式生成。
3. 分布不匹配。视觉 token 和文本 token 存在于隐藏空间的不同区域，产生微妙的对齐问题。

Chameleon 拒绝了这一前提：图像只是来自共享词汇表的离散 token 序列。在交错文档上训练模型，一个损失，一个自回归解码器，你就免费获得了混合模态生成。

## 概念

### VQ-VAE 作为图像 tokenizer

Tokenizer 是一个向量量化变分自编码器。架构：

- 编码器：CNN + ViT，将图像映射到空间特征图，比如 32x32 个维度为 256 的特征。
- 码本：一个 K 个向量的学习词汇表（Chameleon 使用 8192），也是维度 256。
- 量化：对于每个空间特征，通过 L2 距离查找最近的码本条目。将连续特征替换为整数索引。
- 解码器：CNN，将量化特征映射回像素。

训练：VAE 重建损失 + 承诺损失 + 码本损失。码本索引形成图像的离散字母表。

对于 Chameleon：一张图像变成 32*32 = 1024 个 token，从 8192 的词汇表中抽取。与文本 token（来自 LLM 的 BPE 词汇表，比如 32000）拼接。最终词汇表：40192。Transformer 看到一个序列，一个损失。

### 共享词汇表

Chameleon 的词汇表结合了文本 token、图像 token 和模态分隔符。每个 token 有一个单一 ID。输入嵌入层将每个 ID 映射到 D 维隐藏向量。输出投影将隐藏向量映射回词汇表 logits。Softmax 选择下一个 token，无论什么模态。

分隔符很重要：`<image>` 和 `</image>` 标签包围图像 token 序列。生成时，如果模型发出 `<image>`，下游软件知道接下来的 1024 个 token 是 VQ 索引，需要发送给解码器渲染像素。

### 混合模态生成

推理是在共享词汇表中的 next-token 预测。示例提示："画一只猫并描述它。"Chameleon 输出：

```
<image> 4821 1029 2891 ... (1024 个图像 token) </image>
这只猫是橙色的，坐在窗台上...
```

模型自主选择顺序——它可能先产生图像再文本，先文本再图像，或交错排列。相同的解码器，相同的损失。

与适配器 VLM 相比（生成只限于文本），Chameleon 重新开启了模型输出模态的问题。

### 训练稳定性——QK-Norm、dropout、LayerNorm 排序

早期融合训练在大规模上不稳定。Chameleon 的论文记录了三个技巧：

- QK-Norm。在注意力内部，点积之前，对查询和键投影应用 LayerNorm。防止深层 logit 幅度爆炸。被多个 2024 年后的大模型使用。
- Dropout 放置。在每个残差加法之后放置 dropout，而不仅仅是在注意力和 MLP 之后。当来自图像 token 的梯度可能占主导时，需要更多正则化。
- LayerNorm 排序。残差分支上的 Pre-LN（标准），加上最后一个块跳跃连接上的额外 LN。稳定最后一层的梯度流。

没有这些技巧，34B 参数的 Chameleon 训练在多个 checkpoint 处发散。有了它们，它收敛。训练配方与架构本身同样是贡献。

### Tokenizer 的重建天花板

VQ-VAE 是有损的。在 8192 个码本条目和每张 512x512 图像 1024 个 token 下，重建 PSNR 约在 26-28 dB。这足以识别图像生成，但明显差于连续空间扩散（Stable Diffusion 3 达到 32+ dB）。

Tokenizer 是瓶颈。更好的 tokenizer（MAGVIT-v2、IBQ、SBER-MoVQGAN）提升了天花板。Emu3（课程 12.12）仅通过更好的 tokenizer 就达到了 SDXL 质量。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（早期融合，共享词汇表）：
- 一个损失，一个解码器。
- 生成混合模态输出。
- Tokenizer 是质量天花板。
- 昂贵：每张生成图像在推理路径上需要 VQ-VAE 解码器。

BLIP-2 / LLaVA（晚期融合，独立塔）：
- 视觉输入，仅文本输出。
- 重用预训练 LLM。
- 没有 tokenizer 瓶颈用于理解。
- 便宜：单次前向传播。

按任务选择。如果你需要图像生成，选 Chameleon 家族。如果你只需要理解，适配器 VLM 更简单，重用更多预训练计算。

### Fuyu 和 AnyGPT

Fuyu（Adept，2023 年）是一种相关方法：完全跳过独立的视觉编码器，将原始图像 patch 通过 LLM 的输入投影作为 token 馈入，没有 tokenizer。比 Chameleon 更简单，但失去了共享词汇表输出生成。

AnyGPT（Zhan 等人，2024 年）将 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每种使用相同的 VQ-VAE 技巧，共享 Transformer。任意到任意生成。在课程 12.16 中有更多介绍。

## 使用它

`code/main.py` 构建了一个玩具端到端早期融合模型：

- 一个微小的 VQ-VAE 风格量化器，将 8x8 patch 映射到码本索引（K=16）。
- 一个共享词汇表（文本 id 0..31）+（图像 id 32..47）+（分隔符 48，49）。
- 一个玩具自回归解码器（bigram 表），在合成描述 + 图像 token 序列上训练。
- 采样循环，给定提示后交替输出文本 + 图像 token。

代码有意让 Transformer 保持微小（bigrams），这样你可以从头到尾追踪信号流。

## 交付物

本课程产生 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定一个产品规格（仅理解 vs 理解 + 生成，所需图像质量，成本预算），它在 Chameleon 家族（早期融合）和 LLaVA 家族（晚期融合）之间做出选择，并用定量经验法则证明合理性。

## 练习

1. Chameleon 使用 K=8192 码本条目和每张 512x512 图像 1024 个 token。估计与 24 位 RGB 图像相比的压缩比率。它是有损的吗？损失多少？

2. 一张 4K 图像（3840x2160）在相同的 VQ-VAE 密度下产生多少个图像 token？Chameleon 风格的模型能一次推理调用生成 4K 图像吗？什么先出问题——上下文、tokenizer 质量还是 KV 缓存？

3. 在纯 Python 中实现 QK-Norm。给定一个 64 维查询和键，显示 LayerNorm 前后的点积。为什么幅度控制在深层很重要？

4. 阅读 Chameleon 第 2.3 节关于训练稳定性的内容。描述论文在没有 QK-Norm 的情况下在 34B 观察到的确切故障模式。"范数爆炸"的迹象是什么？

5. 扩展玩具解码器，使其在纯文本提示下输出混合模态响应。测量在训练数据分布为 60% 先文本 / 40% 先图像的情况下，模型选择先图像 vs 先文本的频率。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 早期融合 | "统一 token" | 图像被转换为离散 token，从第一步起与 Transformer 的词汇表共享 |
| VQ-VAE | "图像 tokenizer" | CNN + ViT + 码本，将图像映射到 Transformer 可以预测的整数索引 |
| 共享词汇表 | "一本词典" | 覆盖文本 + 图像 + 模态分隔符的单一 token ID 空间 |
| QK-Norm | "注意力稳定器" | 对查询和键在点积前应用 LayerNorm，防止范数爆发 |
| 混合模态生成 | "文本 + 图像输出" | 一次推理中自主产生交错文本和图像 token 的推理过程 |
| 码本大小 | "K 个条目" | VQ-VAE 可以量化到的离散向量数量；在压缩和保真度之间权衡 |
| Tokenizer 天花板 | "重建限制" | 解码 VQ token 可达到的最佳 PSNR；限制了模型的图像质量 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan 等人 — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu 等人 — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan 等人 — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
