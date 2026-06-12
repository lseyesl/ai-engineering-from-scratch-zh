# Emu3：用于图像和视频生成的 Next-Token 预测

> BAAI 的 Emu3（Wang 等人，2024 年 9 月）是 2024 年应该结束扩散与自回归之争的成果。一个单一的 Llama 风格仅解码器 Transformer，仅在 next-token-prediction 目标上训练，跨越文本 + VQ 图像 token + 3D VQ 视频 token 的统一词汇表，在图像生成上击败了 SDXL，在感知上击败了 LLaVA-1.6。没有 CLIP 损失。没有扩散调度。Classifier-free guidance 在推理时用于提高质量，但核心训练目标是 teacher forcing 下的 next-token 预测。发表在 Nature 上。本课程阅读 Emu3 的论点——为什么更好的 tokenizer 加上规模就是一切——并与扩散方法进行对比。

**类型：** 学习
**语言：** Python（标准库，3D 视频 tokenizer 数学 + 自回归采样器框架）
**前置要求：** Phase 12 · 11（Chameleon）
**时间：** ~120 分钟

## 学习目标

- 解释为什么 Emu3 的单损失 next-token 目标有效，尽管长期以来一直认为扩散是图像质量所必需的。
- 描述 3D 视频 tokenizer：时空 VQ 码本是什么样子的，为什么 patch 跨越时间。
- 比较 Emu3 vs Stable Diffusion XL 在（训练计算、推理成本、质量天花板）方面的差异。
- 说出同一个 Emu3 模型扮演的三个角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题

到 2024 年为止的传统智慧：图像生成需要扩散。论点是：离散图像 token 丢失太多信息，无法重建细节，自回归采样在数千个 token 上累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种形式的扩散。Chameleon（课程 12.11）在小规模上部分反驳了这一点，但在质量上没有匹配 SDXL。

Emu3 正面攻击了这一论点。声称：更好的视觉 tokenizer + 足够的规模 + next-token 损失 = 在同一模型中同时做感知和生成，且图像生成击败扩散。

这个赌注在发表时是有争议的。两年后，开源统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）是研究的默认路径；生产前沿模型似乎使用某种变体。

## 概念

### Emu3 tokenizer

关键成分是视觉 tokenizer。Emu3 训练了一个自定义的 IBQ 类 tokenizer（逆向瓶颈量化器，SBER-MoVQGAN 家族），每个 token 的分辨率降低为 8x8。一张 512x512 图像变成 64x64 = 4096 个 token，码本大小 32768。

这比 Chameleon 的 1024 个 token 每张 512x512 图像、K=8192 更大，但每 token 更便宜（更小的码本查找，更简单的编解码器）。关键指标：重建 PSNR 为 30.5 dB，与 Stable Diffusion 的连续潜空间 32 dB 具有竞争力。

对于视频：一个 3D VQ tokenizer 将时空 patch（4x4x4 像素）编码为一个整数。一个 4 秒的片段在 8 FPS 下具有 32 帧；在 256x256 下，4x 空间和 4x 时间缩减，token 数量为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 个 token。

Tokenizer 质量是天花板。Emu3 的贡献部分在于"我们训练了一个非常好的 tokenizer。"

### 单损失训练

Emu3 使用一个目标：在跨越文本 token、2D 图像 token 和 3D 视频 token 的共享词汇表上的 next-token 预测。训练期间权重按模态特定因子相乘以平衡贡献，但损失函数是相同的。

训练混合：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：类似。
- 纯文本：标准 NTP。

模型从数据分布中学习何时生成图像 token vs 文本 token。生成从模型在 `<image>` 标签后预测图像 token 中涌现出来。

### Classifier-free guidance 和温度

自回归图像生成在推理时使用 classifier-free guidance (CFG) 会得到很大改善。Emu3 使用它：生成两次，一次带完整描述，一次带空描述，用指导权重（通常 3.0-7.0）混合 logits。这与扩散使用的 CFG 技巧相同，被借用到自回归设置中。

温度很重要：太高，有伪影；太低，模式崩溃。Emu3 推荐感知任务温度为 1.0，图像生成温度为 0.8。

### 三个角色，一个模型

Emu3 以三个功能不同的 API 形式发布，但底层是同一套权重：

- Emu3-Gen。图像生成。输入文本，输出图像 token。
- Emu3-Chat。VQA 和描述。输入图像（token），输出文本。
- Emu3-Stage2。视频生成和视频 VQA。输入文本或视频，输出文本或视频。

没有任务特定的头部。只是不同的提示模板。相同的 checkpoint。

### 基准测试

来自 Emu3 论文（2024 年 9 月）：

- 图像生成：在 MJHQ-30K FID（5.4 vs 5.6）上击败 SDXL，GenEval 总体（0.54 vs 0.55——统计平局），Deep-Eval 的综合指标持平。
- 图像感知：在 VQAv2（75.1 vs 72.4）上击败 LLaVA-1.6，在 MMMU 上大致匹配。
- 视频生成：4 秒片段质量在与 Sora 时代公开基准模型竞争的 FVD 水平。

数字并非总是胜利——Emu3 在这里换一个点，在那里丢一个点——但"next-token 预测就是一切"的说法在模态之间是可辩护的。

### 计算成本

Emu3 在约 3000 亿多模态 token 上使用 7B 参数模型训练。GPU 小时数大致与 Llama-2-7B 预训练相当（在 A100 级硅片上 2000-4000 GPU 年）。像 Stable Diffusion 3 这样的扩散模型在类似预算中训练，但需要单独的文本编码器和更复杂的流水线。

在推理时，Emu3 每张图像比 SDXL 慢：4096 个图像 token 以 30 tok/s 计算，每张 512x512 图像约 2 分钟，而 SDXL 为 2-5 秒。推测解码和 KV 缓存优化缩小了差距，但没有完全消除。自回归图像生成计算量大；这是存在的权衡。

### 为什么它重要

Emu3 的深层贡献是概念性的。如果 next-token 预测在图像生成上可以扩展到匹配扩散，那么统一模型路径（一个损失，一个骨干网，任何模态）就是可行的。未来的模型不需要单独的文本编码器、单独的扩散调度器、单独的 VAE。一个 Transformer，每个模态一个 tokenizer，扩展。

Show-o、Janus-Pro 和 InternVL-U 都建立或挑战这一论点。中国实验室（BAAI、DeepSeek）到 2025 年比美国实验室更积极地朝这个方向发表成果。

## 使用它

`code/main.py` 构建了两个玩具组件：

- 一个 2D vs 3D VQ tokenizer 计数计算器：给定（分辨率、patch、片段长度、FPS），计算图像 vs 视频的 token 数量。
- 一个带有 classifier-free guidance 和温度的自回归图像 token 采样器。

CFG 实现与 Emu3 的配方一致——用指导权重混合条件和无条件 logits。

## 交付物

本课程产生 `outputs/skill-token-gen-cost-analyzer.md`。给定一个生成产品规格（图像或视频，目标分辨率，质量等级，延迟预算），它计算 token 数量、推理成本，并在 Emu3 家族与扩散之间做出选择。

## 练习

1. Emu3 在 8x8 缩减下每张 512x512 图像产生 4096 个 token。计算 1024x1024 和 2048x2048 的对应值。推理延迟会发生什么变化？

2. 阅读 Emu3 第 3.3 节关于视频 tokenizer 的内容。描述 3D VQ patch 形状以及为什么是 4x4x4 而不是 8x8x1。

3. Classifier-free guidance 权重 5.0 vs 3.0：有什么视觉效果？在 `code/main.py` 中追踪数学原理。

4. 计算 Emu3-7B 在 300B token 下的训练 FLOPs，并与 Stable Diffusion 3 进行比较。哪个训练成本更高？

5. Emu3 在 FID 上击败了 SDXL，但在 VQAv2 上不如专门的 VLM。解释为什么统一损失方法在不同的基准上与专家模型相比显示出不同的优势。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| Next-token 预测 | "NTP" | 标准自回归损失：给定 token[0..i] 预测 token[i+1]；在 token 化后适用于每种模态 |
| IBQ tokenizer | "逆向瓶颈量化器" | 一类 VQ-VAE，具有更大的码本（32768+）和比 Chameleon 更好的重建质量 |
| 3D VQ | "时空量化器" | 由（时间，行，列）索引的码本；一个 token 覆盖一个 4x4x4 像素立方体 |
| Classifier-free guidance | "CFG" | 用权重 gamma 混合条件和无条件 logits；在推理时提升图像质量 |
| 统一词汇表 | "共享 token" | 文本 + 图像 + 视频都从同一个整数空间中抽取；模型预测接下来出现的是哪种模态 |
| MJHQ-30K | "图像生成基准" | Midjourney 质量基准，3 万个提示；Emu3 在此报告 FID |

## 延伸阅读

- [Wang 等人 — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun 等人 — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu 等人 — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu 等人 — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian 等人 — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
