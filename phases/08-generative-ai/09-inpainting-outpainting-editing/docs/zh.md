# 修补、扩展与图像编辑

> 文生图生成新东西。修补修复旧东西。在生产环境中，70% 的收费图像工作是编辑——替换背景、移除标志、扩展画布、重新生成手部。修补是扩散模型真正发挥价值的地方。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 8 · 07（潜在扩散），阶段 8 · 08（ControlNet & LoRA）
**时间：** ~75 分钟

## 问题

客户发来一张完美的产品照片，但背景中有一个分散注意力的标志。你想要擦除标志，并保持其他所有像素完全相同。你不能从头运行文生图——结果会有不同的颜色、不同的光照、不同的产品角度。你想要*仅*重新生成遮罩区域，并且希望重新生成的内容尊重周围的上下文。

这就是修补。变体包括：

- **修补。** 在遮罩内部重新生成，保持外部像素不变。
- **扩展。** 在遮罩外部（或画布之外）重新生成，保持内部不变。
- **图像编辑。** 重新生成整张图像但保持对原始图像的语义或结构保真度（SDEdit、InstructPix2Pix）。

2026 年的每个扩散管线都提供修补模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们都基于相同的原理工作。

## 概念

![修补：带上下文保留重新注入的遮罩感知去噪](../assets/inpainting.svg)

### 朴素的方法（以及为什么它不对）

运行带有遮罩的标准文生图。在每个采样步骤，用前向扩散的干净图像替换带噪潜在变量中未遮罩的区域。它能工作……但效果很差。边界伪影会渗透进来，因为模型没有关于遮罩区域内是什么的信息。

### 正确的修补模型

训练一个修改后的 U-Net，接受 9 个输入通道而不是 4 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是 VAE 编码的源图像的副本加上一个单通道遮罩。训练时，你随机遮罩图像的区域，并训练模型仅对遮罩区域去噪，而未遮罩区域作为干净的条件化信号给出。推理时，模型可以"看到"遮罩区域周围的内容，并产生连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种 9 通道（或类似）输入。Diffusers 中的 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng 等人，2022）——免费编辑

向源图像添加噪声直到某个中间 `t`，然后从 `t` 向下运行反向链到 0，同时使用新的提示。无需重新训练。起始 `t` 的选择在保真度和创作自由度之间权衡：

- `t/T = 0.3` → 与源图像几乎相同，小的风格变化
- `t/T = 0.6` → 中等编辑，保留粗糙结构
- `t/T = 0.9` → 从接近噪声生成，最小的源图像保留

### InstructPix2Pix（Brooks 等人，2023）

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。推理时，以输入图像和文本指令（"让它变成日落"、"添加一条龙"）为条件。两个 CFG 尺度：图像尺度和文本尺度。

### RePaint（Lugmayr 等人，2022）

保持一个标准的无条件扩散模型。在每个反向步骤，重新采样——偶尔跳回更嘈杂的状态并重新生成。避免边界伪影。当你没有训练好的修补模型时使用。

## 动手实现

`code/main.py` 在 5 维数据上实现了一个玩具 1-D 修补方案。我们在 5-D 混合数据上训练一个 DDPM，其中每个样本是来自两个簇之一的 5 个浮点数。推理时，我们"遮罩"5 个维度中的 2 个，在每一步注入未遮罩三个维度的含噪前向版本，并仅重新生成遮罩的维度。

### 步骤 1：5-D DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 步骤 2：在所有 5 维上训练去噪器

标准的 DDPM。网络为 5-D 含噪输入输出 5-D 噪声预测。

### 步骤 3：推理时，遮罩感知的反向

```python
def inpaint_step(x_t, mask, clean_source, alpha_bars, t, rng):
    # 用干净的源图像的加噪版本替换未遮罩的维度
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_source[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...然后对 x_t 运行正常的反向步骤
```

这是朴素的方法，在玩具 1-D 数据上有效。真实的图像修补使用 9 通道输入，因为纹理一致性更重要。

### 步骤 4：扩展

扩展是遮罩反转的修补：遮罩新的（先前不存在的）画布，用原始图像填充其余部分。训练目标完全相同。

## 陷阱

- **接缝。** 朴素的方法留下可见的边界，因为梯度信息无法跨越遮罩流动。修复：将遮罩膨胀 8-16 像素，或使用正确的修补模型。
- **遮罩泄漏。** 如果条件化图像的未遮罩区域质量低或含噪，它会污染遮罩内的生成内容。稍微去噪或模糊。
- **CFG 与遮罩大小相互作用。** 在小遮罩上的高 CFG = 饱和的补丁。对小编辑降低 CFG。
- **SDEdit 保真悬崖。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能会失去主体的身份。进行扫描并设置检查点。
- **提示不匹配。** 提示应该描述*整个*图像，而不仅仅是新内容。"一只猫坐在椅子上"而不是"一只猫"。

## 使用

| 任务 | 管线 |
|------|----------|
| 移除物体，小遮罩 | SD-Inpaint 或 Flux-Fill，标准提示 |
| 替换天空 | SD-Inpaint + "日落时的蓝天" |
| 扩展画布 | SDXL 扩展模式（8px 羽化）或使用扩展遮罩的 Flux-Fill |
| 重新生成手/脸 | SD-Inpaint + 重新描述主体的提示 + ControlNet-Openpose |
| 改变一个区域的风格 | 在遮罩区域上使用 SDEdit，`t/T=0.5` |
| "让它变成日落" | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 遮罩 → SD-Inpaint |
| 超高保真度 | Flux-Fill 或 GPT-Image（托管）用于最难的案例 |

SAM（Meta 的 Segment Anything，2023）+ 扩散修补是 2026 年的背景移除管线。SAM 2（2024）适用于视频。

## 产出

保存 `outputs/skill-editing-pipeline.md`。技能接受原始图像 + 编辑描述 + 可选遮罩（或 SAM 提示）并输出：遮罩生成方法、基础模型、CFG 尺度（图像 + 文本）、SDEdit-t 或修补模式，以及 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，将遮罩维度的比例从 0.2 变化到 0.8。在哪个比例时修补质量（遮罩维度中的残差）等于无条件生成？
2. **中等。** 实现 RePaint：每 10 个反向步骤，跳回 5 步（添加噪声）并重新去噪。测量它是否减少了遮罩边缘的边界残差。
3. **困难。** 使用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill 在 20 个人脸重新生成任务上的表现。分别评分姿态遵循度和身份保持度。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 修补 | "填充空洞" | 在遮罩内重新生成；保持外部像素不变。 |
| 扩展 | "扩展画布" | 在画布外重新生成；保持内部不变。 |
| 9 通道 U-Net | "正确的修补模型" | 输入为 `noisy \| encoded-source \| mask` 的 U-Net。 |
| SDEdit | "带噪声级别的 img2img" | 添加噪声到时间 `t`，用新提示去噪。 |
| InstructPix2Pix | "纯文本编辑" | 在（图像、指令、输出）三元组上微调的扩散模型。 |
| RePaint | "无需重新训练" | 在反向过程中定期重新加噪以减少接缝。 |
| SAM | "分割一切" | 通过点击或框生成遮罩；与修补配合使用。 |
| Flux-Kontext | "带上下文的编辑" | 接受参考图像 + 指令以进行编辑的 Flux 变体。 |

## 生产说明：编辑管线对延迟敏感

编辑图像的用户期望亚 5 秒的往返时间。在 L4 上，1024² 的 30 步 SDXL-Inpaint 需要 3-4 秒，加上 SAM 遮罩生成（约 200 ms）和 VAE 编码/解码（合计约 500 ms）。用生产框架来说，这是 TTFT 受限而非吞吐量受限——批次大小为 1，低并发，最小化每个阶段：

- **SAM-H 是慢的那个。** SAM-H 在 1024² 时约 200 ms；SAM-ViT-B 约 40 ms，质量损失很小。SAM 2（视频）增加了时间开销；不要将其用于单图像编辑。
- **尽可能跳过编码。** `pipe.image_processor.preprocess(img)` 将图像编码为潜在变量。如果你有来自上次生成的潜在变量（在迭代式编辑 UI 中很常见），通过 `latents=...` 直接传递它们以跳过一次 VAE 编码。
- **遮罩膨胀对吞吐量也很重要。** 小遮罩意味着大部分 U-Net 前向传播被浪费了（无论如何未遮罩的像素被钳制）。`diffusers` 的 `StableDiffusionInpaintPipeline` 无论怎样都运行完整的 U-Net；只有 9 通道正确修补变体利用了遮罩计算。
- **Flux-Kontext 是 2025 年的答案。** 针对 `(source_image, instruction)` 的单次前向传播——不需要单独的遮罩，不需要 SDEdit 噪声扫描。在 H100 上，它在 ~1.5 秒内完成一次编辑。架构上的教训：合并阶段。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 无需训练的修补。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文本指令编辑。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，遮罩源。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 视频 SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 注意力级别编辑。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 年工具。
