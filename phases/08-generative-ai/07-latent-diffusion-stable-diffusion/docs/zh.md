# 潜在扩散与 Stable Diffusion

> 在 512×512 图像上进行像素空间扩散是一种计算战争罪。Rombach 等人（2022）注意到你不需要所有 786k 维度来生成一张图像——你需要足够的维度来捕捉语义结构，然后用一个单独的解码器来处理其余部分。在 VAE 的潜在空间内运行扩散。这一个想法就是 Stable Diffusion。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 8 · 02（VAE），阶段 8 · 06（DDPM），阶段 7 · 09（ViT）
**时间：** ~75 分钟

## 问题

在 512² 上进行像素空间扩散意味着 U-Net 在形状为 `[B, 3, 512, 512]` 的张量上运行。每个采样步骤对于一个 500M 参数的 U-Net 大约是 ~100 GFLOPS。五十步就是每张图像 5 TFLOPS。在十亿张图像上训练，计算账单是荒谬的。

这些 FLOPs 中的大部分用于将感知上不重要的细节推过网络——那些有损 VAE 可以压缩掉的高频纹理。Rombach 的想法：训练一次 VAE（*第一阶段*），冻结它，然后在 4 通道 64×64 的潜在空间中完全运行扩散（*第二阶段*）。相同的 U-Net。1/16 的像素。约 64 倍更少的 FLOPs，得到可比较的质量。

这就是 Stable Diffusion 的配方。SD 1.x / 2.x 在 `64×64×4` 的潜在变量上使用了一个 860M 的 U-Net，SDXL 在 `128×128×4` 上使用了一个 2.6B 的 U-Net，SD3 将 U-Net 替换为带有流匹配的扩散 Transformer（DiT）。Flux.1-dev（Black Forest Labs，2024）提供了一个 12B 参数的 DiT-MMDiT。所有这些都在相同的两阶段基础上运行。

## 概念

![潜在扩散：VAE 压缩 + 在潜在空间中的扩散](../assets/latent-diffusion.svg)

**两个阶段，分别训练。**

1. **阶段 1 — VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩：在每个空间轴上 8 倍下采样 + 调整通道，使总潜在大小约为像素数的 1/16。损失 = 重构（L1 + LPIPS 感知）+ KL（权重较小，所以 `z` 不会被迫过于高斯化，因为我们不需要从 `z` 精确采样）。通常使用对抗损失训练，使得解码图像清晰锐利。

2. **阶段 2 — 在 `z` 上扩散。** 将 `z = E(x_real)` 视为数据。训练一个 U-Net（或 DiT）来对 `z_t` 去噪。推理时：通过扩散采样 `z_0`，然后 `x = D(z_0)`。

**文本条件化。** 两个额外的组件。一个冻结的文本编码器（SD 1.x 的 CLIP-L、SD 2/XL 的 CLIP-L+OpenCLIP-G、SD3 和 Flux 的 T5-XXL）。一个交叉注意力注入：每个 U-Net 块接收 `[Q = 图像特征，K = V = 文本 token]` 并将它们混合。token 是文本影响图像的唯一途径。

**损失函数与第 6 课完全相同。** 相同的 DDPM / 流匹配噪声 MSE。你只需切换数据域。

## 架构变体

| 模型 | 年份 | 骨干网络 | 潜在形状 | 文本编码器 | 参数 |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77 tokens） | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 蒸馏 | 128×128×4 | 相同 | 1-4 步采样 |
| SD3 | 2024 | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT 蒸馏 | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 步 |

趋势：用 DiT（潜在补丁上的 Transformer）替换 U-Net，扩大文本编码器（T5 在提示遵循上击败 CLIP），增加潜在通道（4 → 16 提供更多细节空间）。

```figure
noise-schedule
```

## 动手实现

`code/main.py` 将玩具 1-D "VAE"（恒等编码器 + 解码器，用于演示；真正的 VAE 将是卷积网络）叠加在第 6 课的 DDPM 之上，并添加了带有无分类器引导的类别条件化。它展示了相同的扩散损失无论你在原始 1-D 值上运行还是在编码值上运行都有效——这是关键洞见。

### 步骤 1：编码器/解码器

```python
def encode(x):    return x * 0.5          # 玩具"压缩"到更小的尺度
def decode(z):    return z * 2.0
```

真正的 VAE 有训练好的权重。出于教学目的，这个线性映射足以展示扩散在 `z` 上操作而不关心原始数据空间。

### 步骤 2：在 `z` 空间扩散

与第 6 课相同的 DDPM。网络看到的数据是 `z = E(x)`。采样 `z_0` 后，用 `D(z_0)` 解码。

### 步骤 3：无分类器引导

训练期间，10% 的概率丢弃类别标签（替换为空 token）。推理时，同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（完全多样性），`w = 3` = 默认，`w = 7+` = 饱和 / 过度锐化。

### 步骤 4：文本条件化（概念，非代码）

将类别标签替换为冻结的文本编码器输出。通过交叉注意力将文本嵌入馈入 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是类别条件扩散模型和 Stable Diffusion 之间唯一实质性的区别。

## 陷阱

- **VAE 尺度不匹配。** SD 1.x VAE 有一个缩放常数（`scaling_factor ≈ 0.18215`）在编码后应用。忘记这个会导致 U-Net 在方差严重错误的潜在变量上训练。每个检查点都自带一个。
- **错误使用文本编码器。** SD3 需要 T5-XXL，使用 >=128 tokens，退回到仅使用 CLIP 是有损的。始终检查 `use_t5=True`，否则提示保真度会崩溃。
- **混合潜在空间。** SDXL、SD3、Flux 使用不同的 VAE。在 SDXL 潜在变量上训练的 LoRA 无法在 SD3 上工作。Hugging Face diffusers 0.30+ 拒绝加载不匹配的检查点。
- **CFG 过高。** `w > 10` 会产生饱和、油腻的图像，并以多样性为代价过度拟合提示。最佳点是 `w = 3-7`。
- **负面提示泄漏。** 空的负面提示成为空 token；填充的负面提示成为 `ε_uncond`。这两者不同；有些管线静默默认为空 token。

## 使用

2026 年生产技术栈：

| 目标 | 推荐的骨干网络 |
|--------|----------------------|
| 窄域、配对数据、从头训练模型 | SDXL 微调（LoRA / 全参数）——部署最快 |
| 开放域文生图、开放权重 | Flux.1-dev（12B，Apache / 非商业）或 SD3.5-Large |
| 最快推理、开放权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 最佳提示遵循、托管服务 | GPT-Image / DALL-E 3（仍然），Midjourney v7，Imagen 4 |
| 编辑工作流 | Flux.1-Kontext（2024 年 12 月）— 原生接受图像 + 文本 |
| 研究、基线 | SD 1.5 — 古老但研究充分 |

## 产出

保存 `outputs/skill-sd-prompter.md`。技能接受文本提示 + 目标风格并输出：模型 + 检查点、CFG 尺度、采样器、负面提示、分辨率、可选的 ControlNet/IP-Adapter 组合以及一个每一步 QA 检查清单。

## 练习

1. **简单。** 以引导尺度 `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。按类别记录平均样本。在哪个 `w` 下类别均值偏离了真实数据均值？
2. **中等。** 将玩具线性编码器替换为 tanh-MLP 编码器/解码器对，并带有重构损失。在新的潜在变量上重新训练扩散。样本质量会改变吗？
3. **困难。** 用 diffusers 设置真实的 Stable Diffusion 推理：加载 `sdxl-base`，运行 30 步 Euler 采样，CFG=7，计时。然后切换到 `sdxl-turbo`，4 步，CFG=0。相同主题，不同质量——描述发生了什么变化以及为什么。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 第一阶段 | "VAE" | 训练好的编码器/解码器对；将 512² 压缩到 64²。 |
| 第二阶段 | "U-Net" | 在潜在空间上的扩散模型。 |
| CFG | "引导尺度" | `(1+w)·ε_cond - w·ε_uncond`；调节条件化强度。 |
| 空 token | "空提示嵌入" | 用于 `ε_uncond` 的无条件嵌入。 |
| 交叉注意力 | "文本如何进入" | 每个 U-Net 块将文本 token 作为 K 和 V 进行注意力操作。 |
| DiT | "扩散 Transformer" | 用潜在补丁上的 Transformer 替换 U-Net；扩展性更好。 |
| MMDiT | "多模态 DiT" | SD3 的架构：文本和图像流通过联合注意力连接。 |
| VAE 缩放因子 | "神奇数字" | 将潜在变量除以约 5.4，使扩散在单位方差空间中运行。 |

## 生产说明：在 8GB 消费级 GPU 上运行 Flux-12B

参考 Flux 集成是"我有一个消费级 GPU，我能部署它吗？"的权威配方。技巧是生产推理文献中列出的同一个三旋钮配方，应用于扩散 DiT：

1. **交错加载。** Flux 有三个网络，它们从不需要同时在 VRAM 中共存：T5-XXL 文本编码器（fp32 中约 10 GB）、CLIP-L（小）、12B MMDiT 和 VAE。先编码提示，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。消费级 8GB GPU 一次只能容纳一个阶段。
2. **通过 bitsandbytes 的 4 位量化。** 在 T5 编码器和 DiT 上都使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存减少 8 倍，根据 Aritra 的基准测试，文生图的质量下降是难以察觉的（链接在 notebook 中）。
3. **CPU 卸载。** `pipe.enable_model_cpu_offload()` 在每个前向传播推进时自动在 CPU 和 GPU 之间交换模块。增加 10-20% 的延迟，但使得管线能够实际运行。

内存核算：`10 GB T5 / 8 = 1.25 GB` 量化后，`12 B params × 0.5 bytes = ~6 GB` 量化 DiT，加上激活值。用 stas00 的话说，这是 TP=1 推理的极端端——没有模型并行，最大量化。对于生产环境，你会在 H100 上运行 TP=2 或 TP=4；对于单个开发者的笔记本电脑，这就是配方。

## 延伸阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 家族。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 上述每个检查点的参考实现。
