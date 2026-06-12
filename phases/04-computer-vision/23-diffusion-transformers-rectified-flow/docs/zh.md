# 扩散 Transformer 与纠正流

> U-Net 并不是扩散的秘密。用 Transformer 替换它，将噪声调度换成直线流，突然你就得到了 SD3、FLUX 和 2026 年的每个文本到图像模型。

**类型：** 学习 + 构建
**语言：** Python
**前置知识：** 第四阶段第10课（扩散 DDPM），第四阶段第14课（ViT），第七阶段第02课（自注意力）
**时间：** ~75分钟

## 学习目标

- 梳理从 U-Net DDPM（第10课）到扩散 Transformer（DiT）、MMDiT（SD3）和单+双流 DiT（FLUX）的演进
- 解释纠正流：为什么噪声和数据之间的直线轨迹让模型可以用 20 步而不是 1000 步采样
- 实现一个微型 DiT 块和一个纠正流训练循环，两者都在 100 行以内
- 通过架构、参数量和许可区分模型变体（SD3、FLUX.1-dev、FLUX.1-schnell、Z-Image、Qwen-Image）

## 问题

第 10 课使用 U-Net 去噪器构建了一个 DDPM。这个方案在 2020-2023 年占主导地位：U-Net + beta 调度 + 噪声预测损失。它产生了 Stable Diffusion 1.5 和 2.1 以及 DALL-E 2。

每个 2026 年的最先进文本到图像模型都已经超越了它。Stable Diffusion 3、FLUX、SD4、Z-Image、Qwen-Image、Hunyuan-Image——没有使用 U-Net 的。它们使用扩散 Transformer（DiT）。SD3 和 FLUX 还将 DDPM 噪声调度换成了纠正流，这拉直了从噪声到数据的路径，并使得使用一致性或蒸馏变体进行 1-4 步推理成为可能。

这种转变很重要，因为它是基于扩散的图像生成变得可控、提示准确（SD3/SD4 解决了文本渲染）和生产快速的原因。理解 DiT + 纠正流就是理解 2026 年生成式图像堆栈。

## 概念

### 从 U-Net 到 Transformer

```mermaid
flowchart LR
    subgraph UNET["DDPM U-Net (2020)"]
        U1["卷积编码器"] --> U2["卷积瓶颈"] --> U3["卷积解码器"]
    end
    subgraph DIT["DiT (2023)"]
        D1["图块嵌入"] --> D2["Transformer 块"] --> D3["反向图块化"]
    end
    subgraph MMDIT["MMDiT (SD3, 2024)"]
        M1["文本流"] --> M3["联合注意力<br/>（每模态独立权重）"]
        M2["图像流"] --> M3
    end
    subgraph FLUX["FLUX (2024)"]
        F1["双流块<br/>（文本+图像独立）"] --> F2["单流块<br/>（拼接 + 共享权重）"]
    end

    style UNET fill:#e5e7eb,stroke:#6b7280
    style DIT fill:#dbeafe,stroke:#2563eb
    style MMDIT fill:#fef3c7,stroke:#d97706
    style FLUX fill:#dcfce7,stroke:#16a34a
```

- **DiT**（Peebles & Xie，2023）— 用类似 ViT 的 Transformer（在潜图块上）替换 U-Net。通过自适应层归一化（AdaLN）进行条件化。
- **MMDiT**（SD3，Esser 等人，2024）— 两个流，文本和图像标记有独立权重，共享联合注意力。
- **FLUX**（Black Forest Labs，2024）— 前 N 个块类似 SD3 的双流，后面的块拼接并共享权重（单流），以提高更深层的效率。
- **Z-Image**（2025）— 一个高效的 60 亿参数单流 DiT，挑战"不计代价地扩展"。

### 纠正流（一段话）

DDPM 将前向过程定义为一个带噪 SDE，其中 `x_t` 逐渐被破坏。学习到的反向过程是第二个 SDE，通过 1000 个小步求解。

纠正流定义了干净数据和纯噪声之间的**直线**插值：

```
x_t = (1 - t) * x_0 + t * epsilon,     t in [0, 1]
```

训练一个网络预测速度 `v_theta(x_t, t) = epsilon - x_0`——沿着从干净数据到噪声的直线路径的前向方向（`dx_t/dt`）。在采样期间，你从噪声向数据反向积分这个速度。得到的 ODE 更接近直线，因此采样所需的积分步数大大减少。

SD3 称之为**纠正流匹配**。FLUX、Z-Image 和大多数 2026 年模型使用相同的目标。典型推理：20-30 步 Euler（确定性）对比旧 DDPM 方案中的 50+ 步 DDIM。蒸馏 / turbo / schnell / LCM 变体将其降至 1-4 步。

### AdaLN 条件化

DiT 通过**自适应层归一化**以时间步和类别/文本为条件：从条件化向量预测 `scale` 和 `shift`，在 LayerNorm 之后应用它们。比 U-Net 中的 FiLM 风格调制更简洁，并且是每个现代 DiT 的默认选择。

```
cond -> MLP -> (scale, shift, gate)
norm(x) * (1 + scale) + shift, then residual add * gate
```

### SD3 和 FLUX 中的文本编码器

- **SD3** 使用三个文本编码器：两个 CLIP 模型 + T5-XXL。嵌入被拼接并作为文本条件输入图像流。
- **FLUX** 使用一个 CLIP-L + T5-XXL。
- **Qwen-Image / Z-Image** 变体使用它们自己的内部文本编码器，与其基础 LLM 对齐。

文本编码器是 SD3/FLUX 比 SD1.5 理解提示好得多的主要原因。仅 T5-XXL 就有 47 亿参数。

### 无分类器引导仍然有效

纠正流改变了采样器，而不是条件化。无分类器引导（训练时 10% 概率丢弃文本，推理时混合条件和无条件预测）与纠正流完全相同。大多数 2026 年模型使用引导尺度 3.5-5——低于 SD1.5 的 7.5，因为纠正流模型默认更紧密地遵循提示。

### Consistency、Turbo、Schnell、LCM

同一思想的四个名称：将慢速多步模型蒸馏为快速少步模型。

- **LCM（潜在一致性模型）** — 训练一个学生模型，从任何中间 `x_t` 一步预测最终 `x_0`。
- **SDXL Turbo / FLUX schnell** — 1-4 步模型，使用对抗性扩散蒸馏训练。
- **SD Turbo** — OpenAI 风格的一致性模型，适配到潜在扩散。

任何新模型的生产服务都会提供"全质量"检查点和"turbo / schnell"变体。Schnell（德语中"快"的意思，Black Forest Labs 的惯例）在 1-4 步内运行，适合实时流水线。

### 2026 年模型格局

| 模型 | 大小 | 架构 | 许可 |
|-------|------|--------------|---------|
| Stable Diffusion 3 Medium | 2B | MMDiT | SAI Community |
| Stable Diffusion 3.5 Large | 8B | MMDiT | SAI Community |
| FLUX.1-dev | 12B | 双 + 单流 DiT | 非商用 |
| FLUX.1-schnell | 12B | 同上，蒸馏 | Apache 2.0 |
| FLUX.2 | — | FLUX.1 迭代 | 混合 |
| Z-Image | 6B | S3-DiT（可扩展单流） | 宽松 |
| Qwen-Image | ~20B | DiT + Qwen 文本塔 | Apache 2.0 |
| Hunyuan-Image-3.0 | ~80B | DiT | 研究 |
| SD4 Turbo | 3B | DiT + 蒸馏 | SAI Commercial |

FLUX.1-schnell 是 2026 年的开源默认选择。Z-Image 是效率领先者。FLUX.2 和 SD4 是目前的质量顶端。

### 为什么这个阶段转变很重要

DDPM + U-Net 有效。DiT + 纠正流**更好、更快、扩展更干净**。这种转变与 NLP 中从 RNN 到 Transformer 的转变相似：两种架构都能解决同一个问题，但 Transformer 可以扩展并且现在占主导地位。2026 年每篇关于图像、视频或 3D 生成的论文都使用 DiT 形状的去噪器，通常还使用纠正流目标。U-Net DDPM 现在主要是教学性的（第 10 课）。

## 构建

### 第一步：带 AdaLN 的 DiT 块

```python
import torch
import torch.nn as nn


class AdaLNZero(nn.Module):
    """
    带门的自适应 LayerNorm。从条件化预测 (scale, shift, gate)。
    初始化使整个块以恒等开始（"零初始化"）。
    """
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Linear(cond_dim, dim * 3)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, x, cond):
        scale, shift, gate = self.mlp(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return h, gate.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_ratio=4, cond_dim=192):
        super().__init__()
        self.adaln1 = AdaLNZero(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.adaln2 = AdaLNZero(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond):
        h, gate1 = self.adaln1(x, cond)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1 * a
        h, gate2 = self.adaln2(x, cond)
        x = x + gate2 * self.mlp(h)
        return x
```

`AdaLNZero` 以恒等映射开始，因为其 MLP 权重初始化为零。训练将块从恒等推开；这极大地稳定了深 Transformer 扩散模型。

### 第二步：微型 DiT

```python
def timestep_embedding(t, dim):
    import math
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TinyDiT(nn.Module):
    def __init__(self, image_size=16, patch_size=2, in_channels=3, dim=96, depth=4, heads=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, cond_dim=dim) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        n = x.size(0)
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2) + self.pos
        t_emb = self.time_mlp(timestep_embedding(t, self.pos.size(-1)))
        for blk in self.blocks:
            x = blk(x, t_emb)
        x = self.norm_out(x)
        x = self.head(x)
        return self._unpatchify(x, n)

    def _unpatchify(self, x, n):
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        x = x.view(n, h, w, p, p, -1).permute(0, 5, 1, 3, 2, 4).reshape(n, -1, h * p, w * p)
        return x
```

### 第三步：纠正流训练

```python
import torch.nn.functional as F

def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon

    target_velocity = epsilon - x0
    pred_velocity = model(x_t, t)

    loss = F.mse_loss(pred_velocity, target_velocity)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

与 DDPM 的噪声预测损失（第 10 课）比较：结构相同，目标不同。不是预测噪声 `epsilon`，而是预测**速度** `epsilon - x_0`，它沿着直线插值从数据指向噪声。

### 第四步：Euler 采样器

纠正流是一个 ODE。Euler 方法是最简单的，对于一个训练良好的纠正流模型，在 20+ 步时几乎与高阶求解器一样准确。

```python
@torch.no_grad()
def rectified_flow_sample(model, shape, steps=20, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x - dt * v
        t = t - dt
    return x
```

20 步。在训练好的模型上，这产生与 1000 步 DDPM 相当的样本。

### 第五步：端到端冒烟测试

```python
import numpy as np

def synthetic_blobs(num=200, size=16, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((num, 3, size, size), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for i in range(num):
        cx, cy = rng.uniform(4, size - 4, size=2)
        r = rng.uniform(2, 4)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        colour = rng.uniform(-1, 1, size=3)
        for c in range(3):
            out[i, c][mask] = colour[c]
    return torch.from_numpy(out)
```

在此之上用纠正流训练一个 `TinyDiT`。500 步后，采样的输出看起来像模糊的颜色斑点。

## 使用

对于使用 FLUX / SD3 / Z-Image 的真实图像生成，`diffusers` 以统一 API 提供每一个：

```python
from diffusers import FluxPipeline, StableDiffusion3Pipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to("cuda")

out = pipe(
    prompt="一只金毛犬在海啸中冲浪，超写实，工作室灯光",
    guidance_scale=0.0,           # schnell 没有用 CFG 训练
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save("surf.png")
```

三行代码。`FLUX.1-schnell` 只需四步。将模型 ID 换成 `black-forest-labs/FLUX.1-dev`，使用 CFG 以 20-30 步获得更高质量。

对于 SD3：

```python
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
).to("cuda")
out = pipe(prompt, guidance_scale=3.5, num_inference_steps=28).images[0]
```

## 交付

本课产出：

- `outputs/prompt-dit-model-picker.md` — 在 SD3、FLUX.1-dev、FLUX.1-schnell、Z-Image、SD4 Turbo 之间根据质量、延迟和许可约束进行选择。
- `outputs/skill-rectified-flow-trainer.md` — 编写带 AdaLN DiT 和 Euler 采样的纠正流完整训练循环。

## 练习

1. **（简单）** 在上述合成斑点数据集上训练 TinyDiT 500 步。比较使用 10、20 和 50 步 Euler 产生的样本。
2. **（中等）** 通过将学习到的类别嵌入拼接到时间嵌入来添加文本条件化（按颜色分 10 个斑点"类别"）。使用类别 0、5 和 9 采样，验证颜色匹配。
3. **（困难）** 计算来自相同大小网络、在相同数据上训练相同步数的纠正流和 DDPM 版本的生成样本之间的 Fréchet 距离（FID 代理）。报告哪个收敛更快。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| DiT | "扩散 Transformer" | 取代 U-Net 作为扩散去噪器的 Transformer；在图块化潜变量上操作 |
| AdaLN | "自适应层归一化" | 通过学习到的 scale、shift、gate 在 LayerNorm 后应用的时间步/文本条件化；每个现代 DiT 的标准 |
| MMDiT | "多模态 DiT (SD3)" | 文本和图像标记的独立权重流，共享联合自注意力 |
| 单流 / 双流 | "FLUX 技巧" | 前 N 块双流（每模态独立权重），后 N 块单流（拼接 + 共享权重）以提高效率 |
| 纠正流 | "直线噪声到数据" | 数据和噪声之间的线性插值；网络预测速度；推理所需 ODE 步数更少 |
| 速度目标 | "epsilon - x_0" | 纠正流中的回归目标；从干净数据指向噪声 |
| CFG 引导 | "无分类器引导" | 混合条件和无条件预测；在纠正流模型中仍然使用 |
| Schnell / turbo / LCM | "1-4 步蒸馏" | 从全质量模型蒸馏的小步变体；生产实时 |

## 延伸阅读

- [Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748) — DiT 论文
- [Scaling Rectified Flow Transformers (Esser et al., SD3 paper)](https://arxiv.org/abs/2403.03206) — MMDiT 和大规模纠正流
- [FLUX.1 model card and technical report (Black Forest Labs)](https://huggingface.co/black-forest-labs/FLUX.1-dev) — 双 + 单流细节
- [Z-Image: Efficient Image Generation Foundation Model (2025)](https://arxiv.org/html/2511.22699v1) — 60 亿参数单流 DiT
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) — 每个扩散设计权衡的参考
- [Latent Consistency Models (Luo et al., 2023)](https://arxiv.org/abs/2310.04378) — LCM-LoRA 如何实现 4 步推理
