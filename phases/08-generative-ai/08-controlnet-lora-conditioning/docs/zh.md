# ControlNet、LoRA 与条件化

> 文本本身是一个笨拙的控制信号。ControlNet 让你能够克隆一个预训练的扩散模型，并用深度图、姿态骨架、涂鸦或边缘图像来引导它。LoRA 让你通过训练 1000 万个参数就能微调一个 2B 参数的模型。它们一起将 Stable Diffusion 从一个玩具变成了 2026 年每家机构都在部署的图像管线。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 8 · 07（潜在扩散），阶段 10（从头实现 LLM——为 LoRA 打基础）
**时间：** ~75 分钟

## 问题

像"一个穿红裙子的女人在繁忙的街道上遛狗"这样的提示，没有告诉模型狗*在哪里*、女人是什么*姿态*、或者街道的*视角*。文本只能固定你需要描述一张图像所需信息的约 10%。其余的是视觉的，无法用文字有效描述。

为每种信号（姿态、深度、边缘、分割）从头训练一个新的条件模型是不可行的。你想要保持 2.6B 参数的 SDXL 骨干网络冻结，附加一个读取条件化输入的小型侧网络，让它推动骨干网络的中间特征。这就是 ControlNet。

你还想让模型学习新概念（你的脸、你的产品、你的风格），而无需重新训练整个模型。你想要一个小 100 倍的增量更新。这就是 LoRA——低秩适配器，插入到现有的注意力权重中。

ControlNet + LoRA + 文本 = 2026 年从业者的工具箱。大多数生产图像管线在一个 SDXL / SD3 / Flux 基础上叠加 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## 概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet（Zhang 等人，2023）

取一个预训练的 SD。*克隆* U-Net 的编码器一半。冻结原始网络。训练克隆部分以接受额外的条件化输入（边缘、深度、姿态）。通过*零卷积*跳跃连接（1×1 卷积，初始化为零——从无操作开始，学习增量）将克隆部分连接回原始网络的解码器一半。

```
SD U-Net 解码器：   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 从恒等映射开始——即使训练前也没有损害。在 100 万（提示、条件、图像）三元组上使用标准扩散损失进行训练。

每种模态的 ControlNet 作为小型侧模型提供（SDXL 约 360M，SD 1.5 约 70M）。你可以在推理时组合它们：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu 等人，2021）

对于模型中的任何线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力层通常使用秩 4-16，重度微调使用秩 64-128。新增参数量：`2 · d · r` 而不是 `d²`。对于 `d=640`、`r=16` 的 SDXL 注意力层：每个适配器 2 万个参数而不是 41 万——减少 20 倍。在整个模型层面：一个 LoRA 通常是 20-200MB，而基础模型是 5GB。

推理时你可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是正常范围。多个 LoRA 可以叠加相加（但通常它们的交互方式是非线性的）。

### IP-Adapter（Ye 等人，2023）

一个微型适配器，接受*图像*作为条件（与文本一起）。使用 CLIP 图像编码器产生图像 token，通过交叉注意力与文本 token 一起注入。每个基础模型约 20MB。让你能够"以这张参考图像的风格生成图像"而无需 LoRA。

## 可组合性矩阵

| 工具 | 控制什么 | 大小 | 何时使用 |
|------|------------------|------|-------------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格 |
| IP-Adapter | 参考图像的风格或主体 | 20MB | 无法用文字描述的外观 |
| 文本反演 | 将单一概念作为新 token | 10KB | 遗留方法，大部分已被 LoRA 取代 |
| DreamBooth | 在主体上进行全参数微调 | 2-5GB | 强身份绑定，高计算量 |
| T2I-Adapter | 更轻量的 ControlNet 替代 | 70MB | 边缘设备、推理预算有限 |

ControlNet ≈ 空间控制。LoRA ≈ 语义控制。两者都用。

## 动手实现

`code/main.py` 在一维数据上模拟了两种机制：

1. **LoRA。** 一个预训练的线性层 `W`。冻结它。训练一个低秩 `B @ A`，使得 `W + BA` 匹配一个目标线性层。展示 `r = 1` 就足以完美学习一个秩-1 修正。

2. **ControlNet 精简版。** 一个"冻结的基础"预测器和一个读取额外信号的"侧网络"。侧网络的输出由一个初始化为零的可学习标量门控（我们的零卷积版本）。训练并观察门控值逐渐增长。

### 步骤 1：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W 被冻结；A、B 是可训练的低秩因子。
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 步骤 2：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate 初始化为 0
h = base(x) + gated
```

在第 0 步，输出与基础网络完全相同。早期训练缓慢更新 `gate`——不会发生灾难性漂移。

## 陷阱

- **LoRA 过度缩放。** `α = 2` 或 `α = 3` 是一种常见的"让它更强"的 hack，会产生过度风格化/破碎的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 以权重 1.0 使用姿态 ControlNet 和以权重 1.0 使用深度 ControlNet 通常会导致过冲。权重之和 ≈ 1.0 是一个安全的默认值。
- **LoRA 用在错误的基础模型上。** SDXL LoRA 在 SD 1.5 上静默地不起作用，因为注意力维度不匹配。Diffusers 0.30+ 会发出警告。
- **文本反演漂移。** 在一个检查点上训练的 token 在另一个检查点上严重漂移。LoRA 的可移植性更好。
- **LoRA 权重合并与存储。** 你可以将 LoRA 烘焙到基础模型权重中，以获得更快的推理（无需运行时相加），但你会失去在运行时缩放 `α` 的能力。保留两种版本。

## 使用

| 目标 | 2026 年管线 |
|------|---------------|
| 复现品牌的艺术风格 | 在约 30 张精选图像上训练的 LoRA，秩 32 |
| 把我的脸放进生成的图像 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿态 + 提示 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知的构图 | ControlNet-Depth + SD3 |
| 参考图 + 提示 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + 修补（第 9 课） |
| 快速 1 步风格 | SDXL-Turbo 上的 LCM-LoRA |

## 产出

保存 `outputs/skill-sd-toolkit-composer.md`。技能接受一个任务（输入素材：提示、可选的参考图像、可选的姿态、可选的深度、可选的涂鸦）并输出工具栈、权重和可重现的种子协议。

## 练习

1. **简单。** 在 `code/main.py` 中，将 LoRA 秩 `r` 从 1 变化到 4。在哪个秩上 LoRA 完全匹配一个秩-2 目标增量？
2. **中等。** 在两个目标变换上训练两个独立的 LoRA。将它们一起加载并展示它们的相加交互。交互在什么时候会打破线性？
3. **困难。** 使用 diffusers 组合：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。在工具栈权重变化时测量 FID 与提示遵循度之间的权衡。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| ControlNet | "空间控制" | 克隆的编码器 + 零卷积跳跃连接；读取条件化图像。 |
| 零卷积 | "从恒等映射开始" | 初始化为零的 1×1 卷积；ControlNet 从无操作开始。 |
| LoRA | "低秩适配器" | `W + B @ A`，`r << d`；参数比全参数微调少 100 倍。 |
| 秩 r | "旋钮" | LoRA 压缩；4-16 典型值，64+ 用于重度个性化。 |
| α | "LoRA 强度" | LoRA 增量的运行时缩放。 |
| IP-Adapter | "参考图像" | 通过 CLIP-图像 token 的小型图像条件化适配器。 |
| DreamBooth | "全参数主体微调" | 在约 30 张主体图像上训练完整模型。 |
| 文本反演 | "新 token" | 仅学习一个新的词嵌入；遗留方法，大部分已被取代。 |

## 生产说明：LoRA 热替换、ControlNet 通道、多租户服务

一个真实的文生图 SaaS 在同一个基础检查点上提供数百个 LoRA 和十几个 ControlNet。服务问题看起来很像 LLM 多租户（生产文献在连续批处理和 LoRAX / S-LoRA 下涵盖了 LLM 案例）：

- **热替换 LoRA，不要合并。** 将 `W' = W + α·B·A` 合并到基础模型中可以得到约 3-5% 的每步推理加速，但会冻结 `α` 和基础模型。将 LoRA 作为秩-r 增量保持在 VRAM 中热替换；diffusers 提供了 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 用于按请求激活。交换成本是 `2 · d · r · num_layers` 的权重——MB 级别，亚秒级。
- **ControlNet 作为第二条注意力通道。** 克隆的编码器与基础模型并行运行。每个权重为 1.0 的两个 ControlNet = 每步两次额外的前向传播，而不是一次合并的前向传播。批次大小的余量呈二次方下降。每个活跃的 ControlNet 预算约 1.5 倍的步成本。
- **量化 LoRA 也可以。** 如果你已经量化了基础模型（参见第 7 课，Flux on 8GB），LoRA 增量也可以干净地量化为 8 位或 4 位。QLoRA 风格的加载让你可以在 4 位 Flux 基础上叠加 5-10 个 LoRA 而不会撑爆内存。

Flux 特有：Niels 的 Flux-on-8GB notebook 将基础模型量化为 4 位；在该量化基础上使用 `weight_name="pytorch_lora_weights.safetensors"` 叠加风格 LoRA（`pipe.load_lora_weights("user/style-lora")`）仍然有效。这是 2026 年大多数 SaaS 机构部署的配方。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初为 LLM 设计；移植到扩散模型）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的更轻量替代方案。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参考管线。
