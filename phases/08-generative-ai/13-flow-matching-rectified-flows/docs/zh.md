# 流匹配与修正流

> 扩散模型需要 20-50 个采样步骤，因为它们在噪声到数据的路径上沿着一条弯曲的路径前进。流匹配（Lipman 等人，2023）和修正流（Liu 等人，2022）训练了直线路径。更直的路径意味着更少的步骤意味着更快的推理。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切换到了流匹配。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 8 · 06（DDPM），阶段 1 · 微积分
**时间：** ~45 分钟

## 问题

DDPM 的反向过程是从 `N(0, I)` 回到数据分布的 1000 步随机游走。DDIM 将其坍缩为 20-50 步确定性步。你想要更少的步数——最好是一步。阻碍在于求解反向过程的 ODE 是刚性的；路径是弯曲的。

如果你能训练模型使得从噪声到数据的路径是一条*直线*，那么从 `t=1` 到 `t=0` 的单步欧拉迭代就能工作。流匹配直接构建了这一点：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值，训练向量场 `v_θ(x, t)` 匹配其时间导数，在推理时积分。

修正流（Liu 2022）更进一步：使用一个逐步使 ODE 越来越接近线性的重新流过程反复拉直路径。经过两次重新流迭代，2 步采样器就能匹配 50 步 DDPM 的质量。

## 概念

![流匹配：噪声和数据之间的直线插值](../assets/flow-matching.svg)

### 直线流

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data` 和 `x_1 ~ N(0, I)`。沿这条直线的导数恒定：

```
dx_t / dt = x_1 - x_0
```

定义一个神经向量场 `v_θ(x_t, t)` 并训练它匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是**条件流匹配**损失（Lipman 2023）。训练无需模拟：你从不展开 ODE。只需采样 `(x_0, x_1, t)` 并进行回归。

### 采样

推理时，*反向*积分学习的向量场：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，欧拉步进到 `t=0`。

### 修正流（Liu 2022）

直线流有效，但学习到的路径*并非真正直线*——它们会弯曲，因为许多 `x_0` 可能映射到同一个 `x_1`。修正流的重新流步骤：

1. 使用随机配对训练流模型 v_1。
2. 通过从 `x_1` 积分 v_1 到其终点 `x_0` 来采样 N 对 `(x_1, x_0)`。
3. 在这些配对示例上训练 v_2。因为配对现在是"ODE 匹配"的，它们之间的直线插值真正更平坦。
4. 重复。

在实践中，2 次重新流迭代就能达到接近线性，实现 2-4 步推理。SDXL-Turbo、SD3-Turbo、LCM 都是基于流匹配蒸馏的模型。

### 为什么这在 2024 年赢了图像生成

三个原因：

1. **无需模拟的训练**——训练期间无需 ODE 展开，实现起来微不足道。
2. **更好的损失几何**——直线路径具有一致的信噪比，而 DDPM 的 ε-损失在调度边缘有很差的 SNR。
3. **更快的推理**——4-8 步达到 SDXL-Turbo 质量；1 步通过一致性蒸馏。

## 流匹配与 DDPM——精确联系

带高斯条件路径的流匹配是*带有特定噪声调度的*扩散。选择 `x_t = α(t) x_0 + σ(t) x_1` 调度，流匹配恢复为 Stratonovich 重新表述的扩散，其中 `v = α'·x_0 - σ'·x_1`。对于高斯路径，两者在代数上等价。

流匹配增加的是：目标的*清晰性*（一个简单的速度）、一个更干净的损失，以及实验非高斯插值的自由度。

## 动手实现

`code/main.py` 在双峰高斯混合上实现 1-D 流匹配。向量场 `v_θ(x, t)` 是一个以直线为目标的微型 MLP。推理时，以 1、2、4 和 20 步欧拉积分并比较样本质量。

### 步骤 1：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # 反向传播 + 更新
```

### 步骤 2：多步推理

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 步骤 3：比较步数

期望 4 步采样器已经匹配 20 步的质量——这对延迟来说是一个巨大的进步。

## 陷阱

- **时间参数化。** 流匹配使用 `t ∈ [0, 1]`，其中 `t=0` 是数据，`t=1` 是噪声。DDPM 使用 `t ∈ [0, T]`，其中 `t=0` 是数据，`t=T` 是噪声。方向相同，尺度不同。论文经常搞错这一点。
- **调度选择。** 修正流的直线是"流匹配的调度"，但你可以使用余弦或 logit-normal t-采样（SD3 这样做）以获得更好的尺度覆盖。
- **重新流成本。** 为重新流生成配对数据集是每个样本的完整推理传递。只有当你真的需要 1-2 步推理时才做重新流。
- **无分类器引导仍然适用。** 只需在线性组合中用 v 替换 ε：`v_cfg = (1+w) v_cond - w v_uncond`。

## 使用

| 用例 | 2026 年技术栈 |
|----------|-----------|
| 文生图，最佳质量 | 流匹配：SD3、Flux.1-dev |
| 文生图，1-4 步 | 蒸馏流匹配：Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| 实时推理 | 从流匹配基础进行一致性蒸馏（LCM、PCM） |
| 音频生成 | 流匹配：Stable Audio 2.5、AudioCraft 2 |
| 视频生成 | 流匹配与扩散混合（Sora、Veo、Stable Video） |
| 科学/物理（粒子轨迹、分子） | 流匹配 + 等变向量场 |

当一篇论文在 2025-2026 年说"比扩散更快"时，它几乎总是流匹配 + 蒸馏。

## 产出

保存 `outputs/skill-fm-tuner.md`。技能接受一个扩散风格的模型规范并将其转换为流匹配训练配置：调度选择、时间采样分布（均匀/logit-normal）、优化器、重新流计划、目标步数和评估协议。

## 练习

1. **简单。** 运行 `code/main.py` 并比较 1 步 vs 20 步 MSE 与真实数据分布的对比。
2. **中等。** 从均匀 `t` 采样切换到 logit-normal（将采样集中在中间 t 值）。模型质量是否提高了？
3. **困难。** 实现一次重新流迭代：通过积分第一个模型生成配对的 (x_0, x_1)，在配对数据上训练第二个模型，并比较 1 步样本质量。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 流匹配 | "直线扩散" | 训练 `v_θ(x, t)` 以匹配插值路径上的 `x_1 - x_0`。 |
| 修正流 | "重新流" | 拉直学习到的流的迭代过程。 |
| 速度场 | "v_θ" | 模型的输出——移动 `x_t` 的方向。 |
| 直线插值 | "路径" | `x_t = (1-t)·x_0 + t·x_1`；简单的目标导数。 |
| 欧拉采样器 | "一阶 ODE 求解器" | 最简单的积分器；当路径直的时候工作良好。 |
| Logit-normal t | "SD3 采样" | 将 `t` 采样集中在梯度最强的中间值。 |
| 一致性蒸馏 | "一步采样器" | 训练学生模型将任意 `x_t` 直接映射到 `x_0`。 |
| 带速度的 CFG | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`；相同的技巧，新的变量。 |

## 生产说明：Flux.1-schnell 是最快的流匹配

流匹配在生产上的胜利是 Flux.1-schnell——一个流匹配的 DiT 蒸馏到 1-4 个推理步骤，同时保持 Flux-dev 级别的质量。Niels 的"在 8GB 机器上运行 Flux" notebook 是参考部署配方：T5 + CLIP 编码，量化 MMDiT 去噪（schnell 4 步 vs dev 50 步），VAE 解码。成本核算：

| 变体 | 步数 | L4 上 1024² 的延迟 | 总 FLOPs（相对） |
|---------|-------|------------------------|------------------------|
| Flux.1-dev（原始） | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08×（快 12 倍） |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2 步 | 2 | ~0.3 s | 0.03× |

生产规则：**流匹配基础 + 蒸馏 = 2026 年快速文生图的默认选择。** 每个主要厂商都提供这种组合：SD3-Turbo（SD3 + 流 + 蒸馏）、Flux-schnell（Flux-dev + 修正流拉直）、CogView-4-Flash。纯扩散基础仅存在于遗留检查点中。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 修正流。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — 流匹配。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，大规模修正流。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 涵盖 FM + 扩散的通用框架。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 扩散/流的一步蒸馏。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — Turbo 变体。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 生产中的流匹配。
