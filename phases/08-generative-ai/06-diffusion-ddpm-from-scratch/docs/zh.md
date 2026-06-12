# 扩散模型——从头实现 DDPM

> Ho、Jain、Abbeel（2020）给了这个领域一个无法抗拒的配方。通过一千个小步用噪声摧毁数据。训练一个神经网络来预测噪声。在推理时逆转这个过程。今天，每一个主流的图像、视频、3D 和音乐模型都在这个循环上运行，可能在其上叠加了流匹配或一致性技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 3 · 02（反向传播），阶段 8 · 02（VAE）
**时间：** ~75 分钟

## 问题

你想要一个 `p_data(x)` 的采样器。GAN 玩一个经常发散的极小极大博弈。VAE 从高斯解码器产生模糊样本。你真正想要的是一个训练目标，它 (a) 是单一的稳定损失（没有鞍点，没有极小极大），(b) 是 `log p(x)` 的下界（所以你有似然），以及 (c) 产生匹配 SOTA 质量的样本。

Sohl-Dickstein 等人（2015）有一个理论上的答案：定义一个马尔可夫链 `q(x_t | x_{t-1})` 逐渐添加高斯噪声，并训练一个反向链 `p_θ(x_{t-1} | x_t)` 去噪。Ho、Jain、Abbeel（2020）表明损失可以简化为一行——预测噪声——并清理了数学。在 2020 年这还是一个好奇的尝试。在 2021 年它产生了最先进的样本。在 2022 年它变成了 Stable Diffusion。在 2026 年它是基础。

## 概念

![DDPM：前向噪声，反向去噪](../assets/ddpm.svg)

**前向过程 `q`。** 在 `T` 个小步中添加高斯噪声。封闭形式——数学可处理的原因——是累积步骤也是高斯分布：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，针对一个 `β_t` 调度。选择 `β_t` 从 1e-4 到 0.02 线性变化，T=1000 步，`x_T` 近似为 `N(0, I)`。

**反向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)` 预测添加的噪声。给定 `x_t`，通过以下方式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 是 `sqrt(β_t)` 或一个学习到的方差。这个表达式看起来很丑陋，但它只是代数——根据后验 `q(x_{t-1} | x_t, x_0)` 求解 `x_{t-1}`，并用噪声预测的估计替换 `x_0`。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，选择一个随机 `t`，采样 `ε ~ N(0, I)`，通过封闭形式一步计算带噪的 `x_t`，并对噪声进行回归。一个损失，没有极小极大，没有 KL，没有重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 到 `1` 迭代反向步骤。完成。

## 为什么有效

三个直觉：

1. **去噪容易；生成困难。** 在 `t=T` 时，数据是纯噪声——网络只需要解决一个微不足道的问题。在 `t=0` 时，网络只清理几个像素。在中间的 `t`，问题很难，但网络从每个噪声层级通过相同的权重获得了许多梯度流。

2. **伪装下的分数匹配。** Vincent（2011）证明预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即*分数*。反向 SDE 使用这个分数沿密度梯度上升——一个向高概率区域的引导随机游走。

3. **ELBO 简化为简单的 MSE。** 完整的变分下界在每个时间步有一个 KL 项。通过 DDPM 的参数化，这些 KL 项简化为对噪声预测的 MSE，带有特定的系数；Ho 去掉了这些系数（称之为"简单"损失），质量反而*提高了*。

```figure
diffusion-denoise
```

## 动手实现

`code/main.py` 实现了一个 1-D 的 DDPM。数据是一个双峰混合。"网络"是一个微型 MLP，输入 `(x_t, t)`，输出预测的噪声。训练是一行损失。采样迭代反向链。

### 步骤 1：前向调度（封闭形式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 步骤 2：一步采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 步骤 3：一个训练步

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 步骤 4：反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于一个使用 40 个时间步和 24 单元 MLP 的 1-D 问题，这在大约 200 个 epoch 内就学会了双峰混合。

## 时间步条件化

网络需要知道它在去噪哪个时间步。两个标准选项：

- **正弦嵌入。** 类似 Transformer 的位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过一个 MLP，广播到网络中。
- **FiLM / 组归一化条件化。** 将嵌入投影到每通道的缩放/偏置（FiLM），作用于每个块。

我们的玩具代码使用正弦嵌入 → 拼接。生产级 U-Net 使用 FiLM。

## 陷阱

- **调度非常重要。** 线性 `β` 是 DDPM 的默认选择，但余弦调度（Nichol & Dhariwal，2021）在相同计算量下给出更好的 FID。如果质量停滞不前，切换调度。
- **时间步嵌入很脆弱。** 将原始 `t` 作为浮点数传递对玩具 1-D 有效，但对图像失效；始终使用适当的嵌入。
- **V-预测 vs ε-预测。** 在狭窄区域（非常小或非常大的 t），`ε` 的信噪比很差。V-预测（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 都使用它。
- **无分类器引导。** 推理时，同时计算条件和无条件的 `ε`，然后使用 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，其中 `w ≈ 3-7`。在第 8 课中详细介绍。
- **1000 步太多了。** 生产环境使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。见第 12 课。

## 使用

| 角色 | 2026 年的典型技术栈 |
|------|-----------------------|
| 图像像素空间扩散（小型、玩具） | DDPM + U-Net |
| 图像潜在扩散 | VAE 编码器 + U-Net 或 DiT（第 7 课） |
| 视频潜在扩散 | 时空 DiT（Sora、Veo、WAN） |
| 音频潜在扩散 | Encodec + 扩散 Transformer |
| 科学（分子、蛋白质、物理） | 等变扩散（EDM、RFdiffusion、AlphaFold3） |

扩散是通用的生成骨干。流匹配（第 13 课）是 2024-2026 年的竞争者，通常在相同质量下在推理速度上胜出。

## 产出

保存 `outputs/skill-diffusion-trainer.md`。技能接受一个数据集 + 计算预算，并输出：调度（线性/余弦/sigmoid）、预测目标（ε/v/x）、步数、引导尺度、采样器家族和评估协议。

## 练习

1. **简单。** 在 `code/main.py` 中将 T 从 40 改为 10。样本质量（输出的可视化直方图）如何下降？在哪个 T 值下，双峰结构会坍缩？
2. **中等。** 从 ε-预测切换到 v-预测。重新推导反向步骤。比较最终的样本质量。
3. **困难。** 添加无分类器引导。以类别标签 `c ∈ {0, 1}` 为条件，在训练期间 10% 的概率丢弃它，并在采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。在 `w = 0, 1, 3, 7` 时测量条件模式命中率。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 前向过程 | "添加噪声" | 固定的马尔可夫链 `q(x_t \| x_{t-1})`，摧毁数据。 |
| 反向过程 | "去噪" | 学习到的链 `p_θ(x_{t-1} \| x_t)`，重构数据。 |
| β 调度 | "噪声阶梯" | 每步方差；线性、余弦或 sigmoid。 |
| α̅ | "Alpha bar" | 累积乘积 `∏(1 - β)`；给出从 `x_0` 到 `x_t` 的封闭形式。 |
| 简单损失 | "噪声上的 MSE" | `\|\|ε - ε_θ(x_t, t)\|\|²`；所有变分推导都归结为此。 |
| ε-预测 | "预测噪声" | 输出是添加的噪声；标准 DDPM。 |
| V-预测 | "预测速度" | 输出是 `α·ε - σ·x`；在 t 上的条件化更好。 |
| DDPM | "那篇论文" | Ho 等人 2020；线性 β、1000 步、U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50 步，相同的训练目标。 |
| 无分类器引导 | "CFG" | 混合条件和无条件的噪声预测以放大条件化。 |

## 生产说明：扩散推理是一个步数问题

DDPM 论文运行 T=1000 反向步。没有人在生产中这样做。每个真实的推理栈都选择三种策略之一——每种策略都清晰地映射到生产框架中的"延迟来自哪里"：

1. **更快的采样器，相同的模型。** DDIM（20-50 步）、DPM-Solver++（10-20）、UniPC（8-16）。反向循环的直接替换；训练的 `ε_θ` 权重不变。延迟降低 20-50 倍。
2. **蒸馏。** 训练一个学生模型在更少的步数中匹配教师：渐进式蒸馏（2 → 1）、一致性模型（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。延迟再降低 5-10 倍，需要重新训练。
3. **缓存和编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA 注意力、bf16 权重。每步延迟降低约 2 倍。可与 (1) 和 (2) 叠加。

对于生产扩散服务器，预算对话与生产 LLM 文献描述相同：延迟是 `num_steps × step_cost + VAE_decode`，吞吐量是 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；TPOT 等价物是完整响应时间，因为图像生成从用户角度看是"一次性"的。

## 延伸阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 扩散论文，领先于时代。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，更少的步数。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — 余弦调度，学习方差。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — 分类器引导。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 统一符号，最干净的配方。
