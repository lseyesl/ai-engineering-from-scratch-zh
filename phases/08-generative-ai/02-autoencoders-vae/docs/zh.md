# 自编码器与变分自编码器（VAE）

> 普通自编码器压缩然后重建。它死记硬背。它不能生成。加上一个技巧——强制编码看起来像高斯分布——你就得到了一个采样器。这一个技巧，即 `z = μ + σ·ε` 的重参数化，就是你 2026 年使用的每个潜在扩散和流匹配图像模型在输入端都有一个 VAE 的原因。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 3 · 02（反向传播），阶段 3 · 07（CNN），阶段 8 · 01（分类法）
**时间：** ~75 分钟

## 问题

将 784 像素的 MNIST 数字压缩为 16 个数字的编码，然后重建。普通自编码器可以出色地完成重建 MSE，但编码空间是一团乱麻。在编码空间中选取一个随机点，解码它，你得到的是噪声。它没有采样器。它是一个伪装成压缩模型的模型。

你真正想要的是：(a) 编码空间是一个你可以采样的干净、平滑的分布——比如各向同性高斯 `N(0, I)`，(b) 解码任何样本产生一个合理的数字，以及 (c) 编码器和解码器仍然能很好地压缩。三个目标，一种架构，一个损失。

Kingma 2013 年的 VAE 通过训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)`，通过 KL 惩罚将该分布拉向先验 `N(0, I)`，然后在解码前从 `q(z|x)` 采样 `z` 来解决这个问题。在推理时，丢弃编码器，采样 `z ~ N(0, I)`，解码。KL 惩罚正是迫使编码空间被结构化的原因。

在 2026 年，VAE 很少单独发布——它们在原始图像质量上已被扩散超越——但它们是每个潜在扩散模型（SD 1/2/XL/3、Flux、AudioCraft）的首选编码器。学会 VAE，你就学会了使用的每个图像流水线的不可见的第一层。

## 概念

![自编码器 vs VAE：重参数化技巧](../assets/vae.svg)

**自编码器。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。编码空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。这些定义了 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从 `q(z|x)` 采样不可微。将采样重写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 加上非参数噪声的确定性函数——梯度流经 `μ` 和 `σ`。

**损失。** 证据下界（ELBO），两项：

```
loss = 重建 + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建推动 `x̂` 向 `x` 靠拢。KL 推动 `q(z|x)` 向先验靠拢。它们互相权衡。小的 β（<1）= 更清晰的样本，编码空间不那么高斯。大的 β（>1）= 更干净的编码空间，更模糊的样本。β-VAE（Higgins 2017）使这个旋钮闻名，并开启了解耦表示研究。

**采样。** 推理时：抽取 `z ~ N(0, I)`，通过解码器前向传播。一次前向传播——不像扩散那样需要迭代采样。

```figure
vae-latent-grid
```

## 动手实现

`code/main.py` 无需 numpy 或 torch 实现了一个微型 VAE。输入是从 8 维中的 2 分量高斯混合中抽取的 8 维合成数据。编码器和解码器是单隐藏层 MLP。我们实现 tanh 激活、前向传播、损失和手写的反向传播。不是生产代码——教学目的。

### 步骤 1：编码器前向

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

使用 `log σ²` 而不是 `σ`，这样网络输出是无约束的（σ 的 softplus 是一个陷阱——梯度在 σ ≈ 0 时消失）。

### 步骤 2：重参数化和解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 步骤 3：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

精确的封闭形式 KL，因为两个分布都是高斯分布。不要进行数值积分。2026 年仍然有人发布使用蒙特卡洛 KL 估计的代码——这慢了 3 倍且毫无理由。

### 步骤 4：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## 陷阱

- **后验坍缩。** KL 项如此激进地将 `q(z|x) → N(0, I)`，以至于 `z` 携带了关于 `x` 的零信息。解决方案：β-退火（从 β=0 开始，逐渐增加到 1）、free bits、或在非活跃维度上跳过 KL。
- **模糊样本。** 高斯解码器似然意味着 MSE 重建，这对于 L2（均值）是贝叶斯最优的——一组合理数字的均值是一个模糊的数字。解决方案：离散解码器（VQ-VAE、NVAE），或仅将 VAE 作为编码器并在潜在空间上堆叠扩散（这就是 Stable Diffusion 所做的）。
- **β 太大，太早。** 参见后验坍缩。从 β≈0.01 开始并逐渐增加。
- **潜在维度太小。** MNIST 用 16-D，ImageNet 256² 用 256-D，ImageNet 1024² 用 2048-D。Stable Diffusion 的 VAE 将 512×512×3 压缩为 64×64×4（空间面积 32 倍下采样，通道 32 倍下采样）。

## 使用

2026 年 VAE 堆栈：

| 情况 | 选择 |
|-----------|------|
| 用于扩散的图像—潜在编码器 | Stable Diffusion VAE（`sd-vae-ft-ema`）或 Flux VAE |
| 音频—潜在编码器 | Encodec（Meta）、SoundStream 或 DAC（Descript） |
| 视频潜在变量 | Sora 的时空补丁、Latte VAE、WAN VAE |
| 解耦表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散潜在变量（用于 transformer 建模） | VQ-VAE、RVQ（ResidualVQ） |
| 用于生成的连续潜在变量 | 普通 VAE，然后在该潜在空间中条件化一个流/扩散模型 |

潜在扩散模型是一个 VAE，其编码器和解码器之间有一个扩散模型。VAE 做粗略压缩，扩散模型做繁重的工作。同样的模式也适用于视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen transformer）。

## 产出

保存 `outputs/skill-vae-trainer.md`。

技能接受：数据集概况 + 潜在维度目标 + 下游用途（重建、采样或潜在扩散输入）并输出：架构选择（普通/β/VQ/RVQ）、β 调度、潜在维度、解码器似然（高斯 vs 分类）和评估计划（重建 MSE、每维 KL、`q(z|x)` 和 `N(0, I)` 之间的 Fréchet 距离）。

## 练习

1. **简单。** 将 `code/main.py` 中的 `β` 更改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终的重建 MSE 和 KL。哪个 β 对你的合成数据是帕累托最优的？
2. **中等。** 用伯努利似然（交叉熵损失）替换高斯解码器似然。在相同合成数据的二值化版本上比较样本质量。
3. **困难。** 将 `code/main.py` 扩展为迷你 VQ-VAE：将连续 `z` 替换为在 K=32 条目的码本中的最近邻查找。比较重建 MSE 并报告使用了多少码本条目（码本坍缩是真实存在的）。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 自编码器 | 编码—解码网络 | `x → z → x̂`，学习 MSE。不是生成式。 |
| VAE | 带采样器的自编码器 | 编码器输出分布，KL 惩罚塑造编码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；当 `q = p(z\|x)` 时紧致。 |
| 重参数化 | `z = μ + σ·ε` | 将随机节点重写为确定性 + 纯噪声。使反向传播能够通过采样。 |
| 先验 | `p(z)` | 潜在变量的目标分布，通常为 `N(0, I)`。 |
| 后验坍缩 | "KL 项赢了" | 编码器忽略 `x`，输出先验；解码器必须幻觉。 |
| β-VAE | 可调 KL 权重 | `loss = recon + β·KL`。β 越高 = 更解耦但更模糊。 |
| VQ-VAE | 离散潜在变量 | 用最近码本向量替换连续 `z`；使 transformer 建模成为可能。 |

## 生产说明：VAE 是扩散服务器中最热门的路径

在 Stable Diffusion / Flux / SD3 流水线中，VAE 每个请求被调用两次——一次编码（如果做 img2img / 修补）和一次解码。在 1024² 时，解码器传递通常是整个流水线中最大的激活内存峰值，因为它将 `128×128×16` 的潜在变量上采样回 `1024×1024×3`。两个实际后果：

- **切片或分块解码。** `diffusers` 暴露了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。分块以微小的接缝伪影换取 `O(tile²)` 内存而不是 `O(H·W)`。在消费级 GPU 上处理 1024²+ 时必不可少。
- **bf16 解码器，fp32 数值用于最终缩放。** SD 1.x VAE 以 fp32 发布，在转换为 fp16 后*静默产生 NaN*（在 1024²+ 时）。SDXL 提供了 `madebyollin/sdxl-vae-fp16-fix`——始终优先选择 fp16-fix 变体或使用 bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 解耦 β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 最先进的图像 VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为编码器。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 标准。
