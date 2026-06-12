# GAN — 生成器 vs 判别器

> Goodfellow 在 2014 年的技巧是彻底跳过密度。两个网络。一个造假。一个抓假。它们互相博弈直到假货与真货无法区分。它本不该有效。它经常不工作。但当它有效时，其样本仍然是窄域文献中最清晰的。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 3 · 02（反向传播），阶段 3 · 08（优化器），阶段 8 · 02（VAE）
**时间：** ~75 分钟

## 问题

VAE 产生模糊样本，因为它们的 MSE 解码器损失对于*均值*图像是贝叶斯最优的——而许多合理数字的均值是一个模糊的数字。你需要一个奖励*合理性*的损失，而不是像素级接近任何单一目标。合理性没有封闭形式。你必须学习它。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像和伪造品。训练一个生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号是 `D` 当前认为使某物看起来真实的东西。这个信号随着 `G` 的改进而更新，追逐一个移动的目标。如果两个网络收敛，`G` 已经学习了数据分布，而从未写下 `log p(x)`。

这就是对抗训练。数学上是一个极小极大博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

在 2026 年，GAN 不再是 SOTA 生成器（扩散和流匹配夺走了这个桂冠）。但 StyleGAN 2/3 仍然是有史以来最清晰的人脸模型，GAN 判别器被用作扩散训练中的*感知损失*，而对抗训练驱动了快速的 1 步蒸馏（SDXL-Turbo、SD3-Turbo、LCM），让你能够提供实时扩散。

## 概念

![GAN 训练：极小极大中的生成器和判别器](../assets/gan.svg)

**生成器 `G(z)`。** 将噪声向量 `z ~ N(0, I)` 映射到样本 `x̂`。一个解码器形状的网络（稠密或转置卷积）。

**判别器 `D(x)`。** 将样本映射到标量概率（或分数）。真实 → 1，伪造 → 0。

**损失。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。在真实=1、伪造=0 上的二元交叉熵。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和*形式（原始的 `log(1 - D(G(z)))` 在 `D` 确定时会饱和并杀死梯度）。

**训练循环。** `D` 一步，`G` 一步。重复。

**为什么有效。** 如果 `G` 完美匹配 `p_data`，那么 `D` 无法做得比随机更好，各处输出 0.5；`G` 不再获得梯度。均衡。

**为什么失效。** 模式坍缩（`G` 找到了 `D` 无法分类的一个模式并永远生成它）、梯度消失（`D` 学习太快，`log D` 饱和）、训练不稳定（学习率、批次大小、任何东西）。

## 使 GAN 工作的变体

| 年份 | 创新 | 修复 |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv、批归一化、LeakyReLU——第一个稳定架构。 |
| 2017 | WGAN、WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替换 BCE。修复梯度消失。 |
| 2017 | 谱归一化 | Lipschitz 约束判别器。2026 年仍在使用。 |
| 2018 | Progressive GAN | 先训练低分辨率，再添加层。首个百万像素结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化。固定域照片级真实感的 SOTA。 |
| 2021 | StyleGAN3 | 无混叠、平移等变——2026 年仍然是人脸黄金标准。 |
| 2022 | StyleGAN-XL | 条件式、类别感知、更大规模。 |
| 2024 | R3GAN | 用更强的正则化重新品牌化；无需技巧即可在 1024² 上工作。 |

```figure
gan-minimax
```

## 动手实现

`code/main.py` 在一维数据上训练一个微型 GAN：两个高斯分布的混合。生成器和判别器是单隐藏层 MLP。我们手工实现前向、反向和极小极大循环。目标是看到两个关键失败模式（模式坍缩 + 梯度消失）的发生。

### 步骤 1：非饱和损失

原始的 Goodfellow 损失 `log(1 - D(G(z)))` 在 D 高置信度地将 G 的伪造品分类为伪造时趋于 0。此时 G 的梯度基本上为零——G 无法改进。非饱和形式 `-log D(G(z))` 有相反的渐近线：当 D 确定时它爆发，给 G 提供强信号。

```python
def g_loss(d_fake):
    # 最大化 log D(G(z))  <=>  最小化 -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 步骤 2：每生成器一步对应一个判别器步

```python
for step in range(steps):
    # 训练 D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # 训练 G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # 新的伪造品
    update_G(fake_batch)
```

G 使用新的伪造品，否则梯度是过时的。

### 步骤 3：观察模式坍缩

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] 模式坍缩：一个模式已被饿死")
```

典型症状：两个真实模式之一不再被生成。判别器不再纠正它，因为它从未被视为伪造品。

## 陷阱

- **判别器太强。** 将 D 的学习率降低 2-5 倍，或添加实例/层噪声。如果 D 达到 >95% 准确率，G 就死了。
- **生成器记忆了一个模式。** 向 D 输入添加噪声，使用小批量判别层，或切换到 WGAN-GP。
- **批归一化泄漏统计量。** 真实批次 + 伪造批次流过同一 BN 层会混合它们的统计量。改用实例归一化或谱归一化。
- **Inception 分数博弈。** FID 和 IS 在低样本数时噪声很大。评估时使用 ≥10k 样本。
- **一次性采样对于条件任务是谎言。** 你仍然需要 CFG 尺度、截断技巧和重新采样才能获得可用的输出。

## 使用

2026 年 GAN 堆栈：

| 情况 | 选择 |
|-----------|------|
| 照片级真实感人脸，固定姿态 | StyleGAN3（最清晰，最小） |
| 动漫 / 风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像翻译 | Pix2Pix / CycleGAN（阶段 8 · 04）或 ControlNet（阶段 8 · 08） |
| 快速 1 步文生图 | 扩散的对抗蒸馏（SDXL-Turbo、SD3-Turbo） |
| 扩散训练器内的感知损失 | 图像裁剪上的小型 GAN 判别器 |
| 任何多模态、开放式的任务 | 不要——使用扩散或流匹配 |

GAN 清晰但狭窄。一旦你的领域开放——照片、任意文本提示、视频——切换到扩散。对抗技巧作为组件（感知损失、蒸馏）继续存在，而不是独立的生成器。

## 产出

保存 `outputs/skill-gan-debugger.md`。技能接受一个失败的 GAN 运行（损失曲线、样本网格、数据集大小）并输出可能原因的排序列表、一行修复方案和重新运行协议。

## 练习

1. **简单。** 使用默认设置运行 `code/main.py`。然后将 `D_LR = 5 * G_LR` 并重新运行。G 的损失多快坍缩到常数？
2. **中等。** 将 Goodfellow BCE 损失替换为 WGAN 损失：`loss_D = E[D(fake)] - E[D(real)]`、`loss_G = -E[D(fake)]`，并将 D 的权重裁剪到 `[-0.01, 0.01]`。训练是否更稳定？比较挂钟收敛时间。
3. **困难。** 将一维示例扩展到二维数据（环上的 8 个高斯混合）。跟踪生成器在第 1k、5k、10k 步捕获了 8 个模式中的多少个。实现小批量判别并重新测量。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 生成器（G） | "G" | 噪声到样本网络，`G: z → x̂`。 |
| 判别器（D） | "D" | 分类器 `D: x → [0, 1]`，真实 vs 伪造。 |
| 极小极大 | "博弈" | 联合目标的 `min_G max_D`。 |
| 非饱和损失 | "修复方案" | 对 G 使用 `-log D(G(z))` 而非 `log(1 - D(G(z)))`。 |
| 模式坍缩 | "G 记住了一件事" | 生成器仅产生少数不同输出，尽管数据多样。 |
| WGAN | "Wasserstein" | 用推土机距离 + 梯度惩罚替换 BCE；梯度更平滑。 |
| 谱归一化 | "Lipschitz 技巧" | 约束 D 的权重范数以限制其斜率；稳定训练。 |
| StyleGAN | "有效的那一个" | 映射网络 + AdaIN；最佳水平人脸模型，2026 年仍然如此。 |

## 生产说明：一次性推理是 GAN 的持久优势

GAN 在开放域生成的样本质量上不再胜出，但它们在推理成本上仍然胜出。在生产推理文献的词汇中，一个 GAN 有：

- **没有预填充，没有解码阶段。** 单一 `G(z)` 前向传播。TTFT ≈ 总延迟。
- **没有 KV 缓存压力。** 唯一的状态是权重。批次大小由激活内存限制，而非缓存。
- **平凡的连续批处理。** 由于每个请求占用相同的固定 FLOPs，服务器目标占用的静态批次通常是最优的。不需要运行中调度器。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年快速文生图的主导技术：它将 20-50 步的扩散流水线坍缩为 1-4 步 GAN 风格的前向传播，同时保持扩散基础的分布。对抗损失作为一种训练时的旋钮而存活，用于将慢速生成器变成快速生成器。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 原始 GAN 论文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 第一个稳定架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — 谱归一化。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo。
