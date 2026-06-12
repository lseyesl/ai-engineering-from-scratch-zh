# 评估——FID、CLIP 分数、人类偏好

> 每个生成模型排行榜都引用 FID、CLIP 分数和来自人类偏好竞技场的胜率。每个数字都有一个有决心的研究者可以利用的失败模式。如果你不了解这些失败模式，你就无法区分真正的改进和作弊运行。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 8 · 01（分类法），阶段 2 · 04（评估指标）
**时间：** ~45 分钟

## 问题

生成模型根据*样本质量*和*条件遵循度*来评判。两者都没有封闭形式的度量。你的模型必须渲染 10,000 张图像；某物必须为它们分配数字；你必须在不同模型家族、不同分辨率和不同架构下信任这些数字。三个指标在 2014-2026 年的淘汰中存活下来：

- **FID（Fréchet Inception Distance）。** 在 Inception 网络的特征空间中，真实分布和生成分布之间的距离。越低越好。
- **CLIP 分数。** 生成图像的 CLIP 图像嵌入和提示的 CLIP 文本嵌入之间的余弦相似度。越高越好。衡量提示遵循度。
- **人类偏好。** 将两个模型在相同提示上进行头对头比较，让人类（或 GPT-4 级模型）选择更好的一个，聚合为 Elo 分数。

你还会看到：IS（Inception 分数，大部分已弃用）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每个都修正了前一个的一个失败。

## 概念

![FID、CLIP 和偏好：三个轴，不同的失败模式](../assets/evaluation.svg)

### FID——样本质量

Heusel 等人（2017）。步骤：

1. 提取 N 张真实图像和 N 张生成图像的 Inception-v3 特征（2048-D）。
2. 为每个集合拟合高斯分布：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：特征空间中两个多元高斯之间的 Fréchet 距离。越低 = 分布越相似。

失败模式：
- **小 N 时的偏差。** FID 是特征分布上的均方——小 N 低估协方差，给出虚假的低 FID。始终使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 在 ImageNet 上训练。远离 ImageNet 的领域（人脸、艺术、文本图像）会产生无意义的 FID。使用领域特定的特征提取器。
- **作弊。** 过度拟合 Inception 先验会给出低 FID 而没有视觉质量改进。使用 CMMD 击败它。

### CLIP 分数——提示遵循度

Radford 等人（2021）。对于生成的图像 + 提示：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在 30k 张生成的图像上取平均 → 一个可在模型之间比较的标量。

失败模式：
- **CLIP 自身的盲点。** CLIP 的组合推理能力较弱（"蓝色球体上的红色立方体"经常失败）。模型可以在 CLIP 分数上排名靠前而实际上不遵循复杂提示。
- **短提示偏差。** 短提示在现实中有更多的 CLIP 图像匹配。较长的提示在机械上具有较低的 CLIP 分数。
- **提示作弊。** 在提示中包含"高质量、4k、杰作"会抬高 CLIP 分数而不改善图像-文本绑定。

CMMD（Jayasumana 等人，2024）修复了其中一些问题：使用 CLIP 特征代替 Inception，使用最大均值差异代替 Fréchet。能更好地检测细微的质量差异。

### 人类偏好——真实标准

选择一个提示池。用模型 A 和模型 B 生成图像。向人类（或一个强大的 LLM 评判器）展示配对。将胜出情况聚合成 Elo 或 Bradley-Terry 分数。基准：

- **PartiPrompts（Google）：** 1,600 个多样化提示，12 个类别。
- **HPSv2：** 107k 个人工标注，广泛用作自动化代理。
- **ImageReward：** 137k 提示-图像偏好对，MIT 许可。
- **PickScore：** 在 Pick-a-Pic 260 万偏好上训练。
- **聊天机器人竞技场风格的图像竞技场：** https://imagearena.ai/ 等。

失败模式：
- **评判者的方差。** 非专家与专家有不同的偏好。两者都使用。
- **提示分布。** 精心挑选的提示有利于某一家族。始终记录。
- **LLM 评判者的奖励作弊。** GPT-4 评判者会被漂亮但错误的输出所迷惑。与人类进行三角验证。

## 一起使用

一个生产级评估报告应包括：

1. 对 10-30k 样本与保留的真实分布的 FID（样本质量）。
2. 相同样本与其提示的 CLIP 分数 / CMMD（遵循度）。
3. 在前一模型的盲测竞技场中的胜率（总体偏好）。
4. 失败模式分析：50 个随机抽样输出，标记已知问题（手部解剖、文本渲染、一致的物体计数）。

任何一个单独的指标都是谎言。三个相互佐证的指标 + 定性评估才是真正的声明。

## 动手实现

`code/main.py` 在合成"特征向量"上实现了 FID、类似 CLIP 分数的计算和 Elo 聚合（我们使用 4-D 向量作为 Inception 特征的替代品）。你将看到：

- 小 N 和大 N 下的 FID 计算——偏差。
- 特征池之间的余弦相似度作为"CLIP 分数"。
- 从合成偏好流中得到的 Elo 更新规则。

### 步骤 1：四行 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 步骤 2：CLIP 风格的余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 步骤 3：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 陷阱

- **FID 在 N=1000 时。** 在 N 低于 10k 时启发式不可靠。报告低 N 版本 FID 的论文是在作弊。
- **跨分辨率比较 FID。** Inception 的 299×299 调整大小改变了特征分布。仅在匹配的分辨率下比较。
- **报告一个种子。** 最少运行 3 个种子。报告标准差。
- **通过负面提示抬高 CLIP 分数。** 一些管线通过过度拟合提示来提升 CLIP。检查视觉饱和度。
- **提示重叠导致的 Elo 偏差。** 如果两个模型在训练期间都见过基准提示，Elo 是没有意义的。使用保留的提示集。
- **人类评估付费群体的偏差。** Prolific、MTurk 的标注者偏年轻/技术友好。与招募的艺术/设计专家结合。

## 使用

2026 年生产级评估协议：

| 支柱 | 最低要求 | 推荐 |
|--------|---------|-------------|
| 样本质量 | 10k 样本 vs 保留真实数据的 FID | + 5k 样本 CMMD + 每类别子集 FID |
| 提示遵循度 | 30k 样本的 CLIP 分数 | + HPSv2 + ImageReward + VQA 风格问答 |
| 偏好 | 200 个与基线的盲测配对 | + 2000 配对人工 + LLM 评判者 + 聊天机器人竞技场 |
| 失败分析 | 50 个手工标记 | 500 个手工标记 + 自动安全分类器 |

报告中的全部四个支柱 = 真正的声明。任何一个单独 = 营销。

## 产出

保存 `outputs/skill-eval-report.md`。技能接受一个新模型检查点 + 基线并输出一个完整的评估计划：样本大小、指标、失败模式探针和签字标准。

## 练习

1. **简单。** 运行 `code/main.py`。在相同的合成分布上比较 N=100 与 N=1000 时的 FID。报告偏差大小。
2. **中等。** 从合成 CLIP 风格的特征实现 CMMD（公式参见 Jayasumana 等人，2024）。比较与 FID 相比对质量差异的敏感性。
3. **困难。** 复现 HPSv2 设置：从 Pick-a-Pic 的一个子集中取 1000 个图像-提示对，在偏好上微调一个小型基于 CLIP 的评分器，并测量其与保留集的契合度。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | 真实与生成 Inception 特征的高斯拟合之间的 Fréchet 距离。 |
| CLIP 分数 | "文本-图像相似度" | CLIP 图像和文本嵌入之间的余弦相似度。 |
| CMMD | "FID 的替代" | CLIP 特征 MMD；偏差较小，无高斯假设。 |
| IS | "Inception 分数" | exp KL(p(y|x) || p(y))；在现代模型上相关性差，已弃用。 |
| HPSv2 / ImageReward / PickScore | "学习到的偏好代理" | 在人类偏好上训练的小模型；用作自动化评判者。 |
| Elo | "国际象棋等级分" | 成对胜利的 Bradley-Terry 聚合。 |
| PartiPrompts | "基准提示集" | 1,600 个 Google 策划的提示，涵盖 12 个类别。 |
| FD-DINO | "自监督替代" | 使用 DINOv2 特征的 FD；对非 ImageNet 领域更好。 |

## 生产说明：评估也是推理工作负载

在 10k 样本上运行 FID 意味着生成 10k 张图像。对于在单个 L4 上以 1024² 运行的 50 步 SDXL 基础模型，这是约 11 小时的单请求推理。评估预算是真实的，其框架与离线推理场景完全相同（最大化吞吐量，忽略 TTFT）：

- **硬性批处理，忘记延迟。** 离线评估 = 在内存允许的最大批次大小下进行静态批处理。在 80GB H100 上使用 `num_images_per_prompt=8` 的 `pipe(...).images` 比单请求快 4-6 倍。
- **缓存真实特征。** 真实参考集上的 Inception（FID）或 CLIP（CLIP 分数、CMMD）特征提取*运行一次*，存储为 `.npz` 文件。不要在每次评估时重新计算。

对于 CI / 回归门控：每个 PR 在 500 样本子集上运行 FID + CLIP 分数（约 30 分钟）；每晚运行完整 10k FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失败模式调查。
