# CLIP 与对比式视觉语言预训练

> OpenAI 的 CLIP（2021 年）证明了一个足以驱动未来五年的核心思想：仅使用噪杂的网络图文对和对比损失，将图像编码器和文本编码器对齐到同一向量空间。零监督标签。4 亿对数据。由此产生的嵌入空间实现了零样本分类、图文检索，并作为视觉塔接入每一个 2026 年的 VLM。SigLIP 2（2025 年）用 sigmoid 替代了 softmax，以更低的成本超越了 CLIP。本课程将讲解从 InfoNCE 到 sigmoid 成对损失的数学原理，并用标准库 Python 实现训练步骤。

**类型：** 构建
**语言：** Python（标准库，InfoNCE + sigmoid 损失实现）
**前置要求：** Phase 12 · 01（ViT patch），Phase 7（Transformer）
**时间：** ~180 分钟

## 学习目标

- 从互信息推导出 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释为什么 sigmoid 成对损失（SigLIP）可以扩展到 batch 32768+，而无需 softmax 所需的 all-gather 通信开销。
- 通过构建文本模板（`a photo of a {class}`）并取余弦相似度的 argmax，运行零样本 ImageNet 分类。
- 说出 CLIP / SigLIP 预训练给你的四个杠杆：batch 大小、温度、提示模板、数据质量。

## 问题

CLIP 之前的视觉是监督式的。收集带标签的数据集（ImageNet：120 万张图像，1000 个类别），训练 CNN，发布。标签昂贵，标签偏向于标注者能达成一致的内容，并且标签不能在没有微调的情况下迁移到新任务。

互联网上的图像-描述数据有十亿以上的松散标注对，免费可用。一张金毛犬的图片配上替代文本"我的狗 Max 在公园里"携带了监督信号——文本描述了图像。问题是：你能将此转化为有用的训练吗？

CLIP 的答案是：将图文对视为一个匹配任务。给定一批 N 张图像和 N 个描述，学习将每张图像与它自己的描述匹配，区别于 N-1 个干扰项。监督信号是"这两个东西属于一起；这 N-1 个不属于。"没有类别标签。没有人工标注。只有一个对比损失。

由此产生的嵌入空间做了比 CLIP 训练目标更多的事情。ImageNet 零样本有效是因为"一张猫的照片"的嵌入靠近那些从未被明确标记为猫的猫图片附近。这就是催生了每一个 2026 年 VLM 的赌注。

## 概念

### 双编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 Transformer，每个描述输出一个 D 维向量。

两个塔都将其输出归一化为单位长度。相似度是 `cos(f(x), g(y)) = f(x)^T g(y)`，因为两者都是单位向量。

对于一批 N 个（图像，描述）对，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是一个学习到的温度（CLIP 初始化为 0.07；在对数空间中学习）。

### InfoNCE 损失

CLIP 在行和列上使用对称的交叉熵：

```
loss_i2t = CE(S, labels=identity)     # 每张图像的正样本是它自己的描述
loss_t2i = CE(S^T, labels=identity)   # 每个描述的正样本是它自己的图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 迫使每张图像与它自己的描述匹配的程度超过 batch 中的所有其他描述。"负样本"是所有其他 batch 项。更大的 batch = 更多负样本 = 更强的信号。CLIP 以 batch 32k 训练；规模很重要。

### 温度

`tau` 控制 softmax 的锐度。低 tau → 尖锐分布，难负样本挖掘效果。高 tau → 柔和，所有样本都有贡献。CLIP 学习 log(1/tau)，并限制下界以防止坍缩。SigLIP 2 固定初始 tau，使用学习到的偏置代替。

### 为什么 sigmoid 扩展性更好（SigLIP）

Softmax 需要整个相似度矩阵保持同步。在分布式训练中，你必须将每个嵌入 all-gather 到每个副本，然后执行 softmax。这对通信来说是世界大小的二次复杂度。

SigLIP 用逐元素 sigmoid 替代 softmax：对于每对 `(i, j)`，损失是一个二分类问题——"这些是匹配对吗？"正类标签是对角线，其他都是负样本。损失是：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 如果 `i == j`，否则为 0。每对损失是独立的。不需要 all-gather。每个 GPU 计算其本地块并求和。SigLIP 2 可以以较低的成本扩展到 batch 32k-512k，而 CLIP 需要成比例地增加通信。

### 零样本分类

给定 N 个类别名称，为每个类别构建一个文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入你的图像。取余弦相似度的 argmax = 预测类别。无需在目标类上训练。

提示模板很重要。CLIP 的原始论文每个类别使用了 80 个模板（普通、艺术、照片、绘画等）并对嵌入取平均。获得了 +3 ImageNet 点数。现代用法通常选择一个或两个模板。

### 线性探测和微调

零样本是一个基线。线性探测（在冻结的 CLIP 特征之上为你的目标类训练一个线性层）在领域内任务上优于零样本。完全微调在领域内优于线性探测，但可能损害零样本迁移。三种方案，三种权衡。

### SigLIP 2：NaFlex 和密集特征

SigLIP 2（2025 年）增加了：
- NaFlex：单个模型处理可变的宽高比和分辨率。
- 更好的分割和深度估计密集特征，目标是作为 VLM 中的冻结骨干网使用。
- 多语言：在 100+ 种语言上训练，而 CLIP 仅限英语。
- 10 亿参数规模，而 CLIP 最高为 4 亿。

在 2026 年的开源 VLM 中，SigLIP 2 SO400m/14 是默认的视觉塔。CLIP 仍然是纯图文检索的默认选择，前提是特定的 LAION-2B 训练分布与你的查询模式匹配。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021 年）：与 CLIP 相同的想法，18 亿对规模，90% 是噪杂数据。证明了噪杂数据可以规模缩放。OpenCLIP（LAION）：CLIP 在 LAION-400M / 2B 上的开源复现，多种规模，是首选的开源 checkpoint。EVA-CLIP：从掩码图像建模初始化；是 VLM 的强骨干网。BASIC：Google 的 CLIP+ALIGN 混合体。都是同一家族，只是数据和调优不同。

### 零样本的天花板

CLIP 类模型在 ImageNet 零样本上大约封顶在 76%（CLIP-G，OpenCLIP-G）。要超越需要更大的数据（SigLIP 2 达到 80%+）或架构变化（监督头，更多参数）。基准测试正在饱和；真正的价值在于下游 VLM 消费的嵌入空间。

```figure
multimodal-fusion
```

## 使用它

`code/main.py` 实现了：

1. 一个玩具双编码器（基于哈希的图像特征，文本字符特征），让你无需 numpy 就能看到 InfoNCE 的形状。
2. 纯 Python 实现的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定性）。
3. 用于比较的 sigmoid 成对损失。
4. 一个零样本分类例程：计算与一组文本提示的余弦相似度，取 argmax 进行预测。

运行它并观察损失曲线。绝对数值是玩具级别的；但形状与真实 CLIP 训练器发出的信号匹配。

## 交付物

本课程产生 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和一个目标类列表，它使用 CLIP 模板构建文本提示，用指定的 checkpoint（例如 `openai/clip-vit-large-patch14`）嵌入两侧，并返回 top-1 / top-5 预测及相似度分数。该技能拒绝对提示列表中不存在的类别做出声明。

## 练习

1. 手动为 4 对 batch 实现 InfoNCE。构建 4x4 相似度矩阵，运行 softmax，提取对角线，计算交叉熵。验证你的 Python 实现与这个手动计算一致。

2. SigLIP 在温度之外还使用了一个偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 存在严重的类别不平衡（每行负样本远多于正样本）时，`b` 扮演什么角色？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 为猫 vs 狗构建一个零样本分类器。尝试两个提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板集成是否优于单个模板？

4. 计算在 512-GPU 运行、batch 32k 下，softmax InfoNCE 与 sigmoid 成对损失的通信成本。哪个是 O(N) 扩展，哪个是 O(N^2)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 缩放定律论文（arXiv:2212.07143，Cherti 等人）。从图表中重现他们在数据缩放方面的结论：在固定模型大小下，ImageNet 零样本准确率与训练数据大小之间的对数线性关系是怎样的？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| InfoNCE | "对比损失" | 在 batch 相似度矩阵上的交叉熵；每个项的正样本是其配对项，负样本是其他所有项 |
| Sigmoid 损失 | "SigLIP 损失" | 逐对二元交叉熵；无 softmax，无 all-gather，在分布式训练中成本低 |
| 温度 | "tau" | 在 softmax/sigmoid 之前缩放 logits 的标量；控制分布的锐度 |
| 零样本 | "无微调分类" | 使用文本提示构建类别嵌入，通过余弦相似度分类；无需在目标类上训练 |
| 提示模板 | "a photo of a ..." | 围绕类名的文本脚手架；影响零样本准确率 1-5 个点 |
| 双编码器 | "双塔" | 一个图像编码器 + 一个文本编码器，在共享的 D 维空间中输出 |
| 难负样本 | "难以区分的干扰项" | 与正样本足够相似的负样本，模型需要努力才能将它们分离开 |
| 线性探测 | "冻结 + 一层" | 仅在冻结特征之上训练一个线性分类器；衡量特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的能力，无需调整大小即可摄入任何宽高比和分辨率的图像 |
| 温度缩放 | "对数参数化的 tau" | CLIP 将 `log(1/tau)` 参数化以使梯度行为良好；限制下界以防止 tau 接近零时坍缩 |

## 延伸阅读

- [Radford 等人 — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai 等人 — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia 等人 — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用噪杂网络数据进行规模缩放。
- [Cherti 等人 — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 缩放定律。
