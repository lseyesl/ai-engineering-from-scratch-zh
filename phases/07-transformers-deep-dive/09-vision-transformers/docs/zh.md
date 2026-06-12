# 视觉 Transformer（ViT）

> 图像是补丁的网格。句子是 token 的网格。同一个 transformer 两者皆可处理。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 7 · 05（完整 Transformer），阶段 4 · 03（CNN），阶段 4 · 14（视觉 Transformer 入门）
**时间：** ~45 分钟

## 问题

在 2020 年之前，计算机视觉意味着卷积。ImageNet、COCO 和检测基准上的每个 SOTA 都使用 CNN 骨干网络。Transformer 是为语言准备的。

Dosovitskiy 等人（2020）——"An Image is Worth 16x16 Words"——展示了你可以完全放弃卷积。将图像切片为固定大小的补丁，将每个补丁线性投影为嵌入，将序列送入标准的 transformer 编码器。在足够的规模下（ImageNet-21k 预训练或更大），ViT 匹配或超越基于 ResNet 的模型。

ViT 是 2026 年一个更广泛模式的开始：一种架构，多种模态。Whisper 将音频 token 化。ViT 将图像 token 化。机器人使用动作 token。视频使用像素 token。Transformer 不在意——给它一个序列，它就学习。

到 2026 年，ViT 及其后继者（DeiT、Swin、DINOv2、ViT-22B、SAM 3）拥有大部分的视觉领域。CNN 在边缘设备和延迟敏感任务上仍然胜出。其他一切都在堆栈中的某处有一个 ViT。

## 概念

![图像 → 补丁 → token → transformer](../assets/vit.svg)

### 步骤 1 — 补丁化

将 `H × W × C` 的图像拆分为 `N × (P·P·C)` 的扁平补丁序列。典型设置：`224 × 224` 图像，`16 × 16` 补丁 → 196 个每个包含 768 个值的补丁。

```
图像 (224, 224, 3) → 14 × 14 网格的 16x16x3 补丁 → 196 个长度为 768 的向量
```

补丁大小是杠杆。更小的补丁 = 更多 token，更好的分辨率，二次注意力成本。更大的补丁 = 更粗糙，更廉价。

### 步骤 2 — 线性嵌入

单个学习矩阵将每个扁平补丁投影到 `d_model`。相当于核大小为 `P`、步长为 `P` 的卷积。在 PyTorch 中，这实际上就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——一个 2 行的实现。

### 步骤 3 — 前置 `[CLS]` token，添加位置嵌入

- 前置一个可学习的 `[CLS]` token。其最终隐藏状态是用于分类的图像表示。
- 添加可学习的位置嵌入（ViT 原始版本）或二维正弦嵌入（后来的变体）。
- 在 2024 年之后，RoPE 被扩展到 2D 用于位置，有时没有显式嵌入。

### 步骤 4 — 标准 transformer 编码器

堆叠 L 个 `LayerNorm → 自注意力 → + → LayerNorm → MLP → +` 模块。与 BERT 相同。没有视觉特定层。这是该论文的教学核心。

### 步骤 5 — 头部

对于分类：取 `[CLS]` 隐藏状态 → 线性层 → softmax。对于 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用补丁嵌入。

### 重要的变体

| 模型 | 年份 | 变化 |
|-------|------|--------|
| ViT | 2020 | 原始版本。固定补丁大小，全局注意力。 |
| DeiT | 2021 | 蒸馏；仅在 ImageNet-1k 上可训练。 |
| Swin | 2021 | 带移动窗口的层级结构。固定的次二次成本。 |
| DINOv2 | 2023 | 自监督（无标签）。最佳通用视觉特征。 |
| ViT-22B | 2023 | 22B 参数；扩展定律适用。 |
| SigLIP | 2023 | ViT + 语言对，sigmoid 对比损失。 |
| SAM 3 | 2025 | 分割一切；ViT-Large + 可提示掩码解码器。 |

### 为什么花了很长时间

ViT 需要*大量*数据才能匹配 CNN，因为它没有 CNN 的任何归纳偏置（平移不变性、局部性）。如果没有 >1 亿张标注图像或强大的自监督预训练，CNN 在同等计算下仍然胜出。DeiT 在 2021 年通过蒸馏技巧解决了这个问题；DINOv2 在 2023 年通过自监督永久解决了它。

## 动手实现

参见 `code/main.py`。纯标准库的补丁化 + 线性嵌入 + 健全性检查。没有训练——任何实际规模的 ViT 都需要 PyTorch 和数小时的 GPU 时间。

### 步骤 1：假图像

一个 24 × 24 的 RGB 图像，作为 `(R, G, B)` 元组的行列表。我们使用 6×6 补丁 → 16 个补丁，每个 108 维嵌入向量。

### 步骤 2：补丁化

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：按行主序遍历网格。每个 ViT 都使用这个顺序。

### 步骤 3：线性嵌入

将每个扁平补丁乘以一个随机的 `(patch_flat_size, d_model)` 矩阵。验证在添加 `[CLS]` 后的输出形状为 `(N_patches + 1, d_model)`。

### 步骤 4：计算实际 ViT 的参数量

打印 ViT-Base 的参数量：12 层、12 头、d=768、patch=16。与 ResNet-50（约 25M）进行比较。ViT-Base 约为 86M。ViT-Large 约为 307M。ViT-Huge 约为 632M。

## 使用

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # 图像表示
```

**DINOv2 嵌入是 2026 年图像特征的默认选择。** 冻结骨干网络，训练一个小型头部。适用于分类、检索、检测、字幕生成。Meta 的 DINOv2 检查点在每个非文本视觉任务上表现优于 CLIP。

**补丁大小选择。** 小模型使用 16×16（ViT-B/16）。密集预测（分割）使用 8×8 或 14×14（SAM、DINOv2）。非常大的模型使用 14×14。

## 产出

参见 `outputs/skill-vit-configurator.md`。该技能根据数据集大小、分辨率和计算预算为新的视觉任务选择 ViT 变体和补丁大小。

## 练习

1. **简单。** 运行 `code/main.py`。验证补丁数等于 `(H/P) * (W/P)`，扁平补丁维度等于 `P*P*C`。
2. **中等。** 实现 2D 正弦位置嵌入——为每个补丁的 `row` 和 `col` 分别生成两个独立的正弦编码，然后拼接起来。将它们输入一个小型 PyTorch ViT，并在 CIFAR-10 上与可学习位置嵌入的准确率进行比较。
3. **困难。** 构建一个 3 层 ViT（PyTorch），在 1,000 张 MNIST 图像上训练，使用 4×4 补丁。测量测试准确率。现在在相同的 1,000 张图像上添加 DINOv2 预训练（简化版：仅训练编码器从掩码补丁预测补丁嵌入）。准确率是否提高？

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 补丁（Patch） | "视觉 transformer 的 token" | 图像中 `P × P × C` 区域的像素值扁平向量。 |
| 补丁化（Patchify） | "切割 + 展平" | 将图像切片为非重叠补丁，将每个展平为向量。 |
| `[CLS]` token | "图像摘要" | 前置的可学习 token；其最终嵌入是图像表示。 |
| 归纳偏置（Inductive bias） | "模型假设了什么" | ViT 的先验知识比 CNN 少；需要更多数据来弥补差距。 |
| DINOv2 | "自监督 ViT" | 使用图像增强 + 动量教师无需标签进行训练。2026 年最佳通用图像特征。 |
| SigLIP | "CLIP 的继承者" | 使用 sigmoid 对比损失训练的 ViT + 文本编码器；在同等计算下优于 CLIP。 |
| Swin | "窗口化 ViT" | 带局部注意力 + 移动窗口的层级 ViT；次二次复杂度。 |
| 寄存器 token（Register tokens） | "2023 年的技巧" | 一些额外的可学习 token，用于吸收注意力陷阱；改善 DINOv2 特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的注册 token 修复。
