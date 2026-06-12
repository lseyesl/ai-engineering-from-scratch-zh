# Show-o 与离散扩散统一模型

> Transfusion 混合了连续和离散表示。Show-o（Xie 等人，2024 年 8 月）走了另一条路：文本 token 使用因果 next-token 预测，图像 token 使用 MaskGIT 风格的掩码离散扩散。两者都位于一个带有混合注意力掩码的 Transformer 内部。结果在一个骨干网、每个模态一个 tokenizer、一个损失公式（next-token 扩展到掩码预测）上统一了 VQA、文生图、图像修复和混合模态生成。本课程讲解 Show-o 的设计——为什么掩码离散扩散是一种并行的、少步图像生成器——并与 Transfusion 和 Emu3 进行对比。

**类型：** 学习
**语言：** Python（标准库，掩码离散扩散采样器）
**前置要求：** Phase 12 · 13（Transfusion）
**时间：** ~120 分钟

## 学习目标

- 解释掩码离散扩散：均匀掩码 token 然后要求 Transformer 恢复它们的调度。
- 比较并行图像解码（Show-o、MaskGIT）与自回归图像解码（Chameleon、Emu3）在速度和质量上的差异。
- 说出 Show-o 在一个 checkpoint 中处理的三个任务：T2I、VQA、图像修复。
- 选择掩码调度（余弦、线性、截断）并推理其对样本质量的影响。

## 问题

Transfusion 的双损失训练有效，但具有更棘手的动态——连续扩散损失与离散 NTP 损失的数字规模不同。平衡损失权重是一个超参数搜索。架构有效但复杂。

Show-o 的答案：保持两种模态都是离散的（像 Chameleon），但通过掩码离散扩散并行生成图像，而不是顺序生成。训练目标变成一个单一的掩码 token 预测，自然地泛化了 next-token 预测。

## 概念

### 掩码离散扩散（MaskGIT）

原始的 Chang 等人（2022 年）MaskGIT 技巧很优雅。从一个完全掩码的图像（每个 token 是特殊的 `<MASK>` id）开始。每一步，并行预测所有掩码 token，然后保留 top-K 最自信的预测并重新掩码其余的。经过约 8-16 次迭代，所有 token 都被填充。每一步取消掩码的 token 数量的调度经过调优——余弦调度效果很好。

训练很简单：从 [0, 1] 均匀采样一个掩码比率，应用于图像的 VQ token，训练 Transformer 恢复被掩码的那些。正是 BERT 对文本所做的，扩展到图像生成。

### Show-o：一个 Transformer，混合掩码

Show-o 将 MaskGIT 放入一个因果语言模型 Transformer 内部。注意力掩码是：

- 文本 token：因果的（标准 LLM）。
- 图像 token：在图像块内完全双向（因此掩码 token 在预测期间可以看到每个其他图像 token）。
- 文本到图像：文本关注之前的图像，图像关注之前的文本。

训练交替进行：
1. 文本序列上的标准 NTP。
2. T2I 样本：文本 → 带掩码图像 token 的图像，掩码 token 预测损失。
3. VQA 样本：图像 → 带掩码文本 token 的文本（实际上就是 NTP）。

统一损失是 `<MASK>` token 上的交叉熵，这涵盖了文本 NTP（只有最后一个 token 被"掩码"）和图像掩码扩散（随机子集被掩码）。

### 并行采样

Show-o 在大约 16 步中生成一张图像，而不是大约 1000 步（每 token 自回归）或大约 20 步（扩散）。每一步，并行预测所有掩码 token；提交 top-K 最自信的；重复。

比较：
- Chameleon / Emu3（token 上的自回归）：N_tokens 次前向传播，每张图像通常 1024-4096 次。
- Transfusion（连续扩散）：约 20 步，每一步是整个 Transformer 前向传播。
- Show-o（掩码离散扩散）：约 16 步，每一步是整个 Transformer 前向传播。

Show-o 在类似规模的模型上比 Chameleon 更快，步数大致匹配 Transfusion，但每步成本更低（离散词汇表 logits vs 连续 MSE 损失）。

### 一个 checkpoint 中的任务

Show-o 在推理时支持四个任务，通过提示格式选择：

- 文本生成：标准自回归文本输出。
- VQA：图像输入，文本输出。
- T2I：文本输入，通过掩码离散扩散输出图像。
- 图像修复：部分 token 被掩码的图像，填充缺失部分。

图像修复能力来自掩码预测训练的免费附带品。掩码 VQ-token 网格的一个区域，输入其余部分加上一个文本提示，预测掩码 token。

### 掩码调度

每步取消掩码的 token 数量的调度决定了质量。Show-o 推荐余弦：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

在第 0 步，所有 token 被掩码（比率 1.0）。在第 T 步，没有 token 被掩码。余弦将质量集中在预测信息最丰富的中等范围比率上。线性调度也有效，但更快达到平台期。

### Show-o2

Show-o2（2025 年后续，arXiv 2506.15564）扩展了 Show-o：更大的 LLM 基础，更好的 tokenizer，改进的掩码调度。相同的架构模式。

### Show-o 的位置

在 2026 年的分类中：

- 离散 token + NTP：Chameleon、Emu3。简单但推理慢。
- 离散 token + 掩码扩散：Show-o、MaskGIT、LlamaGen、Muse。并行采样，仍然有 tokenizer 带来的有损。
- 连续 + 扩散：Transfusion、MMDiT、DiT。最高质量，更复杂的训练。
- 连续 + VLM 中的流匹配：JanusFlow、InternVL-U。最新。

按任务选择：当你想在一个开源模型中用合理的速度获得 T2I + 图像修复 + VQA 时选 Show-o；当质量至上且你能承担双损失管道时选 Transfusion。

## 使用它

`code/main.py` 模拟了 Show-o 采样：

- 一个 16 个 VQ token 的玩具网格。
- 一个模拟的"Transformer"基于提示和当前未掩码的 token 预测 logits。
- 在 8 步中使用余弦调度的并行掩码采样。
- 打印中间状态（掩码模式演化）和最终 token。

运行它，观察掩码逐步溶解。

## 交付物

本课程产生 `outputs/skill-unified-gen-model-picker.md`。给定一个需要理解（VQA、描述）和生成（T2I、图像修复）且受开源权重约束的产品，在 Show-o 家族、Transfusion/MMDiT 家族和 Emu3/Chameleon 家族之间进行选择，附具体权衡。

## 练习

1. 掩码离散扩散在大约 16 步中采样。为什么不是 1 步？如果在第 0 步就取消所有掩码，会出什么问题？

2. 图像修复通过掩码扩散免费获得。提出一个产品用例（真实或假设），其中 Show-o 的图像修复胜过专门模型。

3. 余弦调度 vs 线性调度：对于 T=8，追踪每步未掩码的 token 数量。哪个更平衡？

4. 一张 512x512 的 Show-o 图像是 1024 个 token。在词汇表 K=16384 下，模型输出 1024 * log2(16384) = 14,336 比特（约 1.75 KiB）的数据。Stable Diffusion 输出 512*512*24 比特 = 6,291,456 比特（约 768 KiB）的原始像素。压缩比是多少，它带来了什么质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的类条件自回归图像模型与 Show-o 的掩码方法有何不同？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 掩码离散扩散 | "MaskGIT 风格" | 训练预测掩码 token；推理时，迭代地取消掩码最自信的预测 |
| 余弦调度 | "取消掩码调度" | 推理步骤中掩码比率的衰减；将置信度增长集中在中等范围 |
| 并行解码 | "所有 token 同时" | 每一步在一次前向传播中预测完整的掩码 token 序列，然后提交 top-K |
| 混合注意力 | "因果 + 双向" | 在文本 token 上是因果的、在图像块内是双向的掩码 |
| 图像修复 | "填充生成" | 以部分 token 被掩码的图像为条件，预测缺失的 token；从训练目标免费获得 |
| 提交率 | "每步 Top-K" | 每次迭代中声明"已完成"的 token 数量；控制推理与质量的权衡 |

## 延伸阅读

- [Xie 等人 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang 等人 — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun 等人 — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang 等人 — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
