# 从头构建 Transformer — 期末项目

> 十三节课。一个模型。没有捷径。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 7 · 01 至 13。请勿跳过。
**时间：** ~120 分钟

## 问题

你已经读了每一篇论文。你已经实现了注意力、多头拆分、位置编码、编码器和解码器模块、BERT 和 GPT 损失、MoE、KV 缓存。现在让它们在一个真实任务上协同工作。

期末项目：在字符级语言建模任务上端到端地训练一个小的仅解码器 transformer。它阅读莎士比亚。它生成新的莎士比亚。它足够小，可以在笔记本上在 10 分钟内训练完成。它足够正确，换成更大的数据集和更长的训练就能得到一个真正的 LM。

这是本课程的"nanoGPT"。它并非原创——Karpathy 2023 年的 nanoGPT 教程是每个学生至少写一次的参考实现。我们沿用其形式并根据我们已学内容进行重构。

## 概念

![从头构建 Transformer 模块图](../assets/capstone.svg)

架构，已注释：

```
输入 token (B, N)
   │
   ▼
token 嵌入 + 位置嵌入  ◀── 第 04 课（RoPE 可选）
   │
   ▼
┌──── 模块 × L ────────────────────┐
│  RMSNorm                          │  ◀── 第 05 课
│  多头注意力（因果）               │  ◀── 第 03 + 07 课（因果掩码）
│  残差                             │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── 第 05 课
│  残差                             │
└────────────────────────────────── ┘
   │
   ▼
最终 RMSNorm
   │
   ▼
lm_head（与 token 嵌入共享权重）
   │
   ▼
logits (B, N, V)
   │
   ▼
偏移一位交叉熵                    ◀── 第 07 课
```

### 我们交付的内容

- `GPTConfig` — 一个配置所有超参数的地方。
- `MultiHeadAttention` — 因果、批处理、可选的 Flash 风格路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` — 现代 FFN。
- `Block` — 预归一化、残差包裹的注意力 + FFN。
- `GPT` — 嵌入、堆叠模块、LM 头、generate()。
- 使用 AdamW、余弦 LR、梯度裁剪的训练循环。
- 莎士比亚文本上的字符级分词器。

### 我们不交付的内容

- RoPE — 在第 04 课中概念上实现过。这里为了简单使用可学习位置嵌入。练习要求你替换为 RoPE。
- 生成过程中的 KV 缓存 — 每个生成步骤重新计算整个前缀上的注意力。较慢但更简单。练习要求你添加 KV 缓存。
- Flash Attention — PyTorch 2.0+ 在输入匹配时自动调度；我们使用 `F.scaled_dot_product_attention`。
- MoE — 每模块单个 FFN。你在第 11 课中见过 MoE。

### 目标指标

在 Mac M2 笔记本上，一个 4 层、4 头、d_model=128 的 GPT 在 `tinyshakespeare.txt` 上训练 2,000 步：

- 训练损失从约 4.2（随机）收敛到约 1.5，大约需要 6 分钟。
- 采样输出看起来像莎士比亚风格：古词、换行、像"ROMEO:"这样的人名涌现。
- 验证损失（保留的最后 10% 文本）与训练损失接近；在此大小/预算下没有过拟合。

## 动手实现

本课使用 PyTorch。安装 `torch`（CPU 版本即可）。参见 `code/main.py`。脚本处理：

- 如果缺少则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器。
- 训练/验证分割为 90/10。
- 在支持的硬件上使用 bf16 自动类型转换的训练循环。
- 训练完成后的采样。

### 步骤 1：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。极小的词表。适合 4 字节的 vocab_size。没有 BPE，没有分词器问题。

### 步骤 2：模型

参见 `code/main.py`。模块是第 05 课的标准内容——预归一化、RMSNorm、SwiGLU、因果 MHA。4/4/128 的参数量：约 800K。

### 步骤 3：训练循环

获取一个随机的长度为 256 的 token 窗口批次。前向传播。偏移一位交叉熵。反向传播。AdamW 步。记录。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 步骤 4：采样

给定一个提示，重复前向传播、从 top-p logits 采样、追加、继续。在 500 个 token 后停止。

### 步骤 5：读取输出

2,000 步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但是莎士比亚风格。对于约 800K 参数和在笔记本上运行 6 分钟来说，这是一个明显的胜利。

## 使用

这个期末项目是一个参考架构。将其交付到真实产品的三个扩展：

1. **更换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词表大小从 65 跃升到约 50,000。模型能力需要按比例扩大。
2. **在更大的语料库上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单个 A100 上训练 10B token 对于一个 125M 参数的 GPT 大约需要 24 小时。
3. **添加 RoPE + KV 缓存 + Flash Attention。** 下面的练习带你一步步完成。

最终生成一个能生成流利英语的 125M 参数 GPT。不是前沿模型。但同样的代码路径——只是更大——正是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年用于训练研究检查点的方法。

## 产出

参见 `outputs/skill-transformer-review.md`。该技能根据之前 13 课的内容审查从头实现的 transformer 的正确性。

## 练习

1. **简单。** 运行 `code/main.py`。验证你训练好的模型的最后一步验证损失低于 2.0。将 `max_steps` 从 2,000 改为 5,000——验证损失是否持续改善？
2. **中等。** 用 RoPE 替换可学习位置嵌入。在 `MultiHeadAttention` 内部对 Q 和 K 应用旋转。训练并验证验证损失至少不高于之前。
3. **中等。** 在采样循环中实现 KV 缓存。使用缓存和不使用缓存分别生成 500 个 token。在笔记本上挂钟时间应改进 5-20 倍。
4. **困难。** 为模型添加一个预测下一个加一个 token 的第二头（MTP——DeepSeek-V3 的多 token 预测）。联合训练。它有帮助吗？
5. **困难。** 将每个模块中的单个 FFN 替换为 4 个专家的 MoE。路由器 + top-2 路由。在匹配活跃参数下观察验证损失如何变化。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy 的教程仓库" | 最小仅解码器 transformer 训练代码，约 300 行；规范参考。 |
| tinyshakespeare | "标准玩具语料库" | 约 1.1 MB 文本；自 2015 年以来每个字符级 LM 教程都使用它。 |
| 共享嵌入（Tied embeddings） | "共享输入/输出矩阵" | LM 头权重 = token 嵌入矩阵的转置；节省参数，改善质量。 |
| bf16 自动类型转换 | "训练精度技巧" | 前向/反向在 bf16 中运行，优化器状态保持在 fp32；自 2021 年以来标准。 |
| 梯度裁剪（Gradient clipping） | "阻止尖峰" | 将全局梯度范数限制在 1.0；防止训练崩溃。 |
| 余弦 LR 调度 | "2020+ 默认" | LR 线性上升（预热）然后余弦形状衰减到峰值的 10%。 |
| MFU | "模型 FLOP 利用率" | 实现的 FLOPs / 理论峰值；2026 年稠密 40%、MoE 30% 为强。 |
| 验证损失（Val loss） | "保留损失" | 模型未见过的数据上的交叉熵；过拟合检测器。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 经典注解实现。
