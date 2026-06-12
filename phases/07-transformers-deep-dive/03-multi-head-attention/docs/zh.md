# 多头注意力

> 一个注意力头一次学习一种关系。八个头学习八种。头是免费的。多用几个。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 7 · 02（从头实现自注意力）
**时间：** ~75 分钟

## 问题

单个自注意力头计算一个注意力矩阵。该矩阵捕获一种关系——通常是训练信号下使损失最小化的那种。如果你的数据同时包含主谓一致、共指消解、长距离篇章关系和语法分块，一个头会把它们全部涂抹到单一的 softmax 分布中，丢失一半的信号。

2017 年 Vaswani 论文的解决方案：并行运行多个注意力函数，每个函数有自己的 Q、K、V 投影，并拼接输出。每个头在维度为 `d_model / n_heads` 的更小子空间中操作。总参数量保持不变。表达能力提升。

多头注意力是 2026 年每个 transformer 的默认配置。唯一的争论是关于**多少个**头，以及键和值是否共享投影（分组查询注意力、多查询注意力、多头潜在注意力）。

## 概念

![多头注意力拆分、关注、拼接](../assets/multi-head-attention.svg)

**拆分。** 取形状为 `(N, d_model)` 的 `X`。投影到 Q、K、V，每个形状为 `(N, d_model)`。重塑为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。转置为 `(n_heads, N, d_head)`。

**并行关注。** 在每个头内运行缩放点积注意力。每个头产生 `(N, d_head)`。这些头在嵌入的不同子空间上操作，在注意力计算期间从不相互通信。

**拼接和投影。** 将头堆叠回 `(N, d_model)`，并乘以形状为 `(d_model, d_model)` 的学习输出矩阵 `W_o`。`W_o` 是头混合的地方。

**为什么有效。** 每个头可以专精而无需与其他头竞争表示预算。2019-2024 年的探针研究表明了不同的头角色：位置头、关注前一个 token 的头、复制头、命名实体头、归纳头（这是上下文学习的基础）。

**2026 年的变体系谱：**

| 变体 | Q 头 | K/V 头 | 被谁使用 |
|---------|---------|-----------|---------|
| 多头（MHA） | N | N | GPT-2、BERT、T5 |
| 多查询（MQA） | N | 1 | PaLM、Falcon |
| 分组查询（GQA） | N | G（例如 N/8） | Llama 2 70B、Llama 3+、Qwen 2+、Mistral |
| 多头潜在（MLA） | N | 压缩为低秩 | DeepSeek-V2、V3 |

GQA 是现代默认选择，因为它将 KV 缓存内存减少了 `N/G` 倍，同时保持了近乎完整的质量。MLA 更进一步，将 K/V 压缩到潜在空间，然后在计算时投影回来——增加 FLOPs，但节省更多内存。

```figure
multihead-split
```

## 动手实现

### 步骤 1：从已有的单头注意力中拆分出头

取第 02 课的 `SelfAttention`，并用拆分/拼接对包装它。参见 `code/main.py` 的 numpy 实现；逻辑如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一个 reshape 和一个 transpose。没有循环。这正是 PyTorch 在 `nn.MultiheadAttention` 底层所做的。

### 步骤 2：每头运行缩放点积注意力

每个头得到自己的 Q、K、V 切片。注意力变成一个批处理矩阵乘法：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在实际硬件上 `Qh @ Kh.transpose(...)` 是一个 `bmm`。GPU 将其视为形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的单个批处理矩阵乘法。增加头是免费的。

### 步骤 3：分组查询注意力变体

只有键和值投影发生变化。Q 有 `n_heads` 组；K 和 V 有 `n_kv_heads < n_heads` 组，并被重复以匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

在推理时，这节省了内存，因为只有 `n_kv_heads` 份拷贝存在于 KV 缓存中，而不是 `n_heads`。Llama 3 70B 使用 64 个查询头和 8 个 KV 头——8 倍的缓存缩小。

### 步骤 4：探针每个头学到了什么

对包含 4 个头的短句运行 MHA。对每个头，打印 `(N, N)` 注意力矩阵。即使随机初始化，你也会看到不同的头挑出不同的结构——这既是信号作用，也部分源于子空间中的旋转对称性。

## 使用

在 PyTorch 中，一行版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention 在 CUDA 上自动调度 Flash Attention。
# 对于 GQA，传入形状为 (B, n_heads, N, d_head) 的 Q 和形状为
# (B, n_kv_heads, N, d_head) 的 K、V。PyTorch 处理重复。
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**多少个头？** 来自 2026 年生产模型的经验法则：

| 模型大小 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小（~125M） | 768 | 12 | 64 |
| 基础（~350M） | 1024 | 16 | 64 |
| 大（~1B） | 2048 | 16 | 128 |
| 前沿（~70B） | 8192 | 64 | 128 |

`d_head` 几乎总是 64 或 128。它是一个头能"看到"多少的单位。低于 32，头开始与缩放因子 `sqrt(d_head)` 斗争；高于 256，你就失去了"多小专家"的优势。

## 产出

参见 `outputs/skill-mha-configurator.md`。该技能根据参数预算、序列长度和部署目标，为新的 transformer 推荐头数、KV 头数和投影策略。

## 练习

1. **简单。** 从 `code/main.py` 中取 MHA，在固定 `d_model=64` 的情况下将 `n_heads` 从 1 改为 16。在合成复制任务上绘制小型单层模型的损失。更多的头是否有助于、平台期还是损害性能？
2. **中等。** 实现 MQA（一个 KV 头在所有查询头之间共享）。测量参数量相比完整 MHA 下降了多​​少。计算在推理时 N=2048 的情况下 KV 缓存大小缩减了多少。
3. **困难。** 实现一个小型的多头潜在注意力：将 K、V 压缩到秩为 `r` 的潜在空间，将潜在向量存储在 KV 缓存中，在注意力时解压缩。在什么 `r` 下，缓存内存会低于完整 MHA 的 1/8，同时质量保持在验证 ppl 的 1 bit 以内？

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 头（Head） | "一个注意力回路" | 维度为 `d_head = d_model / n_heads` 的一个 Q/K/V 投影，带有自己的注意力矩阵。 |
| d_head | "头维度" | 每头隐藏宽度；生产中几乎总是 64 或 128。 |
| 拆分/合并 | "重塑技巧" | `(N, d_model) ↔ (n_heads, N, d_head)` 围绕注意力的 reshape+transpose。 |
| W_o | "输出投影" | 拼接头后应用的 `(d_model, d_model)` 矩阵；头在此处混合。 |
| MQA | "一个 KV 头" | 多查询注意力：单个共享 K/V 投影。最小的 KV 缓存，有一定质量损失。 |
| GQA | "Llama 2 以来的默认选择" | 分组查询注意力，使用 `n_kv_heads < n_heads`；重复以匹配 Q。 |
| MLA | "DeepSeek 的绝招" | 多头潜在注意力：K、V 压缩为低秩潜在向量，在注意力时解压缩。 |
| 归纳头（Induction head） | "上下文学习背后的回路" | 一对检测先前出现并复制其后内容并的头。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始多头规范。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 如何在训练后将 MHA 转换为 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA 以及为什么它在缓存内存上优于 MHA/GQA。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 从机制角度观察头实际做了什么。
