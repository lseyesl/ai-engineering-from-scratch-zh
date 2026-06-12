# 完整 Transformer — 编码器 + 解码器

> 注意力是主角。其他一切——残差连接、归一化、前馈网络、交叉注意力——是让你能堆叠深层的脚手架。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 7 · 02（自注意力），阶段 7 · 03（多头注意力），阶段 7 · 04（位置编码）
**时间：** ~75 分钟

## 问题

单层注意力是一个特征提取器，不是模型。每层一次矩阵乘法不足以处理语言。你需要深度——而没有合适的管道，深度就会出问题。

2017 年的 Vaswani 论文打包了六项设计决策，将一层注意力变成了可堆叠的模块。此后的每个 transformer——仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5）——都继承了相同的骨架。到 2026 年，这些模块已经得到改进（RMSNorm、SwiGLU、预归一化、RoPE），但骨架是相同的。

本课就是关于这个骨架。接下来的课程会对其进行专门化——第 06 课讲编码器，第 07 课讲解码器，第 08 课讲编码器-解码器。

## 概念

![编码器和解码器模块内部结构，已连线](../assets/full-transformer.svg)

### 六个组件

1. **嵌入 + 位置信号。** Token → 向量。通过 RoPE（现代）或正弦（经典）注入位置。
2. **自注意力。** 每个位置关注每个其他位置。在解码器中是掩码的。
3. **前馈网络（FFN）。** 逐位置的双层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比为 4×。
4. **残差连接。** `x + sublayer(x)`。没有它，梯度在大约 6 层之后就会消失。
5. **层归一化。** `LayerNorm` 或 `RMSNorm`（现代）。稳定残差流。
6. **交叉注意力（仅解码器）。** 查询来自解码器，键和值来自编码器输出。

```figure
transformer-block
```

### 编码器模块（由 BERT、T5 编码器使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── 残差 ──────┘
```

编码器是双向的。没有掩码。所有位置都能看到所有位置。

### 解码器模块（由 GPT、T5 解码器使用）

```
x → LN → MHA(掩码自注意力) → + → LN → MHA(与编码器的交叉注意力) → + → LN → FFN → + → out
```

解码器每层有三个子层。中间的子层——交叉注意力——是信息从编码器流向解码器的唯一位置。在纯解码器架构（GPT）中，交叉注意力被省略，你只需掩码自注意力 + FFN。

### 预归一化 vs 后归一化

原始论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。后归一化在 2019 年左右失宠——没有仔细的预热很难深层训练。预归一化（`LN` 在子层*之前*）是 2026 年的默认选择：Llama、Qwen、GPT-3+、Mistral 都使用它。

### 2026 年现代化模块

Vaswani 2017 使用了 LayerNorm + ReLU。现代堆栈已经替换了这两者。生产模块实际的样子：

| 组件 | 2017 | 2026 |
|-----------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活 | ReLU | SwiGLU |
| FFN 扩展 | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数匹配） |
| 位置 | 正弦绝对 | RoPE |
| 注意力 | 完整 MHA | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 去掉了 LayerNorm 的均值中心化（少一次减法），节省了计算，经验上至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 论文中一致优于 ReLU/GELU FFN，困惑度提升约 0.5 点。

### 参数量

对于一个 `d_model = d` 和 FFN 扩展比 `r` 的模块：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- 归一化：可忽略

在 `d = 4096, r = 2.6, layers = 32`（大致相当于 Llama 3 8B）时，总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = 每层约 1.5B 参数 × 32 ≈ 7B`（加上嵌入和输出头）。与公布的参数量一致。

## 动手实现

### 步骤 1：构建块

使用第 03 课中的小型 `Matrix` 类（为独立运行已复制到本文件）：

- `layer_norm(x, eps=1e-5)` — 减去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。没有均值减法。
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

参见 `code/main.py` 的完整连线。

### 步骤 2：连接 2 层编码器和 2 层解码器

堆叠它们。将编码器输出传递到每个解码器的交叉注意力中。在输出投影之前添加最终的 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 步骤 3：在玩具示例上运行前向传播

传入一个 6-token 源序列和一个 5-token 目标序列。验证输出形状为 `(5, vocab)`。没有训练——本课关注的是架构，而不是损失。

### 步骤 4：替换为 RMSNorm + SwiGLU

用 RMSNorm 和 SwiGLU 替换 LayerNorm 和 ReLU-FFN。确认形状仍然匹配。这是用一个函数替换完成的 2026 年现代化。

## 使用

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 年生产代码自己编写模块，因为：

- Flash Attention 在注意力内部调用，而不是通过 `nn.MultiheadAttention`。
- GQA / MLA 不在 stdlib 参考中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 的默认值。

HF `transformers` 有你应该阅读的干净参考模块：`modeling_llama.py` 是 2026 年仅解码器模块的规范实现。它大约 500 行，值得通读一次。

**编码器 vs 解码器 vs 编码器-解码器——何时选择：**

| 需求 | 选择 | 示例 |
|------|------|---------|
| 分类、嵌入、文本问答 | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、聊天、代码、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5、BART、Whisper |

仅解码器在语言中胜出，因为它扩展最干净，同时处理理解和生成。当输入具有清晰的"源序列"同一性（翻译、语音识别、结构化任务）时，编码器-解码器仍然是最好的。

## 产出

参见 `outputs/skill-transformer-block-reviewer.md`。该技能根据 2026 年默认值审查新的 transformer 模块实现，并标记缺失的部分（预归一化、RoPE、RMSNorm、GQA、FFN 扩展比）。

## 练习

1. **简单。** 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 下计算你的 encoder_block 中的参数量。通过实现模块并使用 `sum(p.numel() for p in block.parameters())` 进行验证。
2. **中等。** 从后归一化切换到预归一化。初始化两者，并在随机输入上测量 12 个堆叠层后的激活范数。后归一化的激活应该会爆炸；预归一化的应该保持有界。
3. **困难。** 在玩具复制任务（反转复制 `x`）上实现一个 4 层编码器-解码器。训练 100 步。报告损失。替换为 RMSNorm + SwiGLU + RoPE——损失是否下降？

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 模块（Block） | "一个 transformer 层" | 包含归一化 + 注意力 + 归一化 + FFN 的堆栈，由残差连接包裹。 |
| 残差（Residual） | "跳跃连接" | `x + f(x)` 输出；使梯度流经深层堆栈。 |
| 预归一化（Pre-norm） | "先归一化，再操作" | 现代做法：`x + sublayer(LN(x))`。无需预热技巧即可训练更深。 |
| RMSNorm | "不带均值的 LayerNorm" | 除以 RMS；少一次操作，相同的经验稳定性。 |
| SwiGLU | "每个人都切换到的 FFN" | `Swish(W1 x) ⊙ W3 x → W2`。在 LM 困惑度上优于 ReLU/GELU。 |
| 交叉注意力（Cross-attention） | "解码器如何看到编码器" | Q 来自解码器，K/V 来自编码器输出的 MHA。 |
| FFN 扩展（FFN expansion） | "中间 MLP 有多宽" | 隐藏大小与 d_model 的比率，通常为 4（LayerNorm）或 2.6（SwiGLU）。 |
| 去偏置（Bias-free） | "去掉 +b 项" | 现代堆栈在线性层中省略偏置；轻微 PPL 改进，更小的模型。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始模块规范。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么预归一化在深层优于后归一化。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026 年仅解码器模块的规范实现。
