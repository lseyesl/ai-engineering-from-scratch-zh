# 注意力变体 — 滑动窗口、稀疏、差分

> 完整注意力是一个圆。每个 token 看到每个 token，内存付出了代价。四种变体改变了圆的形状，收回了成本的一半。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 7 · 02（自注意力），阶段 7 · 03（多头），阶段 7 · 12（KV 缓存 / Flash Attention）
**时间：** ~60 分钟

## 问题

完整注意力在序列长度上需要 `O(N²)` 内存和 `O(N²)` 计算。对于一个 128K 上下文的 Llama 3 70B，每层有 160 亿个注意力条目，乘以 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 的激活内存，但没有改变算术成本——每个 token 仍然关注每个其他 token。

三类变体改变了注意力矩阵本身的拓扑：

1. **滑动窗口注意力（SWA）。** 每个 token 关注一个固定的邻居窗口，而不是整个前缀。内存和计算降至 `O(N · W)`，其中 `W` 是窗口。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long。
2. **稀疏 / 块注意力。** 只有选定的 `(i, j)` 对被评分；其余被强制设为零权重。Longformer、BigBird、OpenAI sparse transformer。
3. **差分注意力。** 用独立的 Q/K 投影计算两个注意力映射，一个减去另一个。消除了将权重渗入前几个 token 的"注意力陷阱"。微软的 DIFF Transformer（2024）。

这些方法共存。一个 2026 年的前沿模型经常混合它们：大多数层是 SWA-1024，每五层是一个全局完整注意力，还有一些是清理检索的差分头。Gemma 3 的 5:1 SWA 对全局比率是当前教科书式的默认值。

## 概念

### 滑动窗口注意力（SWA）

位置 `i` 的查询只关注 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）中的位置。窗口外的 token 在分数矩阵中得到 `-inf`。

```
完整因果：                滑动窗口（W=4）：
位置 0-7                   位置 0-7, W=4
    0 1 2 3 4 5 6 7          0 1 2 3 4 5 6 7
0 | x                  0 |  x
1 | x x                1 |  x x
2 | x x x              2 |  x x x
3 | x x x x            3 |  x x x x
4 | x x x x x          4 |    x x x x
5 | x x x x x x        5 |      x x x x
6 | x x x x x x x      6 |        x x x x
7 | x x x x x x x x    7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，分数矩阵期望有 1024 × 8192 个非零行——8 倍的缩减。

**KV 缓存随 SWA 缩小。** 每层只需要保留最后 `W` 个 token 的 K 和 V。对于类似 Gemma-3 的配置（1024 窗口，128K 上下文），KV 缓存缩减 128 倍。

**质量成本。** 纯 SWA transformer 在长距离检索上表现不佳。修复方法：在 SWA 层之间穿插完整注意力层。Gemma 3 使用 5:1 SWA：全局。Mistral 7B 使用因果 SWA 堆栈，其中信息通过重叠窗口"向前流动"——每层将有效感受野扩展 `W`，经过 `L` 层后，模型可以关注 `L × W` 个 token 之前。

### 稀疏 / 块注意力

提前选择一个 `N × N` 的稀疏模式。三种经典形状：

- **局部 + 步进（OpenAI sparse transformer）。** 关注最后 `W` 个 token 以及之前的每 `stride` 个 token。以 `O(N · sqrt(N))` 的计算量捕获局部和长距离信息。
- **Longformer / BigBird。** 局部窗口 + 一小部分全局 token（例如 `[CLS]`），它们关注每个人并被每个人关注 + 随机稀疏连接。在同等质量下经验性地获得 2 倍上下文。
- **原生稀疏注意力（DeepSeek，2025）。** 学习哪些 `(Q, K)` 块重要；在核级别跳过零块。与 FlashAttention 兼容。

稀疏注意力是一个核工程问题。数学很简单（掩码分数矩阵）；优势来自于从不将零条目加载到 SRAM 中。FlashAttention-3 和 2026 年的 FlexAttention API 使自定义稀疏模式在 PyTorch 中成为一等公民。

### 差分注意力（DIFF Transformer，2024）

常规注意力有一个"注意力陷阱"问题：softmax 强制每行总和为 1，因此不想特别关注任何内容的 token 将权重倾倒在第一个 token（或前几个）上。这偷走了本应用于真正内容的容量。

差分注意力通过计算**两个**注意力映射并相减来解决这个问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是一个学习的标量（通常为 0.5-0.8）。A1 捕获真实内容权重；A2 捕获陷阱。相减消除了陷阱，将权重重新分配给相关 token。

报告的结果（微软 2024）：困惑度降低 5-10%，相同训练长度下有效上下文长度增加 1.5-2 倍，大海捞针检索更清晰。

### 变体比较

| 变体 | 计算量 | KV 缓存 | 与完整注意力的质量对比 | 生产使用 |
|---------|---------|----------|-----------------|----------------|
| 完整注意力 | O(N²) | O(N) 每层 | 基线 | 每个模型的默认层 |
| SWA（窗口 1024） | O(N·W) | O(W) 每层 | -0.1 ppl，配合全局层效果好 | Gemma 2/3、Phi-3-Long |
| 局部 + 步进稀疏 | O(N·√N) | 混合 | 类似 SWA | OpenAI sparse transformer、Longformer |
| BigBird（局部+全局+随机） | 约 O(N) | 混合 | 在 2 倍上下文中匹配完整 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 活跃比例) | O(N) | 在 0.05 ppl 以内 | DeepSeek-V3.2、2025 |
| 差分注意力 | O(2·N²) | O(2N) | -5 到 -10% ppl | DIFF Transformer、2026 早期模型 |

```figure
gqa-kv-sharing
```

## 动手实现

参见 `code/main.py`。我们实现一个因果掩码比较器，在玩具序列上并排显示完整、SWA、局部+步进和差分注意力。

### 步骤 1：完整因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

来自第 07 课的基线。下三角；对角线以上为零权重。

### 步骤 2：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数——`window`。对于 `window >= n`，你恢复完整因果注意力。对于 `window = 1`，每个 token 只关注自己。

### 步骤 3：局部 + 步进稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

稠密局部窗口加上每 `stride` 个 token 回到序列开头的 token。随着额外层数增加，感受野以对数步长增长。

### 步骤 4：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力传递，用一个学习到的混合系数相减。在代码中我们比较单头注意力和差分注意力的注意力陷阱热力图，观察陷阱的消失。

### 步骤 5：KV 缓存大小

在 `N = 131072` 时打印每种变体的每层缓存大小。SWA 和稀疏变体下降 10-100 倍。差分注意力翻倍。有意识地支付你的内存账单。

## 使用

2026 年生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 以 5:1 混合 SWA（窗口=1024）和全局层。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接受一个掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这编译成一个自定义的 Triton 核。对于常见模式，速度在 FlashAttention-3 的 10% 以内，而掩码函数是一个 Python 可调用对象。

**何时选择每种：**

- **纯完整注意力** — 每层直到约 16K 上下文，或当检索质量最重要时。
- **SWA + 全局混合** — 长上下文（>32K），训练和推理内存受限。32K 以上的 2026 年默认值。
- **稀疏块注意力** — 自定义核，自定义模式。为专门工作负载保留（检索、音频）。
- **差分注意力** — 任何注意力陷阱污染有害的工作负载（长上下文 RAG、大海捞针）。

## 产出

参见 `outputs/skill-attention-variant-picker.md`。该技能根据目标上下文长度、检索需求和训练/推理计算概况为新模型选择注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 的 SWA 将每行最后 4 个 token 之外的所有内容置零。验证 `window=n` 逐位重现完整因果注意力。
2. **中等。** 在第 07 课的期末项目之上实现 `window=1024` 的因果 SWA。在 tinyshakespeare 上训练 1,000 步。验证损失相对于完整注意力退化了多少？峰值内存下降了多​​少？
3. **困难。** 在期末项目模型中实现 Gemma-3 风格的 5:1 层混合（5 SWA，1 全局）。在匹配参数下，比较与纯 SWA 和纯全局基线的损失、内存和生成质量。
4. **困难。** 实现每头带学习 `λ` 的差分注意力。在合成检索任务（一个目标，2,000 个干扰项）上训练。与匹配参数的单头注意力基线比较检索准确率。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 滑动窗口注意力（SWA） | "局部注意力" | 每个查询关注其最后 `W` 个 token；KV 缓存缩小到 `O(W)`。 |
| 有效感受野 | "模型能看到多远之前" | 在 `L` 层 SWA 堆栈中，窗口为 `W`，最多可达 `L × W` 个 token。 |
| Longformer / BigBird | "局部 + 全局 + 随机" | 带有少数始终注意的全局 token 的稀疏模式；早期的长上下文方法。 |
| 原生稀疏注意力 | "DeepSeek 的核技巧" | 学习块级稀疏性；在核级别跳过零块同时保持质量。 |
| 差分注意力 | "两个映射，一个相减" | DIFF Transformer：减去一个学习的 `λ` 倍的第二注意力映射从第一个，以消除注意力陷阱。 |
| 注意力陷阱（Attention sink） | "权重泄漏到 token 0" | Softmax 归一化强制行总和为 1；无信息查询将权重倾倒在位置 0。 |
| FlexAttention | "掩码即 Python" | PyTorch 2.5+ API，将任意掩码函数编译成 FlashAttention 形状的核。 |
| 层类型混合 | "5:1 SWA 对全局" | 在堆栈中交错稀疏和完整注意力层，以在更低内存下保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 规范滑动窗口 + 全局 token 论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 局部 + 全局 + 随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的局部+步进模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:全局混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 5:1 混合，窗口=1024，现已成为教科书式默认。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的学习稀疏注意力。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — "使用"部分中掩码即可调用模式的 API 参考。
