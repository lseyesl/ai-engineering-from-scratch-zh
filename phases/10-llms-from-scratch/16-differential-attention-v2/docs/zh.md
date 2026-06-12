# 差分注意力（V2）

> Softmax 注意力会在每个不匹配的 token 上分配少量概率。在 10 万个 token 上，这种噪声累积起来会淹没信号。差分 Transformer（Ye 等人，ICLR 2025）通过将注意力计算为两个 softmax 的差来修复这一问题，从而减去共享的噪声基底。DIFF V2（微软，2026 年 1 月）是生产栈的重写：解码延迟与基线 Transformer 匹配，无需自定义内核，兼容 FlashAttention。本课程从 V1 到 V2 端到端讲解，包含一个可以在 stdlib Python 中运行的差分操作工作玩具实现。

**类型:** 构建
**语言:** Python (stdlib)
**前置要求:** 第 7 阶段·第 02 课（自注意力），第 7 阶段·第 15 课（注意力变体），第 10 阶段·第 14 课（架构解读）
**时间:** ~60 分钟

## 学习目标

- 精确说明为什么 softmax 注意力存在噪声基底，以及它如何随上下文长度增长。
- 推导差分注意力公式，并解释减法为何能抵消共享的噪声分量同时保留信号。
- 梳理 V1 到 V2 的差异：哪些变快了、哪些变简单了、哪些变得更稳定了，以及为什么每个变化对于生产预训练都是必要的。
- 在纯 Python 中从头实现差分注意力，并在合成的信号加噪声查询上经验性地验证噪声消除特性。

## 问题

标准的 softmax 注意力有一个数学属性，在大规模下会变成运营上的麻烦。对于查询 `q`，注意力权重为 `softmax(qK^T / sqrt(d))`。Softmax 永远无法产生精确的零——每个不匹配的 token 都会获得一些正的质量。这种残余质量就是噪声，它随上下文长度而扩展。在 128k token 处，即使每个不匹配 token 只获得 0.001% 的概率，127,999 个这样的 token 合计贡献了大约 12% 的总量。模型必须学会绕过随上下文增长的噪声基底。

经验上，这表现为注意力头干扰：长上下文 RAG 中的幻觉引用、10 万 token 检索任务中的"迷失在中间"的失败，以及在 32k 以上的大海捞针基准测试中微妙的准确率下降。差分 Transformer 论文（arXiv:2410.05258, ICLR 2025）测量了这一差距：DIFF Transformer 实现了比同尺寸基线更低的困惑度、更高的长上下文准确率和更少的幻觉。

DIFF V1 有三个问题使其无法进入前沿预训练流水线。其值缓存必须在每个解码步骤加载两次，它需要破坏 FlashAttention 兼容性的自定义 CUDA 内核，以及它的每头 RMSNorm 在 70B 以上规模的长周期训练中不稳定。DIFF V2（微软 unilm 博客，2026 年 1 月 20 日）修复了所有三个问题。本课程讲解两个版本，构建差分算子，并在一个玩具查询上基准测试噪声消除效果。

## 概念

### Softmax 的噪声基底

对于查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有 `w_i` 会为零。如果 `k_i` 与 `q` 完全无关，得分 `q . k_i` 不会是 0——它以方差 `||q||^2 / d` 在零附近波动。经过 softmax 归一化后，每个无关 token 仍然对加权和贡献 `O(1/N)`。无关 token 的总贡献是 `O((N-1)/N) = O(1)`——这不是一个小量。

模型想要的是类似硬 top-k 的东西：匹配 token 上高权重，其他地方接近零权重。Softmax 过于平滑，无法直接做到这一点。

### 差分思想

将每个头的 Q 和 K 投影分成两部分：Q = (Q_1, Q_2) 和 K = (K_1, K_2)。计算两个注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法抵消了这两个图共享的任何噪声分布。如果两个图在 127k 个无关 token 上都有大致均匀的权重（在随机初始化时它们确实如此），这些噪声会被抵消。信号——在少数实际相关 token 上的尖峰权重——只有在两个图中以相同幅度出现时才会被抵消，而一旦模型训练，这种情况就不会发生。

`lambda` 是一个每头可学习的标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是负数。`lambda_init` 默认为一个小的正数，如 0.8。

### 为什么这相当于噪声消除

想象两个有噪声的麦克风录制同一个声音。两者都拾取了说话者加上相关的背景噪声。将一个减去另一个，共享的噪声就会消失。声音保留下来，因为两个信号在相位或幅度上存在足够的差异，防止了完全抵消。每头的 `lambda` 精确地学习这种平衡。

### V1 与 V2：差异对比

V1 保持了与基线 Transformer 相等的参数量。为了每头获得两个查询，它将头维度减半。这牺牲了头的表达能力，并且——更麻烦的是——将每头的值缓存减半。解码必须每步加载值缓存两次（每个 softmax 分支一次）。结果：尽管参数量相同，解码速度却比基线慢。

V2 将查询头的数量翻倍，同时保持 KV 头数量不变（从上投影中借用参数）。头维度与基线保持一致。减法之后，额外的维度被投影回以匹配基线 Transformer 的 O_W 投影。同时发生了三件事：

1. 解码速度与基线匹配（KV 缓存只加载一次）。
2. FlashAttention 无需更改即可运行（无需自定义内核）。
3. 解码时的算术强度增加（每次从 HBM 加载的字节执行更多计算）。

V2 还移除了 V1 用于稳定减法的每头 RMSNorm。在 70B 级别的预训练规模下，该 RMSNorm 在训练后期变得不稳定。V2 用一个更简单的初始化方案代替了它，无需额外模块即可保持训练稳定。

### 何时使用

| 工作负载 | 收益 |
|----------|------|
| 长上下文 RAG（64k+） | 更清晰的注意力图，更少的幻觉引用 |
| 大海捞针基准测试 | 在 32k 以上有显著的准确率提升 |
| 多文档问答 | 减少跨文档干扰 |
| 8k 下的代码补全 | 边际收益，不值得架构改变 |
| 短聊天（< 4k） | 与基线基本无区别 |

其价值随上下文长度增长。在 4k token 下，噪声基底足够小，标准注意力没问题。在 128k 下，它正在损害你的模型。

### 与其他 2026 旋钮的兼容性

| 特性 | 与 DIFF V2 兼容？ |
|------|------------------|
| GQA | 是（V2 增加 Q 头，而非 KV 头） |
| MLA（DeepSeek） | 原则上可行，尚无已发表的论文将两者结合 |
| MoE | 是（注意力独立于 MLP 块） |
| RoPE | 是（无需更改） |
| YaRN / 长上下文扩展 | 是（这正是 DIFF 最有帮助的地方） |
| FlashAttention | V2 中是（V1 中否） |
| 推测解码 | 是（注意力变化对推测解码循环不可见） |

```figure
differential-attention
```

## 构建

`code/main.py` 在纯 Python 中实现差分注意力。一个具有已知信号加噪声结构的玩具查询让你可以直接测量噪声消除比。

### 第 1 步：标准 softmax 注意力

Stdlib 矩阵运算：列表的列表、手动矩阵乘法、带数值稳定性最大减法 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 第 2 步：将 Q、K 分成两半

V1 风格：将头维度减半。V2 风格：保持头维度不变，将头数量翻倍。玩具实现使用 V1 以获得教学上的清晰性——数学是相同的，只有记账方式不同。

### 第 3 步：两个 softmax 分支 + 减法

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以是负数。这没问题——值缓存仍然处理带符号的贡献。后续的 V 投影吸收符号。

### 第 4 步：噪声消除测量

构建一个长度为 1024 的合成序列。将信号 token 放在已知位置，其余填充噪声。计算（a）标准 softmax 注意力在信号位置上的权重和（b）差分注意力的权重。测量两者的信噪比。DIFF 注意力可靠地产生更高的信噪比，根据两个分支的训练差异程度，高出 3 到 10 倍。

### 第 5 步：V1 与 V2 的参数量核算

给定一个配置（hidden=4096, heads=32, d_head=128），打印：

- 基线 Transformer：Q、K、V 各为 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 各为 `hidden * hidden`，V 为 `hidden * hidden`（不变），头维度内部减半。增加了每头 `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q 为 `2 * hidden * hidden`，K 为 `hidden * hidden`，V 为 `hidden * hidden`。额外维度在 O_W 之前投影回原始大小。增加了相同的 `lambda` 参数。

玩具测量 V2 的额外参数成本（每个注意力块大约 `hidden * hidden` 额外参数）并打印。

## 使用

截至 2026 年 4 月，DIFF V2 尚未在每个生产推理服务器中上线，但 vLLM 和 SGLang 的集成正在进行中。同时，这种模式出现在：

- 微软内部的长上下文生产模型中。
- 几个针对 256k+ 上下文的开源模型训练运行的研究复现。
- 将 DIFF 注意力与交替层的滑动窗口注意力相结合的混合架构。

2026 年何时使用：

- 从头训练一个面向 64k+ 有效上下文的新模型。从一开始就添加差分注意力；后期重新训练代价高昂。
- 微调一个长上下文模型，其中"迷失在中间"的失败主导了你的评估。在 Q 投影上的 LoRA 可以近似 DIFF 结构。

何时不使用：

- 你在服务一个预训练的密集模型，该模型具有稳定的长上下文性能。重新训练成本在现有权重上很少能收回回报。
- 你的上下文始终在 16k 以下。噪声基底可以忽略。

## 交付物

本课程产出 `outputs/skill-diff-attention-integrator.md`。给定一个模型架构、目标上下文长度、幻觉画像和训练预算，它为将差分注意力添加到新的预训练运行或 LoRA 微调中制定集成计划。

## 练习

1. 运行 `code/main.py`。验证差分注意力报告的信噪比在合成查询上高于标准 softmax 注意力。改变噪声幅度，显示标准注意力变得不可用的交叉点。

2. 计算从基线到 DIFF V1 以及从基线到 DIFF V2 的参数量变化，针对 7B 类模型（hidden=4096, heads=32, d_head=128, 32 层）。显示哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 论文的第 3 节（arXiv:2410.05258）和 DIFF V2 Hugging Face 博客的第 2 节。用两句话解释为什么 V1 的每头 RMSNorm 是必要的，以及为什么 V2 可以移除它而不会导致训练发散。

4. 实现一个消融实验：计算 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完全减法）下的差分注意力。在合成查询上，测量信噪比在整个扫描过程中的变化。识别出最大化信噪比的 `lambda`。

5. 将玩具扩展到 GQA + DIFF V2。选择 8 个 KV 头和 32 个 Q 头。展示 KV 缓存大小与具有相同（8，32）配置的基线 GQA 模型匹配。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 差分注意力 | "两个 softmax 相减" | 将 Q、K 分成两半，计算两个 softmax 图，将第二个（乘以 lambda 缩放后）从第一个中减去，然后乘以 V |
| 噪声基底 | "softmax 的非零尾部" | softmax 分配给每个无关 token 的 O(1/N) 权重，在长上下文中总计为 O(1) |
| lambda | "减法缩放因子" | 每头可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | "ICLR 2025 版本" | 原始差分 Transformer；将头维度减半以保持参数量，需要自定义内核，解码较慢 |
| DIFF V2 | "2026 年 1 月修复" | 将 Q 头翻倍，保持 KV 头不变；匹配基线解码速度，与 FlashAttention 兼容 |
| 每头 RMSNorm | "V1 稳定器" | V1 在差分后应用的额外归一化；V2 移除了它以防止训练后期不稳定性 |
| 信噪比 | "多少注意力被浪费了" | 真实信号位置权重与无关位置平均权重之比 |
| 迷失在中间 | "长上下文失败模式" | 检索准确率在长上下文中间部分的文档上下降的经验现象——DIFF 注意力减少了这种情况 |
| 算术强度 | "每加载字节的 FLOPs" | V2 通过每次 KV 加载增加查询数量来提高的比率；对内存受限的解码很重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 原始论文，包含噪声消除理论和长上下文消融实验
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — 生产栈重写，匹配基线解码，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 关于减法为何恢复预训练注意力结构的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — 参数共享变体
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF 所减去的基线 Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — DIFF 注意力所针对的长上下文基准测试
