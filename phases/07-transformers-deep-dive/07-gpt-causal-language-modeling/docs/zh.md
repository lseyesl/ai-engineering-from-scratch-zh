# GPT — 因果语言建模

> BERT 查看两侧。GPT 只看到过去。三角掩码是现代 AI 中影响最深远的一行代码。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 7 · 02（自注意力），阶段 7 · 05（完整 Transformer），阶段 7 · 06（BERT）
**时间：** ~75 分钟

## 问题

语言模型回答一个问题：给定前 `t-1` 个 token，token `t` 的概率分布是什么？在这个信号上训练——下一个 token 预测——你得到一个可以一次一个 token 地生成任意文本的模型。

为了并行地在整个序列上端到端训练它，每个位置的预测只能依赖于更早的位置。否则模型会通过查看答案而简单作弊。

因果掩码实现了这一点。它是一个在 softmax 之前添加到注意力分数的上三角 `-inf` 值矩阵。经过 softmax 后，这些位置变为 0。每个位置只能关注自身和更早的位置。因为你只需对整个序列应用一次，你可以在一次前向传播中得到 N 个并行的下一个 token 预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是具有相同核心循环的仅解码器因果 transformer。只是更大、数据更好、RLHF 更好。

## 概念

![因果掩码创建三角形注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定长度为 `N` 的序列，构建一个 `N × N` 的矩阵：

```
M[i, j] = 0       如果 j <= i
M[i, j] = -inf    如果 j > i
```

在 softmax 之前将 `M` 加到原始注意力分数上。`exp(-inf) = 0`，因此被掩码的位置贡献零权重。注意力矩阵的每一行是仅对先前位置的概率分布。

实现成本：一次 `torch.tril()` 调用。计算时间：纳秒级。对该领域的影响：一切。

### 并行训练，串行推理

训练：一次前向传播整个 `(N, d_model)` 序列，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列方向并行。这就是 GPT 训练可以扩展的原因——你在一次 GPU 传递中处理一批中的 1M 个 token。

推理：你一个 token 一个 token 地生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV 缓存（第 12 课）保存了 `t1…tn` 的隐藏状态，这样你就不用在每一步重新计算它们。但推理时的串行深度 = 输出长度。这就是自回归的代价，也是为什么解码是每个 LLM 的延迟瓶颈。

### 损失——偏移一位

给定 tokens `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对于每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是整个序列的交叉熵。

你听说过的每个 transformer LM 都在这个损失上训练。预训练、微调、SFT——相同的损失，不同的数据。

### 解码策略

训练后，采样选择比人们想象的更重要。

| 方法 | 作用 | 何时使用 |
|--------|--------------|-------------|
| 贪心（Greedy） | 每一步取 argmax | 确定性任务、代码补全 |
| 温度（Temperature） | 将 logits 除以 T，再采样 | 创意任务，T 越高 = 多样性越高 |
| Top-k | 仅从 top-k 个 token 中采样 | 消除低概率尾部 |
| Top-p（核采样） | 从累积概率 ≥ p 的最小集合中采样 | 2020+ 默认；适应分布形状 |
| Min-p | 保留 `p > min_p * max_p` 的 token | 2024+；比 top-p 更好地拒绝长尾 |
| 推测解码（Speculative decoding） | 草稿模型提出 N 个 token，大模型验证 | 同等质量下延迟降低 2-3 倍 |

2026 年，min-p + 温度 0.7 是开放权重模型的合理默认值。推测解码是任何生产推理堆栈的入门门槛。

### "GPT 配方"成功的原因

1. **仅解码器。** 没有编码器开销。每层一次注意力 + FFN 传递。
2. **扩展。** 124M → 1.5B → 175B → 万亿。Chinchilla 扩展定律（第 13 课）告诉你怎么分配计算。
3. **上下文学习。** 在 6B–13B 左右涌现。模型可以在不微调的情况下遵循少样本示例。
4. **RLHF。** 对人类偏好的后训练将原始预训练文本转化为聊天助手。
5. **预归一化 + RoPE + SwiGLU。** 大规模稳定训练。

自 GPT-2 以来，核心架构变化不大。一切有趣的事情都发生在数据、规模和后训练上。

```figure
causal-mask
```

## 动手实现

### 步骤 1：因果掩码

参见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前将其添加到注意力分数中。这就是整个机制。

### 步骤 2：2 层的类 GPT 模型

堆叠两个解码器模块（掩码自注意力 + FFN，无交叉注意力）。添加 token 嵌入、位置编码和解嵌入（与 token 嵌入矩阵共享——自 GPT-2 以来的标准技巧）。

### 步骤 3：端到端的下一个 token 预测

在一个包含 20 个 token 的玩具词表上，在每个位置产生 logits。计算与偏移一位目标的交叉熵损失。没有梯度——这是前向传播的健全性检查。

### 步骤 4：采样

实现贪心、温度、top-k、top-p、min-p。在固定提示上运行每种方法并比较输出。采样函数只需 10 行。

## 使用

PyTorch，2026 年惯用写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 运行前向传播，提取最后一个位置的 logits，采样下一个 token，追加它，然后重复。每个生产级 LLM 推理堆栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都实现相同的循环并进行大量优化——批处理预填充、连续批处理、KV 缓存分页、推测解码。

**GPT vs BERT，各一行：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失决定了模型是否可以生成。

## 产出

参见 `outputs/skill-sampling-tuner.md`。该技能为新的生成任务选择采样参数，并在需要确定性解码时进行标记。

## 练习

1. **简单。** 运行 `code/main.py` 并验证因果注意力矩阵在 softmax 后是下三角的。抽查：第 3 行应该只在列 0-3 有权重。
2. **中等。** 为宽度 4 实现束搜索。比较束搜索-4 与贪心搜索在 10 个短提示上的困惑度。束搜索总是更优吗？（提示：通常对翻译是，对开放聊天不是。）
3. **困难。** 实现推测解码：使用小型 2 层模型作为草稿，6 层模型作为验证器。在 100 个长度为 64 的补全上测量挂钟加速。确认输出与验证器的贪心搜索匹配。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 因果掩码（Causal mask） | "三角形" | 添加到注意力分数的上三角 `-inf` 矩阵，使得位置 `i` 只能看到位置 `≤ i`。 |
| 下一个 token 预测（Next-token prediction） | "损失" | 模型分布与每个位置的真实下一个 token 之间的交叉熵。 |
| 自回归（Autoregressive） | "一次生成一个" | 将输出作为输入反馈；仅在训练时并行，生成时不行。 |
| Logits | "softmax 前的分数" | 语言模型头在 softmax 之前的原始输出；采样在这些值上进行。 |
| 温度（Temperature） | "创造力旋钮" | 将 logits 除以 T；T→0 = 贪心，T→∞ = 均匀。 |
| Top-p | "核采样" | 截断分布到总和 ≥ p 的最小集合；从剩余部分采样。 |
| Min-p | "比 top-p 更好" | 保留 `p ≥ min_p × max_p` 的 token；根据分布锐利程度自适应截断。 |
| 推测解码（Speculative decoding） | "草稿 + 验证" | 廉价模型提出 N 个 token；大模型并行验证。 |
| 教师强制（Teacher forcing） | "训练技巧" | 在训练期间，输入真实的前一个 token，而不是模型的预测。每个 seq2seq LM 的标准做法。 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 和上下文学习。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 推测解码论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 规范因果 LM 参考代码。
