# 推测解码 — 草稿、验证、重复

> 自回归解码是串行的。每个 token 等待前一个。推测解码打破了链条：一个廉价模型草拟 N 个 token，昂贵模型在一次前向传播中验证所有 N 个。当草稿正确时，你为 N 次生成支付一次大模型前向传播。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 7 · 07（GPT 因果 LM），阶段 7 · 12（KV 缓存和 Flash Attention）
**时间：** ~60 分钟

## 问题

一个 70B LLM 在 H100 上采样一个 token 需要约 30 ms。一个 3B 草稿模型需要约 3 ms。如果我们让 3B 草稿模型提前生成 5 个 token，然后运行 70B 模型*一次*来验证所有 5 个，总耗时是 `5×3 + 30 = 45 ms` 以获得最多 5 个被接受的 token——相比之下直接生成需要 `5×30 = 150 ms`。这就是完整的推测解码方案：用少量的额外 GPU 内存（草稿模型）换取 2-4 倍更低的解码延迟。

这个技巧必须保持分布不变。Leviathan 等人（2023）和 Chen 等人同时提出的推测采样保证了输出序列与大模型自己生成的结果**分布完全相同**。没有质量折衷。只是更快。

四个家族主导了 2026 年推理的草稿-验证器配对：

1. **原始推测（Leviathan 2023）。** 独立的草稿模型（例如 Llama 3 1B）+ 验证器（例如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 验证器上的多个解码头并行预测位置 `t+1..t+k`。没有独立的草稿模型。
3. **EAGLE 家族（Li 2024，2025）。** 轻量级草稿，重用验证器的隐藏状态；接受率比原始方案更高；典型 3-4 倍。
4. **Lookahead 解码（Fu 2024）。** Jacobi 迭代；完全不需要草稿模型。自推测。小众但无需依赖。

2026 年每个生产推理堆栈默认都带有推测解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都至少支持原始 + EAGLE-2。

## 概念

### 核心算法

给定一个验证器 `M_q` 和一个更便宜的草稿模型 `M_p`：

1. 设 `x_1..x_k` 为已解码的前缀。
2. **草稿**：使用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，带有草稿概率 `p_1..p_N`。
3. **并行验证**：在 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 上运行 `M_q` 一次，得到位置 `k+1..k+N+1` 的验证器概率 `q_1..q_{N+1}`。
4. **从左到右接受/拒绝每个草稿 token**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 首次拒绝时：从归一化的"残余"分布 `(q_j - p_j)_+` 采样 `t_j`。`j` 之后的所有草稿都被丢弃。
6. 在接受了所有 `N` 个 token 时：从 `q_{N+1}` 采样一个额外的 token `t_{N+1}`（免费奖励 token）。

残余分布技巧是保持输出分布恰好与 `M_q` 从头采样相同的数学洞见。

### 什么决定了加速

设 `α` = 每个草稿 token 的期望接受率。设 `c` = 草稿与验证器的成本比。每步：

- 朴素生成每 token 需要 1 次大模型调用。
- 推测生成每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 需要 1 次大模型调用（当 α 较高时）。

在 `α = 0.75` 和 `N = 5` 时的典型经验法则：大模型调用减少 3 倍。草稿成本是 5 倍廉价。总挂钟时间下降约 2.5 倍。

**α 取决于：**

- 草稿模型逼近验证器的程度。同家族/同训练数据显著提高 α。
- 解码策略。贪心草稿对贪心验证器：高 α。温度采样：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受更多（可预测）；自由形式创意写作接受更少。

### Medusa — 无需草稿模型的草稿

Medusa 用验证器上的额外输出头替换草稿模型。在位置 `t`：

```
共享主干 → 隐藏状态 h_t
    ├── head_0: 预测 t+1 处的 token（标准 LM 头）
    ├── head_1: 预测 t+2 处的 token
    ├── head_2: 预测 t+3 处的 token
    ├── head_3: 预测 t+4 处的 token
```

每个头输出自己的 logits。在推理时，你从每个头采样得到候选序列，然后使用一种树注意力方案一次考虑所有候选延续，通过一次前向传播进行验证。

优点：没有第二个模型。缺点：增加了可训练参数；需要一个监督微调阶段（约 1B token）；接受率略低于使用良好草稿模型的原始推测。

### EAGLE — 通过重用隐藏状态获得更好的草稿

EAGLE-1/2/3（Li 等人，2024-2025）使草稿模型成为一个极小的 transformer（通常 1 层），它接收验证器最后一层的隐藏状态。因为草稿模型看到了验证器的特征表示，它的预测与验证器的输出分布高度相关。接受率从约 0.6（原始）攀升到 0.85 以上。

EAGLE-3（2025）添加了候选延续的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认推测路径。

### KV 缓存之舞

验证在一次前向传播中将 `N` 个草稿 token 馈送到验证器中。这将验证器的 KV 缓存扩展了 `N` 个条目。如果某些草稿被拒绝，你必须将缓存回滚到已接受的前缀长度。

生产实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）使用临时 KV 缓冲区处理这个问题。先写入，接受时提交。概念上不难，但很繁琐。

## 动手实现

参见 `code/main.py`。我们实现核心的推测采样算法（拒绝步骤 + 残余分布），包含：

- 一个"大模型"，它对手工编码的分布进行确定性的 softmax（这样我们可以分析性地验证接受数学）。
- 一个"草稿模型"，它是大模型的扰动版本。
- 一个接受/拒绝循环，产生与直接采样相同的边际分布。

### 步骤 1：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是一个均匀随机数。`q_prob` 是验证器对草稿 token 的概率。`p_prob` 是草稿模型的概率。Leviathan 定理表明，这个伯努利决策，随后在拒绝时从残余分布采样，精确地保持了验证器的分布。

### 步骤 2：残余分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素从 `q` 中减去 `p`，将负值钳位到零，重新归一化。在任何拒绝时从此分布采样。

### 步骤 3：一个推测步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个被接受 → 一个奖励 token → 在一次验证器传递中产生六个 token。

### 步骤 4：测量接受率

在不同草稿质量水平上运行 10,000 个推测步骤。绘制接受率与草稿和验证器分布之间的 KL 散度。你应该看到一个清晰的单调关系。

### 步骤 5：验证分布等价性

经验验证：推测循环产生的 token 直方图应与直接从验证器采样的直方图匹配。这就是实践中的 Leviathan 定理。卡方检验在采样误差范围内确认。

## 使用

生产环境中：

```bash
# 使用 EAGLE 的 vLLM
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# 使用原始草稿模型的 vLLM
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中，TensorRT-LLM 拥有最快的 Medusa 路径。`faster-whisper` 为 Whisper-large 封装了带有小型草稿的推测解码。

**选择草稿：**

| 策略 | 何时选择 | 加速 |
|----------|--------------|---------|
| 原始草稿（1B/3B Llama 系列） | 快速原型，无需训练 | 1.8-2.3× |
| Medusa 头 | 你可以微调验证器 | 2-3× |
| EAGLE-2 / 3 | 生产环境，最高速度 | 3-4× |
| Lookahead | 无需草稿、无需训练、无需额外参数 | 1.3-1.6× |

**何时不进行推测解码：**

- 1-5 个 token 的单序列生成。开销占主导。
- 高度创意 / 高温采样（α 下降）。
- 内存受限的部署（草稿模型增加 VRAM）。

## 产出

参见 `outputs/skill-spec-decode-picker.md`。该技能为新的推理工作负载选择推测解码策略（原始 / Medusa / EAGLE / lookahead）和调参参数（N、草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。在 50,000 个 token 上确认推测 token 分布在卡方 p > 0.05 范围内与验证器的直接采样分布匹配。
2. **中等。** 对于 `α = 0.5, 0.7, 0.85`，绘制加速比（每大模型前向的 token 数）作为 `N` 的函数。确定每个 α 的最优 N。（提示：每次验证调用的期望 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个小型 Medusa：取第 14 课的期末项目 GPT，添加 3 个预测位置 t+2, t+3, t+4 的额外 LM 头。使用联合多头损失在 tinyshakespeare 上训练。与通过截断同一模型制作的原始草稿比较接受率。
4. **困难。** 实现回滚：从一个 10-token 前缀 KV 缓存开始，输入 5 个草稿 token，模拟在位置 3 的拒绝。验证你的缓存读取在下一次迭代时正确匹配"前缀 + 前 2 个已接受的草稿"。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 草稿模型（Draft model） | "廉价的那个" | 提出候选 token 的较小模型；通常比验证器便宜 10-50 倍。 |
| 验证器（Verifier） | "大的那个" | 目标模型，其分布我们需要保持；每个推测步骤运行一次。 |
| 接受率（α） | "草稿正确的频率" | 验证器接受草稿的每 token 概率。典型 0.7-0.9。 |
| 残余分布（Residual distribution） | "拒绝时的回退" | `(q - p)_+` 归一化；在拒绝时从此采样保持验证器的分布。 |
| 奖励 token（Bonus token） | "免费的那个" | 当所有 N 个草稿都被接受时，从验证器的下一步分布中再采样一个。 |
| Medusa | "无需草稿的推测" | 验证器上的多个 LM 头并行预测位置 t+1..t+k。 |
| EAGLE | "隐藏状态草稿" | 以验证器最后一层隐藏状态为条件的小型 transformer 草稿。 |
| Lookahead 解码 | "Jacobi 迭代" | 使用不动点迭代的自推测；无需草稿模型。 |
| 树注意力（Tree attention） | "一次验证多个候选" | 同时考虑多个草稿延续的分支验证。 |
| KV 回滚 | "撤销被拒绝的草稿" | 临时 KV 缓冲区；接受时提交，拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法和等价定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 同时引入；简洁的伯努利拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；树注意力验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；隐藏状态条件草稿。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead，无需草稿的方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 所有四种策略的规范生产参考。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的参考代码。
