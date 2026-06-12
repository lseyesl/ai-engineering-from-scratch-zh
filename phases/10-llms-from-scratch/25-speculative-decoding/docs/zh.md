# 推测解码与 EAGLE

> 前沿 LLM 生成一个 token 需要对数十亿参数做一次完整前向传播。该前向传播严重过度配置：大多数时候，一个小得多的模型就能正确猜出接下来 3-5 个 token，大模型只需要*验证*猜测。当猜测正确时，你用一次前向传播的代价获得了 5 个 token。推测解码（Leviathan et al. 2023）使这一点精确化，EAGLE-3（2025）将接受率推至约每验证 4.5 个 token——在匹配输出分布的情况下实现 4-5 倍加速。

**Type:** 构建
**Languages:** Python（使用 numpy）
**Prerequisites:** 第 10 阶段第 12 课（推理优化），第 10 阶段第 04 课（预训练 Mini-GPT）
**Time:** ~75 分钟

## 问题所在

70B 级模型在 H100 上的解码吞吐量通常为 40-80 token/秒。每个 token 需要一次完整前向传播，从 HBM 读取所有模型权重。你不能在不改变输出的情况下缩小模型。你不能在内存限制之外增加批大小。你卡住了——除非你能让模型每次前向传播输出不止一个 token。

自回归生成看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但存在一个并发机会。如果你有一个廉价预测器说"接下来 4 个 token 可能是 [a, b, c, d]"，你可以在**大模型的一次前向传播**中验证所有 5 个位置，并接受最长匹配前缀。

Leviathan, Kalia, Matias（2023，"Fast Inference from Transformers via Speculative Decoding"）通过巧妙的接受/拒绝规则使这一点精确化，该规则保持目标模型的采样分布。相同的输出分布，2-4 倍更快。

## 核心概念

### 双模型设置

- **目标模型** `M_p`：你实际想要采样的大、慢、高质量模型。分布：`p(x)`。
- **草稿模型** `M_q`：小、快、质量较低的模型。分布：`q(x)`。小 5-30 倍。

每步：

1. 草稿模型自回归地提出 `K` 个 token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型对所有 `K+1` 个位置并行运行一次前向传播，为每个提出的 token 产生 `p(x_k)`。
3. 通过下面的修正拒绝采样规则从左到右逐个接受/拒绝每个 token。接受最长匹配前缀。
4. 如果任何 token 被拒绝，从修正分布中采样替换并停止。否则从 `p(· | x_1...x_K)` 采样一个奖励 token。

如果草稿完美匹配目标，每次目标前向获得 K+1 个 token。如果草稿在第 1 位就错了，你只得到 1 个 token。

### 精确性规则

推测解码在分布上**可证明等价于从 p 采样**。拒绝规则：

```
对于每个草稿 token x_t：
r ~ Uniform(0, 1)
如果 r < p(x_t) / q(x_t)：
    接受 x_t
否则：
    从残差分布采样替换：(p - q)+ / ||(p - q)+||_1
    停止
```

其中 `(p - q)+` 表示逐点差的正部。当草稿和目标一致（`p ≈ q`）时，接受率接近 1。当它们不一致时，残差分布的构造保证整体采样仍然精确为 `p`。

**贪心情况。** 对于 temperature=0 采样，只需检查 `argmax(p) == x_t`。如果是，接受；如果不是，输出 `argmax(p)` 并停止。

### 预期加速比

如果草稿模型的 token 级接受率为 `α`，每次目标前向传播产生的预期 token 数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)  # K = 草稿长度，α ∈ [0, 1]
```

在 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 个 token 每次前向。单次目标前向的代价大约为 `cost_q * K + cost_p`（K 步草稿加一次目标验证）。如果 `cost_p >> cost_q * K`，加速比为 `3.36× / 1 = 3.36×` 吞吐量。

唯一真正的参数是 `α`，它完全取决于草稿-目标对齐度。好的草稿就是一切。

### 训练草稿：蒸馏

随机小模型做不了好草稿。标准配方是从目标蒸馏：

1. 选择小架构（70B 目标用约 1B，7B 目标用约 500M）。
2. 在大型文本语料上运行目标模型；存储其下一 token 分布。
3. 用 KL 散度对目标的分布训练草稿（不是对真实 token）。

结果：`α` 在代码上通常为 0.6-0.8，在自然语言对话上为 0.7-0.85。生产中加速 2-3 倍。

### EAGLE：树状草稿 + 特征复用

Li, Wei, Zhang, Zhang（2024，"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）观察到标准推测解码的两个低效之处：

1. 草稿做 K 步串行计算，每步全栈。但草稿可以复用目标最近验证时的特征（隐藏状态）——目标已经计算了丰富的表示，草稿却从头重新推导。
2. 草稿输出线性链。如果草稿能输出候选*树*（每个节点多个猜测），目标的单次前向传播可以通过树注意力掩码并行验证多条候选路径，并选择最长接受的分支。

EAGLE-1 改动：
- 草稿输入 = 目标在位置 t 的最终隐藏状态，而非原始 token。
- 草稿架构 = 1 个 Transformer 解码器层（而非独立小模型）。
- 输出 = 每深度 K = 4-8 个候选的树，深度 4-6。

EAGLE-2（2024）添加动态树拓扑：树在草稿不确定处变宽，在确定处保持窄。在不增加验证成本的情况下提升 `α_effective`。

EAGLE-3（Li et al. 2025，"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）移除了固定顶层特征依赖，用新的"测试时模拟"损失训练草稿——草稿在匹配目标测试时分布的输出上训练，而非教师强制训练分布。接受率从 0.75（EAGLE-2）升至 0.82（EAGLE-3），平均每验证 token 数从 3.0 升至 4.5。

### 树注意力验证

当草稿输出树时，目标模型使用**树注意力掩码**在单次前向传播中验证——一种编码树拓扑而非纯线的因果掩码。每个 token 只关注其在树中的祖先。验证仍是一次前向、一次矩阵乘法；拓扑掩码只多占少量 KV 条目。

```
root
 / \
a   b
/ \ / \
c d e f
```

如果 `a, b` 是竞争的第一 token 候选，`c, d, e, f` 是第二 token 候选，所有六个位置在一次前向传播中验证。输出是任何被接受路径上的最长前缀。

### 何时有效，何时无效

**有效：**
- 可预测文本的对话/补全（代码、常见英语、结构化输出）。`α` 较高。
- 解码期间有未使用 GPU 计算的设置（内存受限阶段）。树状草稿利用可用 FLOPs。

**无效 / 无收益：**
- 高度随机输出（高温创意写作）。`α` 降至约 `1/|vocab|`。
- 非常高并发批服务——批处理已经填满 FLOPs，树验证几乎没有空间。
- 非常小的目标模型，草稿并不比它小多少。

生产团队通常报告对话 2-3 倍挂钟加速，代码生成 3-5 倍，创意写作接近零。

```figure
speculative-decoding
```

## 构建它

`code/main.py`：

- 一个参考 `speculative_decode(target, draft, prompt, K, temperature)`，实现精确拒绝规则并验证其保持目标分布（经验 KL < 0.01 vs 普通目标采样）。
- 一个 EAGLE 风格树草稿器，用 top-p 分支构建深度 K 树。
- 一个树注意力掩码构建器，为验证器生成正确的因果模式。
- 一个接受率测试工具，在微型 LM 上运行两者（从 GPT-2-medium 目标蒸馏一个 GPT-2-small）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """一轮推测解码。返回已接受 token 列表。"""
    # 1. 草拟 K 个 token
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. 目标在每个草稿位置 + 1 个额外位置计算 p
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. 从左到右接受/拒绝
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
        return accepted
    # 4. 全部 K 个被接受 → 从目标采样奖励 token
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用它

- **vLLM** 和 **SGLang** 提供一流的推测解码。标志：`--speculative_model`、`--num_speculative_tokens`。通过 `--spec_decoding_algorithm eagle` 标志支持 EAGLE-2/3。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B 草拟），`meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B 草拟）。
- **Medusa 头**（Cai et al. 2024，"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）：不用草稿模型，在目标本身上添加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 交付它

本课程产出 `outputs/skill-speculative-tuning.md`——一个分析目标模型工作负载并选择草稿模型、K（草稿长度）、树宽度、温度以及何时回退到普通解码的技能。

## 练习

1. 实现精确拒绝规则并经验验证。通过 `speculative_decode` 和普通目标采样各运行 10K 样本；计算两个输出分布之间的 TV 距离。应 < 0.01。

2. 计算加速公式。给定固定 `α` 和 `K`，绘制每次目标前向的预期 token 数。找出 α ∈ {0.5, 0.7, 0.9} 的最优 K。

3. 训练一个小草稿。取 124M GPT-2 目标，在 100M token 上用 KL 损失蒸馏 30M GPT-2 草稿。在留出文本上测量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格树状草稿。不用链，让草稿在每个深度输出 top-3 分支。构建树注意力掩码。验证目标接受最长正确分支。

5. 测量失败模式。在 temperature=1.5（高随机性）下运行推测解码。展示 α 崩溃，算法因草稿开销比普通解码更慢。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 目标模型 | "大模型" | 你想要采样的慢、高质量模型（p 分布） |
| 草稿模型 | "推测器" | 小、快的预测器（q 分布）；小 5-30 倍 |
| K / 草稿长度 | "前瞻" | 每次验证 pass 的推测 token 数 |
| α / 接受率 | "命中率" | 草稿提案被接受的逐 token 概率 |
| 精确拒绝规则 | "接受测试" | 保持目标分布的 r < p/q 比较 |
| 残差分布 | "修正的 p-q" | (p - q)+ / ||(p - q)+||_1，拒绝时从中采样的分布 |
| 树状草稿 | "分支推测" | 草稿输出候选树，用树结构注意力掩码一次验证 |
| 树注意力掩码 | "拓扑掩码" | 编码树拓扑的因果掩码，使每个节点只关注其祖先 |
| Medusa 头 | "并行头" | 目标本身上的 K 个额外预测头；无需单独草稿模型 |
| EAGLE 特征复用 | "隐藏状态草稿" | 草稿输入是目标的最后隐藏状态而非原始 token，缩小草稿 |
| 测试时模拟损失 | "EAGLE-3 训练" | 在匹配目标测试时分布的输出上训练草稿，而非教师强制 |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则和理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 的同期推测采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 草稿模型的并行头替代方案
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征复用和树状草稿
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时测试时匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/前瞻解码，一种无需推测器的替代方案
