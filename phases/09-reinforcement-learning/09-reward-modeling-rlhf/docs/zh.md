# 奖励建模与 RLHF

> 人类无法为"好的助手回复"写出奖励函数，但他们可以比较两个回复并选出更好的一个。将奖励模型拟合到这些比较结果上，然后让语言模型通过 RL 与之对抗。Christiano 2017。InstructGPT 2022。将 GPT-3 变成 ChatGPT 的配方。到 2026 年，它正在被 DPO 取代——但心智模型保留了下来。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 5 · 05（情感分析），阶段 9 · 08（PPO）
**时间：** ~45 分钟

## 问题

你在下一个 token 预测目标上训练了一个语言模型。它写出语法正确的英语。但也会撒谎、绕圈子、拒绝不拒绝。你不能通过更多预训练来解决这个问题——网上的文本是问题本身，而不是解药。

你想要一个*标量奖励*来表明"对于指令 X，回复 A 比回复 B 更好"。手动编写那个奖励函数是不可能的。"有帮助"不是一个关于 token 的闭式表达式。但人类可以比较两个输出并标记偏好。以规模化的方式收集这种数据成本低廉。

RLHF（Christiano 等人 2017 年；Ouyang 等人 2022 年）将偏好转换为奖励模型，然后通过 PPO 让 LM 对抗该奖励进行优化。分三步：SFT → RM → PPO。这就是交付 ChatGPT、Claude、Gemini 和每个其他对齐 LLM 的配方（2023–2025 年）。

到 2026 年，PPO 步骤大部分已被 DPO（阶段 10 · 08）取代，因为它更便宜且在对齐微调方面几乎一样好。但*奖励模型*这部分仍然是每个 Best-of-N 采样器、每个基于可验证奖励的 RL 管道以及每个使用过程奖励模型的推理模型的基础。理解了 RLHF，你就理解了整个对齐技术栈。

## 概念

**阶段 1：监督式微调（SFT）。** 从预训练基座模型开始。在目标行为的人工编写演示（指令遵循回复、有帮助的回答等）上进行微调。结果：一个*偏向好行为*但仍然有无限动作空间的模型 `π_SFT`。

**阶段 2：奖励模型训练。**

- 收集针对提示 `x` 的回复对 `(y_+, y_-)`，由人类标记为"y_+ 优于 y_-"。
- 训练一个奖励模型 `R_φ(x, y)` 为 `y_+` 分配更高的分数。
- 损失：**Bradley-Terry 成对逻辑回归**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid。奖励之差隐含偏好对数几率。BT 自 1952 年（Bradley-Terry）以来一直是标准，也是现代 RLHF 中的主导选择。

- `R_φ` 通常从 SFT 模型初始化，顶部加一个标量头。同一个 transformer 骨干；一个单线性层输出奖励。

**阶段 3：带 KL 惩罚的 PPO 对抗 RM。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保留一个冻结的*参考* `π_ref = π_SFT`。
- 回复 `y` 结束时的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT`——它是一个*正则化项*，而非硬性信任区域。`β` 通常为 `0.01`–`0.05`。
- 使用此奖励运行 PPO（第 8 课）。优势在 token 级别的轨迹上计算，但 RM 只对整个回复评分。

**为什么需要 KL？** 没有它，PPO 会愉快地找到奖励黑客策略——RM 只在分布内的补全上训练过。一个分布外的回复可能获得比任何人类编写的回复更高的分数。KL 使 `π_θ` 保持在 RM 曾训练过的流形附近。它是 RLHF 中最重要的单个旋钮。

**2026 年现状：**

- **DPO**（Rafailov 2023）：闭式代数将阶段 2+3 坍缩为一个基于偏好数据的监督损失。无需 RM，无需 PPO。在对齐基准上取得相同质量，仅需极少的计算量。在阶段 10 · 08 中涵盖。
- **GRPO**（DeepSeek 2024–2025）：使用组相对基线替代评论家的 PPO，奖励来自*验证器*（代码运行 / 数学答案匹配）而非人类训练的 RM。推理模型的主导方法。在阶段 9 · 12 中涵盖。
- **过程奖励模型（PRMs）：** 对部分解法（每个推理步骤）评分，用于 RLHF 和 GRPO 变体以支持推理。
- **宪法式 AI / RLAIF：** 使用一个对齐的 LLM 代替人类生成偏好。扩展偏好预算。

## 动手实现

本课使用微型的合成"提示"和"回复"，表示为字符串。RM 是一个基于 token 包表示之上的线性评分器。没有真正的 LLM——管道的*形状*才是关键，规模不重要。

### 步骤 1：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实的 RLHF 中，这由人类标注者替代。但其形态——`(prompt, preferred_response, rejected_response)`——是相同的。

### 步骤 2：Bradley-Terry 奖励模型

线性分数：`R(x, y) = w · bag(y)`。训练以最小化 BT 成对对数损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

经过几百次更新后，`w` 为好词 token 分配正权重，为坏词分配负权重。

### 步骤 3：在 RM 之上的 PPO 风格策略

我们的玩具策略从词汇表中产生单个 token。我们在 RM 下对 token 评分，计算 `log π_θ(token | prompt)`，添加 KL 到参考的惩罚，并应用裁剪后的 PPO 替代目标。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # 在 theta 上做 ppo 风格更新，将 reward 视为回报
    ...
```

### 步骤 4：监控 KL

每次更新追踪平均 `KL(π_θ || π_ref)`。如果它超过 `~5-10`，策略已经远离 `π_SFT`——要么 `β` 太低，要么奖励黑客攻击已经开始。这是真实 RLHF 中的首要诊断指标。

### 步骤 5：使用 TRL 的生产配方

一旦你理解了玩具管道，以下是用户用真实库编写相同循环的方式。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——`RewardTrainer` 用于阶段 2，`PPOTrainer`（内置对参考的 KL）用于阶段 3。

```python
# 阶段 2：从成对偏好训练奖励模型
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# 数据集行：{"prompt", "chosen", "rejected"} — Bradley-Terry 格式
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# 阶段 3：带 KL 惩罚的 PPO 对抗 RM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 冻结

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats 包含：mean_kl, clip_frac, value_loss — PPO 的三个诊断指标
```

库为你做了三件事。`adap_kl_ctrl=True` 实现自适应 β 调度：如果观察到的 KL 超过 `target_kl`，β 翻倍；如果低于一半，β 减半。参考模型按约定冻结——你一定不能意外地与 `policy` 共享参数。价值头与策略共享同一骨干（`AutoModelForCausalLMWithValueHead` 附加了一个标量 MLP 头），这就是 TRL 分别报告 `policy/kl` 和 `value/loss` 的原因。

## 陷阱

- **过度优化 / 奖励黑客攻击。** RM 不完美；`π_θ` 找到分数很高但实际很差的对抗性补全。症状：奖励无限增长而人类评估分数停滞或下降。修复：提前停止、提高 β、扩展 RM 训练数据。
- **长度黑客。** 在有用回复上训练的 RM 往往隐含地奖励长度。策略学会填充回复。补救：长度归一化奖励，或使用对长度感知的 RM 进行 RLAIF。
- **RM 太小。** RM 至少需要和策略一样大。一个微小的 RM 无法忠实地对策略的输出评分。
- **KL 调参。** β 太低 → 策略漂移和奖励黑客攻击。β 太高 → 策略几乎没有变化。标准技巧是使用*自适应* β，针对每步固定的 KL。
- **偏好数据噪声。** 约 30% 的人类标注是嘈杂或模糊的。通过在经过一致性筛选的数据上训练 RM，或对 BT 使用温度参数来校准。
- **离策略问题。** 第一个轮次后 PPO 数据略微离策略。像第 8 课那样监控裁剪比例。

## 使用

2026 年 RLHF 的分层结构：

| 层 | 目标 | 方法 |
|-------|--------|--------|
| 指令遵循、有帮助、无害 | 对齐 | DPO（阶段 10 · 08）优先于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 使用验证器奖励的 GRPO（阶段 9 · 12）。 |
| 长时间跨度多步任务 | 智能体 | 使用过程奖励模型的 PPO / GRPO 对步骤评分。 |
| 安全 / 拒绝行为 | 安全 | 使用单独安全 RM 的 RLHF-PPO，或宪法式 AI。 |
| 推理时的 Best-of-N | 快速对齐 | 在解码时使用 RM；无需策略训练。 |
| 奖励蒸馏 | 推理计算 | 在冻结的 LM 之上训练小型"奖励头"。 |

RLHF 在 2022–2024 年曾是*那*个方法。到 2026 年，生产级对齐管道以 DPO 为先，PPO 仅用于 RM 密集型或安全关键的步骤。

## 产出

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: 为语言模型设计 RLHF / DPO / GRPO 对齐管道，包括 RM、KL 和数据策略。
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

给定一个基座 LM、目标行为（对齐 / 推理 / 拒绝 / 智能体）以及偏好或验证器预算，输出：

1. 阶段。SFT？RM？DPO？GRPO？附理由。
2. 偏好或验证器来源。人类、AI 反馈、基于规则、单元测试通过或奖励蒸馏。
3. KL 策略。固定 β、自适应 β 或 DPO（隐式 KL）。
4. 诊断指标。平均 KL、奖励稳定性、过度优化防护（留出的人类评估）。
5. 安全门。红队测试集、拒绝率、安全 RM 与有用性 RM 分开。

拒绝在没有 KL 监控的情况下交付 RLHF-PPO。拒绝使用比目标策略更小的 RM。拒绝仅基于长度的奖励。标记任何未保留盲法人类评估集的管道为缺乏过度优化保护。
```

## 练习

1. **简单。** 在 `code/main.py` 中，在 500 个合成偏好对上训练 Bradley-Terry 奖励模型。在留出的 100 对上测量成对准确率。应超过 90%。
2. **中等。** 以 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对每个 β，绘制 RM 分数与 KL-to-reference 随更新的关系。哪些运行出现了奖励黑客？
3. **困难。** 在相同偏好数据上实现 DPO（闭式偏好似然损失），并与 RLHF-PPO 管道在计算量和最终 RM 分数方面进行比较。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| RLHF | "对齐 RL" | 三步 SFT + RM + PPO 管道（Christiano 2017，Ouyang 2022）。 |
| 奖励模型（RM） | "评分网络" | 通过 Bradley-Terry 拟合到成对偏好的学习标量函数。 |
| Bradley-Terry | "成对逻辑损失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL 惩罚 | "靠近参考" | 奖励中的 `β · KL(π_θ \|\| π_ref)`；反奖励黑客正则化器。 |
| 奖励黑客 | "Goodhart 定律" | 策略利用 RM 缺陷；症状：奖励上升，人类评估持平。 |
| RLAIF | "AI 标注的偏好" | 标注来自另一个 LM 而非人类的 RLHF。 |
| PRM | "过程奖励模型" | 对部分推理步骤评分；用于推理管道。 |
| 宪法式 AI | "Anthropic 的方法" | 由明确规则指导的 AI 生成的偏好。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 开启 RLHF 的论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 早期的 RLHF 用于摘要。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年 RLHF 后的默认选择。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 和自我批评循环。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 生产级 `RewardTrainer` 和 `PPOTrainer`。阅读训练器源码以了解自适应 KL 和价值头细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — 三步管道的经典图解教程。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 该库；`examples/` 中有针对 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 奖励假说视角；思考奖励黑客的必要先备知识。
