# 近端策略优化（PPO）

> A2C 在每次更新后丢弃所有展开数据。PPO 将策略梯度包装在裁剪过的重要性比率中，使得你可以在相同数据上做 10+ 个轮次而不会让策略爆炸。Schulman 等人（2017 年）。到 2026 年仍是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 9 · 06（REINFORCE），阶段 9 · 07（演员-评论家）
**时间：** ~75 分钟

## 问题

A2C（第 7 课）是在策略的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 需要从*当前* `π_θ` 采样的数据。做一次更新后，`π_θ` 就变了；你使用的数据现在变成离策略的了。重用它们会导致梯度有偏。

展开是昂贵的。在 Atari 上，一次跨 8 个环境 × 128 步的展开 = 1024 条转移和十几秒的环境时间。在一个梯度步后就把它们丢弃是浪费的。

信任区域策略优化（TRPO，Schulman 2015）是第一个修复方案：约束每次更新，使得新旧策略之间的 KL 散度保持在 `δ` 以下。理论上干净，但每次更新需要共轭梯度求解。2026 年没有人运行 TRPO。

PPO（Schulman 等人 2017 年）用简单的裁剪目标替代了硬性信任区域约束。多一行代码。每次展开十个轮次。没有共轭梯度。理论保证足够好。九年后它仍然是从 MuJoCo 到 RLHF 的默认策略梯度算法。

## 概念

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略与收集数据的策略的似然比。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略采取 `a_t` 的可能性是旧策略的两倍。

**裁剪过的替代目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项：

- 如果优势 `A_t > 0` 且比率试图增长超过 `1 + ε`，裁剪使梯度变平——不要将一个好的动作推得比 `+ε` 更远地超出旧概率。
- 如果优势 `A_t < 0` 且比率试图增长超过 `1 - ε`（意味着我们会使一个坏动作相比其裁剪后的减少变得更可能），裁剪限制梯度——不要将一个坏动作推得比 `-ε` 更低。

`min` 处理另一个方向：如果比率已经朝*有益*的方向移动，你仍然得到梯度（在会伤害你的那一侧没有裁剪）。

典型 `ε = 0.2`。将目标绘制为 `r_t` 的函数：一个分段线性函数，在"好的一侧"有平坦的屋顶，在"坏的一侧"有平坦的地板。

**完整的 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的演员-评论家结构。三个系数，通常 `c_v = 0.5`，`c_e = 0.01`，`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中收集 `N × T` 条转移，每环境 `T` 步。
2. 计算优势（GAE），将它们冻结为常数。
3. 将 `π_{θ_old}` 冻结为当前 `π_θ` 的快照。
4. 对于 `K` 个轮次，对于 `(s, a, A, V_target, log π_old(a|s))` 的每个小批量：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 梯度步。
5. 丢弃展开数据。返回步骤 1。

`K = 10` 和小批量大小为 64 是标准的超参数集。PPO 很鲁棒：确切数字在 ±50% 范围内很少影响结果。

**KL 惩罚变体。** 原始论文提出了一个使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，其中 `β` 根据观察到的 KL 调整。裁剪版本成为主导；KL 变体在 RLHF 中幸存下来（在那里对参考策略的 KL 是一个独立的约束，你总是想要的）。

## 动手实现

### 步骤 1：在展开时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在展开时只取一次。在更新轮次期间不会改变。

### 步骤 2：计算 GAE 优势（第 7 课）

与 A2C 相同。在整个批次上归一化。

### 步骤 3：裁剪替代目标更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # 反向传播 -surrogate，添加价值损失，减去熵
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # 已裁剪
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"裁剪 → 零梯度"模式是 PPO 的核心。如果新策略已经在有益方向上漂移太远，更新就停止。

### 步骤 4：价值和熵

添加对评论家目标的标准 MSE 和对演员的熵奖励，与 A2C 相同。

### 步骤 5：诊断

每次更新要观察的三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]`。如果超过 `0.1`，减少 `K_EPOCHS` 或 `LR`。
- **裁剪比例**——比率在 `[1-ε, 1+ε]` 之外的样本比例。应在 `~0.1-0.3`。如果 `~0`，裁剪从未触发 → 提高 `LR` 或 `K_EPOCHS`。如果 `~0.5+`，你过拟合了展开数据 → 降低它们。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。评论家质量指标。应随着评论家学习向 1 攀升。

## 陷阱

- **裁剪系数调错。** `ε = 0.2` 是事实上的标准。降到 `0.1` 使更新过于胆怯；`0.3+` 会带来不稳定性。
- **轮次太多。** `K > 20` 通常会破坏稳定性，因为策略远离 `π_old`。限制轮次，尤其是对于大型网络。
- **没有奖励归一化。** 大的奖励尺度会蚕食裁剪范围。在计算优势之前归一化奖励（运行标准差）。
- **忘记优势归一化。** 每批次零均值/单位标准差归一化是标准做法。跳过它会破坏大多数基准测试上的 PPO。
- **学习率不衰减。** PPO 受益于线性 LR 衰减到零。恒定 LR 通常更差。
- **重要性比率数学错误。** 始终使用 `exp(log_new - log_old)` 以保证数值稳定性，不要用 `new / old`。
- **梯度符号错误。** 最大化替代目标 = *最小化* `-L^{CLIP}`。符号颠倒是最常见的 PPO bug。

## 使用

PPO 是 2026 年跨大量领域的默认 RL 算法：

| 用例 | PPO 变体 |
|----------|-------------|
| MuJoCo / 机器人控制 | 具有高斯策略、GAE(0.95) 的 PPO |
| Atari / 离散游戏 | 具有分类策略、128 步展开的 PPO |
| LLM 的 RLHF | 具有对参考模型 KL 惩罚、回答结束时 RM 奖励的 PPO |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理 LLM | GRPO（第 12 课）——没有评论家的 PPO 变体 |
| 纯偏好数据 | DPO——PPO+KL 的闭式坍缩，无需在线采样 |

PPO 的*损失形状*——裁剪替代 + 价值 + 熵——是 DPO、GRPO 和几乎每个 RLHF 管道的脚手架。

## 产出

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: 为给定环境生成 PPO 训练配置和诊断计划。
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

给定一个环境和训练预算，输出：

1. 展开大小。`N` 个环境 × `T` 步。
2. 更新调度。`K` 个轮次、小批量大小、LR 调度。
3. 替代参数。`ε`（裁剪）、`c_v`、`c_e`、优势归一化开启。
4. 优势。GAE(`λ`)，明确指定 `γ` 和 `λ`。
5. 诊断计划。KL、裁剪比例、解释方差阈值及警报。

拒绝 `K > 30` 或 `ε > 0.3`（不安全的信任区域）。拒绝任何没有优势归一化或 KL/裁剪监听的 PPO 运行。标记裁剪比例持续高于 0.4 认为出现漂移。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，设置 `ε=0.2, K=4`。在匹配的环境步数下，与 A2C（每个展开一个轮次）比较样本效率。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报与环境步数之间的关系，并追踪每次更新的平均 KL。在这个任务上，`K` 取何值时 KL 会爆炸？
3. **困难。** 将裁剪替代目标替换为自适应 KL 惩罚（如果 `KL > 2·目标` 则 `β` 翻倍，如果 `KL < 目标/2` 则减半）。比较最终回报、稳定性和无需裁剪的程度。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 重要性比率 | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`；与收集数据的策略的偏离程度。 |
| 裁剪替代目标 | "PPO 的主要技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有益侧超出裁剪后梯度变平。 |
| 信任区域 | "TRPO / PPO 意图" | 限制每次更新的 KL 以保证单调改进。 |
| KL 惩罚 | "软信任区域" | 替代 PPO：`L - β · KL(π_θ \|\| π_old)`。自适应 `β`。 |
| 裁剪比例 | "裁剪触发的频率" | 诊断指标——应为 0.1-0.3；超出范围表示调参不当。 |
| 多轮次训练 | "数据重用" | 每个展开 K 个轮次；用方差成本换取样本效率。 |
| 在策略-ish | "主要是在策略" | PPO 名义上是在策略的，但 K>1 轮次安全地使用了略微离策略的数据。 |
| PPO-KL | "另一个 PPO" | KL 惩罚变体；在 RLHF 中使用，其中对参考的 KL 已经是约束条件。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 论文本身。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 每个 PPO 超参数的消融研究。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；PPO 在 RLHF 中的配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 带 PyTorch 的清晰现代阐述。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 许多论文使用的参考单文件 PPO。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 语言模型上 PPO 的生产配方；与第 9 课（RLHF）一起阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37 个代码级优化"论文；哪些 PPO 技巧是承重的，哪些是传说。
