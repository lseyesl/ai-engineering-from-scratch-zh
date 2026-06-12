# RL 用于游戏——AlphaZero、MuZero 与 LLM 推理时代

> 1992 年：TD-Gammon 用纯 TD 在双陆棋上击败人类冠军。2016 年：AlphaGo 击败李世石。2017 年：AlphaZero 从零开始称霸国际象棋、将棋和围棋。2024 年：DeepSeek-R1 证明了用 GRPO 替代 PPO 的相同配方在推理上有效。游戏是驱动本阶段每项突破的基准。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 9 · 05（DQN），阶段 9 · 08（PPO），阶段 9 · 09（RLHF），阶段 9 · 10（MARL）
**时间：** ~120 分钟

## 问题

游戏拥有一切 RL 想要的。干净的奖励（赢/输）。无限的片段（自我对弈重置）。完美的仿真（游戏*就是*模拟器）。离散或小型连续动作空间。强制对抗鲁棒性的多智能体结构。

而且游戏是每项重大 RL 突破的测试平台。TD-Gammon（双陆棋，1992 年）。Atari-DQN（2013 年）。AlphaGo（2016 年）。AlphaZero（2017 年）。OpenAI Five（Dota 2，2019 年）。AlphaStar（星际争霸 II，2019 年）。MuZero（学习到的模型，2019 年）。AlphaTensor（矩阵乘法，2022 年）。AlphaDev（排序算法，2023 年）。DeepSeek-R1（数学推理，2025 年）——游戏 RL 技术在文本上有效的最新证明。

本毕业设计通过一个统一的视角——**自我对弈 + 搜索 + 策略改进**——审视三种里程碑式架构：AlphaZero、MuZero 和 GRPO。每一个都是前一个的泛化；GRPO 特别是 AlphaZero 的配方应用于 LLM 推理，以 token 作为动作，以数学验证作为获胜信号。

## 概念

**统一的循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # 与自己对弈
    policy_target = search.improved_policy(trajectory) # 搜索改进原始策略
    policy_net.update(policy_target, value_target)     # 在搜索输出上监督学习
```

**AlphaZero（2017 年）。** Silver 等人。给定一个已知规则的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：一个塔 `f_θ(s) → (p, v)`。`p` 是合法移动上的先验分布。`v` 是期望的游戏结果。
- 蒙特卡洛树搜索（MCTS）：在每一步，展开可能延续的树。使用 `(p, v)` 作为先验 + 引导。通过 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自我对弈：智能体与自身对弈游戏。在步 `t`，MCTS 的访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人类知识。零手工设计的启发式。一个单一的配方，在各自数千万次自我对弈游戏后，掌握了国际象棋、将棋和围棋。

**MuZero（2019 年）。** Schrittwieser 等人。移除了规则必须已知的要求。

- 不依赖固定环境，而是学习一个*潜在动力学模型* `(h, g, f)`：
  - `h(s)`：将观察编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学习到的潜在空间*中运行。相同的搜索，相同的训练循环。
- 适用于围棋、国际象棋、将棋*和* Atari——一个算法，无需规则知识。

**随机 MuZero（2022 年）。** 添加随机动力学和机会节点；扩展到双陆棋类游戏。

**Muesli、Gumbel MuZero（2022–2024 年）。** 在样本效率和确定性搜索方面的改进。

**GRPO（2024–2025 年）。** DeepSeek-R1 配方。相同 AlphaZero 形状的循环，应用于语言模型推理：

- "游戏"：回答一个数学 / 编码 / 推理问题。"赢" = 验证器（测试用例通过、数值答案匹配）返回 1。
- 策略：LLM。动作：token。状态：提示 + 当前已生成的回复。
- 无评论家（PPO 风格的 V_φ）。取而代之，对每个提示，从策略中采样 `G` 个补全。计算每个的奖励。使用**组相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 对参考策略的 KL 惩罚以防止漂移（像 RLHF 一样）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无需奖励模型、无需评论家、无需 MCTS。组相对基线替代了三者。在推理基准上以极小计算量匹配或超过 PPO-RLHF 质量。

**完整的 R1 配方。** DeepSeek-R1（DeepSeek 2025）在一篇论文中包含两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基座模型开始。无 SFT。直接应用 GRPO，使用两个奖励组件：*准确率奖励*（基于规则的——最终答案是否被解析为正确的数字 / 代码是否通过单元测试）和*格式奖励*（补全是否将其思维链包裹在 `<think>…</think>` 标签中）。经过数千步，平均回复长度从约 100 增长到约 10,000 个 token，数学基准分数攀升到接近 o1-preview 的水平。模型从零开始学习推理。缺点是：其思维链往往难以阅读、混合语言、缺乏风格上的润色。
- **R1。** 通过四阶段管道修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集几千条带有干净格式的长 CoT 演示。在其上监督微调基座模型。这给出了一个可读的起点。
  2. **面向推理的 GRPO。** 应用 GRPO，使用准确率 + 格式奖励，外加一个*语言一致性*奖励以防止语码转换。
  3. **拒绝采样 + 第二轮 SFT。** 从 RL 检查点采样约 600K 推理轨迹，只保留那些具有正确最终答案和可读 CoT 的，并结合约 200K 非推理 SFT 示例（写作、问答、自我认知）。再次微调基座模型。
  4. **全频谱 GRPO。** 再进行一轮 RL，涵盖推理（基于规则的奖励）和通用对齐（基于偏好的有用性/无害性奖励）。

结果在 AIME 和 MATH-500 上以开放权重匹配 o1，并且足够小以便蒸馏。同一篇论文还通过在 R1 推理轨迹上进行 SFT，发布了六个蒸馏稠密模型（Qwen-1.5B 到 Llama-70B）——学生无需 RL。强 RL 教师的蒸馏在学生的规模上始终胜过从头开始的 RL。

**为什么推理用 GRPO 而不是 PPO。** DeepSeekMath 论文（2024 年 2 月）中的三个原因：（1）无需训练价值网络，内存减半；（2）组基线自然处理推理任务产生的稀疏轨迹末端奖励；（3）每个提示的归一化使优势在不同难度的问题之间具有可比性，而 PPO 的单个评论家无法做到。

**无搜索 vs 基于搜索。** 游戏已经分化：

- *完美信息、长视野游戏*（围棋、国际象棋）：仍然基于搜索。AlphaZero / MuZero 占据主导。
- *LLM 推理*：生产中尚无 MCTS；使用完整展开的 GRPO，推理时使用 Best-of-N。过程奖励模型（PRMs）暗示逐步搜索可能会被重新加入。

## 动手实现

`code/main.py` 中的代码实现了**微型 GRPO**——一个带多组采样的赌博机。算法与 LLM 上的相同；只是策略和环境更简单。它教授了*损失*和*组相对优势*，这是 2025 年的创新。

### 步骤 1：微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实 GRPO 中，验证器运行单元测试或检查数学等式。

### 步骤 2：策略：每个提示的 K 个答案 token 上的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于以提示为条件的 LLM 最终层输出。

### 步骤 3：组采样和组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL 惩罚：将 theta 拉向参考
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 2024 年 DeepSeek 的技巧。无需评论家。"基线"是组均值，归一化使用组标准差。

### 步骤 4：与 REINFORCE 基线比较（无价值函数）

相同的设置，相同的计算，朴素的 REINFORCE。GRPO 收敛更快、更稳定。

### 步骤 5：观察熵和 KL

与 RLHF 相同的诊断指标：相对参考的平均 KL、策略熵、奖励随时间变化。这些稳定后，训练就完成了。

## 陷阱

- **通过游戏验证器的奖励黑客。** GRPO 继承了 RLHF 的风险：如果验证器错误或可被利用，LLM 会找到利用方式。鲁棒的验证器（多个测试用例、形式化证明）很重要。
- **组大小太小。** 组基线的方差大致为 `1/√G`。低于 `G = 4`，优势信号有噪声；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的 LLM 补全具有不同的对数概率。通过 token 数量归一化，或使用序列级别的对数概率，或截断到最大长度。
- **纯自我对弈循环。** AlphaZero 风格的训练可能在一般和博弈上陷入支配循环。通过多样化对手池缓解（联赛玩法，第 10 课）。
- **搜索-策略不匹配。** AlphaZero 训练策略来模仿搜索输出。如果策略网络太小，无法表示搜索的分布，训练就会停滞。
- **计算量下限。** MuZero / AlphaZero 需要大量计算。单次消融通常需要数百 GPU 小时。存在用于学习的小型演示（例如 Connect Four 上的 AlphaZero）。
- **验证器覆盖率。** 对有 bug 的方案通过的单元测试会强化那个 bug。设计能捕获边界情况的验证器。

## 使用

按领域划分的 2026 年游戏 RL 格局：

| 领域 | 主导方法 |
|--------|-----------------|
| 双人零和棋盘游戏（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完美信息卡牌游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多玩家策略（Dota、星际争霸） | PPO + 自我对弈 + 联赛（OpenAI Five、AlphaStar） |
| LLM 数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开源复现） |
| LLM 对齐 | DPO / RLHF-PPO（非 GRPO；验证器是偏好而非可验证） |
| 机器人 | PPO + DR（非游戏 RL，但使用相同的策略梯度工具） |
| 组合优化问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

其*配方*——自我对弈、搜索增强的改进、策略蒸馏——横跨文本、像素和物理控制。GRPO 是最年轻的实例；还有更多即将到来。

## 产出

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: 为给定领域设计游戏 RL 或推理 RL 训练管道（AlphaZero / MuZero / GRPO）。
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

给定一个目标（完美信息游戏 / 不完美信息 / Atari / LLM 推理 / 组合优化），输出：

1. 环境匹配。规则已知？马尔可夫？随机？多智能体？决定 AlphaZero vs MuZero vs GRPO。
2. 搜索策略。MCTS（带学习先验的 PUCT）、Gumbel 采样、Best-of-N 或无。
3. 自我对弈计划。对称自我对弈 / 联赛 / 离线数据 / 验证器生成。
4. 目标信号。游戏结果 / 验证器奖励 / 偏好 / 学习到的模型。包含鲁棒性计划。
5. 诊断指标。对基线胜率、ELO 曲线、验证器通过率、相对参考的 KL。

拒绝在不完美信息游戏上使用 AlphaZero（引导至 CFR）。拒绝在没有可信验证器的情况下使用 GRPO。拒绝任何没有固定基线对手集的游戏 RL 管道（否则自我对弈 ELO 未经校准）。
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO 赌博机。在 2 个提示 × 4 个答案 token 上训练。在 `G=8` 的情况下在 < 1,000 次更新内收敛。
2. **中等。** 接入 PPO（裁剪版）和原始 REINFORCE。在相同的赌博机上与 GRPO 比较样本效率和奖励方差。
3. **困难。** 扩展到长度为 2 的"推理链"：智能体发射两个 token，验证器奖励这对组合。测量 GRPO 如何处理跨两步序列的信用分配。（提示：每*完整序列*计算组优势，传播到两个 token 位置。）

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| MCTS | "带学习网络的树搜索" | 蒙特卡洛树搜索；使用学习的 `(p, v)` 先验的 UCB1/PUCT 选择。 |
| AlphaZero | "自我对弈 + MCTS" | 策略-价值网络训练以匹配 MCTS 访问和游戏结果。 |
| MuZero | "学习模型的 AlphaZero" | 相同循环，但在潜在空间中通过学习的动力学模型进行。 |
| GRPO | "无评论家 PPO" | 组相对策略优化；带组均值基线 + KL 的 REINFORCE。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)`——平衡价值估计与先验。 |
| 自我对弈 | "智能体 vs 过去的自己" | 零和博弈的标准方法；对称训练信号。 |
| 联赛玩法 | "基于种群的自我对弈" | 过去 + 当前 + 利用者被采样为对手。 |
| 验证器奖励 | "可验证的 RL" | 奖励来自确定性检查器（测试通过、答案匹配）。 |
| 过程奖励 | "PRM" | 对每个推理步骤评分，而不仅仅是最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 引入 GRPO 和组相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整的四阶段 R1 配方加上 R1-Zero 的消融实验。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 大规模 CFR + 深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开启一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 使用自定义奖励函数应用 GRPO 的生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 多规模 R1 配方的开源复现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — 关于自我对弈、搜索和"设计奖励"的教科书框架，R1 在 LLM 规模上实现了这一点。
