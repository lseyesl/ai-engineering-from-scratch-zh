# 指令遵循作为对齐信号

> 后来对 RLHF 的每一次批评都是针对这一流水线的。在研究优化压力如何扭曲一个代理指标之前，你必须先看到这个代理指标。InstructGPT（Ouyang 等人，2022）定义了参考架构：在指令-响应对上进行监督微调、在成对偏好排名上训练奖励模型、以及针对奖励模型进行 PPO 优化并附带对 SFT 策略的 KL 惩罚。一个 1.3B 的 InstructGPT 比 175B 的 GPT-3 更受青睐。仅这一个结果就是为什么 2026 年每个前沿实验室仍然发布 RLHF 形式的训练后流水线的原因。

**Type:** Learn
**Languages:** Python（stdlib，简易三阶段流水线）
**Prerequisites:** Phase 10 · 06 (SFT)、Phase 10 · 07 (RLHF)、Phase 10 · 08 (DPO)
**Time:** ~45 分钟

## 学习目标

- 说出 InstructGPT 流水线的三个阶段以及每个阶段使用的损失函数。
- 解释为什么 1.3B 的指令微调模型在人类偏好评估中胜过原始的 175B GPT-3。
- 说明第三阶段中 KL 惩罚防止了什么，以及为什么移除它会崩溃到模式寻找行为（mode-seeking behaviour）。
- 描述对齐税（alignment tax）以及 Ouyang 等人用来缓解它的 PPO-ptx 方法。

## 问题

预训练语言模型能补全文本。但它们不能回答问题。问 GPT-3"写一个反转列表的 Python 函数"，你得到的往往是一个提示的继续，因为大部分训练分布是网络文本，继续输出网络文本。模型在做它的工作——但工作本身是错的。

每个严肃实验室用来解决这个问题的代理指标是人类偏好。两个输出交给评分员；评分员选择更好的那个；奖励模型学习评分员的偏好。然后一个 RL 循环将策略移向奖励模型评分高的输出。这就是 InstructGPT 完整论点的三句话总结。论文的其余部分是工程实现。

## 概念

### 阶段 1：监督微调（SFT）

收集提示-响应对，其中响应是善意的人类会写的内容。Ouyang 等人使用了来自标注者和 OpenAI API 的 13k 提示。在此数据上使用标准交叉熵损失对基础模型进行微调。

SFT 带来的：模型现在回答问题，而非继续它们。SFT 没有带来的：当多个答案都合理时，关于评分员偏好哪个答案的任何信号。

### 阶段 2：奖励模型（RM）

对每个提示，从 SFT 模型采样 K 个输出。标注者对它们排序。训练一个奖励模型，对任何提示-响应对进行评分，使得对于 `y_w` 优于 `y_l` 的配对：

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这就是 Bradley-Terry 成对偏好损失。RM 通常从 SFT 模型初始化，将 LM 头替换为一个标量头。

奖励模型很小：对于 175B 的 InstructGPT，6B 就足够了。它们也很脆弱——论文的第 5 节大部分关于在小规模下出现的奖励黑客行为。

### 阶段 3：带 KL 惩罚的 PPO

定义目标函数：

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

使用 PPO 最大化。KL 项防止 `pi` 偏离 SFT 策略太远。没有它，优化器会找到对抗样本——在 RM 下得分高的字符串，不是因为 RM 从未见过它们，而是因为人类实际上并不偏好它们。

KL 系数 `beta` 是 RLHF 中最重要的超参数。太低：奖励黑客。太高：相比 SFT 没有改进。

### 对齐税

RLHF 之后，模型被人类偏好，但在标准基准测试（SQuAD、HellaSwag、DROP）上出现退化。Ouyang 等人称之为对齐税（alignment tax），并用 PPO-ptx 修复：将预训练梯度混入 RL 目标中，使模型不会忘记它从未因其获得奖励的下游任务。

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx 成为标准。Anthropic、DeepMind 和 Meta 都使用某种变体。

### 结果

一个 1.3B 的 InstructGPT（SFT + RM + PPO-ptx）在约 70% 的情况下被标注者认为优于 175B 的基础 GPT-3。在生产流量的隐藏测试提示上差距更大。从这一数字可以读出两件事：

1. 对齐与能力是不同的轴。175B 模型有更强的能力；1.3B 模型有更好的对齐；标注者更喜欢对齐的那个。
2. 能力下限由基础模型决定。你不能通过 RLHF 让基础模型知道它从未见过的事实。

### 为什么这是 Phase 18 的参考点

后面课程中的每一个批评——奖励黑客（第 2 课）、DPO（第 3 课）、谄媚（第 4 课）、CAI（第 5 课）、休眠代理（第 7 课）、对齐伪装（第 9 课）——都是针对这一流水线的某些部分。奖励黑客攻击阶段 2。DPO 合并了阶段 2 和 3。CAI 取代了人类标注者。谄媚表明标注者是一个有偏差的信号。对齐伪装表明策略可以完全绕过阶段 3。如果脑中不先有这个流水线，你无法理解这些批评中的任何一个。

## 使用它

`code/main.py` 在玩具偏好数据上模拟三个阶段。基础"策略"是一个对动作 {A, B, C} 的有偏硬币。阶段 1 SFT 在 200 个提示上模仿标注者的行为。阶段 2 从 500 个成对排名中拟合一个 Bradley-Terry 奖励模型。阶段 3 运行一个简化的 PPO 更新，带有对 SFT 策略的 KL 惩罚。你可以观察奖励上升、KL 散度增长和策略漂移——你也可以关闭 KL 项，在 50 步内看到奖励黑客出现。

关注点：

- `beta = 0.1` vs `beta = 0.0` 时的奖励轨迹。
- 训练步骤中的 KL(pi || pi_SFT)。
- 与标注者偏好相比的最终动作分布。

## 交付物

本课程产出 `outputs/skill-instructgpt-explainer.md`。给定一个 RLHF 流水线描述或论文摘要，它识别出三个阶段中的哪一个是正在被修改的，每个阶段使用什么损失，以及是否存在 KL 惩罚或等效的正则化器。

## 练习

1. 运行 `code/main.py`。设置 `beta = 0.0` 并报告 200 步 PPO 后的动作分布。用一段话解释模式寻找行为。
2. 修改奖励模型，使其对动作 B 有 +0.5 的偏置（模拟奖励 bug）。用 `beta = 0.1` 运行 PPO。KL 惩罚是否阻止了策略利用该偏置？在多大的 `beta` 下，利用变得可见？
3. 阅读 Ouyang 等人（arXiv:2203.02155）图 1。通过运行 PPO 1、5、20、100 步并测量相对于 SFT 模型的偏好，重现标注者偏好曲线。
4. 论文第 4.3 节报告 1.3B InstructGPT 在约 70% 的情况下胜过 175B GPT-3。为什么在隐藏生产提示上的比例会高于标注者自己的提示？
5. 在相同的偏好数据上，用 DPO（Phase 10 · 08）替换 PPO 损失。比较最终的策略漂移（KL 到 SFT）和最终奖励。在相同的奖励水平下，哪种方法漂移更远？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| SFT | "指令微调" | 阶段 1：在提示-响应对上进行交叉熵微调 |
| Reward model | "RM" | 在 (提示, 响应) 上使用 Bradley-Terry 成对标签训练的标量回归器 |
| Bradley-Terry | "成对偏好损失" | -log sigmoid(r_w - r_l)；将成对排序简化为二分类 |
| KL penalty | "正则化器" | `beta * KL(pi || pi_SFT)`——使 RL 策略接近 SFT 锚点 |
| PPO-ptx | "带预训练混合的 PPO" | 在 PPO 目标中添加一部分预训练对数似然以抵消对齐税 |
| Alignment tax | "RLHF 退化" | RLHF 后在 RLHF 未针对的标准基准上的表现下降 |
| Labeler preference | "地面真相" | 人类排名的样本；RM 是此数据的统计代理，而非"人类价值观" |

## 延伸阅读

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155)——InstructGPT 论文，是所有后续 RLHF 流水线的基础
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325)——RLHF 用于摘要的前身
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741)——原始的基于偏好的 RL 公式
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862)——Anthropic 对 InstructGPT 流水线的 HH 扩展
