# 直接偏好优化家族

> Rafailov 等人（2023）证明了 RLHF 的最优解在偏好数据方面有一个闭合形式，因此你可以跳过显式的奖励模型，直接优化策略。这一洞见催生了一个家族——IPO、KTO、SimPO、ORPO、BPO——每个修复了 DPO 的一种失败模式。到 2026 年，直接对齐算法在前沿训练后运行中的出货量超过了 PPO。但第 2 课中的过度优化曲线仍然适用：直接对齐算法（DAA）不能逃脱古德哈特定律，它们只是移动了它咬人的位置。

**Type:** Learn
**Languages:** Python（stdlib，六变体偏好损失比较器）
**Prerequisites:** Phase 18 · 01 (InstructGPT)、Phase 18 · 02 (Reward hacking)、Phase 10 · 08 (DPO basics)
**Time:** ~75 分钟

## 学习目标

- 从带 KL 的 RLHF 最优解推导 DPO 闭合形式。
- 说出 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 中的哪种失败模式。
- 区分"隐式奖励差距"和"偏好强度"，并解释为什么 IPO 的恒等映射很重要。
- 解释为什么 Rafailov 等人（NeurIPS 2024）证明 DAA 尽管没有显式 RM 仍然会过度优化。

## 问题

RLHF 目标（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此奖励由最优策略与参考策略的比率隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将其代入 Bradley-Terry 偏好似然中，配分函数 `Z(x)` 被消去，因为它只依赖于 `x`。剩下的损失仅在策略参数中——不再需要奖励模型。这就是 DPO。

问题在于：推导假设最优解是可达的、偏好数据在分布内、参考策略是真实的模式锚点。这些假设没有一个能完全成立。家族的每个成员修复了一个不同的被违反的假设。

## 概念

### DPO（Rafailov 等人，2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出错的地方：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。一个微小的偏好可以产生任意大的差距。
- 损失驱动选定和拒绝的对数概率朝相反方向移动。只要拒绝下降得更快，它可以将选定绝对对数概率推低。这就是退化选定响应（Degraded Chosen Response）现象。
- 分布外偏好（稀有对 vs 稀有对）产生任意的隐式奖励。

### IPO（Azar 等人，2024）

恒等偏好优化（Identity Preference Optimization）将对数 sigmoid 替换为偏好概率上的恒等映射。损失变为一个有界目标上的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边际由 `1/(2 beta)` 限定。偏好强度和隐式奖励差距成比例。没有爆炸。

### KTO（Ethayarajh 等人，2024）

Kahneman-Tversky 优化完全去除了成对结构。给定单个标记输出和一个二元的"可取"或"不可取"信号，它映射到前景理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益和损失（损失厌恶）使用不同权重。好处：你可以使用非配对数据，这要丰富得多。

### SimPO（Meng 等人，2024）

简单偏好优化（Simple Preference Optimization）将训练信号与生成对齐。完全移除参考策略，并按长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

带有边际 `gamma` 以保持稳定。长度归一化移除了利用 DPO 长度偏差失败模式的激励（更长的 `y_w` 按构造产生更大的对数概率差距）。

### ORPO（Hong 等人，2024）

几率比偏好优化（Odds-Ratio Preference Optimization）在标准 SFT 负对数似然中添加了一个偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有参考策略——SFT 项就是正则化器。从基础模型到对齐模型单阶段训练。无需单独的 SFT 检查点。

### BPO（ICLR 2026 投稿，OpenReview id=b97EwMUWu7）

识别了退化选定响应（Degraded Chosen Responses）问题：DPO 保持了排名 `y_w > y_l`，但 `y_w` 的绝对对数概率可能下降。BPO 添加了一行修正，惩罚选定响应上的向下移动。报告在数学推理上比 DPO 在 Llama-3.1-8B-Instruct 上准确率提升 +10.1%。

### 普适结果：DAA 仍然过度优化

Rafailov 等人"Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms"（NeurIPS 2024）使用 DPO、IPO、SLiC 在多个数据集上跨 KL 预算训练了策略。黄金奖励 vs KL 曲线具有相同的 Gao 等人峰值-下降形状。隐式奖励在训练期间查询了分布外样本；KL 正则化不能稳定这一点。

DAA 不能逃脱古德哈特定律。它们改变了咬人的表面——从"奖励模型过度优化"变为"参考策略比率过度优化。"通用的修复——更好的数据、集成、提前停止——适用于两者。

### 如何选择（2026）

- 如果你有大量成对偏好数据：DPO 配合保守的 beta，如果长度偏差明显则用 SimPO。
- 如果你有非配对的二元反馈：KTO。
- 如果你想要从基础模型开始的单阶段流水线：ORPO。
- 如果你在 DPO 日志中看到退化选定对数概率：BPO。
- 如果偏好强度变化很大且 DPO 正在饱和：IPO。

每个实验室在所有五种上运行一个电池测试，并为每个任务选择胜出者。没有理由认为数学推理和安全的最优解是相同的。

```figure
dpo-margin
```

## 使用它

`code/main.py` 在一个玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实偏好强度按配对变化。每种损失在相同的 500 对样本上使用一个小型 softmax 策略进行优化。绘制每种方法的最终胜率、选定对数概率漂移和隐式奖励分布。

## 交付物

本课程产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计信息（配对 vs 非配对、变 vs 均匀偏好强度、长度分布）和目标（单阶段或 SFT 后偏好），推荐一个偏好损失并报告它防范的失败模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终选定对数概率下降。BPO 应该保持更高的选定绝对概率——验证这一点。
2. 修改偏好数据，使所有对具有相同的强度。六种方法中哪种最鲁棒？哪种退化？解释 IPO 在此处的优势。
3. 使被拒绝的响应平均长度是选定的 2 倍。在不改变其他任何东西的情况下，数值上展示 DPO 的长度利用和 SimPO 的修复。
4. Rafailov 等人（NeurIPS 2024）声称 DAA 会过度优化。重现一个单点版本：绘制选定减去的 KL 散度，并观察大 beta 下 DPO 的过度优化。
5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写出 BPO 对 DPO 添加的一行修正。对照 `code/main.py` 中的实现确认。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| DPO | "无奖励模型的 RLHF" | 从 RLHF 最优解闭合形式推导的损失；仅策略参数 |
| Implicit reward | "对数比率" | `beta * log(pi(y|x) / pi_ref(y|x))`——DPO 隐含的奖励 |
| IPO | "有界 DPO" | 将对数 sigmoid 替换为恒等映射；隐式奖励差距上限为 `1/(2 beta)` |
| KTO | "非配对 DPO" | 带有损失厌恶的单个标签上的前景理论效用 |
| SimPO | "无参考 DPO" | 长度归一化对数似然 + 边际；无参考策略 |
| ORPO | "单阶段 DPO" | NLL + 几率比偏好项；从基础模型一次训练 |
| BPO | "保选定 DPO" | DPO 加上对减少选定响应绝对对数概率的惩罚 |
| Degraded Chosen | "选定下降" | DPO 降低选定对数概率，只要拒绝下降得更快 |
| DAA | "直接对齐算法" | 任何跳过显式 RM 的偏好损失方法 |

## 延伸阅读

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036)——IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
