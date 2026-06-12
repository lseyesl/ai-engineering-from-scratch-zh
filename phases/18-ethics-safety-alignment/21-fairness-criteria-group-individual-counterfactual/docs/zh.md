# 公平性标准——群体、个体、反事实

> 三种框架构成了公平性文献的结构。群体公平（Group fairness）：人口均等（demographic parity）、均等几率（equalized odds）、条件使用准确率均等（conditional use accuracy equality）——受保护群体间的平均比率相等。个体公平（Individual fairness，Dwork 等人 2012）：相似的个体获得相似的决策；决策映射上的 Lipschitz 条件。反事实公平（Counterfactual fairness，Kusner 等人 2017）：如果当敏感属性被反事实地改变时决策不变，则对个体而言该决策是公平的。2024 年理论结果（NeurIPS 2024）：存在固有的 CF-准确率权衡；一个模型无关的方法可以将最优但不公平的预测器转化为 CF 预测器，且准确率损失有界。回溯反事实（arXiv:2401.13935，2024 年 1 月）：避免要求对合法受保护属性进行干预的新范式。哲学调和（ICLR Blogposts 2024）：使用因果图，满足某些群体公平度量蕴含反事实公平。

**Type:** Learn
**Languages:** Python（stdlib，三个标准比较）
**Prerequisites:** Phase 18 · 20（偏见）、Phase 02（经典 ML）
**Time:** ~60 分钟

## 学习目标

- 陈述三种群体公平性标准（人口均等、均等几率、条件使用准确率均等）和一个不可能性结果。
- 通过 Dwork 等人 2012 年的 Lipschitz 公式描述个体公平性。
- 描述反事实公平性及其对因果图的依赖。
- 解释回溯反事实以及它为什么绕过了对受保护属性进行干预的问题。

## 问题

第 20 课是关于测量偏见。第 21 课是关于定义测量应服务的公平性标准。三种框架给出了结构上不同的标准——一个模型可能是群体公平的而个体不公平的，反事实公平的而群体不公平的。选择标准是一个政策决策；没有标准是普遍最优的。

## 概念

### 群体公平

- **人口均等（Demographic parity）。** 对所有群体 P(Y=1 | A=a) = P(Y=1 | A=a')。相同的接受率。
- **均等几率（Equalized odds）。** 对所有群体 P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a')。跨群体相同的 TPR 和 FPR。
- **条件使用准确率均等（Conditional use accuracy equality）。** 对所有群体 P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a')。跨群体相同的预测价值。

不可能性（Chouldechova，Kleinberg-Mullainathan-Raghavan 2017）：在不平等的基率下，这三个标准无法同时满足。

### 个体公平

Dwork 等人 2012。如果存在某个 Lipschitz 常数 L，使得对于所有 x, x'，有 |f(x) - f(x')| <= L * d(x, x')，则决策映射 f 相对于任务特定的相似性度量 d 是个体公平的。相似的个体获得相似的决策。

需要定义 d。这是政策问题，而非统计问题。

### 反事实公平

Kusner 等人 2017。如果在总体的因果模型下，当 i 的敏感属性被反事实地改变时决策不变，则对个体 i 而言该决策是反事实公平的。

需要一个因果 DAG。DAG 是建模选择。反事实公平的合理性取决于 DAG 的合理性。

### CF 与准确率的权衡

NeurIPS 2024 理论：反事实公平与预测准确率之间存在固有的权衡。一个模型无关的方法可以将最优但不公平的预测器转化为 CF 预测器，代价是有限的准确率损失。准确率损失取决于最优但不公平预测器中敏感属性系数的大小。

### 回溯反事实

arXiv:2401.13935（2024 年 1 月）。传统的反事实要求对敏感属性进行干预——"如果这个人性别不同，决策会改变吗？"在法律上，这存在问题：在分类法中不能对受保护属性进行干预。

回溯反事实反转了方向：不是对属性进行干预，而是问该个体的哪些实际特征的组合会产生反事实结果。这绕过了法律上的反对意见。

### 哲学调和

ICLR Blogposts 2024。手头有因果图时，满足某些群体公平度量蕴含反事实公平。这三个框架不是正交的；它们是同一底层因果结构的不同侧面。

这并未解决不可能性定理（不平等的基率仍然阻止同时满足群体公平）。但它表明"群体"与"个体/反事实"之间的表面对立部分是由于未对因果模型进行显式处理而产生的。

### 在 Phase 18 中的位置

第 20 课是偏见测量。第 21 课是公平性定义。第 22 课是隐私（差分隐私）。第 23 课是水印。这些是与分配相关的课程，补充了与欺骗相关的第 7-11 课。

## 使用它

`code/main.py` 构建了一个带有敏感属性和不平等基率的玩具二分类数据集。在简单分类器上计算人口均等、均等几率和条件使用准确率均等。观察到三个指标不一致。应用以人口均等为目标的重新加权，并观察其对其他两个指标的代价。

## 交付物

本课程产出 `outputs/skill-fairness-criterion.md`。给定一个公平性声明或政策，识别声明的标准、模型在声明的不平等基率下能否满足其余标准、以及声明依赖的因果 DAG。

## 练习

1. 运行 `code/main.py`。报告默认数据上的三个群体指标。应用以人口均等为目标的重新加权并重新报告。
2. 使用非敏感���性上的 L2 实现 Dwork 等人 2012 年的个体公平性指标。报告有多少对在 Lipschitz 常数 L=1 下违反条件。
3. 阅读 Kusner 等人 2017 年。为简历评分构建一个简单的双特征因果 DAG，并识别其蕴含的反事实公平条件。
4. 2024 年的回溯反事实论文避免了对受保护属性的干预。描述一个在法律合规中这很重要的场景。
5. ICLR 2024 的调和论认为群体和反事实公平是同一结构的不同侧面。从 `code/main.py` 的三个标准中选择两个，并说明使它们等价的因果假设。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Demographic parity | "均等比率" | P(Y=1 | A=a) 跨群体相等 |
| Equalized odds | "均等 TPR/FPR" | 跨群体相等的真阳率和假阳率 |
| Conditional use accuracy | "均等 PPV/NPV" | 跨群体相等的预测价值 |
| Individual fairness | "Lipschitz 条件" | 相似个体获得相似决策 |
| Counterfactual fairness | "因果改变不变性" | 在反事实属性改变下决策不变 |
| Backtracking counterfactual | "通过实际数据解释" | 从结果向后推理的反事实，而非从属性向前推理 |
| Impossibility theorem | "三者冲突" | Chouldechova / KMR 2017：不平等的基率下群体标准互斥 |

## 延伸阅读

- [Dwork et al. — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913)——个体公平
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856)——反事实公平
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056)——不可能性
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935)——受保护属性干预的新范式
