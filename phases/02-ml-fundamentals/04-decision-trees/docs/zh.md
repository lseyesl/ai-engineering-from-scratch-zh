# 决策树与随机森林

> 决策树只是一个流程图。但是一片森林却是 ML 中最强大的工具之一。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 1（课程 09 信息论、06 概率）
**时间：** ~90 分钟

## 学习目标

- 实现基尼不纯度、熵和信息增益的计算，以找到最优的决策树分割
- 从零开始构建带预剪枝控制（最大深度、最小样本数）的决策树分类器
- 使用自助采样和特征随机化构建随机森林，并解释为什么它能降低方差
- 比较 MDI 特征重要性与置换重要性，并识别 MDI 何时有偏

## 问题

你有表格数据。行是样本，列是特征，还有一个你想预测的目标列。你可以用神经网络。但是，对于表格数据，基于树的模型（决策树、随机森林、梯度提升树）始终优于深度学习。结构化数据上的 Kaggle 竞赛由 XGBoost 和 LightGBM 主导，而不是 Transformer。

为什么？树能处理混合的特征类型（数值型和类别型）而无需预处理。它们无需特征工程即可处理非线性关系。它们是可解释的：你可以查看树并确切知道为什么做出某个预测。而随机森林——它对许多树进行平均——在中等规模的数据集上对过拟合有很强的抵抗力。

本课程使用递归分割从零开始构建决策树，然后在其基础上构建随机森林。你将实现分割标准（基尼不纯度、熵、信息增益）背后的数学原理，并理解为什么弱学习器的集成会成为一个强学习器。

## 概念

（概念部分翻译：决策树原理、分割标准、信息增益、停止条件、随机森林、特征重要性等）

## 构建它

代码包含在 `code/trees.py` 中。关键实现：

### 第 1 步：基尼不纯度和熵

```python
import math

def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum(
        (c / n) * math.log2(c / n) for c in counts.values() if c > 0
    )
```

### 第 2 步：信息增益和最佳分割

```python
def information_gain(parent_labels, left_labels, right_labels, criterion="gini"):
    measure = gini_impurity if criterion == "gini" else entropy
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = measure(parent_labels)
    child_impurity = (
        (n_left / n) * measure(left_labels) +
        (n_right / n) * measure(right_labels)
    )
    return parent_impurity - child_impurity
```

### 第 3 步：决策树类

递归分割、预测和特征重要性追踪。

### 第 4 步：随机森林类

自助采样、特征随机化和多数投票。

实现细节请参考完整代码文件。

## 使用它

用 scikit-learn，训练随机森林只需要三行：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"准确率: {rf.score(X_test, y_test):.4f}")
print(f"特征重要性: {rf.feature_importances_}")
```

在实践中，梯度提升树（XGBoost、LightGBM、CatBoost）通常比随机森林更强，因为它们顺序构建树，每棵树纠正前一棵树的错误。但是随机森林更难配置错误，几乎不需要超参数调优。

## 交付物

本课程产出 `outputs/prompt-tree-interpreter.md`——一个为业务利益相关者解释决策树分割的提示词。

## 关键术语

| 术语 | 含义 |
|------|------|
| 决策树 | 通过学习一系列 if/else 分割将特征空间划分为矩形区域的模型 |
| 基尼不纯度 | 在一个节点上随机样本被误分类的概率。0 = 纯，0.5 = 二分类的最大不纯度 |
| 熵 | 节点中的信息量。0 = 纯，1.0 = 二分类的最大不确定性 |
| 信息增益 | 分割后不纯度的降低量。选择分割的贪心标准 |
| 预剪枝 | 通过设置最大深度、最小样本数或最小增益阈值来提前停止树的生长 |
| 后剪枝 | 先让树完全生长，然后移除不能提高验证性能的子树 |
| Bagging | 自助聚合。在每个不同的有放回随机样本上训练模型 |
| 随机森林 | 决策树的集成，每棵树在自助样本上训练，每个分割使用随机特征子集 |
| 特征重要性 (MDI) | 每个特征贡献的总不纯度减少，跨所有树和节点求和 |
| 置换重要性 | 当某特征的值被随机打乱时准确率的下降。对于有噪声的特征比 MDI 更可靠 |

## 延伸阅读

- [Breiman: Random Forests (2001)](https://link.springer.com/article/10.1023/A:1010933404324)——原始随机森林论文
- [Grinsztajn et al.: Why do tree-based models still outperform deep learning on tabular data? (2022)](https://arxiv.org/abs/2207.08815)——树与神经网络在表格任务上的严格比较
- [scikit-learn Decision Trees 文档](https://scikit-learn.org/stable/modules/tree.html)——带有可视化工具的实用指南
- [XGBoost: A Scalable Tree Boosting System (2016)](https://arxiv.org/abs/1603.02754)——主导 Kaggle 的梯度提升论文
