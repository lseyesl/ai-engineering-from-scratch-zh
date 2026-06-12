# 支持向量机

> 在两个类别之间找到最宽的街道。这就是全部想法。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 1（课程 08 优化、14 范数与距离、18 凸优化）
**时间：** ~90 分钟

## 学习目标

- 使用合页损失和梯度下降在原始形式下实现线性 SVM
- 解释最大间隔原理，并从训练后的模型中识别支持向量
- 比较线性核、多项式核和 RBF 核，并解释核技巧如何避免显式的高维映射
- 评估 C 参数在间隔宽度和分类错误之间的权衡

## 问题

你有两个类别的数据点，需要画一条线（或超平面）将它们分开。可能有无限多条线都能做到。你应该选哪条？

选间隔最大的那条。间隔是决策边界与每侧最近数据点之间的距离。更宽的间隔意味着分类器更自信，对未见数据的泛化能力更强。

这个直觉引出了支持向量机，ML 中数学最优美的算法之一。SVM 在深度学习之前是主要的分类方法，并且仍然是小数据集、高维数据以及需要原则性强、理论理解透彻且具有理论保证的模型时的最佳选择。

SVM 直接连接到阶段 1：优化是凸的（课程 18），间隔用范数衡量（课程 14），核技巧利用点积来处理非线性边界，而无需在高维空间中计算。

## 概念

### 最大间隔分类器

给定线性可分的数据，标签 y_i 属于 {-1, +1}，特征向量 x_i，我们希望找到超平面 w^T x + b = 0 来分离类别。

点到超平面的距离为：distance = |w^T x_i + b| / ||w||

对于正确分类的点：y_i * (w^T x_i + b) > 0。间隔是超平面到两侧最近点距离的两倍。

优化问题：

```
最大化    2 / ||w||     (间隔宽度)
约束     y_i * (w^T x_i + b) >= 1  对所有 i
```

等价地（最小化 ||w||^2 更易优化）：

```
最小化    (1/2) ||w||^2
约束     y_i * (w^T x_i + b) >= 1  对所有 i
```

这是一个凸二次规划问题。它有唯一的全局解。恰好位于间隔边界上的数据点（y_i * (w^T x_i + b) = 1）就是支持向量。它们是唯一决定决策边界的点。移动或移除任何非支持向量点，边界不会改变。

### 软间隔：用 C 参数处理噪声

真实数据很少是完美可分的。有些点可能在边界的错误一侧，或在间隔内部。软间隔公式通过引入松弛变量允许违规。

```
最小化    (1/2) ||w||^2 + C * sum(xi_i)
约束     y_i * (w^T x_i + b) >= 1 - xi_i
         xi_i >= 0  对所有 i
```

松弛变量 xi_i 衡量点 i 违反间隔的程度。C 控制权衡：

| C 值 | 行为 |
|------|------|
| C 大 | 严重惩罚违规。窄间隔，更少误分类。容易过拟合 |
| C 小 | 允许更多违规。宽间隔，更多误分类。可能欠拟合 |

C 是正则化强度的倒数。大的 C = 更少的正则化。小的 C = 更多的正则化。

### 合页损失：SVM 的损失函数

软间隔 SVM 可以重写为无约束优化：

```
最小化    (1/2) ||w||^2 + C * sum(max(0, 1 - y_i * (w^T x_i + b)))
```

项 max(0, 1 - y_i * f(x_i)) 就是合页损失。当点被正确分类且在间隔外时为零；当点在间隔内或被误分类时为线性损失。

合页损失产生稀疏解（只有支持向量有非零贡献）。逻辑损失使用所有数据点。这使得 SVM 在预测时内存效率更高。

### 对偶形式和核技巧

SVM 问题的拉格朗日对偶形式只涉及数据点之间的点积 x_i . x_j。这是关键洞察。将每个点积替换为核函数 K(x_i, x_j)，SVM 就可以学习非线性边界，而无需显式计算变换。

```
线性核：     K(x, z) = x . z
多项式核：   K(x, z) = (x . z + c)^d
RBF 核：     K(x, z) = exp(-gamma * ||x - z||^2)
```

RBF 核将数据映射到无限维空间。输入空间中接近的点核值接近 1。相距很远的点核值接近 0。它可以学习任何光滑的决策边界。

## 构建它

代码包含在 `code/svm.py` 中。实现包括线性 SVM（合页损失 + 梯度下降）、核函数和间隔分析。

### 第 1 步：合页损失和梯度

```python
def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n
```

### 第 2 步：通过梯度下降的线性 SVM

```python
class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0
        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * (-y[i])

    def predict(self, X):
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]
```

## 使用它

用 scikit-learn：

```python
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
])
clf.fit(X_train, y_train)
print(f"准确率: {clf.score(X_test, y_test):.4f}")
print(f"支持向量: {clf['svm'].n_support_}")
```

重要提示：在训练 SVM 之前一定要对特征进行缩放。SVM 对特征幅度很敏感，因为间隔依赖于 ||w||，未缩放的特征会扭曲几何。

## 关键术语

| 术语 | 含义 |
|------|------|
| 支持向量 | 最接近决策边界的训练点。唯一决定超平面的点 |
| 间隔 | 决策边界到最近支持向量的距离。SVM 最大化这个距离 |
| 合页损失 | max(0, 1 - y*f(x))。正确分类且在间隔外时为零，否则为线性惩罚 |
| C 参数 | 间隔宽度和分类错误之间的权衡。大 C = 窄间隔，小 C = 宽间隔 |
| 软间隔 | 允许通过松弛变量违反间隔的 SVM 公式。处理不可分数据 |
| 核技巧 | 在高维特征空间中计算点积，而无需显式映射到该空间 |
| RBF 核 | K(x, z) = exp(-gamma * \|\|x-z\|\|^2)。映射到无限维。学习任意光滑边界 |
| 对偶形式 | SVM 问题的重新表述，只依赖于数据点之间的点积。使核技巧成为可能 |

## 延伸阅读

- [Vapnik: The Nature of Statistical Learning Theory (1995)](https://link.springer.com/book/10.1007/978-1-4757-3264-1)——SVM 和统计学习的基础文本
- [Cortes & Vapnik: Support-vector networks (1995)](https://link.springer.com/article/10.1007/BF00994018)——原始 SVM 论文
- [scikit-learn SVM 文档](https://scikit-learn.org/stable/modules/svm.html)——带有实现细节的实用指南
