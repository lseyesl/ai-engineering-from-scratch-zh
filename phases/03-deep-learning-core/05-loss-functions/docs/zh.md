# 损失函数

> 损失函数告诉你的网络它有多糟糕。选择正确的损失函数，网络知道如何改进。选择错误的损失函数，它在解决错误的问题。

**类型：** 构建
**语言：** Python
**前置知识：** 课程 03.03（反向传播）
**时间：** ~60 分钟

## 学习目标

- 从头实现 MSE、MAE、BCE、交叉熵和 Huber 损失
- 将损失函数匹配到任务类型（回归对比二分类对比多分类）
- 通过解释交叉熵梯度为何永不饱和来诊断输出层学习失败
- 检测多分类问题中 one-hot 编码与概率输出的形状不匹配

## 问题

你的网络预测了 2.5。正确答案是 2.0。有多糟糕？"有点糟糕"是不够的。你的优化器需要评估差异的一个数字和一个方向。这个度量就是损失函数。

损失函数量化了预测与目标的距离。在训练期间，反向传播计算 d(损失)/d(每个参数)。如果你选择了一个与你的问题不匹配的损失函数，梯度可能在数学意义上"正确"却指向了错误的方向。例如，当你的目标是分类时使用 MSE——梯度对阈值附近的错误惩罚不足，网络学会了过于自信。

损失函数主要有两类：回归损失（连续输出）和分类损失（离散标签）。每类都有微妙之处，理解这些区别决定了网络是快速收敛还是毫无进展地振荡。

## 概念

### 回归损失

当你的网络预测一个连续值时：

| 损失 | 公式 | 优势 | 劣势 |
|------|---------|----------|---------|
| MSE | (y - ŷ)² | 梯度强，惩罚大错误 | 对异常值敏感 |
| MAE | |y - ŷ| | 对异常值鲁棒 | 梯度恒定（不随着靠近而缩小） |
| Huber | 靠近 0 时为二次，远离时为线性 | 两者的最佳部分 | 需要调优 δ |

MSE 对大错误的惩罚超过小错误，因为平方。这使得它对异常值敏感——一个离群点可以支配损失。MAE 对所有错误一视同仁，但当预测接近目标时，梯度不会缩小，使其难以达到精确收敛。

Huber 损失结合了它们的好处：在某个阈值 δ 内使用 MSE（平滑到达零），超出 δ 时使用 MAE（对异常值鲁棒）。

### 分类损失

当你的网络将事物分类时：

**二元交叉熵（BCE）** 适用于二分类：
```
BCE = -(y * log(ŷ) + (1 - y) * log(1 - ŷ))
```

**分类交叉熵** 适用于多分类：
```
CCE = -log(ŷ_c)  其中 c 是正确的类别
```

交叉熵的关键属性：它的梯度与 ŷ - y 成正比。当 ŷ 远离 y 时，梯度很大。当 ŷ 接近 y 时，梯度缩小到零。梯度永远不会饱和或消失——即使经过 100 层，输出层的梯度仍然很强。

### 为什么分类任务使用交叉熵而非 MSE

直觉：交叉熵梯度 = ŷ - y。MSE 梯度 = (ŷ - y) * ŷ * (1 - ŷ)（当与 sigmoid 配对时）。额外的 ŷ * (1 - ŷ) 项是对 sigmoid 导数的惩罚。当 sigmoid 饱和（接近 0 或 1）时，该项消失，梯度消失。用 MSE 训练的分类器在早期就停滞不前，因为梯度在输出层饱和。交叉熵没有这个乘法惩罚——它只在残差上流动。

### 多分类形状匹配

一个常见错误：logits 形状是 (n_classes,)，但目标是一个标签索引或 one-hot 编码向量。交叉熵需要形状匹配。典型设置是 logits → softmax → 交叉熵，其中 logits 与 one-hot 目标进行比较。

```
网络输出：    [0.1, 0.7, 0.2]  (logits)
Softmax：     [0.18, 0.62, 0.20] (概率)
目标：          [0,   1,   0]  (one-hot)
损失 = -log(0.62) = 0.48
```

## 构建它

### 第 1 步：回归损失

```python
def mse_loss(predicted, target):
    diff = predicted - target
    return diff * diff

def mse_gradient(predicted, target):
    return 2 * (predicted - target)

def mae_loss(predicted, target):
    return abs(predicted - target)

def mae_gradient(predicted, target):
    return 1.0 if predicted > target else -1.0

def huber_loss(predicted, target, delta=1.0):
    diff = predicted - target
    if abs(diff) <= delta:
        return 0.5 * diff * diff
    return delta * (abs(diff) - 0.5 * delta)
```

### 第 2 步：分类损失

```python
def binary_cross_entropy(predicted, target):
    eps = 1e-15
    predicted = max(eps, min(1 - eps, predicted))
    return -(target * math.log(predicted) + (1 - target) * math.log(1 - predicted))

def bce_gradient(predicted, target):
    return predicted - target

def cross_entropy_loss(logits, target_class):
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    total = sum(exps)
    probs = [e / total for e in exps]
    return -math.log(probs[target_class] + 1e-15)

def categorical_cross_entropy(probs, target_one_hot):
    return -sum(t * math.log(p + 1e-15) for t, p in zip(target_one_hot, probs))
```

### 第 3 步：可视化损失景观

创建一个空间网格并绘制每个损失函数的形状。比较 MSE 与 MAE：MSE 是抛物线，MAE 是 V 形。比较 MSE 与交叉熵：交叉熵随着预测向零移动呈指数增长。

### 第 4 步：训练比较实验

在玩具问题上训练两个相同的网络——一个用 MSE，一个用交叉熵。在分类任务上，交叉熵的收敛速度要快得多。绘制训练曲线以查看。

```figure
cross-entropy-loss
```

## 使用它

PyTorch 提供了所有这些作为 `nn` 模块：

```python
import torch.nn as nn

regression_loss = nn.MSELoss()
robust_regression = nn.HuberLoss(delta=1.0)
binary_loss = nn.BCEWithLogitsLoss()
multi_class_loss = nn.CrossEntropyLoss()  # 在内部结合 softmax

loss = multi_class_loss(logits, targets)
```

## 交付物

本课程产出：
- `outputs/prompt-loss-architect.md`——为任何任务和架构选择正确损失函数的可复用提示词

## 练习

1. 计算并打印交叉熵相对于未归一化 logits 的梯度——验证它永不饱和。
2. 在多分类网络的输出层尝试用 MSE 而非交叉熵。报告发生了什么。
3. 实现用于多标签分类的 BCE（每个样本可以有多个正标签）。
4. 实现 focal loss（focal loss 对已正确分类的样本进行降权）——用于类别不平衡的 BCE 的变体。
5. 在 Huber 损失中绘制预测与目标差异的梯度。验证它在 |diff| < δ 时为线性，此后为常数。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| 损失函数 | "目标函数，代价函数" | 为每个（预测，目标）对输出一个标量值的函数，量化网络的误差有多大 |
| MSE | "均方误差" | (y-ŷ)²——回归任务的标准损失，对异常值敏感 |
| Cross-Entropy | "对数损失" | -log(ŷ_c)——分类的标准损失，梯度永不饱和 |
| Huber | "平滑 MAE" | 在 δ 内使用 MSE，之外使用 MAE 的混合损失 |
| One-hot 编码 | "虚拟变量" | 将类别标签表示为向量，在正确类别位置为 1，其余为 0 |

## 延伸阅读

- Goodfellow, Bengio, Courville, "Deep Learning", 第 6.2 节 (https://www.deeplearningbook.org/)
- Hastie, Tibshirani, Friedman, "The Elements of Statistical Learning", 第 2 章
