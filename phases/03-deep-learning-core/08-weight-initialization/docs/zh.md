# 权重初始化

> 两个相同的网络，相同的优化器，相同的数据。一个学习，一个停滞。区别仅在于它们开始时权重有多大。

**类型：** 构建
**语言：** Python
**前置知识：** 课程 03.03（反向传播）、课程 03.04（激活函数）
**时间：** ~60 分钟

## 学习目标

- 针对不同激活函数（sigmoid、tanh、ReLU、GELU）推导并实现 Xavier 和 Kaiming 初始化
- 测量初始化如何影响 10 层深度网络中 10k 个训练步骤后的梯度幅度
- 在 20 层 sigmoid 网络中通过使用 Xavier 初始化将使梯度提升 2^20 倍来解决梯度消失问题
- 检测大权重初始化在方差大于 2/n 时导致梯度爆炸的问题

## 问题

你初始化了你的网络。所有权重在 [-0.01, 0.01] 之间的均匀随机分布。你运行了前向传播。输出如此之小，以至于它们被 sigmoid 压扁到接近零。反向传播梯度因此也是零。你的网络从不动弹。

然后你尝试 [-10, 10] 之间的均匀随机分布。现在 sigmoid 被推向饱和——输出接近 1 或 0，梯度又一次消失。要么太大，要么太小。权重初始化决定了你的网络是否一开始就能学习一个单一的梯度步骤。它通常被当作实现细节，但它是使深度网络可训练的关键因素。

问题是：对于与每层传入连接数量成比例的标准差，你需要一个精确的值。太大得到饱和和爆炸的梯度。太小得到消失的激活和梯度。正确值来自数学分析：推导在若干层后保持信号方差的权重方差。

名称"Xavier"初始化是由 Glorot 和 Bengio 在 2010 年首次系统地研究这个问题的论文中引入的。"Kaiming"初始化由 He 等人在 2015 年推导，专门用于 ReLU 网络。在你的初始化代码中交换一个 sqrt(2) 因子会触发数十层的可训练和停滞之间的差异。

## 概念

### 激活方差分析

对于一个具有 n_in 个输入的神经元，其加权和为 z = Σ(w_i * x_i)。假设输入和权重均值为零且独立，则 z 的方差为：

```
Var(z) = n_in * Var(w) * Var(x)
```

为了保持层间方差，我们需要 Var(z) = Var(x)，这意味着：

```
Var(w) = 1 / n_in
```

所以权重应从均值为 0、方差为 1/n_in 的分布中采样。这就是 Xavier 初始化。

### Xavier（Glorot）初始化

适用于 sigmoid 和 tanh 激活函数：

```
W ~ Uniform(-limit, limit)  其中 limit = sqrt(6 / (n_in + n_out))
W ~ Normal(0, sqrt(2 / (n_in + n_out)))
```

保持前向和反向传播的方差。在 sigmoid 深度网络中至关重要——它使梯度通过 2^N（层数）而非消失。

### Kaiming（He）初始化

专门为 ReLU 激活函数推导。ReLU 将输出方差减半（一半的神经元为 0），所以需要补偿：

```
W ~ Normal(0, sqrt(2 / n_in))
W ~ Uniform(-limit, limit)  其中 limit = sqrt(6 / n_in)
```

额外的 sqrt(2) 因子抵消了 ReLU 的方差减半效应。使用 Kaiming 初始化的 20 层 ReLU 网络保持稳定的激活幅度。使用 Xavier 初始化的相同网络随着每一层逐渐衰减到零。

### 初始化失败模式

| 初始化 | 激活 | 结果 |
|------------|-----------|--------|
| 太小（std = 0.01） | 任意 | 激活在数层后衰减到零，梯度消失 |
| 太大（std = 1.0） | tanh/sigmoid | 所有输出饱和到 ±1，梯度为零 |
| Xavier | Sigmoid/Tanh | 方差保持，稳定训练 |
| Kaiming | ReLU | 方差保持，稳定训练 |
| Kaiming（对 GELU 有 sqrt(2)） | GELU | 略微过多，轻微的高估，但通常仍然有效 |

### 为什么偏置初始化为零

偏置初始化为零，因为它们在训练期间具有大的、持续的梯度。从零开始初始化偏置意味着神经元最初仅依赖于它们的权重。这提供了明确的对称性破坏——不同的权重将不同的神经元驱动到不同的激活。

在实践中："当前神经元饱和了吗？"偏置从 0 开始。如果神经元在初始激活时没有足够活跃，你可以调整偏置。对于使用 ReLU 的深度网络，将最终偏置初始化为一个小正值（如 0.1）可以防止网络早期死亡。

## 构建它

### 第 1 步：初始化函数

```python
import math
import random

def xavier_uniform(n_in, n_out):
    limit = math.sqrt(6.0 / (n_in + n_out))
    return [random.uniform(-limit, limit) for _ in range(n_in * n_out)]

def xavier_normal(n_in, n_out):
    std = math.sqrt(2.0 / (n_in + n_out))
    return [random.gauss(0, std) for _ in range(n_in * n_out)]

def kaiming_uniform(n_in, n_out):
    limit = math.sqrt(6.0 / n_in)
    return [random.uniform(-limit, limit) for _ in range(n_in * n_out)]

def kaiming_normal(n_in, n_out):
    std = math.sqrt(2.0 / n_in)
    return [random.gauss(0, std) for _ in range(n_in * n_out)]
```

### 第 2 步：方差传播实验

构建一个 10 层网络，每层有 100 个神经元。分别使用小（0.01）、Xavier 和大（1.0）初始化。打印每层输出的均值和方差。

```python
def variance_experiment(initializer, name, n_layers=10, n_neurons=100):
    x = [random.gauss(0, 1) for _ in range(n_neurons)]
    outputs = [x]
    for layer in range(n_layers):
        weights = initializer(n_neurons, n_neurons)
        biases = [0.0] * n_neurons
        z = [sum(w * v for w, v in zip(weights[i*n_neurons:(i+1)*n_neurons], outputs[-1])) + biases[i] for i in range(n_neurons)]
        a = [sigmoid(v) for v in z]
        outputs.append(a)
        mean = sum(a) / len(a)
```figure
weight-init-variance
```

        var = sum((v - mean) ** 2 for v in a) / len(a)
        print(f"  Layer {layer+1}: mean={mean:.4f}, var={var:.4f}")
```

### 第 3 步：训练实验

在圆形分类任务上分别使用 Kaiming 初始化和零初始化训练相同的网络。分别使用 ReLU 和 tanh 训练。

### 第 4 步：梯度尺度比较

训练后打印每层的梯度幅度。比较使用 Xavier 初始化与均匀随机初始化的网络。

## 使用它

PyTorch 中的权重初始化：

```python
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
        nn.init.zeros_(m.bias)

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
model.apply(init_weights)
```

PyTorch 的默认初始化已经是 Kaiming 均匀分布——与 ReLU 网络一致。

## 交付物

本课程产出：
- `outputs/skill-initialization-guide.md`——为任何架构选择、调试和实现权重初始化的技能

## 练习

1. 将 Xavier 初始化的网络与零初始化网络的梯度幅度进行比较。哪个更快收敛？
2. 使用 Kaiming 初始化训练一个具有 GELU 激活函数的 20 层网络。测量中间层激活的方差。
3. 修改你的偏置初始化：将偏置设为 0.1（正值）而不是 0.0。ReLU 网络的行为有何变化？
4. 推导 GELU 的理论 Kaiming 方差缩放因子。（提示：GELU 均值不为零）
5. 重新运行方差传播实验，但使用没有活跃层归一化的 50 层。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| 权重初始化 | "参数的起始值" | 梯度下降开始之前设置网络参数的初始值的过程 |
| Xavier（Glorot）初始化 | "sigmoid 的标准初始化" | 从方差为 1/n_in 的分布中采样权重，保持 sigmoid 网络中的方差 |
| Kaiming（He）初始化 | "ReLU 的 Xavier" | 方差为 2/n_in 以补偿 ReLU 将一半神经元归零效果，保持 ReLU 网络中的方差 |
| 梯度消失 | "信号消失" | 随着梯度通过饱和或不适当的初始化传播，其幅度在层间减小到零 |
| 梯度爆炸 | "信号发散" | 随着梯度传播，其幅度在层间不受控制地增长 |
| 方差传播 | "信号如何在网络中流动" | 分析激活和梯度方差如何随层数和初始化方案变化 |

## 延伸阅读

- Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (AISTATS 2010)
- He et al., "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" (ICCV 2015)
