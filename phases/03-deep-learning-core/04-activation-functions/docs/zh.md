# 激活函数

> 没有非线性，你的 100 层网络就是一个花哨的矩阵乘法。激活函数是让神经网络以曲线思考的门。

**类型：** 构建
**语言：** Python
**前置知识：** 课程 03.03（反向传播）
**时间：** ~75 分钟

## 学习目标

- 从头实现 sigmoid、tanh、ReLU、Leaky ReLU、GELU、Swish 和 softmax 及其导数
- 通过测量 10 层以上不同激活函数的激活幅度来诊断梯度消失问题
- 检测 ReLU 网络中的死亡神经元，并解释为什么 GELU 避免了这种失败模式
- 针对给定架构（transformer、CNN、RNN、输出层）选择正确的激活函数

## 问题

堆叠两个线性变换：y = W2(W1x + b1) + b2。展开它：y = W2W1x + W2b1 + b2。就是 y = Ax + c——一个单一的线性变换。无论你堆叠多少线性层，结果都坍缩为一个矩阵乘法。你的 100 层网络与单层具有相同的表示能力。

这不是理论上的好奇心。这意味着一个深度线性网络根本无法学习 XOR，无法分类螺旋数据集，无法识别面部。没有激活函数，深度就是一种幻觉。

激活函数打破了线性。它们通过一个非线性函数扭曲每层的输出，赋予网络弯曲决策边界、近似任意函数和真正学习的能力。但选择错误的激活函数，你的梯度会消失到零（深度网络中的 sigmoid）、爆炸到无穷大（没有仔细初始化的无界激活），或者你的神经元会永久死去（具有大负偏置的 ReLU）。激活函数的选择直接决定了你的网络能否学到任何东西。

## 概念

### 激活函数对比

| 函数 | 公式 | 范围 | 最大导数 | 问题 |
|--------|-------|-------|--------------|---------|
| Sigmoid | 1/(1+e^(-x)) | (0, 1) | 0.25 | 梯度消失 |
| Tanh | (e^x-e^(-x))/(e^x+e^(-x)) | (-1, 1) | 1.0 | 仍会消失 |
| ReLU | max(0, x) | [0, inf) | 1.0 | 死亡神经元 |
| Leaky ReLU | max(αx, x) | (-inf, inf) | 1.0 | 需要调优 α |
| GELU | x*Φ(x) | ~(-0.17, inf) | ~1.0 | 计算稍慢 |
| Swish | x*sigmoid(x) | ~(-0.28, inf) | ~1.2 | -- |

### ReLU：突破

2010 年由 Nair 和 Hinton 推广用于深度学习，它改变了一切。

```
relu(x) = max(0, x)
导数：1 如果 x > 0，0 如果 x <= 0
```

对于正输入没有梯度消失。梯度正好是 1，直接通过。这就是深层网络变得可训练的原因——ReLU 跨层保持梯度大小。

但有一个失败模式：死亡神经元问题。如果一个神经元的加权输入总是负的，它的输出总是零，它的梯度总是零，并且它永远不会更新。它永久死亡了。在实践中，ReLU 网络中 10-40% 的神经元可能在训练过程中死亡。

### GELU：现代默认

高斯误差线性单元。由 Hendrycks 和 Gimpel 在 2016 年提出。BERT、GPT 和大多数现代 transformer 中的默认激活函数。

```
gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

GELU 处处平滑，允许小的负值，并且有一个概率解释：它根据输入在正态分布下为正的概率来加权每个输入。这种平滑门控在 transformer 架构中优于 ReLU，因为它提供了更好的梯度流并完全避免了死亡神经元问题。

### 何时使用哪种激活函数

- Transformer / NLP → GELU
- CNN / 视觉 → ReLU 或 Swish
- RNN / LSTM → Tanh
- 简单 MLP → ReLU
- 二分类输出 → Sigmoid
- 多分类输出 → Softmax
- 回归 → 线性（无激活）

## 构建它

### 第 1 步：实现所有激活函数及其导数

```python
import math

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def gelu_derivative(x):
    phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return phi + x * pdf

def swish(x):
    return x * sigmoid(x)

def softmax(xs):
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]
```

### 第 2 步：梯度消失实验

通过 N 层使用 sigmoid 对比 ReLU 前向传播信号，并测量激活幅度如何变化。

```python
def vanishing_gradient_experiment(activation_fn, name, n_layers=10, n_inputs=5):
    values = [random.gauss(0, 1) for _ in range(n_inputs)]
    for layer in range(n_layers):
        weights = [random.gauss(0, 1) for _ in range(n_inputs)]
        z = sum(w * v for w, v in zip(weights, values))
        activated = activation_fn(z)
        magnitude = abs(activated)
        print(f"  Layer {layer+1:2d}: magnitude = {magnitude:.6f}")
        values = [activated] * n_inputs
```

### 第 3 步：死亡神经元检测器

创建一个 ReLU 网络，通过它传递随机输入，计算有多少神经元从不激活。

```python
def dead_neuron_detector(n_inputs=5, hidden_size=20, n_samples=1000):
    weights = [[random.gauss(0, 1) for _ in range(n_inputs)] for _ in range(hidden_size)]
    biases = [random.gauss(0, 1) for _ in range(hidden_size)]
    fire_counts = [0] * hidden_size
    for _ in range(n_samples):
        inputs = [random.gauss(0, 1) for _ in range(n_inputs)]
        for neuron_idx in range(hidden_size):
            z = sum(w * x for w, x in zip(weights[neuron_idx], inputs)) + biases[neuron_idx]
            if relu(z) > 0:
                fire_counts[neuron_idx] += 1
    dead = sum(1 for c in fire_counts if c == 0)
    print(f"  死亡（从未激活）：{dead}/{hidden_size} ({dead/hidden_size*100:.1f}%)")
```

### 第 4 步：训练比较——Sigmoid vs ReLU vs GELU

在相同数据集上用三种不同激活函数训练相同的两层网络。比较收敛速度。

## 使用它

PyTorch 提供了所有这些作为函数式和模块式两种形式：

```python
import torch.nn.functional as F

relu_out = F.relu(x)
gelu_out = F.gelu(x)
sigmoid_out = torch.sigmoid(x)
swish_out = F.silu(x)  # SiLU = Swish
probs = F.softmax(logits, dim=1)

model = nn.Sequential(
    nn.Linear(10, 64), nn.GELU(),
    nn.Linear(64, 32), nn.GELU(),
    nn.Linear(32, 5),
)
```

## 交付物

本课程产出：
- `outputs/prompt-activation-selector.md`——帮助你为任何架构选择正确的激活函数的可复用提示词

## 练习

1. 实现 Parametric ReLU（PReLU），其中负斜率 alpha 是可学习参数。
2. 用 50 层而非 10 层运行梯度消失实验。
3. 实现 ELU（指数线性单元）。
4. 构建一个"梯度健康监视器"，在训练期间打印警告。
5. 在 XOR 数据集而非圆形上运行训练比较。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| 激活函数 | "非线性部分" | 应用于每个神经元输出的函数，打破线性，使网络能够学习非线性映射 |
| 梯度消失 | "深层网络中梯度消失" | 当激活函数的导数小于 1 时，梯度在层间呈指数级缩小，使早期层无法训练 |
| 死亡神经元 | "停止学习的神经元" | 输入永久为负的 ReLU 神经元，产生零输出和零梯度 |
| ReLU | "将负数裁剪为零" | max(0, x)——通过保持梯度大小使深度学习变得实用的激活函数 |
| GELU | "transformer 激活函数" | 高斯误差线性单元，根据输入为正的概率加权输入的平滑激活函数 |
| Softmax | "将分数转化为概率" | 将 logit 向量归一化为概率分布，其中所有值在 (0,1) 且和为 1 |
| 饱和 | "sigmoid 的平坦部分" | 激活函数导数趋近于零的区域，阻止梯度流动 |

## 延伸阅读

- Nair & Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines" (2010)
- Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016)
- Ramachandran et al., "Searching for Activation Functions" (2017)
- Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010)
