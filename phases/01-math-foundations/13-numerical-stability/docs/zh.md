# 数值稳定性

> 浮点数是一个有漏洞的抽象。它会在训练过程中坑你，而你毫无察觉。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 1，课程 01-04
**时间：** ~120 分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的上溢、下溢和灾难性抵消
- 使用中心有限差分法验证解析梯度与数值梯度的一致性
- 解释为何 bfloat16 比 float16 更适合训练，以及损失缩放如何防止梯度下溢

## 问题

你的模型训练了三小时，然后 loss 变成了 NaN。你加了一行 print 语句。第 9000 步时 logits 还是正常的。第 9001 步它们变成了 `inf`。到第 9002 步，所有梯度都是 `nan`，训练结束了。

或者：你的模型训练完成了，但准确率比论文声称的低 2%。你检查了一切。架构一致。超参数一致。数据一致。问题在于论文用的是 float32，而你用了 float16 但没有做正确的缩放。32 位累积的舍入误差悄悄地吞噬了你的精度。

或者：你从头实现了交叉熵损失。它在小 logits 时工作正常。当 logits 超过 100 时，它返回 `inf`。softmax 溢出了，因为 `exp(100)` 超过了 float32 能表示的范围。每个 ML 框架都用两行代码的技巧处理这个问题。但你不知道这个技巧的存在。

数值稳定性不是一个理论问题。它决定了训练是成功还是悄然失败。你最终要调试的每一个严重的 ML 错误，归根结底都是浮点数问题。

## 概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准将实数存储为浮点值。一个浮点数有三个部分：符号位、指数和尾数（有效数）。

```
Float32 布局（共 32 位）：
[1 符号位] [8 指数] [23 尾数]

值 = (-1)^符号 * 2^(指数 - 127) * 1.尾数
```

尾数决定精度（多少位有效数字）。指数决定范围（数字可以多大或多小）。

```
格式      位数   指数   尾数    十进制位数   范围（近似）
float64   64     11     52      ~15-16       +/- 1.8e308
float32   32     8      23      ~7-8         +/- 3.4e38
float16   16     5      10      ~3-4         +/- 65,504
bfloat16  16     8      7       ~2-3         +/- 3.4e38
```

float32 提供大约 7 位十进制精度。这意味着它可以区分 1.0000001 和 1.0000002，但无法区分 1.00000001 和 1.00000002。7 位之后，一切就都是舍入噪声了。

float16 提供大约 3 位精度。它能表示的最大数是 65,504。这对于 ML 来说小得令人不安，因为 logits、梯度和激活值通常会超过这个值。

bfloat16 是 Google 针对 float16 范围问题的解决方案。它拥有与 float32 相同的 8 位指数（相同的范围，最大 3.4e38），但只有 7 位尾数（精度低于 float16）。对于训练神经网络来说，范围比精度更重要，因此 bfloat16 通常更胜一筹。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法在二进制浮点数中精确表示。以 2 为底，它是一个无限循环小数：

```
0.1 的二进制 = 0.0001100110011001100110011...（无限循环）
```

Float32 将其截断为 23 位尾数。存储的值大约是 0.100000001490116。类似地，0.2 存储为大约 0.200000002980232。它们的和是 0.300000004470348，而不是 0.3。

```
在 Python 中：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 很重要，因为：

1. `if loss < threshold` 这样的 loss 比较可能会给出错误答案
2. 累积许多小值（数千步的梯度更新）会偏离真实总和
3. 如果你用 `==` 比较浮点数，校验和和可重现性测试会失败

解决方法：永远不要用 `==` 比较浮点数。改用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当你减去两个几乎相等的浮点数时，有效数字会抵消，剩下的只有被提升到前导位置的舍入噪声。

```
a = 1.0000001    （在 float32 中存储为 1.00000011920929）
b = 1.0000000    （在 float32 中存储为 1.00000000000000）

真实差值：  0.0000001
计算结果：  0.00000011920929

相对误差：19.2%
```

一次减法就导致 19% 的相对误差。在 ML 中，这发生在你：

- 计算具有较大均值的数据的方差时：`E[x^2] - E[x]^2`（当 E[x] 很大时）
- 减去几乎相等的对数概率时
- 使用过小的 epsilon 计算有限差分梯度时

解决方法：重新排列公式，避免减去大的、接近相等的数。对于方差，使用 Welford 算法或先对数据进行中心化。对于对数概率，始终在 log 空间中计算。

### 上溢和下溢

上溢发生在结果太大而无法表示时。下溢发生在结果太小（比最小可表示正数更接近零）时。

```
Float32 边界：
  最大值：             3.4028235e+38
  最小正数（正常）：   1.175e-38
  最小正数（非规范）： 1.401e-45
  上溢：              任何 > 3.4e38 的值变成 inf
  下溢：              任何 < 1.4e-45 的值变成 0.0
```

`exp()` 函数是 ML 中上溢的主要来源：

```
exp(88.7)  = 3.40e+38   （勉强适合 float32）
exp(89.0)  = inf         （上溢）
exp(-87.3) = 1.18e-38   （勉强高于下溢）
exp(-104)  = 0.0         （下溢为零）
```

`log()` 函数则面向另一个方向：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      （没问题）
log(1e-46) = -inf        （输入下溢为 0，然后 log(0) = -inf）
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。没有正确的技巧，`log(exp(x))` 组合就是一个雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任何 `x_i` 很大，`exp(x_i)` 会上溢。如果所有 `x_i` 都非常负，每个 `exp(x_i)` 都下溢为零，而 `log(0)` 是 `-inf`。

技巧：在指数运算之前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效：减去 `max(x)` 后，最大的指数是 `exp(0) = 1`。不会发生上溢。求和中至少有一项是 1，所以总和至少为 1，`log(1) = 0`。不会发生下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    （加减 c）
= log(sum(exp(x_i - c) * exp(c)))               （exp(a+b) = exp(a)*exp(b)）
= log(exp(c) * sum(exp(x_i - c)))               （提取公因子 exp(c)）
= c + log(sum(exp(x_i - c)))                    （log(a*b) = log(a) + log(b)）
```

设 `c = max(x)`，上溢就被消除了。

这个技巧在 ML 中随处可见：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推理

### 为什么 Softmax 需要最大值减法技巧

Softmax 将 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有这个技巧，[100, 101, 102] 这样的 logits 会导致上溢：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

这些会溢出 float32（最大 ~3.4e38）吗？实际上：
exp(88.7) 已经处于 float32 的极限。
exp(100) 在 float32 中 = inf。
```

使用了技巧后，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全相同。计算是安全的。这不是优化，而是正确性的必要条件。

### NaN 和 Inf：检测与预防

`nan`（非数字）和 `inf`（无穷大）会像病毒一样在计算中传播。梯度更新中的一个 `nan` 会使权重变为 `nan`，进而使后续所有输出都变成 `nan`。训练在一两步之内就结束了。

`inf` 如何出现：
- `exp()` 对一个很大的正数
- 除以零：`1.0 / 0.0`
- `float32` 在累积中上溢

`nan` 如何出现：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数取 `sqrt()`
- 对负数取 `log()`
- 任何涉及已有 `nan` 的运算

检测：

```python
import math

math.isnan(x)       # 如果 x 是 nan 返回 True
math.isinf(x)       # 如果 x 是 +inf 或 -inf 返回 True
math.isfinite(x)    # 如果 x 既不是 nan 也不是 inf 返回 True
```

预防策略：

1. 限制 `exp()` 的输入：`exp(clamp(x, -80, 80))`
2. 在分母中添加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内部添加 epsilon：`log(x + 1e-8)`
4. 使用稳定的实现（log-sum-exp、稳定 softmax）
5. 梯度裁剪以防止权重爆炸
6. 在调试期间每次前向传播后检查 `nan`/`inf`

### 数值梯度检验

解析梯度（来自反向传播）可能有 bug。数值梯度检验通过有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 精度的，远优于仅为 O(h) 的前向差分 `(f(x+h) - f(x)) / h`。

选择 h：太大则近似值不准。太小则灾难性抵消会破坏结果。通常取 `h = 1e-5` 到 `1e-7`。

检验：计算解析梯度与数值梯度的相对差异。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，可能正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

在实现新层或损失函数时，务必检查梯度。PyTorch 为此提供了 `torch.autograd.gradcheck()`。

### 混合精度训练

现代 GPU 拥有专门的硬件（Tensor Cores），其计算 float16 矩阵乘法的速度比 float32 快 2-8 倍。混合精度训练利用了这一特性：

```
1. 维护 float32 主副本的权重
2. 前向传播使用 float16（快速）
3. 使用 float32 计算损失（防止上溢）
4. 反向传播使用 float16（快速）
5. 将梯度缩放到 float32
6. 更新 float32 主权重
```

纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。Float16 会将任何低于 ~6e-8 的值下溢为零。你的模型停止学习，因为所有梯度更新都为零。

解决方法是损失缩放：

```
1. 将损失乘以一个大的缩放因子（例如 1024）
2. 反向传播计算 (loss * 1024) 的梯度
3. 所有梯度都放大了 1024 倍（被推到 float16 下溢阈值之上）
4. 在更新权重之前，将梯度除以 1024
5. 净效果：更新量相同，但不会下溢
```

动态损失缩放会自动调整缩放因子。从一个较大的值（65536）开始。如果梯度上溢到 `inf`，将其减半。如果 N 步都没有发生上溢，将其加倍。

### bfloat16 与 float16：为何 bfloat16 更适合训练

```
float16:   [1 符号位] [5 指数]  [10 尾数]
bfloat16:  [1 符号位] [8 指数]  [7 尾数]
```

float16 精度更高（10 位尾数 vs 7 位），但范围有限（最大 ~65,504）。bfloat16 精度较低，但范围与 float32 相同（最大 ~3.4e38）。

对于训练神经网络：

- 在训练峰值期间，激活值和 logits 经常超过 65,504。float16 会溢出；bfloat16 可以处理。
- 使用 float16 时通常需要损失缩放，但 bfloat16 通常不需要，因为它的范围覆盖了梯度幅值范围。
- bfloat16 是 float32 的简单截断：丢弃尾数的低 16 位。转换简单，指数部分无损。

Float16 更适合推理，因为推理时值有界且精度更重要。bfloat16 更适合训练，因为范围更重要。这就是 TPU 和现代 NVIDIA GPU（A100, H100）原生支持 bfloat16 的原因。

### 梯度裁剪

梯度爆炸发生在梯度通过多层呈指数增长时（常见于 RNN、深度网络和 Transformer）。一个大的梯度可以在一步之内破坏所有权重。

两种裁剪方式：

**按值裁剪：** 独立地限制每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但可能会改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保持梯度的方向。这就是 `torch.nn.utils.clip_grad_norm_()` 所做的。它是标准选择。

典型值：Transformer 使用 `max_norm=1.0`，RL 使用 `max_norm=0.5`，更简单的网络使用 `max_norm=5.0`。

梯度裁剪不是 hack。它是一种安全机制。没有它，一个单一的异常批次就可能产生大到足以毁掉数周训练的梯度。

### 归一化层作为数值稳定器

批量归一化、层归一化和 RMS 归一化通常被描述为帮助训练收敛的正则化器。它们同时也是数值稳定器。

没有归一化，激活值会通过各层呈指数增长或缩小：

```
第 1 层：值在 [0, 1]
第 5 层：值在 [0, 100]
第 10 层：值在 [0, 10,000]
第 50 层：值在 [0, inf]
```

归一化在每一层重新居中和重新缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常为 1e-5）防止在所有激活值相同时除以零。可学习参数 `gamma` 和 `beta` 让网络可以恢复它需要的任何尺度。

这使值在整个网络中保持在数值安全的范围内，既防止了前向传播中的上溢，也防止了反向传播中的梯度爆炸。

### 常见的 ML 数值 Bug

**Bug：经过几个 epoch 后，Loss 变为 NaN。**
原因：logits 变得太大，softmax 溢出。或者学习率太高，权重发散。
修复：使用稳定 softmax（最大值减法），降低学习率，添加梯度裁剪。

**Bug：Loss 卡在 log(num_classes)。**
原因：模型输出接近均匀的概率。通常意味着梯度消失或模型完全没有学习。
修复：检查数据标签是否正确，验证损失函数，检查是否有死掉的 ReLU。

**Bug：验证准确率比预期低 1-3%。**
原因：混合精度没有正确的损失缩放。梯度下溢悄悄地使小更新变为零。
修复：启用动态损失缩放，或切换到 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：死亡的 ReLU 神经元（所有输入为负），或 float16 下溢。
修复：使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**Bug：模型在一个 GPU 上工作正常，但在另一个 GPU 上给出不同结果。**
原因：非确定性的浮点累积顺序。GPU 并行归约在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
修复：接受小的差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度惩罚。

**Bug：损失计算中 `exp()` 返回 `inf`。**
原因：原始 logits 传递给 `exp()` 而没有使用最大值减法技巧。
修复：使用 `torch.nn.functional.log_softmax()`，它在内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的梯度幅度或高于 65,504 的激活值。
修复：使用带损失缩放的混合精度（AMP），或改用 bfloat16。

```figure
logsumexp-stability
```

## 构建它

### 第 1 步：演示浮点精度限制

```python
print("=== 浮点精度 ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"差值: {(0.1 + 0.2) - 0.3:.2e}")
```

### 第 2 步：实现朴素 vs 稳定 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"朴素:  {softmax_naive(safe_logits)}")
print(f"稳定: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"稳定: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### 第 3 步：实现稳定的 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"朴素:  {logsumexp_naive(safe):.6f}")
print(f"稳定: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"稳定: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### 第 4 步：实现稳定的交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"朴素:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"稳定: {cross_entropy_stable(true_class, logits):.6f}")
```

### 第 5 步：梯度检验

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  参数 {i}: 解析={a:.8f} 数值={n:.8f} "
              f"相对误差={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用它

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"原始范数: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"裁剪后范数:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"方向保留: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"警告 {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整的实现及所有边界情况演示，请参见 `code/numerical.py`。

## 交付物

本课程产出：
- `code/numerical.py`，包含稳定 softmax、log-sum-exp、交叉熵、梯度检验和混合精度模拟
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现将在阶段 3 构建训练循环和阶段 4 实现注意力机制时再次出现。

## 练习

1. **灾难性抵消。** 使用朴素公式 `E[x^2] - E[x]^2` 在 float32 下计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后使用 Welford 在线算法计算。比较两者与真实方差 (0.6667) 的误差。

2. **精度探索。** 在 Python 中找到最小的正 float32 值 `x`，使得 `1.0 + x == 1.0`。这就是机器 epsilon。验证它与 `numpy.finfo(numpy.float32).eps` 一致。

3. **Log-sum-exp 边界情况。** 测试你的 `logsumexp_stable` 函数，使用：(a) 所有值相等，(b) 一个值远大于其他值，(c) 所有值都非常负 (-1000)。验证其在朴素版本失败的地方给出正确结果。

4. **检验神经网络层的梯度。** 实现一个简单的线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 验证一个 3x2 权重矩阵的正确性。

5. **损失缩放实验。** 模拟 float16 训练：创建范围在 [1e-9, 1e-3] 的随机梯度，转换为 float16，测量有多少变为零。然后应用损失缩放（乘以 1024），转换为 float16，再缩放回来，再次测量零值比例。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf, nan）的国际标准。每颗现代 CPU 和 GPU 都实现了它。 |
| 机器 epsilon | "精度极限" | 在给定浮点格式中，满足 1.0 + e != 1.0 的最小值 e。对于 float32，约为 1.19e-7。 |
| 灾难性抵消 | "减法导致的精度损失" | 当减去两个几乎相等的浮点数时，有效数字抵消，舍入噪声主导了结果。 |
| 上溢 | "数字太大" | 结果超过最大可表示值，变成 inf。exp(89) 会使 float32 上溢。 |
| 下溢 | "数字太小" | 结果比最小可表示正数更接近零，变成 0.0。exp(-104) 会使 float32 下溢。 |
| Log-sum-exp 技巧 | "先减去最大值" | 通过提取 exp(max(x)) 因子来计算 log(sum(exp(x)))，以防止上溢和下溢。用于 softmax、交叉熵和对数概率计算。 |
| 稳定 softmax | "不会爆炸的 softmax" | 在指数运算之前减去 max(logits)。数值结果相同，不可能上溢。 |
| 梯度检验 | "验证你的反向传播" | 将反向传播的解析梯度与有限差分的数值梯度进行比较，以捕获实现错误。 |
| 混合精度 | "Float16 前向，float32 反向" | 对速度关键的操作使用低精度浮点数，对数值敏感的操作使用高精度浮点数。典型加速为 2-3 倍。 |
| 损失缩放 | "防止梯度下溢" | 在反向传播之前将损失乘以一个大常数，使梯度保持在 float16 的可表示范围内，然后在权重更新前除以相同的常数。 |
| bfloat16 | "大脑浮点数" | Google 的 16 位格式，具有 8 位指数（与 float32 相同的范围）和 7 位尾数（精度低于 float16）。更适合训练。 |
| 梯度裁剪 | "限制梯度范数" | 缩放梯度向量，使其范数不超过阈值。防止梯度爆炸破坏权重。 |
| NaN | "非数字" | 来自未定义操作的浮点特殊值（0/0, inf-inf, sqrt(-1)）。会在所有后续算术中传播。 |
| Inf | "无穷大" | 来自上溢或除以零的浮点特殊值。可以组合产生 NaN（inf - inf, inf * 0）。 |
| 数值梯度 | "暴力求导" | 通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。速度慢但可靠，用于验证。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考，内容密集但完整
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- NVIDIA 发表的论文，介绍了 float16 训练的损失缩放
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch 中混合精度的实用指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Google 为何为 TPU 选择这种格式
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 用于减少浮点求和中舍入误差的算法
