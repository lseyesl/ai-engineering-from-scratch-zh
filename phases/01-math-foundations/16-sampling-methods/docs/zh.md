# 采样方法

> 采样是 AI 探索可能性空间的方式。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 1，课程 06-07（概率，贝叶斯定理）
**时间：** ~120 分钟

## 学习目标

- 仅使用均匀随机数从头实现逆 CDF、拒绝采样和重要性采样
- 构建用于语言模型 token 生成的温度、top-k 和 top-p（核）采样
- 解释重参数化技巧以及它为什么使 VAE 中通过采样的反向传播成为可能
- 运行 Metropolis-Hastings MCMC 从未归一化的目标分布中采样

## 问题

语言模型处理完你的提示后，会产生一个包含 50,000 个 logits 的向量。每个对应词表中的一个 token。现在它必须选择一个。怎么选？

如果它总是选择概率最高的 token，那么每个响应都是一样的。确定性的。无聊的。如果它均匀随机选择，输出就是胡言乱语。答案介于这两个极端之间，而这个中间地带由采样控制。

采样不仅限于文本生成。强化学习通过采样轨迹来估计策略梯度。VAE 通过从学习到的分布中采样并通过随机性进行反向传播来学习潜在表示。扩散模型通过采样噪声并迭代去噪来生成图像。蒙特卡洛方法估计没有封闭解的积分。MCMC 算法探索无法枚举的高维后验分布。

每个生成式 AI 系统都是一个采样系统。采样策略决定了输出的质量、多样性和可控性。本课程从头构建每种主要的采样方法，从均匀随机数开始，到驱动现代 LLM 和生成模型的技术结束。

## 概念

### 为什么采样很重要

采样在 AI 和机器学习中扮演四个基本角色：

**生成。** 语言模型、扩散模型和 GAN 都通过采样产生输出。采样算法直接控制创造力、连贯性和多样性。温度、top-k 和核采样是工程师每天调节的旋钮。

**训练。** 随机梯度下降采样小批量。Dropout 采样要停用的神经元。数据增强采样随机变换。重要性采样重新加权样本以减少强化学习（PPO、TRPO）中的梯度方差。

**估计。** ML 中的许多量没有封闭解。数据分布上的期望损失、基于能量的模型的配分函数、贝叶斯推理中的证据。蒙特卡洛估计通过对样本求平均来近似所有这些量。

**探索。** MCMC 算法探索贝叶斯推理中的后验分布。进化策略采样参数扰动。汤普森采样在多臂老虎机中平衡探索和利用。

核心挑战：你只能直接从简单分布（均匀、正态）中采样。对于其他所有分布，你需要一种方法将简单样本转换为你目标分布的样本。

### 均匀随机采样

每种采样方法都从这里开始。均匀随机数生成器在 [0, 1) 范围内产生值，其中每个等长子区间具有相等的概率。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    对于 0 <= a <= b <= 1

性质：
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个项目的离散集合中均匀采样，生成 U 并返回 floor(n * U)。要从连续范围 [a, b] 中采样，计算 a + (b - a) * U。

关键洞察：一个均匀随机数恰好包含足够的随机性，可以从任何分布中产生一个样本。关键在于找到正确的变换。

### 逆 CDF 方法（逆变换采样）

累积分布函数（CDF）将值映射到概率：

```
F(x) = P(X <= x)

性质：
  F 非递减
  F(-inf) = 0
  F(+inf) = 1
  F 将实数线映射到 [0, 1]
```

逆 CDF 将概率映射回值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 服从目标分布。

```
算法：
  1. 生成 u ~ Uniform(0, 1)
  2. 返回 F_inverse(u)

为什么有效：
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```
PDF：f(x) = lambda * exp(-lambda * x)，   x >= 0
CDF：F(x) = 1 - exp(-lambda * x)

解 F(x) = u 求 x：
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

由于 (1 - U) 和 U 具有相同的分布：
  x = -ln(u) / lambda
```

当你能够给出 F_inverse 的封闭形式时，这种方法完美工作。对于正态分布，没有封闭形式的逆 CDF，所以我们使用其他方法（Box-Muller 或数值近似）。

**离散版本：** 对于离散分布，以累积和的形式构建 CDF，生成 U，并找到累积和超过 U 的第一个索引。这就是课程 06 中 `sample_categorical` 的工作方式。

### 拒绝采样

当无法求 CDF 的逆但可以评估目标 PDF（可能相差一个常数因子）时，拒绝采样有效。

```
目标分布：p(x)  （可评估，可能未归一化）
提议分布：q(x)  （可以从中采样）
界限：M 使得对所有 x 都有 p(x) <= M * q(x)

算法：
  1. 抽样 x ~ q(x)
  2. 抽样 u ~ Uniform(0, 1)
  3. 如果 u < p(x) / (M * q(x))，接受 x
  4. 否则，拒绝并返回步骤 1

接受率 = 1/M
```

界限 M 越紧，接受率越高。在低维中（1-3），拒绝采样效果很好。在高维中，接受率呈指数下降，因为大部分提议体积被拒绝。这就是拒绝采样的维度灾难。

**示例：从截断正态分布中采样。** 在截断范围内使用均匀提议。包络线 M 是该范围内正态 PDF 的最大值。

**示例：从半圆中采样。** 在边界矩形内均匀提议。如果点落在半圆内则接受。这就是蒙特卡洛计算 pi 的方式：接受率等于面积比 pi/4。

### 重要性采样

有时你不需要来自目标分布 p(x) 的样本。你需要估计在 p(x) 下的期望值，而你拥有来自不同分布 q(x) 的样本。

```
目标：估计 E_p[f(x)] = integral of f(x) * p(x) dx

重写：
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

其中 w(x) = p(x) / q(x) 是重要性权重。

估计器：
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    其中 x_i ~ q(x)
```

这在强化学习中至关重要。在 PPO（近端策略优化）中，你在旧策略 pi_old 下收集轨迹，但想要优化新策略 pi_new。重要性权重是 pi_new(a|s) / pi_old(a|s)。PPO 会裁剪这些权重，防止新策略偏离旧策略太远。

重要性采样估计器的方差取决于 q 与 p 的相似程度。如果 q 与 p 差异很大，少量样本会获得极大的权重并主导估计。自归一化重要性采样通过除以权重之和来减少这个问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 蒙特卡洛估计

蒙特卡洛估计通过对随机样本求平均来近似积分。大数定律保证收敛。

```
目标：估计 I = integral of g(x) dx 在区域 D 上

方法：
  1. 从 D 中均匀抽样 x_1, ..., x_N
  2. I ~ (D 的体积 / N) * sum(g(x_i))

误差：O(1 / sqrt(N))   与维度无关
```

误差率与维度无关。这就是为什么蒙特卡洛方法在高维度中占主导地位，而基于网格的积分是不可能的。

**估计 pi：**

```
在 [-1, 1] x [-1, 1] 内均匀抽样 (x, y)
统计落在单位圆内的数量：x^2 + y^2 <= 1
pi ~ 4 * (圆内数量) / (总数量)
```

**估计期望值：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    其中 x_i ~ p(x)

样本均值收敛到真实期望。
估计器的方差 = Var(f(X)) / N
```

### 马尔可夫链蒙特卡洛（MCMC）：Metropolis-Hastings

MCMC 构建一个马尔可夫链，其平稳分布是目标分布 p(x)。经过足够多的步骤后，链中的样本（近似）来自 p(x)。

```
目标：p(x)  （已知到归一化常数）
提议：q(x'|x)  （如何在给定当前状态下提议下一个状态）

Metropolis-Hastings 算法：
  1. 从某个 x_0 开始
  2. 对 t = 1, 2, ..., T：
     a. 提议 x' ~ q(x'|x_t)
     b. 计算接受比率：
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. 以概率 min(1, alpha) 接受：
        - 如果 u < alpha（u ~ Uniform(0,1)）：x_{t+1} = x'
        - 否则：x_{t+1} = x_t
  3. 丢弃前 B 个样本（烧入期）
  4. 返回剩余样本
```

对于对称提议（q(x'|x) = q(x|x')），比率简化为 p(x')/p(x)。这是原始的 Metropolis 算法。

**为什么有效。** 接受规则确保细致平衡：处于 x 并移动到 x' 的概率等于处于 x' 并移动到 x 的概率。细致平衡意味着 p(x) 是链的平稳分布。

**实际考虑：**
- 烧入期：丢弃链达到平衡之前的早期样本
- 稀疏化：每隔 k 个样本保留一个以减少自相关
- 提议尺度：太小则链移动缓慢（接受率高，探索慢）；太大则大多数提议被拒绝（接受率低，卡在原地）
- 高维高斯提议的最优接受率约为 0.234

### Gibbs 采样

Gibbs 采样是多变量分布 MCMC 的一个特例。它不是一次性在所有维度上提议移动，而是每次从其条件分布中更新一个变量。

```
目标：p(x_1, x_2, ..., x_d)

算法：
  对于每次迭代 t：
    抽样 x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    抽样 x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    抽样 x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs 采样要求你可以从每个条件分布 p(x_i | x_{-i}) 中采样。这对于许多模型是直接的：
- 贝叶斯网络：条件分布来自图结构
- 高斯混合：条件分布是高斯分布
- Ising 模型：每个自旋的条件分布仅依赖于其邻居

接受率总是 1（每个提议都被接受），因为从精确的条件分布中采样自然满足细致平衡。

**局限性。** 当变量高度相关时，Gibbs 采样混合缓慢，因为一次更新一个变量无法在分布中做出大的对角移动。

### 温度采样（用于 LLM）

语言模型为词表中的每个 token 输出 logits z_1, ..., z_V。Softmax 将这些转换为概率。温度在 softmax 之前重新缩放 logits：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0：标准 softmax（原始分布）
T -> 0：  argmax（确定性的，总是选最高 logit）
T -> inf：均匀分布（所有 token 等可能）
T < 1.0：锐化分布（更自信，更少多样性）
T > 1.0：平坦化分布（更不自信，更多多样性）
```

**为什么有效。** 除以 T < 1 会放大 logits 之间的差异。如果 z_1 = 2 且 z_2 = 1，除以 T = 0.5 得到 z_1/T = 4 和 z_2/T = 2，使差距更大。经过 softmax 后，最高 logit 的 token 获得更大的份额。

**在实践中：**
- T = 0.0：贪心解码，最适合事实性问答
- T = 0.3-0.7：略微创造性，适合代码生成
- T = 0.7-1.0：平衡，适合一般对话
- T = 1.0-1.5：创意写作、头脑风暴
- T > 1.5：越来越随机，很少有用

温度不会改变哪些 token 是可能的。它改变了分配给每个 token 的概率质量。

### Top-k 采样

Top-k 采样将候选集限制为概率最高的 k 个 token，然后重新归一化并从该受限集中采样。

```
算法：
  1. 计算所有 V 个 token 的 softmax 概率
  2. 按概率对 token 排序（降序）
  3. 仅保留前 k 个 token
  4. 重新归一化：p_i' = p_i / sum(p_j 对于 j 在 top-k 中)
  5. 从重新归一化的分布中采样

k = 1：  贪心解码
k = V：  不过滤（标准采样）
k = 40：典型设置，去除长尾中不太可能的 token
```

Top-k 防止模型选择词表分布长尾中的极端不可能 token（拼写错误、无意义内容）。问题在于：k 是固定的，不依赖于上下文。当模型自信时（一个 token 有 95% 概率），k = 40 仍然允许 39 个备选。当模型不确定时（概率分布在 1000 个 token 上），k = 40 会切断合理的选项。

### Top-p（核）采样

Top-p 采样动态调整候选集大小。它不是保留固定数量的 token，而是保留累积概率超过 p 的最小 token 集合。

```
算法：
  1. 计算所有 V 个 token 的 softmax 概率
  2. 按概率对 token 排序（降序）
  3. 找到最小的 k 使得前 k 个概率之和 >= p
  4. 仅保留那 k 个 token
  5. 重新归一化并采样

p = 0.9：  保留覆盖 90% 概率质量的 token
p = 1.0：  不过滤
p = 0.1：  限制性很强，几乎贪心
```

当模型自信时，核采样保留少量 token（可能 2-3 个）。当模型不确定时，它保留许多（可能 200 个）。这种自适应行为使核采样通常比 top-k 产生更好的文本。

**常见组合：**
- 温度 0.7 + top-p 0.9：良好的通用设置
- 温度 0.0（贪心）：最适合确定性任务
- 温度 1.0 + top-k 50：Fan et al.（2018）原始论文设置

Top-k 和 top-p 可以组合使用。先应用 top-k，然后在剩余集合上应用 top-p。

### 重参数化技巧（用于 VAE）

变分自编码器（VAE）通过将输入编码为潜空间中的分布，从该分布中采样，然后解码样本来学习。问题在于：你无法通过采样操作进行反向传播。

```
标准采样（不可微）：
  z ~ N(mu, sigma^2)

  随机性阻断了梯度流。
  d/d_mu [从 N(mu, sigma^2) 采样] = ???
```

重参数化技巧将随机性与参数分离：

```
重参数化采样：
  epsilon ~ N(0, 1)          （固定的随机噪声，无参数）
  z = mu + sigma * epsilon   （参数的确定性函数）

  现在 z 是 mu 和 sigma 的确定性、可微函数。
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  梯度流过 mu 和 sigma。
```

这是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 具有相同的分布。关键洞察：将随机性移到无参数源（epsilon），然后将样本表示为参数的可微变换。

**在 VAE 训练循环中：**
1. 编码器为每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重建输入
5. 通过步骤 4、3、2、1 反向传播（因为步骤 3 是可微的）

没有重参数化技巧，VAE 无法使用标准反向传播进行训练。这一洞察使 VAE 变得实用。

### Gumbel-Softmax（可微分类采样）

重参数化技巧适用于连续分布（高斯分布）。对于离散分类分布，我们需要不同的方法。Gumbel-Softmax 提供了分类采样的可微近似。

**Gumbel-Max 技巧（不可微）：**

```
从具有对数概率 log(p_1), ..., log(p_k) 的分类分布中采样：
  1. 为每个类别采样 g_i ~ Gumbel(0, 1)
     （g = -log(-log(u))，其中 u ~ Uniform(0, 1)）
  2. 返回 argmax(log(p_i) + g_i)

这会产生精确的分类样本。
```

**Gumbel-Softmax（可微近似）：**

```
用软的 softmax 替换硬的 argmax：
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau（温度）控制近似的精度：
  tau -> 0：  接近独热向量（硬分类）
  tau -> inf：接近均匀分布（1/k, 1/k, ..., 1/k）
  tau = 1.0：软近似
```

Gumbel-Softmax 产生离散样本的连续松弛。输出是一个概率向量（软独热）而不是硬独热。梯度通过 softmax 流动。在训练的前向传播中，你可以使用"直通"估计器：在前向传播中使用硬 argmax，但在反向传播中使用软的 Gumbel-Softmax 梯度。

**应用：**
- VAE 中的离散潜变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 具有离散动作的强化学习

### 分层采样

标准蒙特卡洛采样可能会偶然在样本空间中留下间隙。分层采样通过将空间划分为层并从每个层中采样来强制均匀覆盖。

```
标准蒙特卡洛：
  从 [0, 1] 中均匀采样 N 个点
  某些区域可能有聚集，其他区域有间隙

分层采样：
  将 [0, 1] 分成 N 个相等层：[0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  在每个层内均匀采样一个点
  x_i = (i + u_i) / N   其中 u_i ~ Uniform(0, 1)，i = 0, ..., N-1
```

分层采样总是具有低于或等于标准蒙特卡洛的方差：

```
Var(分层) <= Var(标准蒙特卡洛)

当 f(x) 平滑变化时，改进最大。
对于分段常数函数，分层采样是精确的。
```

**应用：**
- 数值积分（准蒙特卡洛）
- 训练数据划分（确保每折中的类别平衡）
- 带分层的重要性采样（结合两种技术）
- NeRF（神经辐射场）沿相机射线使用分层采样

### 与扩散模型的联系

扩散模型通过采样过程生成图像。前向过程在 T 步中向图像添加高斯噪声，直到变成纯噪声。反向过程学习去噪，逐步恢复原始图像。

```
前向过程（已知）：
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  其中 epsilon ~ N(0, I)

  经过 T 步：x_T ~ N(0, I)  （纯噪声）

反向过程（学习得到）：
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  其中 z ~ N(0, I)

  每个去噪步骤都是一个采样步骤。
```

与本课程方法的联系：
- 每个去噪步骤都使用重参数化技巧（采样噪声，应用确定性变换）
- 噪声计划 {alpha_t} 控制一种温度退火形式
- 训练使用蒙特卡洛估计来近似 ELBO（证据下界）
- 扩散模型中的祖先采样是一个马尔可夫链（每一步仅依赖于当前状态）

整个图像生成过程是迭代采样：从噪声开始，每一步都以学习到的去噪模型为条件，采样一个噪声稍小的版本。

```figure
monte-carlo-pi
```

## 构建它

### 第 1 步：均匀和逆 CDF 采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数样本，验证均值为 1/lambda。

### 第 2 步：拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用拒绝采样从截断正态分布中抽样。通过直方图验证形状。

### 第 3 步：重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用均匀提议估计正态分布下的 E[X^2]。与已知答案（mu^2 + sigma^2）比较。

### 第 4 步：蒙特卡洛估计 pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 第 5 步：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布（两个高斯混合）中采样。可视化链的轨迹。

### 第 6 步：Gibbs 采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 第 7 步：温度采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示对于一组 token logits，温度如何改变输出分布。

### 第 8 步：Top-k 和 top-p 采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 第 9 步：重参数化技巧

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度流经重参数化样本，但不流经直接采样。

### 第 10 步：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低温度如何使输出接近独热向量。

所有实现及可视化的完整代码在 `code/sampling.py` 中。

## 使用它

使用 NumPy 和 SciPy 的生产版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"指数均值: {exponential_samples.mean():.4f} (期望 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"1.96 处的 CDF: {normal.cdf(1.96):.4f}")
print(f"0.975 处的逆 CDF: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"采样到的 token 索引: {token}")
```

对于大规模 MCMC，使用专门的库：
- PyMC：完整的贝叶斯建模，使用 NUTS（自适应 HMC）
- emcee：集成 MCMC 采样器
- NumPyro/JAX：GPU 加速的 MCMC

你从零构建了这些。现在你知道库调用在做什么了。

## 练习

1. 为柯西分布实现逆 CDF 采样。CDF 为 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本，将直方图与真实 PDF 对比。注意其厚尾性（远离中心的极值）。

2. 使用拒绝采样从 Beta(2, 5) 分布中生成样本，使用 Uniform(0, 1) 作为提议分布。绘制接受的样本与真实 Beta PDF 的对比。理论接受率是多少？

3. 使用蒙特卡洛方法估计 sin(x) 从 0 到 pi 的积分，分别使用 1,000、10,000 和 100,000 个样本。比较每个级别的误差。验证误差遵循 O(1/sqrt(N)) 的缩放规律。

4. 实现 Metropolis-Hastings 从 2D 分布 p(x, y) 中采样，该分布正比于 exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2)。绘制样本和链轨迹。尝试不同的提议标准差。

5. 构建一个完整的文本生成演示：给定一个包含 10 个词及其 logits 的词表，使用 (a) 贪心、(b) 温度=0.7、(c) top-k=3、(d) top-p=0.9 生成 20 个 token 的序列。比较 5 次运行中输出的多样性。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| 采样 | "抽取随机值" | 根据概率分布生成值。所有生成式 AI 背后的机制 |
| 均匀分布 | "所有值等可能" | [a, b] 中每个值的概率密度为 1/(b-a)。所有采样方法的起点 |
| 逆 CDF | "概率变换" | F_inverse(U) 将均匀样本转换为来自任何已知 CDF 的分布的样本。精确且高效 |
| 拒绝采样 | "提议并接受/拒绝" | 从简单的提议分布生成，以正比于目标/提议比率的概率接受。精确但浪费样本 |
| 重要性采样 | "重新加权样本" | 使用来自 q(x) 的样本估计在 p(x) 下的期望值，通过每个样本乘以 p(x)/q(x) 权重。RL 中 PPO 的核心 |
| 蒙特卡洛 | "平均随机样本" | 将积分近似为样本平均值。误差 O(1/sqrt(N))，与维度无关 |
| MCMC | "收敛的随机游走" | 构建一个马尔可夫链，其平稳分布就是目标分布。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | "接受上坡，有时接受下坡" | 提议移动，基于密度比接受。细致平衡确保收敛到目标分布 |
| Gibbs 采样 | "一次一个变量" | 从其条件分布中更新每个变量，保持其他变量固定。100% 接受率 |
| 温度 | "置信度旋钮" | 在 softmax 之前将 logits 除以 T。T<1 锐化（更自信），T>1 平坦化（更多样） |
| Top-k 采样 | "保留前 k 个" | 将所有除概率最高的 k 个 token 以外的归零，重新归一化，采样。固定候选集大小 |
| 核采样（top-p） | "保留那些可能的" | 保留累积概率超过 p 的最小 token 集合。自适应候选集大小 |
| 重参数化技巧 | "将随机性移到外部" | 将 z 写为 mu + sigma * epsilon，其中 epsilon ~ N(0,1)。使采样可微。对 VAE 训练至关重要 |
| Gumbel-Softmax | "软分类采样" | 使用 Gumbel 噪声 + 带温度的 softmax 对分类采样进行可微近似 |
| 分层采样 | "强制覆盖" | 将样本空间划分为层，从每个层中采样。方差始终低于朴素蒙特卡洛 |
| 烧入期 | "预热期" | 在链达到其平稳分布之前丢弃的初始 MCMC 样本 |
| 细致平衡 | "可逆性条件" | p(x) * T(x->y) = p(y) * T(y->x)。p 是马尔可夫链平稳分布的充分条件 |
| 扩散采样 | "迭代去噪" | 从噪声开始，应用学习到的去噪步骤来生成数据。每一步都是一个条件采样操作 |

## 延伸阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - 原始 Gumbel-Softmax 论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 核（top-p）采样论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 引入重参数化技巧的 VAE 论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM 将采样与图像生成联系起来
