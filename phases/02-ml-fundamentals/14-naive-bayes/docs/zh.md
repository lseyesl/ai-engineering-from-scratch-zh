# 朴素贝叶斯

> "朴素"的假设是错误的，但它仍然有效。这就是它的美妙之处。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 2，课程 01-07（分类、贝叶斯定理）
**时间：** ~75 分钟

## 学习目标

- 从头实现带拉普拉斯平滑的多项式朴素贝叶斯用于文本分类
- 解释为什么朴素独立性假设在数学上是错误的，但在实践中能产生正确的类别排名
- 比较多項式、伯努利和高斯朴素贝叶斯变体，并为给定的特征类型选择正确的变体
- 在高维稀疏数据上评估朴素贝叶斯与逻辑回归，并解释偏差-方差权衡的作用

## 问题

你需要对文本进行分类。邮件分为垃圾邮件或非垃圾邮件。客户评论分为正面或负面。支持工单分类。你有数千个特征（每个词一个）和有限的训练数据。

大多数分类器在这里举步维艰。逻辑回归需要足够样本来可靠地估计数千个权重。决策树一次在一个词上分裂且严重过拟合。在 10,000 维中的 KNN 毫无意义，因为每个点与其他任何点距离同样远。

朴素贝叶斯处理了这个情况。它做了一个数学上错误的假设（给定类别，每个特征与其他每个特征独立），它仍然在文本分类上优于"更聪明"的模型，尤其是在小训练集上。它在单次数据遍历中完成训练。它可以扩展到数百万个特征。它产生概率估计（尽管由于独立性假设而常常校准不佳）。

理解为什么一个错误假设会导致好的预测，教会你关于机器学习的一些基本原理：最好的模型不是最正确的那个，而是对你的数据具有最佳偏差-方差权衡的那个。

## 概念

### 贝叶斯定理（快速回顾）

贝叶斯定理翻转条件概率：

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们想要 `P(class | features)`——给定文档中的词，文档属于某个类别的概率。我们可以从以下内容计算：
- `P(features | class)`——在该类别的文档中看到这些词的可能性
- `P(class)`——类别的先验概率（通常情况下垃圾邮件有多常见？）
- `P(features)`——证据，对所有类别相同，所以在比较时可以忽略

具有最高 `P(class | features)` 的类别获胜。

### 朴素独立性假设

精确计算 `P(features | class)` 需要估计所有特征一起的联合概率。对于一个包含 10,000 个词的词汇表，你需要估计一个 2^10,000 种可能组合上的分布。不可能。

朴素假设：给定类别，每个特征条件独立。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

而不是一个不可能的联合分布，你估计 n 个简单的每特征分布。每个只需要一个计数。

这个假设显然是错误的。在任何文档中，"机器"和"学习"这两个词都不是独立的。但分类器不需要正确的概率估计。它需要正确的排名——哪个类别的概率最高。独立性假设引入了系统误差，但这些误差对所有类别的影响相似，所以排名保持正确。

### 为什么它仍然有效

三个原因：

1. **排名重于校准。** 分类只需要排名第一的类别正确。即使 P(spam) = 0.99999 而真实概率是 0.7，分类器仍然正确地选择了垃圾邮件。我们不需要正确的概率。我们需要正确的胜者。

2. **高偏差，低方差。** 独立性假设是一个强先验。它严重约束了模型，防止过拟合。在有限的训练数据下，一个略有错误但稳定的模型胜过一个理论上正确但极其不稳定的模型。这就是偏差-方差权衡的作用。

3. **特征冗余相互抵消。** 相关特征提供冗余证据。分类器多算了这个证据，但它也为正确的类别多算了。如果"机器"和"学习"总是一起出现，两者都为"科技"类别提供证据。NB 将它们计数两次，但它是为正确的类计数两次。

第四个实际原因：朴素贝叶斯极快。训练是单次数据遍历，对频率进行计数。预测是矩阵乘法。你可以在几秒内在百万份文档上训练。这种速度意味着你可以更快地迭代，尝试更多特征集，并比使用较慢的模型运行更多实验。

### 逐步数学推导

让我们通过一个具体例子。假设有两个类别：垃圾邮件和非垃圾邮件。我们的词汇表有三个词："免费"、"钱"、"会议"。

训练数据：
- 垃圾邮件中提到"免费"80 次、"钱"60 次、"会议"10 次（总共 150 个词）
- 非垃圾邮件中提到"免费"5 次、"钱"10 次、"会议"100 次（总共 115 个词）
- 40% 的邮件是垃圾邮件，60% 是非垃圾邮件

使用拉普拉斯平滑（alpha=1）：

```
P(免费 | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(钱 | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(会议 | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(免费 | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(钱 | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(会议 | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含："免费"（2 次）、"钱"（1 次）、"会议"（0 次）。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

垃圾邮件以很大优势获胜。"免费"一词出现两次是垃圾邮件的强力证据。注意"会议"不出现对两个对数之和贡献为零（0 * log(P)）——在 Multinomial NB 中，缺失的词没有影响。显式建模词缺失的是 Bernoulli NB。

### 三个变体

朴素贝叶斯有三种形式。每种对 `P(feature | class)` 的建模方式不同。

#### 多项式朴素贝叶斯

将每个特征建模为计数。最适合特征是词频或 TF-IDF 值的文本数据。

```
P(word_i | class) = (class 中 word_i 的计数 + alpha) / (class 中总词数 + alpha * vocab_size)
```

`alpha` 是拉普拉斯平滑。这个变体是文本分类的主力。

#### 高斯朴素贝叶斯

将每个特征建模为正态分布。最适合连续特征。

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别获得每个特征的均值和方差。当特征在每个类别内真的遵循钟形曲线时效果很好。

#### 伯努利朴素贝叶斯

将每个特征建模为二值（存在或缺失）。最适合短文本或二值特征向量。

```
P(word_i | class) = (包含 word_i 的 class 中文档数 + alpha) / (class 总文档数 + 2 * alpha)
```

与多项不同，伯努利显式惩罚某个词的缺失。如果"免费"通常出现在垃圾邮件中但此邮件中没有，伯努利会将其作为反对垃圾邮件的证据。

### 何时使用每个变体

| 变体 | 特征类型 | 最适合 | 示例 |
|---------|-------------|----------|---------|
| 多项式 | 计数或频率 | 文本分类、词袋 | 邮件垃圾过滤、主题分类 |
| 高斯 | 连续值 | 带正态分布的表格数据 | 鸢尾花分类、传感器数据 |
| 伯努利 | 二值 (0/1) | 短文本、二值特征向量 | 短信垃圾过滤、存在/缺失特征 |

### 拉普拉斯平滑

当测试数据中出现一个在训练数据中某一类从未见过的词时会发生什么？

没有平滑：`P(word | class) = 0/N = 0`。一个零乘入整个乘积使 `P(class | features) = 0`，无论所有其他证据如何。单个未见过的词摧毁了整个预测，无论有多少其他证据支持它。

拉普拉斯平滑为每个特征计数添加一个小计数 `alpha`（通常为 1）：

```
P(word_i | class) = (count(word_i, class) + alpha) / (class 中总词数 + alpha * vocab_size)
```

使用 alpha=1，每个词至少获得一个很小的概率。测试邮件中出现"莫名其妙"这个词不再杀死垃圾邮件概率。平滑有一个贝叶斯解释：它相当于在词分布上放置一个均匀的狄利克雷先验。

较高的 alpha 意味着更强的平滑（更均匀的分布）。较低的 alpha 意味着模型更信任数据。Alpha 是一个需要调优的超参数。

| Alpha | 效果 | 何时使用 |
|-------|--------|-------------|
| 0.001 | 几乎没有平滑，信任数据 | 非常大的训练集，预期无未见特征 |
| 0.1 | 轻度平滑 | 大型训练集 |
| 1.0 | 标准拉普拉斯平滑 | 默认起点 |
| 10.0 | 重度平滑，拉平分布 | 非常小的训练集，预期许多未见特征 |

### 对数空间计算

将数百个概率（每个小于 1）相乘会导致浮点数下溢。乘积在浮点数中变为零，即使真实值是一个非常小的正数。

解决方法：在对数空间中工作。不乘概率，而是加它们的对数：

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这将预测变成点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是朴素贝叶斯预测如此之快的原因——它与单层线性模型的操作相同。

### 朴素贝叶斯 vs 逻辑回归

两者都是用于文本的线性分类器。区别在于它们建模的东西。

| 方面 | 朴素贝叶斯 | 逻辑回归 |
|--------|------------|-------------------|
| 类型 | 生成式（建模 P(X|Y)） | 判别式（建模 P(Y|X)） |
| 训练 | 计数频率 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（不足以估计权重） |
| 大数据 | 更差（错误假设有害） | 更好（灵活的边界） |
| 特征 | 假设独立 | 处理相关性 |
| 速度 | 单次遍历，非常快 | 迭代优化 |
| 校准 | 概率差 | 概率更好 |

经验法则：从朴素贝叶斯开始。如果你有足够的数据且 NB 进入瓶颈，切换到逻辑回归。

### 分类流水线

```mermaid
flowchart LR
    A[原始文本] --> B[分词]
    B --> C[构建词汇表]
    C --> D[计数词频]
    D --> E[应用平滑]
    E --> F[计算对数概率]
    F --> G[预测：argmax P(class|词)]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

实际上，我们在对数空间中工作以避免浮点数下溢。不乘许多小概率，而是加它们的对数：

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## 构建它

`code/naive_bayes.py` 中的代码从头实现了 MultinomialNB 和 GaussianNB。

### MultinomialNB

从头实现：

1. **fit(X, y)**：对于每个类别，统计每个特征的频率。添加拉普拉斯平滑。计算对数概率。存储类别先验（类别频率的对数）。
2. **predict_log_proba(X)**：对于每个样本，计算所有类别的 log P(class) + log P(feature_i | class) 之和。这通过矩阵乘法实现：X @ log_probs.T + log_priors。
3. **predict(X)**：返回对数概率最高的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞见：拟合后，预测只是矩阵乘法加一个偏置。这就是朴素贝叶斯如此快速的原因。

### GaussianNB

对于连续特征，我们估计每个类别每个特征的均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测使用每个特征的高斯 PDF，跨特征乘（在对数空间中加）。

### 演示：文本分类

代码生成合成词袋数据，模拟两个类别（科技文章 vs 体育文章）。每个类别有不同的词频分布。MultinomialNB 使用词计数对它们进行分类。

合成数据的工作原理：我们创建 200 个"词"（特征列）。词 0-39 在科技文章中高频、在体育文章中低频。词 80-119 在体育文章中高频、在科技文章中低频。词 40-79 在两者中中等频率。这创建了一个现实场景，其中一些词是强类别指标，而其他是噪声。

### 演示：连续特征

代码生成类似鸢尾花的数据（3 个类别、4 个特征、高斯簇）。GaussianNB 使用每类均值和方差进行分类。

代码还演示了平滑比较、训练大小实验，以及使用混淆矩阵进行逐类精确率/召回率/F1 分析。

### 预测速度

朴素贝叶斯预测是矩阵乘法。对于 n 个样本，d 个特征和 k 个类别：
- MultinomialNB：一次矩阵乘法 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB：n * k 次高斯 PDF 评估，每次涉及 d 个特征 = O(n * d * k)

两者在所有维度上都是线性的。对比 KNN 或 SVM，NB 在预测时快了几个数量级。

## 使用它

使用 sklearn，两个变体都是一行代码：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB 准确率: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB 准确率: {mnb.score(X_test_counts, y_test):.3f}")
```

### TF-IDF 与朴素贝叶斯

TF-IDF（词频-逆文档频率）降低常见词的权重，提高罕见、有辨别力的词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

### 校准 NB 概率

NB 概率校准不佳。如果需要可靠的概率估计，使用 sklearn 的 CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

### 常见陷阱

1. **负特征值。** MultinomialNB 需要非负特征。如果遇到负值，使用 GaussianNB。
2. **零方差特征。** GaussianNB 除以方差。代码向所有方差添加 1e-9 防止崩溃。
3. **类别不平衡。** 强先验可能压倒似然证据。使用 class_prior 参数调整。
4. **特征缩放。** MultinomialNB 和 GaussianNB 都不需要特征缩放。

## 交付物

本课程产出：
- `outputs/skill-naive-bayes-chooser.md`——用于选择正确 NB 变体的决策技能
- `code/naive_bayes.py`——从头实现的 MultinomialNB 和 GaussianNB

### 朴素贝叶斯何时失败

当独立性假设导致不正确的排名时，NB 会失败。这发生在强特征交互（类 XOR 模式）、具有相反证据的高度相关特征，或非常大的训练集（此时判别模型如逻辑回归会超越它）的情况下。

## 练习

1. **平滑实验。** 使用不同 alpha 值训练 MultinomialNB。绘制准确率 vs alpha。
2. **特征独立性测试。** 计算 P(word1|class) * P(word2|class) 并与 P(word1 AND word2|class) 比较。
3. **伯努利实现。** 扩展代码添加 BernoulliNB。比较与 MultinomialNB 的准确率。
4. **NB vs 逻辑回归。** 在文本数据上从 100 到 10,000 样本训练两者，绘制准确率曲线。
5. **垃圾邮件过滤器。** 构建完整的垃圾邮件分类器：分词、构建词汇表、词袋特征、MultinomialNB、用精确率和召回率评估。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| 朴素贝叶斯 | "简单的概率分类器" | 应用贝叶斯定理并假设特征在给定类别条件下独立 |
| 条件独立 | "特征互不影响" | P(A,B|C) = P(A|C) * P(B|C)——知道 B 不会告诉你关于 A 的新信息 |
| 拉普拉斯平滑 | "加一平滑" | 为每个特征添加小计数，防止零概率主导预测 |
| 先验 | "看到数据前的信念" | P(class)——观察任何特征前每个类别的概率 |
| 似然 | "数据拟合的程度" | P(features|class)——给定类别时观察到这些特征的概率 |
| 后验 | "看到数据后的信念" | P(class|features)——观察特征后类别的更新概率 |
| 生成模型 | "建模数据生成方式" | 学习 P(X|Y) 和 P(Y)，然后用贝叶斯定理得到 P(Y|X) |
| 判别模型 | "建模决策边界" | 直接学习 P(Y|X) 而不建模 X 如何生成 |
| 对数概率 | "避免下溢" | 使用 log P 代替 P，防止多个小数的乘积在浮点数中变为零 |

## 延伸阅读

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)
