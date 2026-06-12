# 处理不平衡数据

> 当 99% 的数据是"正常"时，准确率就是一个谎言。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 2，课程 01-09（尤其是评估指标）
**时间：** ~90 分钟

## 学习目标

- 从头实现 SMOTE，并解释合成过采样与随机复制有何不同
- 使用 F1、AUPRC 和 Matthews 相关系数而非准确率来评估不平衡分类器
- 比较类别加权、阈值调整和重采样策略，并为给定的不平衡比率选择合适的方法
- 构建一个完整的不平衡数据流水线，结合 SMOTE、类别权重和阈值优化

## 问题

你构建了一个欺诈检测模型。它达到了 99.9% 的准确率。你庆祝了一番。然后你意识到它对每笔交易都预测"非欺诈"。

这不是一个 bug。当只有 0.1% 的交易是欺诈时，这是理性的做法。模型学会了始终猜测多数类以最小化总体误差。它在技术上是正确的，但完全无用。

这发生在所有真正重要的分类任务中。疾病诊断：1% 的阳性率。网络入侵：0.01% 的攻击。制造缺陷：0.5% 的缺陷。垃圾邮件过滤：20% 的垃圾邮件。客户流失预测：5% 的流失者。少数类越重要，它就越罕见。

准确率失败，因为它将所有正确预测视为同等重要。正确标记合法交易和正确捕获欺诈都计为准确率的一个点。但捕获欺诈是整个模型存在的原因。我们需要迫使模型关注罕见但重要的类的指标、技术和训练策略。

## 概念

### 为什么准确率失败

准确率 = (TP + TN) / (TP + TN + FP + FN)

当 99% 的样本属于多数类时，一个始终预测多数类的模型就得到 99% 准确率。它完全忽略了少数类。

准确率假设类别平衡。当这种假设被严重违反时，它没有提供有用的信息。

### 不平衡下的正确指标

**精确率：** TP / (TP + FP)。在模型标记为"欺诈"的所有样本中，有多少确实是欺诈？
**召回率：** TP / (TP + FN)。在所有实际欺诈中，我们捕获了多少？
**F1 分数：** 精确率和召回率的调和平均数。当两者都不占优势时，平衡两者。
**AUPRC（精确率-召回率曲线下面积）：** 对不同阈值的精确率 vs 召回率绘图。与类别分布无关。对于不平衡数据优于 AUC-ROC。
**MCC（Matthews 相关系数）：** 从 -1（完全分歧）到 +1（完美一致），0 是随机。平衡两类的大小。对于高度不平衡的数据，MCC 是单个指标的最佳选择。

```python
def mcc(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    numerator = tp * tn - fp * fn
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator > 0 else 0
```

### 处理不平衡的三种策略

#### 1. 重采样

**过采样少数类：** 随机复制少数类样本。简单但可能导致过拟合，因为模型看到重复的样本。

**SMOTE（合成少数过采样技术）：** 在少数类样本之间插值来创建合成样本。计算少数类的最近邻之间的特征向量的加权平均。

```python
def smote(X_minority, n_synthetic, k=5):
    synthetic = []
    for _ in range(n_synthetic):
        idx = np.random.randint(len(X_minority))
        point = X_minority[idx]
        neighbors = find_k_nearest(point, X_minority, k)
        neighbor = neighbors[np.random.randint(k)]
        diff = neighbor - point
        synthetic.append(point + np.random.random() * diff)
    return np.array(synthetic)
```

SMOTE 缓解了随机过采样的过拟合问题。生成的样本在原始样本之间插值，而不是复制品。

**欠采样多数类：** 随机丢弃多数类样本。减少训练数据。可能导致信息丢失，因为丢弃了可能有用的多数类样本。

**组合（SMOTE + 欠采样，如 SMOTETomek）：** 对少数类进行过采样，对多数类进行欠采样。结合了两种方法的优点。

#### 2. 算法内方法：类别权重

修改损失函数以对少数类的错误分类施加更高的惩罚。

```python
weight_for_minority = n_majority / n_minority
weight_for_majority = 1.0

# 在交叉熵损失中使用这些权重
loss = -1/N * sum( w_i * (y_i * log(p_i) + (1-y_i) * log(1-p_i)) )
```

在 sklearn 中：

```python
LogisticRegression(class_weight="balanced")
RandomForestClassifier(class_weight="balanced")
SVC(class_weight="balanced")
```

`"balanced"` 权重自动与类别频率成反比：`weight = n / (n_classes * np.bincount(y))`。

类别权重将决策边界移近多数类，给少数类更多"空间"。

#### 3. 阈值调整

对于输出概率的分类器，默认阈值 0.5 假设类别平衡。如果不平衡，这个阈值是次优的。

```python
# 使用验证集找到最佳阈值
thresholds = np.linspace(0.1, 0.9, 100)
best_f1 = 0
best_threshold = 0.5

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t
```

将阈值设置为先验类别概率（如果少数类是 1%，阈值为 0.01）是一个合理的起点。然后使用 F1 或精确率-召回率曲线上的 AUPRC 进行调整。

### 何时使用每种策略

| 策略 | 数据量 | 不平衡比率 | 计算成本 |
|--------|------------|-----------------|--------------|
| 过采样 | 小 | 任何 | 低 |
| SMOTE | 中等 | < 1:100 | 中等 |
| 欠采样 | 大 | < 1:10 | 低 |
| 类别权重 | 任何 | 任何 | 无额外成本 |
| 阈值调整 | 足够用于验证 | 任何 | 低 |

**经验法则：**
- 从类别权重开始。不需要修改数据，与任何模型一起工作。
- 当类别权重不够时，添加 SMOTE。
- 始终在单独的验证集上优化阈值。这是最便宜且最有效的单一技术。

### 常见陷阱

**在重采样后评估。** 如果你对数据进行过采样，然后在原始测试集上评估，指标会被乐观地偏倚。始终在原始的、未修改的测试数据上评估。

**在验证期间过度优化阈值。** 如果你在验证集上调整阈值太多次，你会过拟合验证集。使用单独的保留校准集或交叉验证。

**忽略校准。** 类别权重和重采样改变预测概率的校准。如果需要良好的概率估计，在模型之上应用 Platt 缩放（CalibratedClassifierCV）。

**在极端不平衡下使用 AUC-ROC。** 当少数类非常罕见时，AUC-ROC 可能过于乐观，因为假正例率（FP / (FP + TN)）几乎总是很低。使用 AUPRC 代替，它关注少数类。

### 不平衡学习流水线

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

pipeline = ImbPipeline([
    ("sampler", SMOTE(sampling_strategy="auto")),
    ("classifier", RandomForestClassifier(class_weight="balanced")),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

`imblearn` 的管道确保仅在训练集内进行重采样，防止数据泄露。

## 构建它

`code/imbalanced_data.py` 中的代码从头实现了 SMOTE 以及处理不平衡数据的完整工作流。

### 第 1 步：生成不平衡数据

合成数据具有 5% 少数类比率。

### 第 2 步：SMOTE 实现

```python
def smote(X_minority, n_synthetic, k=5):
    synthetic = []
    for _ in range(n_synthetic):
        idx = np.random.randint(len(X_minority))
        point = X_minority[idx]
        dists = np.sqrt(np.sum((X_minority - point) ** 2, axis=1))
        dists[idx] = np.inf
        nearest = np.argsort(dists)[:k]
        neighbor = X_minority[nearest[np.random.randint(k)]]
        diff = neighbor - point
        synthetic.append(point + np.random.random() * diff)
    return np.array(synthetic)
```

### 第 3 步：比较不平衡场景下的指标

演示准确率、精确率、召回率、F1、MCC 和 AUPRC 如何对全多数基线做出反应。

```figure
class-imbalance
```

### 第 4 步：完整演示

将类别权重、SMOTE 和阈值优化应用于不平衡的欺诈检测数据集。

## 使用它

```python
# 类别权重
model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)

# SMOTE + 类别权重
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model = RandomForestClassifier(class_weight="balanced")
model.fit(X_resampled, y_resampled)

# 阈值优化
y_proba = model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
f1_scores = 2 * precision * recall / (precision + recall)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
y_pred = (y_proba >= best_threshold).astype(int)
```

## 交付物

本课程产出：
- `outputs/skill-imbalanced-learner.md`——用于处理不平衡分类问题的技能

## 练习

1. 生成具有 1% 少数类的合成数据集。训练一个逻辑回归，使用默认阈值 0.5 和根据少数类频率调整的阈值。在 F1 和 MCC 上比较它们。
2. 从头实现 SMOTE 并将其应用于不平衡数据集。将结果与 sklearn 的 SMOTE 进行比较。可视化重采样前后的决策边界。
3. 使用带有类别权重的随机森林并运行前向验证。在没有任何类别权重的情况下重复。类别权重产生多少改进？
4. 在高度不平衡的数据上绘制精确率-召回率曲线。计算 AUPRC 并与 AUC-ROC 进行比较。哪个能更好地捕捉模型性能？
5. 下载真实不平衡数据集（例如，信用卡欺诈检测数据集）。应用 SMOTE + 类别权重 + 阈值优化。以逐步改善的方式记录每个技术带来的 F1 提升。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| 类别不平衡 | "类大小不等" | 一些类别在其他类别数量上远超其他的分布 |
| SMOTE | "合成少数类样本" | 通过在现有少数类样本之间插值创建新的少数类样本 |
| 过采样 | "复制少数类" | 通过复制或合成增加少数类的样本数量 |
| 欠采样 | "丢弃多数类" | 通过随机丢弃减少多数类的样本数量 |
| 类别权重 | "给少数类更高权重" | 在损失函数中增加少数类错误分类的成本 |
| 阈值调整 | "改变决策边界" | 移动决策阈值以更偏好少数类 |
| AUPRC | "精确率-召回率曲线下面积" | 对所有阈值的精确率 vs 召回率绘图；对不平衡数据优于 AUC-ROC |
| MCC | "平衡的相关系数" | Mathews 相关系数；从 -1 到 +1 的单一指标，平衡两类大小 |
| 精确率 | "正例预测的准确性" | TP / (TP + FP) |
| 召回率 | "对正例的覆盖" | TP / (TP + FN) |

## 延伸阅读

- [Chawla et al., SMOTE (2002)](https://www.jair.org/index.php/jair/article/view/10302)
- [imbalanced-learn 文档](https://imbalanced-learn.org/stable/)
- [He & Garcia, Learning from Imbalanced Data (2009)](https://ieeexplore.ieee.org/document/5128907)
