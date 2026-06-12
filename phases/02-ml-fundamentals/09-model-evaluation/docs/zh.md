# 模型评估

> 模型的好坏取决于你衡量它的方式。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 1（概率与分布、ML 统计），阶段 2 课程 1-8
**时间：** ~90 分钟

## 学习目标

- 从头实现 K 折和分层 K 折交叉验证，并解释为什么分层对不平衡数据至关重要
- 从头计算精确率、召回率、F1、AUC-ROC 以及回归指标（MSE、RMSE、MAE、R 平方）
- 解释学习曲线以诊断模型是否遭受高偏差或高方差
- 识别常见的评估错误，包括数据泄露、错误指标选择以及测试集污染

## 问题

你训练了一个模型。它在你的数据上达到 95% 的准确率。它好吗？

可能好，也可能不好。如果你的数据中 95% 属于一个类别，那么一个总是预测该类别的模型就能达到 95% 的准确率，同时完全没用。如果你在训练集上评估，那么 95% 这个数字毫无意义，因为模型只是记住了答案。如果你的数据集有时间分量，而你在切分之前随机打乱了数据，那么你的模型可能用未来的数据预测过去。

模型评估是大多数 ML 项目出错的地方。错误的指标让坏模型看起来很好。错误的切分让模型作弊。错误的比较让你选择了更差的模型。把评估做对不是可选项。它是生产环境可用模型与一看到真实数据就失败的模型之间的分水岭。

## 概念

### 训练集、验证集、测试集

```mermaid
flowchart LR
    A[完整数据集] --> B[训练集 60-70%]
    A --> C[验证集 15-20%]
    A --> D[测试集 15-20%]
    B --> E[拟合模型]
    E --> C
    C --> F[调优超参数]
    F --> E
    F --> G[最终模型]
    G --> D
    D --> H[报告性能]
```

三个切分，三个目的：

- **训练集**：模型从这些数据中学习。模型在训练期间看到这些样本。
- **验证集**：用于调优超参数和在模型之间做选择。模型从未在这些数据上训练，但你的决策受其影响。
- **测试集**：仅在最后被触碰一次，用于报告最终性能。如果你看了测试性能然后回去修改模型，那它就不再是测试集了。它已经变成了第二个验证集。

测试集是你的保留保证——报告的性能反映了模型在真正未见数据上的表现。

### K 折交叉验证

对于小数据集，单次训练/验证切分会浪费数据并给出有噪声的估计。K 折交叉验证使用所有数据进行训练和验证：

```mermaid
flowchart TB
    subgraph 折1["折 1"]
        direction LR
        V1["验证"] --- T1a["训练"] --- T1b["训练"] --- T1c["训练"] --- T1d["训练"]
    end
    subgraph 折2["折 2"]
        direction LR
        T2a["训练"] --- V2["验证"] --- T2b["训练"] --- T2c["训练"] --- T2d["训练"]
    end
    subgraph 折3["折 3"]
        direction LR
        T3a["训练"] --- T3b["训练"] --- V3["验证"] --- T3c["训练"] --- T3d["训练"]
    end
    subgraph 折4["折 4"]
        direction LR
        T4a["训练"] --- T4b["训练"] --- T4c["训练"] --- V4["验证"] --- T4d["训练"]
    end
    subgraph 折5["折 5"]
        direction LR
        T5a["训练"] --- T5b["训练"] --- T5c["训练"] --- T5d["训练"] --- V5["验证"]
    end
    折1 --> R["平均分数"]
    折2 --> R
    折3 --> R
    折4 --> R
    折5 --> R
```

1. 将数据分成 K 个大小相等的折
2. 对于每个折，在 K-1 折上训练，在剩余一折上验证
3. 平均 K 个验证分数

K=5 或 K=10 是标准选择。每个数据点恰好被用于验证一次。平均分数比任何单次切分都更稳定。

**分层 K 折**：在每个折中保持类别分布。如果数据集是 70% 类别 A 和 30% 类别 B，每个折将大致具有相同的比例。这对于不平衡数据集很重要，因为随机切分可能会将少数类样本全部放在一个折中。

### 分类指标

**混淆矩阵**：基础。对于二分类：

|  | 预测为正 | 预测为负 |
|--|---|---|
| 实际为正 | 真正例 (TP) | 假负例 (FN) |
| 实际为负 | 假正例 (FP) | 真负例 (TN) |

从这个矩阵出发，所有其他指标随之而来：

- **准确率** = (TP + TN) / (TP + TN + FP + FN)。正确预测的比例。当类别不平衡时具有误导性。
- **精确率** = TP / (TP + FP)。在被预测为正例的所有样本中，有多少是真正的正例？当假正例代价高时使用（例如，垃圾邮件过滤器将真实邮件标记为垃圾邮件）。
- **召回率**（灵敏度）= TP / (TP + FN)。在所有实际正例中，我们找出了多少？当假负例代价高时使用（例如，癌症筛查漏检肿瘤）。
- **F1 分数** = 2 * 精确率 * 召回率 / (精确率 + 召回率)。精确率和召回率的调和平均数。当两者都不明显占优时平衡两者。
- **AUC-ROC**：受试者工作特征曲线下面积。绘制不同分类阈值下的真正例率对假正例率的曲线。AUC = 0.5 表示随机猜测，AUC = 1.0 表示完美分离。与阈值无关：它衡量模型将正例排列在负例之上的能力，无论你选择哪个截断值。

### 回归指标

- **MSE**（均方误差）= mean((y_true - y_pred)^2)。二次惩罚大误差。对异常值敏感。
- **RMSE**（均方根误差）= sqrt(MSE)。与目标变量单位相同。比 MSE 更容易解释。
- **MAE**（平均绝对误差）= mean(|y_true - y_pred|)。线性处理所有误差。比 MSE 对异常值更鲁棒。
- **R 平方** = 1 - SS_res / SS_tot，其中 SS_res = sum((y_true - y_pred)^2)，SS_tot = sum((y_true - y_mean)^2)。模型解释的方差比例。R² = 1.0 完美。R² = 0.0 表示模型不比总预测均值好。如果模型比均值更差，R² 可能为负。

### 学习曲线

绘制训练和验证分数随训练集大小的变化：

- **高偏差（欠拟合）**：两条曲线都收敛到低分。增加更多数据没有帮助。你需要更复杂的模型。
- **高方差（过拟合）**：训练分数高但验证分数低得多。两者之间的差距很大。增加更多数据应该有帮助。

### 验证曲线

绘制训练和验证分数随超参数的变化：

- 低复杂度：两个分数都低（欠拟合）
- 合适复杂度：两个分数都高且接近
- 高复杂度：训练分数保持高位但验证分数下降（过拟合）

最优超参数值是验证分数峰值所在的位置。

### 常见评估错误

**数据泄露**：测试集的信息泄露到训练中。例如：在切分之前对整个数据集拟合缩放器、在时间序列预测中包含未来数据、使用从目标派生出的特征。总是先切分，再预处理。

**类别不平衡**：99% 的交易是合法的，1% 是欺诈。一个总是预测"合法"的模型达到 99% 的准确率。使用精确率、召回率、F1 或 AUC-ROC 代替。

**错误的指标**：在应该优化召回率时优化准确率（医学诊断），或者在你的数据有严重异常值时优化 RMSE（改用 MAE）。

**不使用分层切分**：对于不平衡数据，随机切分可能将很少的少数类样本放入验证折，给出不稳定的估计。

**测试过于频繁**：每次你查看测试性能并进行调整，你就在过拟合测试集。测试集是一次性的。

```figure
precision-recall-threshold
```

## 构建它

### 第 1 步：训练/验证/测试切分

```python
import random
import math


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
```

### 第 2 步：K 折和分层 K 折交叉验证

```python
def kfold_split(n, k=5, seed=42):
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))

    return folds


def stratified_kfold_split(y, k=5, seed=42):
    random.seed(seed)

    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    for label in class_indices:
        random.shuffle(class_indices[label])

    folds = [{"train": [], "val": []} for _ in range(k)]

    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(indices)
            val_part = indices[start:end]
            train_part = indices[:start] + indices[end:]
            folds[i]["val"].extend(val_part)
            folds[i]["train"].extend(train_part)

    return [(f["train"], f["val"]) for f in folds]


def cross_validate(X, y, model_fn, k=5, metric_fn=None, stratified=False):
    n = len(X)

    if stratified:
        folds = stratified_kfold_split(y, k)
    else:
        folds = kfold_split(n, k)

    scores = []
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        predictions = [model.predict(x) for x in X_val]

        if metric_fn:
            score = metric_fn(y_val, predictions)
        else:
            score = sum(1 for yt, yp in zip(y_val, predictions) if yt == yp) / len(y_val)
        scores.append(score)

    return scores
```

### 第 3 步：混淆矩阵和分类指标

```python
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_curve(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list = []
    fpr_list = []

    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives

    for threshold in thresholds:
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)

        tpr = tp / total_positives if total_positives > 0 else 0.0
        fpr = fp / total_negatives if total_negatives > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list, thresholds


def auc_roc(y_true, y_scores):
    fpr_list, tpr_list, _ = roc_curve(y_true, y_scores)

    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    area = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i - 1]
        height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        area += width * height

    return area
```

### 第 4 步：回归指标

```python
def mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def r_squared(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot
```

### 第 5 步：学习曲线

```python
def learning_curve(X, y, model_fn, metric_fn, train_sizes=None, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    pool_idx = indices[val_size:]

    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    if train_sizes is None:
        train_sizes = [int(len(pool_idx) * r) for r in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]

    train_scores = []
    val_scores = []

    for size in train_sizes:
        subset = pool_idx[:size]
        X_train = [X[i] for i in subset]
        y_train = [y[i] for i in subset]

        model = model_fn()
        model.fit(X_train, y_train)

        train_pred = [model.predict(x) for x in X_train]
        val_pred = [model.predict(x) for x in X_val]

        train_scores.append(metric_fn(y_train, train_pred))
        val_scores.append(metric_fn(y_val, val_pred))

    return train_sizes, train_scores, val_scores
```

### 第 6 步：用于测试的简单分类器，以及完整演示

```python
class SimpleLogistic:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                pred = self.sigmoid(z)
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j]
                self.bias += self.lr * error

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self.sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0


class SimpleLinearRegression:
    def __init__(self, lr=0.001, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(X)

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j] / n
                self.bias += self.lr * error / n

    def predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var) if var > 0 else 1.0
    return [(v - mean) / std for v in values], mean, std


def make_classification_data(n=300, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        label = 1 if (x1 + x2 + random.gauss(0, 0.5)) > 0 else 0
        X.append([x1, x2])
        y.append(label)
    return X, y


def make_regression_data(n=200, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 5)
        target = 3 * x1 + 2 * x2 + random.gauss(0, 2)
        X.append([x1, x2])
        y.append(target)
    return X, y


def make_imbalanced_data(n=300, minority_ratio=0.05, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        if random.random() < minority_ratio:
            x1 = random.gauss(3, 0.5)
            x2 = random.gauss(3, 0.5)
            label = 1
        else:
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            label = 0
        X.append([x1, x2])
        y.append(label)
    return X, y


if __name__ == "__main__":
    X_clf, y_clf = make_classification_data(300)

    print("=== 训练/验证/测试切分 ===")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_clf, y_clf)
    print(f"  训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
    print(f"  训练集类别分布: {sum(y_train)}/{len(y_train)} 正例")
    print(f"  验证集类别分布: {sum(y_val)}/{len(y_val)} 正例")

    model = SimpleLogistic(lr=0.1, epochs=200)
    model.fit(X_train, y_train)

    print("\n=== 分类指标 ===")
    y_pred = [model.predict(x) for x in X_test]
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    print(f"  混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  准确率:  {accuracy(y_test, y_pred):.4f}")
    print(f"  精确率: {precision(y_test, y_pred):.4f}")
    print(f"  召回率:    {recall(y_test, y_pred):.4f}")
    print(f"  F1 分数:  {f1_score(y_test, y_pred):.4f}")

    y_scores = [model.predict_proba(x) for x in X_test]
    auc = auc_roc(y_test, y_scores)
    print(f"  AUC-ROC:   {auc:.4f}")

    print("\n=== K 折交叉验证 (K=5) ===")
    cv_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
    )
    mean_cv = sum(cv_scores) / len(cv_scores)
    std_cv = math.sqrt(sum((s - mean_cv) ** 2 for s in cv_scores) / len(cv_scores))
    print(f"  折分数: {[round(s, 4) for s in cv_scores]}")
    print(f"  均值: {mean_cv:.4f} (+/- {std_cv:.4f})")

    print("\n=== 分层 K 折交叉验证 (K=5) ===")
    strat_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
        stratified=True,
    )
    strat_mean = sum(strat_scores) / len(strat_scores)
    strat_std = math.sqrt(sum((s - strat_mean) ** 2 for s in strat_scores) / len(strat_scores))
    print(f"  折分数: {[round(s, 4) for s in strat_scores]}")
    print(f"  均值: {strat_mean:.4f} (+/- {strat_std:.4f})")

    print("\n=== 不平衡数据：为什么准确率会说谎 ===")
    X_imb, y_imb = make_imbalanced_data(300, minority_ratio=0.05)
    positives = sum(y_imb)
    print(f"  类别分布: {positives} 正例, {len(y_imb) - positives} 负例 ({positives/len(y_imb)*100:.1f}% 正例)")

    always_negative = [0] * len(y_imb)
    print(f"  全负基线:")
    print(f"    准确率:  {accuracy(y_imb, always_negative):.4f}")
    print(f"    精确率: {precision(y_imb, always_negative):.4f}")
    print(f"    召回率:    {recall(y_imb, always_negative):.4f}")
    print(f"    F1 分数:  {f1_score(y_imb, always_negative):.4f}")

    X_tr_i, y_tr_i, X_v_i, y_v_i, X_te_i, y_te_i = train_val_test_split(X_imb, y_imb)
    model_imb = SimpleLogistic(lr=0.5, epochs=500)
    model_imb.fit(X_tr_i, y_tr_i)
    y_pred_imb = [model_imb.predict(x) for x in X_te_i]
    print(f"\n  在不平衡数据上训练的模型:")
    print(f"    准确率:  {accuracy(y_te_i, y_pred_imb):.4f}")
    print(f"    精确率: {precision(y_te_i, y_pred_imb):.4f}")
    print(f"    召回率:    {recall(y_te_i, y_pred_imb):.4f}")
    print(f"    F1 分数:  {f1_score(y_te_i, y_pred_imb):.4f}")

    print("\n=== 回归指标 ===")
    X_reg, y_reg = make_regression_data(200)

    col0 = [x[0] for x in X_reg]
    col1 = [x[1] for x in X_reg]
    col0_s, m0, s0 = standardize(col0)
    col1_s, m1, s1 = standardize(col1)
    X_reg_scaled = [[col0_s[i], col1_s[i]] for i in range(len(X_reg))]

    X_tr_r, y_tr_r, X_v_r, y_v_r, X_te_r, y_te_r = train_val_test_split(X_reg_scaled, y_reg)
    reg_model = SimpleLinearRegression(lr=0.01, epochs=500)
    reg_model.fit(X_tr_r, y_tr_r)
    y_pred_r = [reg_model.predict(x) for x in X_te_r]

    print(f"  MSE:       {mse(y_te_r, y_pred_r):.4f}")
    print(f"  RMSE:      {rmse(y_te_r, y_pred_r):.4f}")
    print(f"  MAE:       {mae(y_te_r, y_pred_r):.4f}")
    print(f"  R 平方: {r_squared(y_te_r, y_pred_r):.4f}")

    mean_baseline = [sum(y_tr_r) / len(y_tr_r)] * len(y_te_r)
    print(f"\n  均值基线:")
    print(f"    MSE:       {mse(y_te_r, mean_baseline):.4f}")
    print(f"    R 平方: {r_squared(y_te_r, mean_baseline):.4f}")

    print("\n=== 学习曲线 ===")
    sizes, train_sc, val_sc = learning_curve(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        metric_fn=accuracy,
    )
    print(f"  {'大小':>6} {'训练':>8} {'验证':>8}")
    for s, tr, va in zip(sizes, train_sc, val_sc):
        print(f"  {s:>6} {tr:>8.4f} {va:>8.4f}")

    print("\n=== 统计模型比较 ===")
    model_a_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=100),
        k=5, metric_fn=accuracy,
    )
    model_b_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=500),
        k=5, metric_fn=accuracy,
    )
    diffs = [a - b for a, b in zip(model_a_scores, model_b_scores)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))
    t_stat = mean_diff / (std_diff / math.sqrt(len(diffs))) if std_diff > 0 else 0.0
    print(f"  模型 A (100 轮) 均值: {sum(model_a_scores)/len(model_a_scores):.4f}")
    print(f"  模型 B (500 轮) 均值: {sum(model_b_scores)/len(model_b_scores):.4f}")
    print(f"  平均差异: {mean_diff:.4f}")
    print(f"  配对 t 统计量: {t_stat:.4f}")
    print(f"  (|t| > 2.78 表示在 p<0.05 且 df=4 时显著)")
```

## 使用它

使用 scikit-learn，评估内置于工作流程中：

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score,
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="f1")
```

从头实现的版本精确展示了交叉验证在做什么（没有魔法，只是循环和索引追踪）、每个指标如何计算（只是计数 TP/FP/TN/FN），以及为什么分层很重要（在每个折中保持类别比例）。库版本增加了并行性、更多评分选项以及与流水线的集成。

## 交付物

本课程产出：
- `outputs/skill-evaluation.md`——涵盖分类和回归模型评估策略的技能

## 练习

1. 实现精确率-召回率曲线：在不同阈值下绘制精确率与召回率。计算平均精确率（PR 曲线下面积）。在不平衡数据集上比较 PR 曲线和 ROC 曲线，并解释何时哪个更有效。
2. 构建嵌套交叉验证循环：外层循环评估模型性能，内层循环调优超参数。用它公平地比较两个模型，而不将验证数据泄露到评估中。
3. 实现用于模型比较的置换检验：打乱标签，重新训练，并测量性能。重复 100 次以构建零分布。计算观测模型性能相对于该分布的 p 值。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| 过拟合 | "记住训练数据" | 模型捕捉训练数据中的噪声，在训练集表现良好但在未见数据上表现差 |
| 交叉验证 | "在不同子集上测试" | 系统轮换哪些数据用于验证，平均所有轮换的结果 |
| 精确率 | "预测为正例中有多少正确" | TP / (TP + FP)：被预测为正例中实际为正例的比例 |
| 召回率 | "找出了多少实际正例" | TP / (TP + FN)：实际正例中被正确识别的比例 |
| AUC-ROC | "模型区分类别的能力" | 所有阈值下真正例率对假正例率的曲线下面积，从 0.5（随机）到 1.0（完美） |
| R 平方 | "解释了多少方差" | 1 -（残差平方和 / 总平方和）：模型捕捉的目标方差比例 |
| 数据泄露 | "模型作弊了" | 在训练中使用了预测时无法获得的信息，导致乐观的评估 |
| 学习曲线 | "性能如何随数据增加变化" | 训练和验证分数随训练集大小的图，揭示欠拟合或过拟合 |
| 分层切分 | "保持类别比例平衡" | 切分数据使每个子集具有与完整数据集相同的各类别比例 |

## 延伸阅读

- [scikit-learn 模型选择指南](https://scikit-learn.org/stable/model_selection.html)——交叉验证、指标和超参数调优的综合参考
- [超越准确率：精确率与召回率（Google ML 速成课程）](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)——带有交互示例的清晰解释
- [交叉验证程序综述（Arlot & Celisse, 2010）](https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.full)——关于不同 CV 策略何时以及为何有效的严谨论述
