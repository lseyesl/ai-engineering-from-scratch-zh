# 特征工程与选择

> 一个好的特征胜过一千个数据点。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 1（ML 统计、线性代数），阶段 2 课程 1-7
**时间：** ~90 分钟

## 学习目标

- 实现数值变换（标准化、最小-最大缩放、对数变换、分箱）并解释每种方法适用的场景
- 为类别特征构建独热编码、标签编码和目标编码，并识别目标编码中的数据泄露风险
- 从头构建 TF-IDF 向量化器，并解释为什么它在文本分类中优于原始词频
- 应用基于过滤器的特征选择（方差阈值、相关性、互信息）来降低维度

## 问题

你有一个数据集。你选了一个算法。你训练了它。结果平庸。你尝试更花哨的算法。仍然平庸。你花了一周调超参数。改善甚微。

然后有人将原始数据转换成更好的特征，一个简单的逻辑回归就击败了你调优过的梯度提升集成。

这种情况一直在发生。在经典 ML 中，数据的表示比算法的选择更重要。一个使用"平方英尺"和"卧室数量"的房价模型，无论学习器多么复杂，都会击败使用"原始地址字符串"的模型。算法只能处理你给它的东西。

特征工程是将原始数据转换为更容易让模型发现模式的表示的过程。特征选择是丢弃那些增加噪声而不增加信号的特征的过程。两者加在一起，是经典 ML 中最高杠杆的活动。

## 概念（概要）

### 数值特征

原始数字很少是模型就绪的。常见变换：缩放（标准化、最小-最大）、对数变换（压缩右偏分布）、分箱（将连续值转换为类别）、多项式特征。

### 类别特征

模型需要数字。类别需要编码：独热编码（每个类别创建一个二值列）、标签编码（将类别映射为整数）、目标编码（用目标均值替换类别）。

### 文本特征

词频向量化：统计每个词在文档中出现的次数。TF-IDF：词频-逆文档频率，根据词在整个语料库中的独特性加权。

### 缺失值处理

删除行、均值/中位数填充、众数填充、指示列、前向/后向填充（时间序列数据）。

### 特征选择

过滤方法（模型前）：相关性、互信息、方差阈值。包装方法（基于模型）：L1 正则化（Lasso）、递归特征消除。

## 构建它

代码包含在 `code/feature_engineering.py` 中。实现了数值变换、类别编码、TF-IDF、缺失值填充、互信息和相关性分析。

## 使用它

用 scikit-learn，这些变换可以组合成管道：

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("encoder", OneHotEncoder(sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["sqft", "age"]),
    ("cat", categorical_pipe, ["neighborhood"]),
])
```

## 关键术语

| 术语 | 含义 |
|------|------|
| 特征工程 | 将原始数据转换为能向模型暴露模式的表示 |
| 标准化 | 减去均值并除以标准差，使特征具有 mean=0, std=1 |
| 独热编码 | 每个类别创建一个二值列，每行恰好有一个列是 1 |
| 目标编码 | 用该类别的目标平均值替换每个类别，使用平滑防止过拟合 |
| TF-IDF | 词频乘以逆文档频率，根据词在整个语料库中的独特性加权 |
| 填充 | 用估计值（均值、中位数、众数或模型预测值）替换缺失值 |
| 特征选择 | 移除增加噪声或冗余的特征，只保留那些与目标有信号关系的 |
| 互信息 | 衡量观测变量 X 时变量 Y 的不确定性减少量 |
| 数据泄露 | 在训练过程中使用了预测时无法获得的信息，给出虚假的乐观结果 |

## 延伸阅读

- [Feature Engineering and Selection (Max Kuhn & Kjell Johnson)](http://www.feat.engineering/)——免费在线书籍
- [scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)——所有标准变换的实用参考
