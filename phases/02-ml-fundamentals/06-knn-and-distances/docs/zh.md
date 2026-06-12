# K 近邻与距离度量

> 存储所有数据。通过观察邻居来预测。最简单且真正有效的算法。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 1（课程 14 范数与距离）
**时间：** ~90 分钟

## 学习目标

- 从零开始实现 KNN 分类和回归，支持可配置的 K 值和距离加权投票
- 比较 L1、L2、余弦和闵可夫斯基距离度量，并为给定数据类型选择合适的度量
- 解释维度灾难，并演示为什么 KNN 在高维空间中性能下降
- 构建 KD 树用于高效的最近邻搜索，并分析它何时优于暴力搜索

## 问题

你有一个数据集。一个新的数据点来了。你需要对其进行分类或预测其值。与其从数据中学习参数（像线性回归或 SVM 那样），你只需找到离新点最近的 K 个训练点，让它们投票。

这就是 K 近邻。没有训练阶段。没有参数需要学习。没有损失函数需要最小化。你存储整个训练集，在预测时计算距离。

它听起来简单得不可能有效。但 KNN 对许多问题出奇地有竞争力，尤其是中小型数据集，深入理解它揭示了基本概念：距离度量的选择（连接到阶段 1 课程 14）、维度灾难以及懒惰学习与急切学习的区别。

KNN 在现代 AI 中也随处可见，只是名字不同。向量数据库对嵌入做 KNN 搜索。检索增强生成（RAG）找到 K 个最近的文档块。推荐系统找到相似的用户或物品。算法是一样的。尺度和数据结构不同。

## 概念

### KNN 的工作原理

给定一个带标签的数据集和一个新的查询点：

1. 计算查询点到数据集中每个点的距离
2. 按距离排序
3. 取 K 个最近的点
4. 对于分类：K 个邻居的多数投票
5. 对于回归：K 个邻居值的平均（或加权平均）

这就是整个算法。没有拟合。没有梯度下降。没有迭代轮次。

### 选择 K

K 是唯一的超参数。它控制偏差-方差权衡：

| K | 行为 |
|---|------|
| K = 1 | 决策边界跟随每个点。零训练误差。高方差。过拟合 |
| 小 K (3-5) | 对局部结构敏感。可捕获复杂边界 |
| 大 K | 更平滑的边界。对噪声更鲁棒。可能欠拟合 |
| K = N | 对每个点预测多数类别。最大偏差 |

常用起点是 K = sqrt(N)（N 为数据集大小）。二分类用奇数 K 以避免平局。

### 距离度量

距离函数定义了"近"的含义。不同的度量产生不同的邻居和不同的预测。

**L2（欧几里得距离）** 是默认值。直线距离。对特征尺度敏感。使用 L2 与 KNN 之前一定要标准化特征。

**L1（曼哈顿距离）** 求绝对差值之和。对异常值比 L2 更鲁棒，因为它不平方差值。

**余弦距离** 衡量向量之间的角度，忽略幅度。对于文本和嵌入数据至关重要。

**闵可夫斯基距离** 用参数 p 推广了 L1 和 L2。

### 加权 KNN

标准 KNN 对所有 K 个邻居赋予相等的权重。但距离为 0.1 的邻居应该比距离为 5.0 的更重要。

距离加权 KNN 按距离的倒数对每个邻居加权：

```
weight_i = 1 / (distance_i + epsilon)
对于分类：加权投票
对于回归：加权平均 = sum(w_i * y_i) / sum(w_i)
```

Epsilon 防止查询点与训练点完全匹配时除以零。

### 维度灾难

KNN 性能在高维空间中下降。这不是一个模糊的担忧，而是一个数学事实。

**问题 1：距离趋同。** 随着维度增加，最大距离与最小距离之比趋近于 1。所有点都变得与查询点同样"远"。

**问题 2：体积爆炸。** 要在一个固定的数据比例内捕获 K 个邻居，你需要将搜索半径扩展到覆盖特征空间的更大比例。

**问题 3：角落主导。** 在 d 维单位超立方体中，大部分体积集中在角落附近，而不是中心。

实际后果：KNN 在约 20-50 个特征以内效果良好。超过这个范围，你需要在应用 KNN 之前进行降维（PCA、UMAP、t-SNE），或者使用利用数据固有低维度的基于树的搜索结构。

### KD 树：快速最近邻搜索

暴力 KNN 计算查询点到每个训练点的距离。每次查询 O(n * d)。对于大数据集来说太慢了。

KD 树沿特征轴递归划分空间。在每一层，它沿一个维度在中位数处分割。

平均查询时间：低维时为 O(log n)。但 KD 树在高维（d > 20）时会退化为 O(n)，因为回溯消除的分支越来越少。

## 构建它

代码包含在 `code/knn.py` 中。实现包括完整的 KNN 分类器和回归器、KD 树和特征缩放。

## 使用它

用 scikit-learn：

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
])
clf.fit(X_train, y_train)
print(f"准确率: {clf.score(X_test, y_test):.4f}")
```

对于大规模最近邻搜索（数百万个向量），使用 FAISS、Annoy 或向量数据库：

```python
import faiss

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
distances, indices = index.search(query_vectors, k=5)
```

## 关键术语

| 术语 | 含义 |
|------|------|
| K 近邻 | 通过找到离查询点最近的 K 个训练点进行预测的非参数算法 |
| 懒惰学习 | 训练时不进行计算。所有工作在预测时完成。KNN 是典型例子 |
| 维度灾难 | 在高维中，距离趋同且邻域扩展到覆盖大部分空间，使 KNN 失效 |
| KD 树 | 沿特征轴递归划分空间的二叉树。低维时 O(log n) 查询 |
| 加权 KNN | 按距离倒数加权的邻居。更近的邻居对预测有更大影响 |
| 特征缩放 | 将特征归一化到可比范围。对 KNN 等基于距离的方法是必需的 |

## 延伸阅读

- [Cover & Hart: Nearest Neighbor Pattern Classification (1967)](https://ieeexplore.ieee.org/document/1053964)——基础的 KNN 论文
- [Friedman, Bentley, Finkel: An Algorithm for Finding Best Matches in Logarithmic Expected Time (1977)](https://dl.acm.org/doi/10.1145/355744.355745)——原始 KD 树论文
- [scikit-learn Nearest Neighbors 文档](https://scikit-learn.org/stable/modules/neighbors.html)——带有算法选择的实用指南
- [FAISS](https://github.com/facebookresearch/faiss)——Meta 的大规模近似最近邻搜索库
