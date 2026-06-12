# ML 流水线

> 模型不是产品。流水线才是。流水线是从原始数据到部署预测的全过程，每一步都必须是可复现的。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 2，课程 12（超参数调优）
**时间：** ~120 分钟

## 学习目标

- 从头构建一个将插补、缩放、编码和模型训练链接到单个可复现对象中的 ML 流水线
- 识别数据泄露场景，并解释流水线如何通过仅在训练数据上拟合变换器来防止数据泄露
- 构建一个对数值和类别特征应用不同预处理的 ColumnTransformer
- 实现流水线序列化，并证明同一拟合流水线在训练和生产中产生相同的结果

## 问题

你有一个笔记本，它加载数据、用中位数填补缺失值、缩放特征、训练模型并输出准确率。它工作正常。你交付了它。

一个月后，有人重新训练模型并得到不同的结果。中位数是在包含测试数据的完整数据集上计算的（数据泄露）。缩放参数没有保存，所以推断使用了不同的统计量。特征工程代码在训练和服务之间被复制粘贴，副本发生了分歧。一个类别列在生产中出现了编码器从未见过的新值。

这些不是假设。它们是 ML 系统在生产中失败的最常见原因。流水线通过将每个变换步骤打包成一个单一的、有序的、可复现的对象来解决所有这些问题。

## 概念

### 什么是流水线

流水线是一系列有序的数据变换步骤后跟一个模型。每个步骤接收上一步的输出作为输入。整个流水线在训练数据上拟合一次。在推断时，同一个拟合的流水线变换新数据并产生预测。

```mermaid
flowchart LR
    A[原始数据] --> B[填补缺失值]
    B --> C[缩放数值特征]
    C --> D[编码类别特征]
    D --> E[训练模型]
    E --> F[预测]
```

流水线保证：
- 变换仅在训练数据上拟合（无泄露）
- 在推断时应用相同的变换
- 整个对象可以作为单个工件序列化和部署
- 交叉验证按折应用流水线，防止微妙的泄露

### 数据泄露：沉默的杀手

当测试集或未来数据的信息污染训练时，就会发生数据泄露。流水线防止最常见的形式。

**有泄露（错误）：**
```python
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test = X_scaled[:800], X_scaled[800:]
y_train, y_test = y[:800], y[800:]
```

缩放器看到了测试数据。均值和标准差包括了测试样本。这夸大了准确率估计。

**正确：**
```python
X_train, X_test = X[:800], X[800:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

使用流水线，你不需要考虑这些。流水线自动处理。

### sklearn 流水线

sklearn 的 `Pipeline` 将变换器和估计器串联起来。它暴露 `.fit()`、`.predict()` 和 `.score()` 方法，这些方法按顺序应用所有步骤。

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

当你调用 `pipe.fit(X_train, y_train)` 时：
1. 缩放器在 X_train 上调用 `fit_transform`
2. 模型在缩放后的 X_train 上调用 `fit`

当你调用 `pipe.predict(X_test)` 时：
1. 缩放器在 X_test 上调用 `transform`（不是 fit_transform）
2. 模型在缩放后的 X_test 上调用 `predict`

缩放器在拟合期间从未看到测试数据。这就是全部要点。

### ColumnTransformer：不同列的不同流水线

真实数据集有需要不同预处理的数值和类别列。`ColumnTransformer` 处理这个问题。

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["age", "income", "score"]),
    ("cat", categorical_pipe, ["city", "gender", "plan"]),
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", GradientBoostingClassifier()),
])
```

OneHotEncoder 中的 `handle_unknown="ignore"` 对于生产环境至关重要。当新类别出现时（模型从未见过的城市），它产生一个零向量而不是崩溃。

### 实验追踪

流水线使训练可复现，但你还需要跨实验追踪发生了什么：使用了哪些超参数、哪个数据集版本、指标是什么、运行的哪个代码。

**MLflow** 是最常见的开源解决方案：

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)

    pipe.fit(X_train, y_train)
    accuracy = pipe.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pipe, "model")
```

每次运行都记录了参数、指标、工件和完整模型。你可以比较运行、复现任何实验并部署任何模型版本。

**Weights & Biases (wandb)** 提供了带有托管仪表板的相同功能：

```python
import wandb

wandb.init(project="my-pipeline")
wandb.config.update({"max_depth": 5, "n_estimators": 100})

pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)

wandb.log({"accuracy": accuracy})
```

### 模型版本管理

在实验追踪之后，你需要管理模型版本。哪个模型在生产中？哪个在预发布？哪个是上周的？

MLflow 的模型注册表提供：
- **版本追踪：** 每个保存的模型获得一个版本号
- **阶段转换：** "预发布"、"生产"、"已归档"
- **审批工作流：** 模型必须被显式提升到生产
- **回滚：** 即时切换回之前的版本

### 使用 DVC 进行数据版本管理

代码用 git 进行版本管理。数据也应该版本管理，但 git 无法处理大文件。DVC（数据版本控制）解决了这个问题。

```
dvc init
dvc add data/training.csv
git add data/training.csv.dvc data/.gitignore
git commit -m "追踪训练数据"
dvc push
```

DVC 将实际数据存储在远程存储（S3、GCS、Azure）中，并在 git 中保存一个记录哈希值的 `.dvc` 小文件。当你检出 git 提交时，`dvc checkout` 恢复使用的确切数据。

这意味着每个 git 提交既固定了代码也固定了数据。完全可复现。

### 可复现的实验

可复现的实验需要四样东西：

1. **固定的随机种子：** 为 numpy、random 和框架（torch、sklearn）设置种子
2. **固定的依赖：** requirements.txt 或 poetry.lock 带有精确版本
3. **版本化的数据：** DVC 或类似工具
4. **配置文件：** 所有超参数在配置中，而不是硬编码

```python
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
```

### 从笔记本到生产流水线

```mermaid
flowchart TD
    A[Jupyter 笔记本] --> B[提取函数]
    B --> C[构建流水线对象]
    C --> D[为超参数添加配置文件]
    D --> E[添加实验追踪]
    E --> F[添加数据验证]
    F --> G[添加测试]
    G --> H[打包用于部署]

    style A fill:#fdd,stroke:#333
    style H fill:#dfd,stroke:#333
```

典型的演进过程：

1. **笔记本探索：** 快速实验、可视化、特征想法
2. **提取函数：** 将预处理、特征工程、评估移动到模块中
3. **构建流水线：** 将变换链接到 sklearn Pipeline 或自定义类中
4. **配置管理：** 将所有超参数移到 YAML/JSON 配置中
5. **实验追踪：** 添加 MLflow 或 wandb 日志记录
6. **数据验证：** 在训练前检查模式、分布和缺失值模式
7. **测试：** 变换器的单元测试，完整流水线的集成测试
8. **部署：** 序列化流水线，包装在 API（FastAPI、Flask）中，容器化

### 常见流水线错误

| 错误 | 为什么糟糕 | 修复 |
|---------|-------------|-----|
| 在切分前对整个数据拟合 | 数据泄露 | 使用 Pipeline 配合 cross_val_score |
| 在流水线外进行特征工程 | 训练 vs 服务不同变换 | 将所有变换放在 Pipeline 中 |
| 不处理未知类别 | 生产中出现新值时崩溃 | OneHotEncoder(handle_unknown="ignore") |
| 硬编码列名 | 模式更改时崩溃 | 从配置中使用列名列表 |
| 无数据验证 | 坏数据上静默错误的预测 | 在预测前添加模式检查 |
| 训练/服务偏差 | 模型在生产中看到不同特征 | 一个 Pipeline 对象用于两者 |

## 构建它

`code/pipeline.py` 中的代码从头构建了一个完整的 ML 流水线：

### 第 1 步：自定义变换器

```python
class CustomTransformer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

### 第 2 步：从头实现流水线

```python
class PipelineFromScratch:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.fit_transform(X_current)
        name, model = self.steps[-1]
        model.fit(X_current, y)
        return self

    def predict(self, X):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)
        name, model = self.steps[-1]
        return model.predict(X_current)
```

### 第 3 步：带流水线的交叉验证

代码演示了带有流水线的交叉验证如何防止数据泄露：缩放器在每个折的训练数据上单独拟合。

### 第 4 步：使用 sklearn 的完整生产流水线

一个包含 `ColumnTransformer`、多条预处理路径和一个模型的完整流水线，通过适当的交叉验证和实验日志记录进行训练。

## 交付物

本课程产出：
- `outputs/prompt-ml-pipeline.md`——构建和调试 ML 流水线的技能
- `code/pipeline.py`——从头到 sklearn 的完整流水线

## 练习

1. 构建一个处理 3 个数值列和 2 个类别列的数据集的流水线。使用 `ColumnTransformer` 对数值列应用中位数插补 + 缩放，对类别列应用众数插补 + 独热编码。使用 5 折交叉验证训练。
2. 故意引入数据泄露：在切分前对整个数据集拟合缩放器。比较交叉验证分数（有泄露的）和流水线交叉验证分数（干净的）。差异有多大？
3. 使用 `joblib.dump` 序列化你的流水线。在单独的脚本中加载它并运行预测。验证预测一致。
4. 向流水线添加一个自定义变换器，为最重要的两个数值列创建多项式特征（2 次）。它应该在流水线中的哪个位置？
5. 为流水线设置 MLflow 追踪。使用不同超参数运行 5 次实验。使用 MLflow UI（`mlflow ui`）比较运行并选择最佳模型。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| 流水线 | "变换链 + 模型" | 拟合的变换器和模型的有序序列，作为一个单元应用以防止泄露 |
| 数据泄露 | "测试信息泄露到训练" | 使用训练集之外的信息来构建模型，夸大性能估计 |
| ColumnTransformer | "每列不同预处理" | 对不同列子集应用不同流水线，组合结果 |
| 实验追踪 | "记录你的运行" | 记录每次训练运行的参数、指标、工件和代码版本 |
| MLflow | "追踪和部署模型" | 用于实验追踪、模型注册表和部署的开源平台 |
| DVC | "数据的 Git" | 大文件的版本控制系统，将哈希值存储在 git 中，数据存储在远程存储中 |
| 模型注册表 | "模型版本目录" | 跟踪模型版本及其阶段标签（预发布、生产、已归档）的系统 |
| 训练/服务偏差 | "它在笔记本里能工作" | 训练和推理期间数据处理方式的差异，导致静默错误 |
| 可复现性 | "相同代码，相同结果" | 从相同代码、数据和配置获得相同结果的能力 |

## 延伸阅读

- [scikit-learn 流水线文档](https://scikit-learn.org/stable/modules/compose.html)——官方的流水线参考
- [MLflow 文档](https://mlflow.org/docs/latest/index.html)——实验追踪和模型注册表
- [DVC 文档](https://dvc.org/doc)——数据版本管理
- [Sculley 等人，机器学习系统中的隐藏技术债（2015）](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)——关于 ML 系统复杂性的开创性论文
- [Google ML 最佳实践：ML 规则](https://developers.google.com/machine-learning/guides/rules-of-ml)——实用的生产 ML 建议
