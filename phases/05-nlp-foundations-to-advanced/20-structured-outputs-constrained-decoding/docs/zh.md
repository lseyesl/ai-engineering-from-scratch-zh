# 自然语言推理与文档分类

> "前提：退款已发放。假设：客户收到了钱。"——这是蕴含、矛盾还是中立？

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 2 · 14（逻辑回归），阶段 5 · 05（情感分析）
**时间：** 约 45 分钟

## 问题

自然语言推理（NLI，也称为文本蕴含识别）确定两个文本之间的关系：前提和假设。给定一个前提"退款已发放"，假设"客户收到了钱"是蕴含的（同义）、矛盾的（否定）还是中立的（未确定）？

NLI 是许多高级 NLP 任务的核心构建模块。它是评估模型理解能力的通用基准，也是文档分类（将文本分配给预定义类别）的基础。

## 概念

**NLI 三种关系：**
- **蕴含**：假设可以从前提中逻辑推导出来。"Refund issued" → "Customer got money back"
- **矛盾**：假设与前提在逻辑上不一致。"Refund issued" → "Customer still waiting"
- **中立**：假设既不是蕴含也不是矛盾的。"Refund issued" → "Customer called support"

**文档分类**在多标签文档分类中，通常隐式使用了 NLI——模型可以评估"文档 X 是否属于类别 Y"的问题。

**零样本分类**使用 NLI 模型进行无需训练的文档分类：将类别标签框架化为假设，并为每个文档确定前提与假设之间的关系。

## 构建

### 步骤 1：基于特征的 NLI 基线

```python
class NLIOverlapBaseline:
    def overlap_score(self, premise, hypothesis):
        p_words = set(premise.lower().split())
        h_words = set(hypothesis.lower().split())
        overlap = len(p_words & h_words)
        total = len(h_words)
        return overlap / total if total > 0 else 0

    def predict(self, premise, hypothesis):
        score = self.overlap_score(premise, hypothesis)
        if score > 0.6:
            return "entailment"
        elif score < 0.2:
            return "contradiction"
        else:
            return "neutral"
```

词重叠基线捕捉到了 NLI 的核心直觉：蕴含 = 假设词在前提中出现；矛盾 = 假设词在前提中不出现；中立 = 部分重叠。简单、可解释，既不太乐观也不太悲观——在大约 50% 的 NLI 测试数据上有效。

### 步骤 2：使用句子编码器的 NLI

```python
from sklearn.linear_model import LogisticRegression

class NLIEncoder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.classifier = LogisticRegression()

    def train(self, premises, hypotheses, labels):
        # 编码并连接前提 + 假设向量
        X = []
        for p, h in zip(premises, hypotheses):
            p_vec = self.encoder.encode(p)
            h_vec = self.encoder.encode(h)
            # 连接，并添加元素级差异和乘积——这是 NLI 编码的标准技巧
            diff = [abs(p_vec[i] - h_vec[i]) for i in range(len(p_vec))]
            prod = [p_vec[i] * h_vec[i] for i in range(len(p_vec))]
            X.append(p_vec + h_vec + diff + prod)
        self.classifier.fit(X, labels)

    def predict(self, premise, hypothesis):
        p_vec = self.encoder.encode(premise)
        h_vec = self.encoder.encode(hypothesis)
        diff = [abs(p_vec[i] - h_vec[i]) for i in range(len(p_vec))]
        prod = [p_vec[i] * h_vec[i] for i in range(len(p_vec))]
        features = p_vec + h_vec + diff + prod
        return self.classifier.predict([features])[0]
```

连接、差异和乘积特征捕获了两种不同的 NLI 信号：差异识别前提中缺少的内容，乘积识别共同关注的内容。这三者的组合构成了一个强分类器的基础。

## 使用

### HuggingFace 零样本分类

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

result = classifier(
    "The product arrived damaged and I want a refund.",
    candidate_labels=["refund", "shipping", "feedback", "complaint"]
)
```

```python
{"labels": ["complaint", "refund", "shipping", "feedback"],
 "scores": [0.85, 0.72, 0.31, 0.15]}
```

`zero-shot-classification` 使用 BART 的 NLI（MNLI 数据集）将标签框架化为假设："This text is about refund." -> 蕴含 vs 矛盾。不需要训练数据就能获得标签。标签总数可达几十个是可行的。

### 文档分类精调

```python
from transformers import AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(class_names)
)

# 在标记数据上精调（标准文本分类流程）
trainer = Trainer(model=model, train_dataset=train_dataset)
trainer.train()
```

精调会在你的分类任务微调完整模型。它通常比零样本方法获得更高的准确率，特别是当类别复杂或领域特定时。精调需要每个类别约 100-1000 个样本。

## 发布

NLI 评估检查清单。

保存为 `outputs/prompt-nli-eval.md`：

```markdown
---
name: nli-eval
description: 提示：评估 NLI 模型——蕴含、矛盾和中立。
phase: 5
lesson: 20
---

评估 NLI 模型输出：

1. 蕴含检测：模型是否能检测到同义信息？（"飞机晚点" → "航班延迟"）
2. 矛盾检测：模型是否能发现否定？("门是开的" → "门是关的")
3. 中立检测：当只有间接推理时，模型是否会产生幻觉？（"John 有一辆车" → "John 住在一栋房子里"——中立，而非蕴含）
4. 否定处理：模型是否理解否定词的影响？（"The customer didn't complain" → "The customer complained"——矛盾而非蕴含）
5. 数量处理：模型是否理解数量差异？（"5 个人" → "2 个人"——矛盾）

检查零样本分类结果：标签是否与人工判断一致？
```

## 练习

1. **简单。** 使用 `pipeline("zero-shot-classification")` 对 3 篇文档进行分类，使用 4 个自定义标签。检查置信分数是否合理。
2. **中等。** 收集 100 个前提-假设对用于 NLI。标注它们为蕴含、矛盾或中立。训练句子编码器 + 逻辑回归分类器。报告测试准确率。
3. **困难。** 比较零样本 NLI 分类与精调 BERT 在文档分类方面的表现。在什么样本量下精调模型开始优于零样本模型？绘制准确率与训练集大小的关系图。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| NLI | 自然语言推理 | 前提与假设之间的蕴含/矛盾/中立分类。 |
| 蕴含 | 逻辑结果 | 假设可以从前提推导出来。 |
| 矛盾 | 不一致 | 假设与前提在逻辑上不一致。 |
| 中立 | 不确定 | 既不是蕴含也不是矛盾。 |
| 零样本分类 | 无需训练的标签 | 使用 NLI 将文本分类到未见过的类别。 |
