# 共指消解 — "它"、"他"、"她"指的是什么？

> "约翰把球递给了玛丽。然后他离开了。"——"他"指的是谁？

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 06（NER）
**时间：** 约 50 分钟

## 问题

共指消解的任务是将指名同一实体的名词短语连接起来。"约翰"、"他"、"那个男人"在文本中可能都指向同一个人。如果 QA 系统读到"约翰把球递给了玛丽。然后他离开了。"然后回答"谁离开了？"，它需要知道"他"= 约翰，而不是玛利亚。

共指消解是 NER 的扩展。NER 定位实体并对其进行分类。共指消解将它们链接起来。

## 概念

**共指（Coreference）** 发生在两个名词短语指向同一真实世界实体时。这可以用作同一个事物的体。提到（每个名词短语出现）被分组为链：["John", "He", "the man"] → 某个真实的人。

**回指（Anaphora）** 是共指的一种特定类型，其中后面的短语（回指语）根据前面的短语（先行词）来理解。"约翰……他"——"约翰"是先行词，"他"是回指语。

**共指链** 是整个文档中一系列相互指向的提及。"约翰" → "他" → "那个男人" → "约翰内斯" 都指向同一个人。

**基于规则的共指**：使用语法和词汇规则的确定性系统。Hobbs 算法（使用依存树）是最早的解决方法之一。准确率有限，但可解释且不依赖数据。

**基于神经网络的共指**：使用 BERT 为其提及分配嵌入，然后以两个提及的嵌入之间的距离作为判断共指可能性的依据。目前最先进的技术在 OntoNotes 上达到约 80% 的 F1 分数。

## 构建

### 步骤 1：基于规则的 Hobbs 共指算法（简化版）

```python
class SimpleHobbsResolver:
    def __init__(self, parser):
        self.parser = parser

    def resolve(self, pronoun, sentence_index, parsed_sentences):
        """
        使用基于规则的方法解析代词。
        简化版——仅搜索当前句子中的先行词。
        """
        sentence = parsed_sentences[sentence_index]
        # 规则 1：在同一句子中找到第一个名词短语（NP）
        # 规则 2：跳过与代词性别/数量不匹配的名词短语
        # 规则 3：遵循最低限度约束（最近的匹配）
        np_candidates = self._find_noun_phrases(sentence)

        for np in np_candidates:
            if self._agreement(np, pronoun):
                return np

        return None

    def _find_noun_phrases(self, parsed_sentence):
        """从成分句中提取名词短语。"""
        return ["John", "Mary"]  # 示例：真实的解析器会使用 NLP 工具

    def _agreement(self, np, pronoun):
        """检查性别和数量的一致性。"""
        male = {"he", "him", "his"}
        female = {"she", "her", "hers"}
        neutral = {"it", "its"}

        if pronoun.lower() in male:
            return np not in female and np not in neutral
        elif pronoun.lower() in female:
            return np not in male and np not in neutral
        return True
```

规则 1：Hobbs 优先选择当前句子中最近的、语法匹配的名词短语。规则 2：如果文本中"约翰"先被提到，且后续句子中使用"他"，那么"约翰"是最可能的先行词。

### 步骤 2：基于神经网络的共指——概念性表示

```python
class NeuralCoref:
    def __init__(self, bert_model):
        self.bert = bert_model

    def mention_pairs(self, tokens):
        """生成所有可能的提及对。"""
        mentions = self._extract_mentions(tokens)
        pairs = []
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                pairs.append((mentions[i], mentions[j]))
        return pairs

    def score_pair(self, mention1, mention2):
        """使用 BERT 表示计算共指分数。"""
        emb1 = self.bert.encode(" ".join(mention1))
        emb2 = self.bert.encode(" ".join(mention2))
        return self._bilinear_score(emb1, emb2)

    def cluster(self, tokens, threshold=0.5):
        """将提及聚类成共指链。"""
        mentions = self._extract_mentions(tokens)
        clusters = []

        for mention in mentions:
            best_cluster = None
            best_score = threshold
            for cluster in clusters:
                for cluster_mention in cluster:
                    score = self.score_pair(mention, cluster_mention)
                    if score > best_score:
                        best_score = score
                        best_cluster = cluster
                        break
            if best_cluster:
                best_cluster.append(mention)
            else:
                clusters.append([mention])

        return clusters
```

现代共指消解系统（如 spaCy 的 `experimental_coref`）使用 BERT 嵌入来生成提及表示，学习一个双线性评分函数来评估两个提及是否属于同一实体，并在解码时使用聚类算法。

## 使用

### spaCy 实验性共指消解

```python
import spacy

nlp = spacy.load("en_core_web_sm")
# 需要安装：pip install spacy-experimental
nlp.add_pipe("experimental_coref")

doc = nlp("John gave the ball to Mary. Then he left.")
for cluster in doc._.coref_clusters:
    print([str(m) for m in cluster])
```

```python
[['John', 'he']]
```

spaCy 将共指消解建模为一个聚类问题。它使用 BERT 样式的编码器生成提及嵌入，并使用双线性注意力来连接共指提及。在 OntoNotes 上的 F1 分数约为 75%。

### HuggingFace 共指消解

```python
from transformers import pipeline

coref = pipeline("coreference-resolution",
                 model="michiyasunaga/BERT-large-coref")

text = "John gave the ball to Mary. Then he left."
result = coref(text)
```

```python
[{'resolved': 'John gave the ball to Mary. Then John left.'}]
```

HuggingFace 的共指管道返回"已解析"的文本——用全名替换代词的文本。这对于下游任务很有用（QA 系统可以直接使用共指解析后的文本），但要注意，重复替换可能导致共指链膨胀（三个句子的文本被替换为"约翰"五次，尽管其中一次替换是不恰当的）。

## 发布

共指消解调试提示。

保存为 `outputs/prompt-coref-debug.md`：

```markdown
---
name: coref-debug
description: 提示：调试共指消解输出。
phase: 5
lesson: 23
---

分析共指消解结果：

1. 准确性：代词是否正确链接到正确的先行词？检查性别和数量一致性。
2. 边界：提及的范围是否准确？（包含所有修饰成分，如"the red car"）。
3. 漏链：有未链接到任何实体的代词吗？
4. 误链：是否有非共指的提及被误链到一起？（如"Apple the company"和"apple the fruit"）
5. 嵌套：共指链是否包含嵌套提及（如在一个更大实体内部引用的实体）？
```

## 练习

1. **简单。** 运行 spaCy 的共指消解管道在"John gave the ball to Mary. Then he left."上。解析的链是否正确？
2. **中等。** 编写一个共指替换函数：使用共指链用完整的名称替换文本中的所有代词。比较替换后文本与原始文本的 QA 准确率。
3. **困难。** 手动收集 10 个包含共指的有歧义的句子。标注每句中"谁做了什么"。与基于规则的共指消解和神经网络共指消解进行比较。这两种方法在你的句子中表现如何？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 共指 | 指向同一实体 | 两个名词短语指向同一真实世界实体。 |
| 回指 | 代词与先行词 | "他"回指"约翰"的一种共指关系。 |
| 先行词 | 被指代的对象 | 代词所指的较早出现的短语。 |
| 共指链 | 提及的链 | 整个文档中指向同一实体的所有提及。 |
| 提及 | 短语出现 | 指向实体的单个名词短语的出现。 |
