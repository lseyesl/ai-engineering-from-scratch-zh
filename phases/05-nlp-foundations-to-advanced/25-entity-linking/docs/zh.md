# 关系抽取与知识图谱

> 文档是分散的。知识图谱是连接的。三元组是连接它们的桥梁。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 06（NER），阶段 5 · 24（实体链接）
**时间：** 约 55 分钟

## 问题

文档包含实体。实体之间存在关系。"埃隆·马斯克是特斯拉的 CEO"定义了一个关系：`(Elon_Musk, CEO_of, Tesla)`。关系抽取（RE）从非结构化文本中提取这些结构化的三元组。

知识图谱（KG）将这些三元组组织成一个图。节点 = 实体。边 = 关系。在完成关系抽取后，可以执行推理（"埃隆在哪些公司担任 CEO？"）和问答。

## 概念

**关系抽取**有两种风格：管道式（先 NER，然后为每对实体分类关系）和联合式（同时识别实体和关系，使用编码器-解码器或序列到序列模型）。

**关系分类**采用一对实体（s，o）并预测关系 r：P(r | s, o, context)。如果有 N 个关系类型，这就是一个 N 路分类问题，加上一个用于无关系的"无关系（None）"类型。

**知识图谱构建**从这样开始：抽取三元组 → 三元组验证 → 三元组去重/融合 → 三元组存储（Neo4j、SPARQL 端点、Grakn）。

## 构建

### 步骤 1：基于模式的关系抽取

```python
import re

class PatternRelationExtractor:
    def __init__(self):
        self.patterns = {
            "born_in": re.compile(r"(\w+) was born in (\w+)"),
            "ceo_of": re.compile(r"(\w+) is (the )?CEO of (\w+)"),
            "located_in": re.compile(r"(\w+) is located in (\w+)"),
        }

    def extract(self, text):
        triples = []
        for relation, pattern in self.patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                subj, obj = match[0], match[-1]
                triples.append((subj, relation, obj))
        return triples
```

```python
>>> extractor = PatternRelationExtractor()
>>> extractor.extract("Elon Musk is the CEO of Tesla. He was born in South Africa.")
[('Elon', 'ceo_of', 'Tesla'), ('Elon', 'born_in', 'South Africa')]
```

基于模式的方法对于格式良好、可预测的句子非常精确。在新闻文本中覆盖率约为 20%，但由于模式的原因非常准确。

### 步骤 2：使用 BERT 对实体对进行关系分类

```python
class RelationClassifier:
    def __init__(self, bert_model, relation_types):
        self.bert = bert_model
        self.relation_types = relation_types

    def extract(self, sentence, head_span, tail_span):
        """对句子中两个实体之间关系的分类。"""
        # 使用特殊标记标记实体：$HEAD$ 和 $TAIL$
        marked = sentence[:head_span[0]] + "$HEAD$" + \
                 sentence[head_span[0]:head_span[1]] + "$HEAD$" + \
                 sentence[head_span[1]:tail_span[0]] + "$TAIL$" + \
                 sentence[tail_span[0]:tail_span[1]] + "$TAIL$" + \
                 sentence[tail_span[1]:]

        # 嵌入标记文本
        encoding = self.bert.encode(marked)

        # 分类层
        logits = self.bert.classify(encoding)

        # softmax
        max_idx = max(range(len(logits)), key=lambda i: logits[i])
        return self.relation_types[max_idx]
```

在将句子传递给 BERT 之前，用特殊标记（`$HEAD$` 和 `$TAIL$`）标记实体，可以让 BERT 知道关注哪里并使用标记嵌入来指导关系分类。

### 步骤 3：基于嵌入的关系抽取

```python
def extract_relations_with_embeddings(encoder, text, entities):
    """使用句子嵌入实体对之间的关系的提取。"""
    sentences = text.split(".")

    triples = []
    for sent in sentences:
        sent_entities = [(start, end, ent_type, text)
                         for start, end, ent_type, text in entities
                         if sent.find(text) != -1]

        for i in range(len(sent_entities)):
            for j in range(i + 1, len(sent_entities)):
                ent1, ent2 = sent_entities[i], sent_entities[j]

                # 在实体之间创建一个关系假设句子
                hypothesis = f"{ent1[3]} {ent2[3]}"
                hyp_emb = encoder.encode(hypothesis)
                sent_emb = encoder.encode(sent)

                # 相似度达到阈值 → 存在关系
                sim = cosine_similarity(hyp_emb, sent_emb)
                if sim > 0.7:
                    triples.append((ent1[3], "related_to", ent2[3]))

    return triples
```

在关系抽取中，"相关"是一个较为宽泛的概念。基于嵌入的方法可以检测到实体之间的"某些关系"，但却无法识别具体是哪种关系类型（需要关系分类器）。

## 使用

### spaCy + 关系抽取

```python
import spacy
from spacy.util import minibatch

nlp = spacy.load("en_core_web_sm")
doc = nlp("Elon Musk founded Tesla in 2003.")

# 使用依存分析提取关系
for token in doc:
    if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
        subj = token.text
        verb = token.head.text
        objs = [child.text for child in token.head.children
                if child.dep_ == "dobj"]
        for obj in objs:
            print(f"({subj}, {verb}, {obj})")
```

```python
('Elon', 'founded', 'Tesla')
```

基于依存分析的提取使用解析树来查找主语-动词-宾语模式。`nsubj → verb → dobj` 是一条直接的语法路径。使用依赖解析可以检测到基于模式的方法无法检测到的关系——"Tesla, founded by Elon Musk," 仍然会通过依赖分析捕捉到。

### 存储到知识图谱

```python
class KnowledgeGraph:
    def __init__(self):
        self.graph = defaultdict(set)  # (subj, rel) → [obj, ...]

    def add_triple(self, subj, rel, obj):
        self.graph[(subj, rel)].add(obj)

    def query(self, subj, rel):
        """查询与给定（主语，关系）相对应的宾语。"""
        return list(self.graph.get((subj, rel), []))

    def all_relations(self, subj):
        """返回与特定主语相关的所有关系。"""
        return [(rel, objs) for (s, rel), objs in self.graph.items() if s == subj]
```

知识图谱存储三元组并支持查询。生产级别的图谱（Neo4j）使用索引、推理规则和 SPARQL 或 Cypher 进行查询。

## 发布

关系抽取验证提示。

保存为 `outputs/prompt-re-validation.md`：

```markdown
---
name: re-validation
description: 提示：验证关系抽取质量。
phase: 5
lesson: 25
---

验证关系抽取（RE）系统的输出：

1. 准确性：抽取的三元组是否正确？（(Elon, born_in, South Africa) ✅，(Elon, CEO_of, Tesla) ✅）
2. 覆盖率：相关的关系是否被遗漏？
3. 关系类型多样性：系统的输出是否只使用了少数几种关系类型？可能存在关系类型不足的问题。
4. 误报：输出中是否包含不正确的三元组？（(Elon, born_in, 2003) ❌）
5. 跨句子抽取：系统是否能处理跨多个句子的关系？（"埃隆成立了特斯拉。它成立于 2003 年。"——"它"→特斯拉）

计算验证集上的精确率、召回率和 F1 分数。
```

## 练习

1. **简单。** 编写一个基于模式的关系抽取器，用于提取 "X was born in Y" 和 "X is the CEO of Y" 这两种关系。在 5 个句子上进行测试。
2. **中等。** 使用 BERT 实现关系分类。在 10 个预定义关系类型（如 `born_in`、`ceo_of`、`located_in`）的小型数据集（100 个句子）上进行训练。在测试集上报告准确率。
3. **困难。** 构建一个完整的管道：从文本中抽取 NER→共指消解→关系抽取→知识图谱。在 10 篇文档上测量端到端的三元组精确率和召回率。共指消解在关系抽取中能产生多大的改善？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 三元组 | (主语, 关系, 宾语) | 表示两个实体之间关系的基本单位。 |
| 关系抽取 | 实体之间 | 识别文本中实体之间关系的（子）任务。 |
| 知识图谱 | 连接图 | 由表示实体及其关系的有向图组成的结构化知识库。 |
| 管道式 | 先 NER，再关系抽取 | 两阶段方法：先检测实体，然后对实体对之间的关系进行分类。 |
| 联合式 | 同时进行 | 同时进行实体识别和关系抽取的端到端模型。 |
