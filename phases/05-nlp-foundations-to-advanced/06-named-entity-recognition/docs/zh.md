# 命名实体识别 — 人物、地点、事物

> 语言中有插槽。你需要知道哪些词填满了它们。

**类型：** 构建
**语言：** Python
**前置要求：** NLTK 和 spaCy 的工作知识
**时间：** 约 50 分钟

## 问题

文本包含"巴黎"、"埃隆·马斯克"、"6000 万美元"、"2026 年 6 月"、"OpenAI"。这些是命名实体——指向现实世界事物的短语。提取它们是从文本转向知识的第一阶段。你需要知道你的文本谈论的是谁（人物）、在哪里（地点）、在什么时候（时间）以及涉及多少钱（金额）。

命名实体识别（NER）是有三项任务：检测边界（"旧金山"是一个实体还是两个？）、分配类型（"巴黎"是地点还是人物？）以及处理嵌套（"哈佛大学"在"哈佛"内部）。

## 概念

NER 是一条流水线：标记化→边界检测→实体类型分类。

**基于规则的方法**使用词汇表（"如果序列匹配我们的地理数据库，则标记为 GPE"）。对于已知实体非常精确，但覆盖率低。对于新实体失败。

**基于特征的方法**使用条件随机场（CRF）——一种序列标记模型，为每个词分配一个 BIO 标签（开始 B-、内部 I-、外部 O）。

特征包括：词本身、词性标签、大小写模式、前缀/后缀、上下文词。特征质量决定了 CRF 质量。正确的特征工程可以将 NER F1 分数从 75% 提升到 90%。

**基于 Transformer 的方法**将问题视为序列标记：每个词元映射到实体标签。上下文嵌入捕获了 CRF 需要使用工程特征获得的信息。现代 NER 使用 BERT + 线性分类器，在标准基准测试上达到 93-96% 的 F1 分数。

```figure
ner-bio-tagging
```

## 构建

### 步骤 1：BIO 标记方案

BIO（开始、内部、外部）是 NER 的标准标记定义。

```python
def bio_tag(tokens, spans):
    """将 tokens 和实体跨度转换为 BIO 标签。"""
    tags = ["O"] * len(tokens)
    entity_map = {span["type"]: span["type"] for span in spans}
    for span in spans:
        start, end = span["start"], span["end"]
        tags[start] = f"B-{span['type']}"
        for i in range(start + 1, end):
            tags[i] = f"I-{span['type']}"
    return tags
```

```python
>>> tokens = ["Elon", "Musk", "visited", "Paris", "in", "June"]
>>> spans = [
...     {"start": 0, "end": 2, "type": "PERSON"},
...     {"start": 3, "end": 4, "type": "GPE"},
...     {"start": 5, "end": 6, "type": "DATE"},
... ]
>>> bio_tag(tokens, spans)
['B-PERSON', 'I-PERSON', 'O', 'B-GPE', 'O', 'B-DATE']
```

BIO 帧将 NER 转换为词级分类问题。一旦序列具有 BIO 标签，NER 就变成了 IOB 标记的序列预测任务。保证边界的关键点是，`I-GPE` 只有在前面有 `B-GPE` 或另一个 `I-GPE` 时才合法。解码时必须强制此约束。

### 步骤 2：简单 CRF 特征

```python
def extract_crf_features(tokens, i):
    """为位置 i 处的 token 提取特征。"""
    word = tokens[i]
    features = {
        "word.lower": word.lower(),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.suffix_2": word[-2:] if len(word) >= 2 else "",
        "word.suffix_3": word[-3:] if len(word) >= 3 else "",
        "word.prefix_2": word[:2],
        "BOS": i == 0,
        "EOS": i == len(tokens) - 1,
    }
    if i > 0:
        features["prev.word.lower"] = tokens[i-1].lower()
        features["prev.word.istitle"] = tokens[i-1].istitle()
    if i < len(tokens) - 1:
        features["next.word.lower"] = tokens[i+1].lower()
    return features
```

CRF 的特征工程是选择提供最高效用并且不只记忆训练数据的转换：

- 大写模式（"Paris" → B-GPE，但"paris" → O 或 B-product）。
- 后缀（"-burg" → B-GPE，"-Corp" → B-ORG）。
- 触发器上下文（"visited" 之后 → B-GPE，"$" 之后 → B-MONEY）。
- 文档级特征（如果一个 token 以大写形式出现一次，则它可能在全文都维持大写形式 —— 这是一致性约束，有助于边界检测）。

### 步骤 3：BERT 的 NER 替换

使用 Transformers 不再需要特征工程。模型自己学习上下文模式。标准实现为每个子词词元添加一个线性分类器头：

```python
# 概念性实现
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english"
)

# BERT 使用子词。在处理之前将 token 对齐到词元。
tokens = ["Elon", "Musk", "visited", "Paris"]
inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
outputs = model(**inputs).logits
predictions = outputs.argmax(-1)
```

挑战在于对齐：BERT 将 "Musk" 分割成 ["Mus", "##k"]，但你必须将预测映射回原始词 token。这就是 NER 评估根据被评估的是基于词元还是基于词而变得复杂的原因。标准 CoNLL-2003 基准测试是基于词的，因此 HuggingFace 的 NER 实现包含对齐逻辑。

## 使用

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Elon Musk visited Paris in June.")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

```
Elon Musk PERSON
Paris GPE
June DATE
```

spaCy 的 NER 在单个文档中每秒处理约 10 万个词——几毫秒。它对于常见的实体类型（人物、组织、GPE、日期、金钱、百分比）是可靠的。需要自定义实体类型？使用 `nlp.add_pipe("entity_ruler")` 添加基于规则的组件。训练自定义 NER 需要约 100 条每种类型的标注示例。

### 常见实体类型

| 类型 | 示例 | 注意 |
|------|----------|------|
| PERSON | "埃隆·马斯克" | 通常为首字母大写序列（有时包含中间名缩写）。 |
| ORG | "OpenAI"，"NASA" | 包含缩写。可能包含分隔符（"AT&T"）。 |
| GPE | "巴黎"，"德国" | 地缘政治实体——城市、国家、州。地理上是地点，政治上是实体。 |
| DATE | "6 月"，"2026 年" | 可能是绝对的（"2026 年 6 月 8 日"）或相对的（"昨天"）。 |
| MONEY | "6000 万美元" | 数量和单位。可能标有货币符号。 |
| PRODUCT | "iPhone 17" | 商标/受保护的产品名称。与 ORG 重叠。 |

### 生产陷阱

**嵌套实体**："哈佛法学院" 内包含一个 ORG（"哈佛"）内包含一个 ORG（"哈佛法学院"）。大多数 NER 系统输出扁平化的结构并丢失了层次关系。spaCy 3+ 使用重叠跨度支持部分解决此问题。如果你的实体层次结构很重要，请选择支持重叠的标注解码器。

**跨文档共指**：第一段中的 "John" 和后面的 "he" 不是 NER 问题——这是共指消解（阶段 5 · 24）。NER 给你的是名称；共指消解将它们链接到同一个现实世界实体。一个常见的错误是责怪 NER 没有完成共指消解的工作。

**漂移**：在 2020 年的新闻文章上训练的 NER 模型无法识别像 "Zoom"（公司，不是动词）这样的词。实体模型漂移的速度与它们所训练的领域一样快。每年在最新数据上评估你的 NER。实体覆盖率下降是重新训练的早期信号。

## 发布

用于在自定义文本上调试 NER 检查的验证提示。

保存为 `outputs/prompt-ner-debug.md`：

```markdown
---
name: ner-debug
description: 在自定义数据集上调试 NER 输出的验证提示。
phase: 5
lesson: 06
---

检查 NER 输出并诊断失败模式。给定文本和 spaCy/HuggingFace 的实体列表：

1. 边界检查：像 "New York" 这样的实体有可能被拆分成 "New/B-ORG York/O" 而不是 "New/B-ORG York/I-ORG" 吗？拆分边界是最常见的 NER bug。
2. 类型检查："Paris" 应标记为 GPE（地点）还是 PERSON（人物）？提供上下文。
3. 漏报（FN）：文本中是否有任何应该被捕获但未被捕获的实体？检查数字（"millions" 有时被遗漏）。
4. 误报（FP）：是否有非实体被标记为实体？检查英文地名与普通词的误报（"Washington" 与 "black" 相对）。
5. 上下文依赖性：模型是否在一个实体具有两种可能类型的句子中混淆了（"Amazon" → ORG vs. LOC）？提及方向性。
```

## 练习

1. **简单。** 对 `"Apple is looking at buying U.K. startup for $1 billion"` 运行 spaCy NER。检查识别的每个实体及其类型。
2. **中等。** 编写一个函数，将 spaCy 的实体存储为三元组（实体文本、实体类型、字符偏移量）。添加一个 `validate()` 方法，检查在开始/结束偏移量处是否存在 `doc.text[start:end] == ent.text`。
3. **困难。** 在自定义文本上运行 spaCy 的 NER。找到三个漏报（NER 遗漏的实体）和三个误报（错误标记）。编写基于规则的修正——使用 `entity_ruler`——以修复它们。测量修正前后的 F1 分数变化。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| BIO | 开始、内部、外部 | 框架——每个词被标记为实体的开始、实体的内部或外部。 |
| 跨度 | 实体边界 | token 序列中实体的开始和结束索引。 |
| CRF | 序列标记 | 为序列预测建模标签依赖关系（B-I 约束）的统计模型。 |
| GPE | 地名 | 地缘政治实体——城市、州、国家。 |
| 对齐 | 词元→词 | 将子词预测映射回原始 token 边距的映射。 |
