# 文本处理 — 分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理是桥梁。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 2 · 14（朴素贝叶斯）
**时间：** 约 45 分钟

## 问题

模型无法阅读 "The cats were running." 这样的句子。它只能读取整数。

每一个 NLP 系统都从同样的三个问题开始：单词从哪里开始？单词的词根是什么？当 "run"、"running"、"ran" 在某些情况下应该视为相同、在某些情况下应该视为不同时，我们该如何处理？

分词做错了，模型就会从垃圾数据中学习。如果你的分词器把 `don't` 当作一个词元，而 `do n't` 当作两个词元，训练分布就会出现分裂。如果你的词干提取器把 `organization` 和 `organ` 归为同一个词干，主题建模就会失败。如果你的词形还原器需要词性上下文但你却没有传入，动词就会被当作名词处理。

本课程从头构建三个预处理步骤，然后展示 NLTK 和 spaCy 如何完成同样的工作，让你看到其中的权衡。

## 概念

三个操作，每个都有自己的任务和失败模式。

**分词（Tokenization）** 将字符串切分成词元。"词元"的定义故意模糊，因为正确的粒度取决于具体任务。词级用于经典 NLP，子词用于 Transformer，字符用于没有空格的语言。

**词干提取（Stemming）** 使用规则去除后缀。快速、激进、简单粗暴。`running -> run`，`organization -> organ`。第二个例子就是它的失败模式。

**词形还原（Lemmatization）** 使用语法知识将单词还原为其词典形式。较慢、更精确，需要查找表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：当速度重要且可以容忍噪声时使用词干提取（搜索索引、粗略分类）；当语义重要时使用词形还原（问答、语义搜索、用户能看到的任何内容）。

```figure
edit-distance
```

## 构建

### 步骤 1：基于正则表达式的分词器

最简单的有用分词器按非字母数字字符切分，同时将标点作为独立的词元保留。不完美，也不终极，但一行代码就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排序：带可选内部撇号的单词（`don't`、`it's`）、纯数字、任何单个非空白非字母数字字符作为独立词元（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式：`3pm` 被分割成 `['3', 'pm']`，因为我们在字母序列和数字序列之间交替。对大多数任务来说已经足够。URL、电子邮件和话题标签都会出问题。对于生产环境，在通用模式之前添加更具体的模式。

### 步骤 2：Porter 词干提取器（仅步骤 1a）

完整的 Porter 算法有五个阶段的规则。仅步骤 1a 就涵盖了最常见的英语后缀，并且展示了模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

自上而下阅读规则。`ies -> i` 规则是为什么 `ponies -> poni`（而非 `pony`）的原因。真正的 Porter 算法有步骤 1b 可以修复这个问题。规则之间存在竞争，较早的规则胜出。顺序比任何单条规则都重要。

### 步骤 3：基于查找表的词形还原器

真正的词形还原需要形态学知识。一个可教学的简化版本使用一个小型词形还原表和后备规则。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个例子是关键的教学点。`watched` 不在我们的表中，我们的后备规则只处理 `ing`。真正的词形还原涵盖 `ed`、不规则动词、比较级形容词、带音变的复数形式（`children -> child`）。这就是为什么生产系统使用 WordNet、spaCy 的形态分析器或完整的形态分析器。

### 步骤 4：将它们串联起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺少的部分是 POS 标注器。阶段 5 · 07（词性标注）将构建一个。目前，将所有内容默认设为 `NOUN` 并承认这个局限性。

## 使用

NLTK 和 spaCy 提供生产版本，每个只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理缩略形式、Unicode 和你的正则表达式会遗漏的边缘情况。`PorterStemmer` 运行全部五个阶段。`WordNetLemmatizer` 需要将 NLTK 的 Penn Treebank 标记方案转换为 WordNet 的缩写集。上面的转换代码是大多数教程跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 将整个流程隐藏在 `nlp(text)` 背后。分词、词性标注和词形还原全部运行。在规模化场景下比 NLTK 更快，开箱即用也更准确。代价是你不能轻易地替换单个组件。

### 何时选择哪种方案

| 场景 | 选择 |
|-----------|------|
| 教学、研究、需要切换组件 | NLTK |
| 生产环境、多语言、速度优先 | spaCy |
| Transformer 流程（无论如何都会用模型的分词器） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人警告你的两个失败模式

大多数教程只教算法就停止了。有两件事会给真正的预处理流程带来麻烦，而且几乎没有教程会覆盖它们。

**可重现性漂移。** NLTK 和 spaCy 在不同版本之间会改变分词和词形还原行为。spaCy 2.x 中输出 `['do', "n't"]` 的代码在 3.x 中可能输出 `["don't"]`。你的模型是在一个分布上训练的，推理却在另一个分布上运行。准确率悄然下降，却没有人知道原因。在 `requirements.txt` 中固定库版本。编写一个预处理回归测试，冻结 20 个样本句子的预期分词结果。每次升级时运行测试。

**训练/推理不匹配。** 使用激进的预处理（小写化、停用词移除、词干提取）进行训练，在原始用户输入上部署，然后看着性能暴跌。这是生产 NLP 中最常见的失败。如果你在训练期间做了预处理，你必须在推理时运行完全相同的函数。将预处理作为一个函数打包在模型包内部，而不是让服务团队重写一个 notebook 单元格。

## 发布

一个可复用的提示，帮助工程师无需阅读三本教科书就能选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: 为 NLP 任务推荐分词、词干提取和词形还原方案。
phase: 5
lesson: 01
---

你负责提供经典 NLP 预处理建议。给定一个任务描述，你输出：

1. 分词选择（正则、NLTK word_tokenize、spaCy 或 Transformer 分词器）。说明原因。
2. 是否需要词干提取、词形还原、两者都要还是都不要。说明原因。
3. 具体的库调用。给出函数名称。如果涉及 NLTK，引用 POS 标注转换代码。
4. 用户应测试的一个失败模式。

拒绝为用户可见文本推荐词干提取。拒绝在没有 POS 标注的情况下推荐词形还原。标记非英语输入，指出需要不同的流程。
```

## 练习

1. **简单。** 扩展 `tokenize` 以将 URL 保持为单个词元。测试：`tokenize("Visit https://example.com today.")` 应该产生一个 URL 词元。
2. **中等。** 实现 Porter 步骤 1b。如果一个单词包含元音并以 `ed` 或 `ing` 结尾，则将其移除。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个使用 WordNet 作为查找表的词形还原器，当 WordNet 没有条目时回退到你的 Porter 词干提取器。在一个标注语料上对比纯 WordNet 和纯 Porter 测量准确率。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 词元 | 一个单词 | 模型消费的任意单元。可以是词、子词、字符或字节。 |
| 词干 | 单词的词根 | 基于规则的词缀去除结果。不一定是真正的单词。 |
| 词形还原形式 | 词典形式 | 你会去查词典的那种形式。需要语法上下文才能正确计算。 |
| 词性标注 | 词性 | NOUN、VERB、ADJ 等类别。精确词形还原需要它。 |
| 形态学 | 词形变化规则 | 单词如何根据时态、数量、格改变形式。词形还原依赖于它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，仅五页，仍然是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 了解真实流程的搭建方式。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边缘情况。
