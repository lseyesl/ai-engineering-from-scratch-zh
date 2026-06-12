# 词性标注与句法分析 — 语法在我们的机器中

> "run" 是一个动词，还是一个名词，还是两者都是？唯一能回答这个问题的是上下文。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 01（文本处理），阶段 5 · 06（NER）
**时间：** 约 60 分钟

## 问题

两个句子："I will run a marathon."，"We went for a run."。相同的词，不同的 POS。词性（POS）标注为每个词分配一个语法标签。依赖分析分配一个句法结构——哪些词修饰哪些词。

为什么需要关心？因为如果你认为 `"run"` 是一个名词，而你的系统将其作为动词处理，那么你的词形还原就会失败，你的关系抽取就会出错，你的翻译就会输出错误的词序。句法是一个棱镜，几乎所有其他 NLP 任务都通过它折射。

## 概念

**词性标注**是一个序列标注问题。每个词获得一个标签：NOUN、VERB、ADJ、DET、ADP、PROPN、ADV、AUX、PRON、CCONJ、SCONJ、NUM、PART、INTJ、X。标签集是 Universal Dependencies（通用依赖标注）的 17 个标签——跨 100 多种语言的标准。

**成分分析**产生一个树：`(S (NP I) (VP (V run) (NP a marathon)))`。它展示了那些构成句子的分组情况。用于语法生成、释义、机器翻译。

**依存分析**产生一个树，其中每个词都是其他词的依赖项：`run → (nsubj I) (dobj marathon) (det a)`。展示了哪些词修饰哪些词——这是语义任务（关系抽取、QA）需要的结构。

## 构建

### 步骤 1：Bigram HMM 用于 POS 标注

隐马尔可夫模型（HMM）将 POS 标注定义为一个噪声通道问题：给定词序列，找到最可能的标签序列。

```python
class HMMTagger:
    def __init__(self):
        self.transition = {}  # P(tag_i | tag_{i-1})
        self.emission = {}    # P(word | tag)
        self.tags = set()

    def train(self, tagged_sentences):
        tag_bigrams = defaultdict(Counter)
        tag_unigrams = Counter()
        word_tag = defaultdict(Counter)

        for sentence in tagged_sentences:
            prev = "<S>"
            for word, tag in sentence:
                tag_bigrams[prev][tag] += 1
                tag_unigrams[tag] += 1
                word_tag[tag][word.lower()] += 1
                prev = tag

        # 计算带平滑的概率
        V = len(set(w for s in tagged_sentences for w, _ in s))
        for prev, next_tags in tag_bigrams.items():
            total = sum(next_tags.values())
            for tag in tag_unigrams:
                self.transition[(prev, tag)] = math.log(
                    (next_tags.get(tag, 0) + 1) / (total + len(tag_unigrams))
                )
        for tag, words in word_tag.items():
            total = sum(words.values())
            for word, count in words.items():
                self.emission[(tag, word)] = math.log(
                    (count + 1) / (total + V)
                )
            self.emission[(tag, "<UNK>")] = math.log(
                1 / (total + V)
            )
```

HMM 从标注数据中学习转移概率（`DET → NOUN` 的概率高于 `DET → VERB`）和发射概率（`"run"` 在 VERB 标签下的概率高于在 DET 标签下的概率）。

### 步骤 2：维特比解码

```python
def viterbi(self, words):
    """使用维特比算法找到最可能的标签序列。"""
    n = len(words)
    dp = [{} for _ in range(n)]
    backpointers = [{} for _ in range(n)]

    # 初始化
    for tag in self.tags:
        dp[0][tag] = self.transition.get(("<S>", tag), -100) + \
                     self.emission.get((tag, words[0].lower()), -100)

    # 递归
    for i in range(1, n):
        for tag in self.tags:
            best_score, best_prev = float("-inf"), None
            for prev_tag in self.tags:
                score = dp[i-1][prev_tag] + \
                        self.transition.get((prev_tag, tag), -100) + \
                        self.emission.get((tag, words[i].lower()), -100)
                if score > best_score:
                    best_score = score
                    best_prev = prev_tag
            dp[i][tag] = best_score
            backpointers[i][tag] = best_prev

    # 回溯
    best_last = max(dp[-1], key=dp[-1].get)
    tags = [best_last]
    for i in range(n - 1, 0, -1):
        tags.insert(0, backpointers[i][tags[0]])
    return tags
```

维特比通过动态规划避免了在所有 17ⁿ 个可能的标签序列上进行穷举搜索。这是序列标注的标准——当转移约束使大多数路径不可能时，搜索空间有效地坍缩。

### 步骤 3：依存分析——计算弧

依存分析找到连接每个词与其父词的箭头。标准方法（Transition-based parsing）从左到右处理句子，使用堆栈和缓冲区，应用三种动作之一。

```python
class ArcEagerParser:
    def __init__(self):
        self.relations = {}  # (child_idx → (parent_idx, relation))

    def parse(self, words, model):
        stack = [0]
        buffer = list(range(1, len(words)))
        heads = [-1] * len(words)
        relations = {}

        while buffer:
            # 简化：使用 oracle 或经过训练的模型来决定
            action = model.predict(stack, buffer, words)

            if action == "SHIFT":
                stack.append(buffer.pop(0))
            elif action == "LEFT_ARC":
                child = stack.pop()
                parent = buffer[0]
                heads[child] = parent
                relations[child] = parent
            elif action == "RIGHT_ARC":
                child = buffer.pop(0)
                parent = stack[-1]
                heads[child] = parent
                relations[child] = parent

        return heads, relations
```

一个简化——真正的解析器需要每个动作的特征（堆栈顶部、缓冲区前端、它们的 POS 标签、左右子节点）。MaltParser 和 spaCy 的解析器使用神经网络来评分动作候选。

## 使用

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I will run a marathon.")

for token in doc:
    print(f"{token.text:8} {token.pos_:6} {token.dep_:8} -> {token.head.text}")
```

```
I        PRON   nsubj    -> run
will     AUX    aux      -> run
run      VERB   ROOT     -> run
a        DET    det      -> marathon
marathon NOUN   dobj     -> run
.        PUNCT  punct    -> run
```

依存分析告诉你 `"marathon"` 是 `"run"` 的直接宾语，`"I"` 是名词主语，`"will"` 是助动词。如果你在做关系抽取，这个结构正是你需要的——事件 `"run"`，主语 `"I"`，宾语 `"marathon"`。

### 两个常用树

**成分树**显示分组：

```
(S
  (NP (PRP I))
  (VP (MD will) (VB run)
    (NP (DT a) (NN marathon)))
  (. .))
```

**依存树**显示修饰关系：

```
      run (VERB)
     /   |    \
   I   will  marathon
 (nsubj)(aux)  |
              a
             (det)
```

## 发布

语法结构提取的调试提示。

保存为 `outputs/prompt-syntax-debug.md`：

```markdown
---
name: syntax-debug
description: 分析 POS 和依赖输出以诊断下游错误。
phase: 5
lesson: 07
---

给定原始文本和 spaCy 的 POS/依赖输出：

1. POS 准确性：`"run"` 被标记为 VERB 还是 NOUN？依赖 ROOT 是否正确？
2. 主语-动词一致性：`"nsubj"` 关系是否存在且语义合理？
3. 介词附着：`"I saw the man with the telescope"` — `"with"` 应修饰 `"man"`（修饰）还是 `"saw"`（工具）？介词附着是最常见的解析歧义。
4. 长距离依赖：对于复杂句子，依存树是否正确地将远距离的修饰语与其父节点连接起来？
5. 解析失败情况：句子非常长（>40 个词）、无标点或高度嵌套时会发生什么？
```

## 练习

1. **简单。** 从 NLTK 加载 Penn Treebank 样本。打印第一个句子的词性标注。
2. **中等。** 使用 spaCy 分析 `"I saw the man with the telescope"` 的依存关系。提取 `"saw"` 的主语和宾语，以及 `"with"` 的修饰对象。这是正确的还是存在歧义？
3. **困难。** 使用 `en_core_web_sm` 的特征在你的 HMM 标注器之上实现一个 transition-based 依存分析器。在一小部分标注数据上训练它，并与 spaCy 的依存分析器比较你的准确率。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| POS | 词性 | 词的语法类别：NOUN、VERB、ADJ、DET 等。 |
| 依存 | 语法修饰 | 树结构，其中边表示修饰关系（nsubj、dobj、aux）。 |
| 成分 | 分组 | 树结构，其中节点是组成句子成分的短语（NP、VP、PP）。 |
| HMM | 隐马尔可夫模型 | 对具有隐藏状态（标签）的序列进行建模的统计模型。 |
| 维特比 | 解码 | 在具有转移约束的隐马尔可夫模型中寻找最可能的标签序列的动态规划算法。 |
| 介词附着 | PP 附着歧义 | "with the telescope" 修饰动词还是前一个名词？每种解析都是正确的，但语义不同。 |
