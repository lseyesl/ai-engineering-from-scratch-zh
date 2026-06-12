# 实体链接 — 将文本连接到知识库

> 文本说的是"牛顿"。知识库里有好几个牛顿：艾萨克·牛顿、牛顿市、牛顿单位。哪个是正确的？

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 06（NER），阶段 5 · 23（共指消解）
**时间：** 约 50 分钟

## 问题

NER 告诉你"华盛顿"是一个实体。实体链接告诉你它是哪一个人：华盛顿州、华盛顿特区还是乔治·华盛顿？实体链接（EL）将文本中提到的实体映射到知识库中的唯一 ID（如维基百科文章）。

EL 解决歧义问题。没有 EL，系统就无法区分"苹果（水果）"和"苹果（公司）"。

## 概念

**候选生成**：对于每个提及，从知识库中找到一个可能的实体列表。"牛顿" → 艾萨克·牛顿、牛顿市、牛顿（单位）、牛顿（马萨诸塞州）。

**实体消歧**：从候选列表中为当前上下文选择最佳实体。使用上下文嵌入、实体流行度和类型兼容性。

**NIL 链接**：当提及没有对应的知识库实体时（新实体且没有维基百科页面）。系统应返回 NIL 而不是随机选择一个实体。

## 构建

### 步骤 1：候选生成

```python
class CandidateGenerator:
    def __init__(self, entity_index):
        self.entity_index = entity_index  # 名称 → [Entity_id, ...]

    def candidates(self, mention, context=None):
        """为提及生成候选实体。"""
        mention_lower = mention.lower()
        candidates = self.entity_index.get(mention_lower, [])

        if not candidates:
            # 回退：去除前面冠词后的模糊搜索
            stripped = mention_lower.lstrip("the ")
            candidates = self.entity_index.get(stripped, [])

        return candidates
```

候选生成提供了一个包含多种可能性的短列表。如果名称是"Newton"，候选列表中包括艾萨克·牛顿（科学家）、牛顿（马萨诸塞州）和牛顿（单位）。基于 TF-IDF 的模糊匹配可以增加覆盖率。

### 步骤 2：使用上下文进行实体消歧

```python
class EntityDisambiguator:
    def __init__(self, encoder, entity_embeddings):
        self.encoder = encoder          # 上下文编码器
        self.entity_embeddings = entity_embeddings  # Entity_id → embedding

    def disambiguate(self, mention, context, candidates):
        """从候选列表中为提及选择最合适的实体。"""
        if len(candidates) == 1:
            return candidates[0]

        # 将上下文编码为向量
        context_emb = self.encoder.encode(context)

        # 选择与上下文相似度最高的候选实体
        best_entity = None
        best_score = -1
        for entity_id in candidates:
            entity_emb = self.entity_embeddings[entity_id]
            score = self._cosine_similarity(context_emb, entity_emb)
            if score > best_score:
                best_score = score
                best_entity = entity_id

        return best_entity
```

如果上下文是"牛顿发现了重力"，候选集中艾萨克·牛顿的嵌入与上下文的相似度接近 0.8，而牛顿市与上下文的相似度接近 0.1。如果上下文是"牛顿距离波士顿 30 分钟车程"，结果则相反。

### 步骤 3：NIL 链接

```python
def link_with_nil(mention, context, candidates, threshold=0.3):
    """如果最佳候选得分过低，返回 NIL。"""
    disambiguator = EntityDisambiguator(encoder, entity_embeddings)
    candidates_list = candidates(mention, context)

    if not candidates_list:
        return "NIL"

    best = disambiguator.disambiguate(mention, context, candidates_list)
    # 在测试期间检查最佳候选的得分
    # 如果低于阈值，在测试时返回 NIL
    # 注意：这需要访问 disambiguator 中的置信度得分
    return best
```

当模型不确定时，应返回 NIL——"我不知道这个实体"——而不是猜测。GED（生成式实体消歧）方法甚至可以对没有标准知识库 ID 的新实体进行自由形式的消歧。

## 使用

### REL（关系实体链接）

```python
import spacy
from rel.relation_extraction import Rel extraction

rel = Rel(mention_detector="spacy")
in_text = "Newton discovered gravity."
relations = rel.extract(in_text)
```

REL 使用维基百科作为参考知识库，并配有预提交的实体索引。它将候选生成过程优化到约 100 个候选实体以内，并使用基于 BERT 的消歧模型在 100 毫秒内完成消歧（在 GPU 上）。

### BLINK（用于 EL 的 BERT）

```python
from blink.main_dense import main_dense

models_path = "models/"
config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": True,
    "top_k": 10,
    "biencoder_model": models_path + "biencoder/pytorch_model.bin",
    "biencoder_config": models_path + "biencoder/config.json",
    "crossencoder_model": models_path + "crossencoder/pytorch_model.bin",
    "crossencoder_config": models_path + "crossencoder/config.json",
    "entity_catalogue": models_path + "entity.jsonl",
    "entity_encoding": models_path + "entity_encoding.t7",
}

results = main_dense(config, examples=[{
    "id": 0,
    "text": "Newton discovered gravity.",
    "label": "unknown",
    "mention": "Newton"
}])
```

BLINK 使用双编码器（独立的提及/上下文编码器 + 实体编码器）进行高效检索，然后使用交叉编码器对所有候选实体进行精细化重排序。

## 发布

实体链接系统的验证检查清单。

保存为 `outputs/prompt-el-verification.md`：

```markdown
---
name: el-verification
description: 提示：验证实体链接系统的质量。
phase: 5
lesson: 24
---

验证实体链接（EL）系统：

1. 候选生成：每个提及是否捕获到正确的知识库 ID？
2. 消歧准确率：多义词（"Apple" → 公司 vs 水果）是否得到正确处理？
3. NIL 检测：如果提及在知识库中不存在，系统是否返回 NIL？
4. 上下文敏感性：同一个提及在不同上下文中是否指向不同的实体？
5. 跨语言支持：EL 是否在非英语文本上有效？或者需要翻译？

计算几个测试样本的准确率和覆盖率。
```

## 练习

1. **简单。** 在"Apple is releasing a new iPhone."和"Apple is a fruit."这两句话上运行实体链接。两个句子中的"Apple"是否链接到不同的实体？
2. **中等。** 在包含 3 个实体（使用 `wikipedia-api` 获取描述）的知识库上实现候选生成函数。测试"Newton"在不同上下文中的表现。
3. **困难。** 实现带有上下文嵌入的实体消歧。在混淆集中评估消歧准确率——包含多义词（"Apple"、"Bass"、"Crane"）的 100 个句子。BLINK 的准确率与平均嵌入相似度相比如何？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 实体 | 知识库条目 | 知识库中的唯一条目（例如，维基百科文章）。 |
| 提及 | 文本中的短语 | 引用实体的文本跨度。 |
| 消歧 | 消除歧义 | 根据上下文从候选实体中选择正确的实体。 |
| NIL | 未知实体 | 没有匹配条目的提及。 |
| KB | 知识库 | 实体的结构化存储库（例如，维基数据、维基百科）。 |
