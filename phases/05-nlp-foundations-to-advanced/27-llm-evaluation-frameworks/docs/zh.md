# 长上下文评估 — 大海捞针

> 上下文窗口越大，寻找相关信息就越困难。"大海捞针"就是衡量这个的能力。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 26（LLM 评估）
**时间：** 约 45 分钟

## 问题

GPT-4 有 128K 的上下文窗口。Gemini 有 1M。Claude 有 200K。但上下文窗口的长度和有效使用之间的比率并不是 1:1。LLM 在上下文中定位相关信息的能力——当相关信息埋藏在 100K 个词元深处时——会随着上下文长度的增加而下降。

"大海捞针"测试是黄金标准评估：将一句事实性声明（"针"）嵌入到一段较长的无关文本（"干草堆"）中，然后询问关于该事实的问题。如果模型在上下文为 128K 时能找到"针"，那么它的有效上下文利用率就是 128K。

## 概念

**有效上下文窗口**是指模型能够可靠地利用其中任意位置的信息的上下文长度。这与最大上下文窗口不同——后者只是模型可以接受而不崩溃的最大输入长度。

**检索位置偏差**是指模型往往在上下文的开头或结尾表现更好（位置偏差）。长上下文评估会测量开头、中间和结尾的性能。

**分心信息**是指输出包含来自"干草堆"的信息而非来自"针"的信息。评估检查幻觉是否存在。

## 构建

### 步骤 1：大海捞针数据生成

```python
import random

class NeedleInHaystack:
    def __init__(self, needle, haystack_sources, context_lengths):
        self.needle = needle
        self.haystack_sources = haystack_sources
        self.context_lengths = context_lengths

    def generate_sample(self, context_length, needle_position=0.5):
        """在干草堆的给定位置插入针。"""
        haystack_text = self._generate_haystack(context_length)

        # 在干草堆中插入针
        insert_idx = int(len(haystack_text) * needle_position)
        sample = (haystack_text[:insert_idx] +
                  " " + self.needle + " " +
                  haystack_text[insert_idx:])

        question = f"根据以上文本，{self._extract_question_from_needle()}？"
        answer = self.needle
        return {"text": sample, "question": question, "answer": answer}

    def _generate_haystack(self, target_length):
        """生成指定长度的干草堆文本。"""
        haystack = ""
        while len(haystack) < target_length:
            haystack += random.choice(self.haystack_sources) + "\n"
        return haystack[:target_length]
```

大海捞针的数据生成需要针（事实）、干草堆（分心物）和插入位置坐标（深度/位置）。通常在多个上下文长度和多个位置上进行测试。

### 步骤 2：大海捞针评估

```python
class NeedleEvaluator:
    def __init__(self, model):
        self.model = model
        self.results = []

    def evaluate(self, test_samples):
        """运行评估并存储结果。"""
        for sample in test_samples:
            context_length = self._count_tokens(sample["text"])
            response = self.model.generate(
                sample["text"] + "\n" + sample["question"]
            )

            correct = sample["answer"].lower() in response.lower()
            self.results.append({
                "context_length": context_length,
                "position": sample.get("position", 0.5),
                "correct": correct,
                "response": response
            })
        return self._summary()

    def _summary(self):
        """汇总不同上下文长度下的准确率。"""
        summary = {}
        for r in self.results:
            length_bin = self._length_bin(r["context_length"])
            if length_bin not in summary:
                summary[length_bin] = {"total": 0, "correct": 0}
            summary[length_bin]["total"] += 1
            if r["correct"]:
                summary[length_bin]["correct"] += 1

        return {
            length: data["correct"] / data["total"]
            for length, data in summary.items()
        }

    def _count_tokens(self, text):
        return len(text.split())
```

大海捞针评估者针对每个上下文长度和位置生成一个二维矩阵（深度 × 位置）。如果模型在长上下文上表现良好，矩阵的绝大部分都是绿色的（准确率高）。

### 步骤 3：多针变体

```python
class MultiNeedleEvaluator(NeedleEvaluator):
    def __init__(self, model, num_needles=3):
        super().__init__(model)
        self.num_needles = num_needles

    def generate_multi_needle_sample(self, needles, haystack_text, positions):
        """在干草堆中插入多个针。"""
        sample = haystack_text
        for needle, pos in sorted(zip(needles, positions), key=lambda x: -x[1]):
            insert_idx = int(len(sample) * pos)
            sample = (sample[:insert_idx] + " " + needle + " " +
                      sample[insert_idx:])

        questions = [f"{n} 这句话是否在以上文本中出现？" for n in needles]
        return {"text": sample, "questions": questions, "answers": needles}
```

多针测试会检查模型是否能够在同一上下文范围内找到多个信息片段。更高的挑战性：要求模型列出所有匹配项。

## 使用

### RULER

```python
# RULER 评估套件的概念性使用
from ruler import Ruler

ruler_model = Ruler(model)

# 多任务评估
multi_needle = ruler_model.evaluate("multi_needle", max_length=128000)
variable_tracking = ruler_model.evaluate("variable_tracking", max_length=128000)
```

RULER 是一种综合性长上下文评估，超越了"大海捞针"的范畴，它使用了四种任务：多针检索、多查询追踪、变量追踪和聚合问题。

### 位置偏差映射

```python
def position_bias_map(model, eval_fn, context_lengths, positions):
    """生成上下文利用的热力图。"""
    results = []
    for cl in context_lengths:
        for pos in positions:
            score = eval_fn(model, cl, pos)
            results.append({"length": cl, "position": pos, "score": score})
    return results
```

位置偏差映射在很多情况下都会显示一个"U形"——模型在上下文开头和结尾的表现更好，但在中间的表现较差。在某些模型上，"中间"的准确率会下降到约 50%，而"开头"和"结尾"的准确率则为 95%。

### 生产环境的注意事项

长上下文评估在生产环境中仍然是一个活跃的研究领域。如果模型在 128K 个词元中的中间位置无法找到"针"，那么对于需要跨 100K 个词元的问答的 RAG 系统来说，就会出现问题。所以很多生产系统会采用分块 + 检索的方法，而不是将整个上下文输入模型。

## 发布

长上下文评估计划。

保存为 `outputs/prompt-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: 提示：规划长上下文 LLM 评估。
phase: 5
lesson: 27
---

为 LLM 规划长上下文评估：

1. 大海捞针：在干草堆中隐藏事实事实。在多个上下文长度（1K、8K、32K、64K、128K）上测试。
2. 多针：在干草堆中隐藏 3-5 个事实。测试模型是否能够找到所有事实。
3. 位置偏差：在开头、25%、50%、75%、结尾处插入针。在开头和结尾的高分，中间的低分表明存在位置偏差。
4. 分心抵抗：在干草堆中加入与针主题相关的事实。测试模型是否会受到分心信息的影响。
5. 推理：要求模型不仅找到针，还要对其进行推理（"是倒着说的吗？"）。

在模型用于长上下文 RAG 或长文档分析之前，请执行此评估。
```

## 练习

1. **简单。** 生成长度为 500 个词元的干草堆文本。插入一个"针"事实。验证是否可以通过字符串搜索找到它。
2. **中等。** 在上下文长度为 1000 和 8000 的位置上评估一个 LLM（通过 API）。准确率在 8000 时是否有所下降？如果有，下降了多少？
3. **困难。** 在 4 个上下文长度（1000、4000、8000、16000）和 5 个深度位置（开头、25%、中间、75%、结尾）上生成位置偏差映射。在热力图中记录结果。哪些长度/位置组合的准确率低于 50%？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 有效上下文窗口 | 可用长度 | 模型可以可靠地利用其中信息的最长上下文。 |
| 位置偏差 | 开头/结尾偏好 | 模型倾向于从上下文的开头或结尾检索信息，而非中间。 |
| 大海捞针 | 检索测试 | 将事实嵌入长文本中并对其进行测试。 |
| 多针 | 多事实检索 | 在同一上下文中搜索多个事实。 |
| 分心信息 | 相关但错误的信号 | LLM 可能会选择与问题主题相关的幻觉信息，而不是正确答案。 |
