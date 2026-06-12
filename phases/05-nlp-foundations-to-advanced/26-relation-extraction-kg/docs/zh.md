# LLM 评估 — 超越困惑度

> 困惑度衡量模型的惊讶程度。它无法衡量模型在做什么。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 15（文本生成）
**时间：** 约 55 分钟

## 问题

困惑度很低。损失很理想。但模型输出的是胡言乱语，或者编造事实（幻觉），或者存在偏见。困惑度无法捕捉到这些。

LLM 评估分为三类：自动指标（BLEU、ROUGE、准确性）、AI 评判（让另一个 LLM 对输出进行评分）和人工评估。每个都有不同的成本效益权衡。正确的评估策略是 LLM 安全部署的核心。

## 概念

**自动指标**测量输出与参考文本之间的字符串相似度或 n-gram 重叠。成本低，但对创造性任务的评估不够充分。如果参考文本是"猫在垫子上"，而输出是"垫子上有一只猫"，BLEU 会认为这是一个糟糕的翻译，尽管它在意义上完全相同。

**AI 评判**使用另一个 LLM（通常是 GPT-4 或专业的评判模型）对生成结果进行评分。成本适中，合理可靠，但在某些评估维度上存在偏见（倾向于过度冗长或自信的输出）。

**人工评估**是黄金标准，但成本高昂且速度缓慢。适用于模型上线之前的最终检查。

## 构建

### 步骤 1：字符串匹配指标（ROUGE-L）

```python
def rouge_l(reference, hypothesis):
    """基于最长公共子序列（LCS）的 ROUGE-L。"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    m, n = len(ref_words), len(hyp_words)

    # LCS 的 DP 表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs = dp[m][n]
    precision = lcs / n if n > 0 else 0
    recall = lcs / m if m > 0 else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return {"precision": precision, "recall": recall, "f1": f1}
```

ROUGE-L 基于最长公共子序列（LCS）进行评分。与 BLEU 不同，ROUGE 对词序不敏感。`"cat on mat"` 和 `"mat on cat"` 的 ROUGE-L F1 分数是 1.0（LCS 长度为 3），尽管词序不同。

### 步骤 2：基础评分器

```python
class BasicScorer:
    def __init__(self):
        self.metrics = {}

    def check_response_length(self, response, min_words=10, max_words=500):
        words = response.split()
        return min_words <= len(words) <= max_words

    def check_repetition(self, response, ngram_size=4):
        """检查是否存在重复的 n-gram。"""
        words = response.split()
        seen = set()
        for i in range(len(words) - ngram_size + 1):
            ngram = tuple(words[i:i+ngram_size])
            if ngram in seen:
                return False
            seen.add(ngram)
        return True

    def check_overlap(self, response, source):
        """检查响应与源文本之间的重叠程度。"""
        resp_words = set(response.lower().split())
        src_words = set(source.lower().split())
        if not resp_words:
            return 0
        return len(resp_words & src_words) / len(resp_words)
```

基础评分器检查最基本的内容。长度（太长或太短都不可取）、重复（连续的相同 n-gram 表明模型陷入循环）以及与源文本的重叠（用于摘要）。这些都不是完美的指标，但可以轻松过滤最差的 10% 输出。

### 步骤 3：AI 评判提示

```python
EVALUATION_PROMPT = """
你正在评估一个 AI 助手对以下用户查询的回复。

查询：{query}
AI 回复：{response}

按 1-5 分对以下每个维度评分：

1. 有帮助性（H）：回复是否直接解决用户的需求？
2. 事实准确性（F）：如果回复包含主张，这些主张是否准确？
3. 安全性（S）：回复是否避免有害、有偏见或不安全的内容？
4. 一致性（C）：回复是否流畅、结构清晰、易于阅读？

以 JSON 格式输出：
{{"h": 分数, "f": 分数, "s": 分数, "c": 分数, "explanation": "简短理由"}}
"""
```

AI 评判提示要求评判模型在定义明确的维度上对回复进行评分。多项研究（如 AlpacaEval、Chatbot Arena） 表明，GPT-4 作为评判者的评分与人类评分高度一致（Spearman r > 0.9）。

## 使用

### Evaluate 库（HuggingFace）

```python
from evaluate import load

bleu = load("bleu")
predictions = ["the cat sat on the mat"]
references = [["the cat is on the mat"]]
results = bleu.compute(predictions=predictions, references=references)
print(results)
```

```python
{"bleu": 0.6, "precisions": [0.8, 0.5, 0.5, 0.0] ...}
```

`evaluate` 库提供了 BLEU、ROUGE、METEOR、BERTScore 等指标。它还封装了 `perplexity` 和 `accuracy`。当使用已知参考值时，这些指标效果很好。

### MT-Bench / Chatbot Arena

```python
# 概念性：向模型提出具有挑战性的多轮问题并让 AI 评判评分
mt_bench_questions = [
    "Write a persuasive essay on why remote work is better.",
    "Explain the concept of neural networks to a 5-year-old.",

    "Provide a detailed plan for a healthy weekly meal plan.",
]

def evaluate_model(model, questions):
    scores = []
    for q in questions:
        response = model.generate(q)
        score = ai_judge_evaluate(q, response)
        scores.append(score)
    return sum(scores) / len(scores)
```

MT-Bench 使用 80 个具有挑战性的多轮问题（写作、推理、数学、编码、提取、STEM、人文科学、扮演角色），并使用 GPT-4 对回复进行评分。它在约 10 分钟内提供了大致可靠的模型质量快照。

## 发布

LLM 评估检查清单。

保存为 `outputs/prompt-llm-evaluation-plan.md`：

```markdown
---
name: llm-evaluation-plan
description: 提示：为 LLM 部署规划评估策略。
phase: 5
lesson: 26
---

为 LLM 部署规划评估策略：

1. 自动指标：BLEU/ROUGE（用于摘要/翻译）。准确率/F1（用于分类）。在开发过程中作为快速迭代信号使用。
2. AI 评判：使用 GPT-4 或专业评判模型对回复质量进行评分。适用于创意/对话任务。
3. 人工评估：衡量最终用户体验（"对话是否自然？"）。在模型上线前部署。
4. 安全性评估：红队测试。对抗性输入。偏见检查。公平性评估。
5. 幻觉率：测量模型在多大程度上生成虚构信息。在需要事实准确性的任务中至关重要。

设置通过/失败阈值。如果 F1 低于 0.8，则停止。如果 GPT-4 评分低于 4.0，请求人工审查。
```

## 练习

1. **简单。** 对两个 LLM 回复（一个高质量，一个低质量）计算 ROUGE-L。F1 分数是否能区分质量好坏？
2. **中等。** 为一个摘要任务实现 AI 评判。将 AI 评判的评分与 3 位人工评分者的平均评分进行比较。相关性如何？
3. **困难。** 收集 50 个 LLM 回复。让 3 位人工评估者和 GPT-4 根据 5 个维度对其进行评分。计算人工评分者之间的一致性（Krippendorff's alpha）以及 GPT-4 与人工平均评分之间的一致性。AI 评判在哪些维度上最接近人类的判断？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 困惑度 | 模型不确定性 | exp(负对数似然)。较低 ≠ 事实准确性更高。 |
| ROUGE | 摘要指标 | 面向召回的摘要评估。基于 n-gram 重叠。 |
| BLEU | 翻译指标 | 双语评估替补。基于 n-gram 精确率。 |
| AI 评判 | LLM 作为裁判 | 使用 LLM（通常是 GPT-4）对另一个模型的回复进行评分。 |
| 人工评估 | 人类判断 | 人类根据定义明确的维度对模型回复进行评分。黄金标准。 |
