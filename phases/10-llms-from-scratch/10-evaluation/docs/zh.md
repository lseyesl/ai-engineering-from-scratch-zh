# 评估：基准测试、评测与 LM Harness

> 古德哈特定律：当一个衡量指标成为目标时，它就不再是一个好的衡量指标。每个前沿实验室都在刷基准。MMLU 分数在上升，而模型仍然无法可靠地数出"strawberry"中有多少个 R。唯一重要的评估是你的评估——在你的任务上，用你的数据。

**类型：** 构建
**语言：** Python
**前置条件：** 第10阶段，第01-05课（从零构建 LLM）
**时间：** ~90 分钟

## 学习目标

- 构建自定义评估框架，对语言模型运行多选题和开放式基准测试
- 解释为什么标准基准（MMLU、HumanEval）会饱和且无法区分前沿模型
- 实现具有适当指标的任务特定评估：精确匹配、F1、BLEU 和 LLM-as-judge 评分
- 设计针对特定用例的自定义评估套件，而非仅依赖公开排行榜

## 问题所在

MMLU 于 2020 年发布，包含 57 个学科的 15,908 道题。三年内，前沿模型就饱和了它。GPT-4 得分 86.4%。Claude 3 Opus 得分 86.8%。Llama 3 405B 得分 88.6%。排行榜压缩到 3 分的范围内，其中差异是统计噪声，而非真正的能力差距。

与此同时，这些相同的模型在一个 10 岁孩子不假思索就能完成的任务上失败。在 MMLU 上得分 88.7% 的 Claude 3.5 Sonnet，最初无法数出"strawberry"中的字母数——这个任务需要零世界知识、零推理，只需字符级迭代。HumanEval 用 164 道题测试代码生成。模型在它上面得分 90%+，却仍然产生任何初级开发者都能捕获的边缘情况崩溃代码。

基准性能与实际可靠性之间的差距是 LLM 评估的核心问题。基准告诉你模型在基准上的表现。它们几乎不告诉你该模型在你的特定任务、特定数据、特定失败模式下会如何表现。如果你在构建客服机器人，MMLU 无关紧要。如果你在构建代码助手，HumanEval 只覆盖函数级生成——它对调试、重构或跨文件解释代码毫无说明。

你需要自定义评估。不是因为基准无用——它们对粗略的模型选择有用——而是因为最终评估必须精确匹配你的部署条件。

## 核心概念

### 评估全景

评估有三种类别，各有不同的成本和信号质量。

**基准测试**是标准化测试套件。MMLU、HumanEval、SWE-bench、MATH、ARC、HellaSwag。你用模型跑基准得到一个分数。优势：每个人都用相同的测试，所以你可以比较模型。劣势：模型和训练数据日益污染这些基准。实验室在包含基准问题的数据上训练。分数上升。能力未必。

**自定义评估**是你为特定用例构建的测试套件。你定义输入、期望输出和评分函数。法律文档摘要器在法律文档上评估。SQL 生成器在你的数据库模式上评估。这些创建成本高，但它们是唯一能预测生产性能的评估。

**人类评估**使用付费标注者根据有用性、正确性、流畅性和安全性等标准评判模型输出。开放式任务中自动评分失败时的金标准。Chatbot Arena 已收集超过 200 万个人类偏好投票，覆盖 100+ 个模型。缺点：成本（每次判断 $0.10-$2.00）和速度（数小时到数天）。

```mermaid
graph TD
    subgraph Eval["评估全景"]
        direction LR
        B["基准测试\n(MMLU, HumanEval)\n便宜、标准化\n可被刷分、过时"]
        C["自定义评估\n你的任务、你的数据\n最高信号\n构建成本高"]
        H["人类评估\n(Chatbot Arena)\n金标准\n慢、贵"]
    end

    B -->|"粗略模型选择"| C
    C -->|"模糊情况"| H

    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
```

### 为什么基准会失效

三种机制导致基准分数不再反映真实能力。

**数据污染。** 训练语料抓取互联网。基准问题存在于互联网上。模型在训练期间看到了答案。这不是传统意义上的作弊——实验室不会故意包含基准数据。但网络规模的抓取使排除几乎不可能。

**应试教学。** 实验室为基准性能优化训练配比。如果训练混合中有 5% 是 MMLU 风格的多选题，模型就学会了格式和答案分布。MMLU 是四选一多选题。模型学到答案分布大约在 A/B/C/D 间均匀分布，这甚至在模型不知道答案时也有帮助。

**饱和。** 当每个前沿模型在基准上得分 85-90% 时，基准就不再有区分力。剩余 10-15% 的问题可能是模糊的、标注错误的，或需要冷僻领域知识。MMLU 从 87% 提高到 89% 可能意味着模型多记住了两道冷僻题，而非变得更聪明。

### 困惑度：快速健康检查

困惑度衡量模型对一个 token 序列的惊讶程度。形式上，它是指数化的平均负对数似然：

```
PPL = exp(-1/N * sum(log P(token_i | context)))
```

困惑度为 10 意味着模型在每个 token 位置上平均像在 10 个选项中均匀选择一样不确定。越低越好。GPT-2 在 WikiText-103 上的困惑度约为 30。GPT-3 约为 20。Llama 3 8B 约为 7。

困惑度适用于在同一测试集上比较模型，但它有盲点。一个模型可以通过擅长预测常见模式而具有低困惑度，同时在罕见但重要的模式上表现糟糕。它也不说明指令遵循、推理或事实准确性。将其用作健全性检查，而非最终裁决。

### LLM-as-Judge

使用强模型评估弱模型的输出。想法很简单：让 GPT-4o 或 Claude Sonnet 在 1-5 分制上对回复的正确性、有用性和安全性评分。使用 GPT-4o-mini 每次判断约 $0.01，与人类判断的相关性出奇地好——在大多数任务上约 80% 一致。

评分提示比模型更重要。模糊的提示（"给这个回复评分"）产生噪声分数。带有评分标准的结构化提示（"如果答案事实正确且引用来源则评 5 分，正确但无来源则 4 分，部分正确则 3 分……"）产生一致、可复现的分数。

失败模式：评判模型表现出位置偏差（在成对比较中偏好第一个回复）、冗长偏差（偏好更长的回复）和自我偏好（GPT-4 对 GPT-4 输出的评分高于等效的 Claude 输出）。缓解措施：随机化顺序、按长度归一化、使用与被评估模型不同的评判模型。

### ELO 评分与成对比较

Chatbot Arena 的方法。向同一提示展示来自不同模型的两个回复。人类（或 LLM 评判者）选择更好的一个。从数千次这样的比较中，为每个模型计算 ELO 评分——与象棋使用的系统相同。

ELO 优势：相对排名比绝对评分更可靠，优雅处理平局，且比独立评分每个输出用更少的比较即可收敛。截至 2026 年初，Chatbot Arena 排名显示 GPT-4o、Claude 3.5 Sonnet 和 Gemini 1.5 Pro 在顶部的 ELO 分差在 20 分以内。

```mermaid
graph LR
    subgraph ELO["ELO 评分流程"]
        direction TB
        P["提示"] --> MA["模型 A 输出"]
        P --> MB["模型 B 输出"]
        MA --> J["评判者\n(人类或 LLM)"]
        MB --> J
        J --> W["A 胜 / B 胜 / 平局"]
        W --> E["ELO 更新\nK=32"]
    end

    style P fill:#1a1a2e,stroke:#0f3460,color:#fff
    style J fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### 评估框架

**lm-evaluation-harness**（EleutherAI）：标准的开源评估框架。支持 200+ 基准。用一条命令对任何 Hugging Face 模型运行 MMLU、HellaSwag、ARC 等。被 Open LLM Leaderboard 使用。

**RAGAS**：专门用于 RAG 流程的评估框架。衡量忠实度（答案是否与检索到的上下文匹配？）、相关性（检索到的上下文是否与问题相关？）和答案正确性。

**promptfoo**：配置驱动的提示工程评估。在 YAML 中定义测试用例，对多个模型运行，获得通过/失败报告。适用于提示的回归测试——确保提示更改不会破坏现有测试用例。

### 构建自定义评估

生产中唯一重要的评估。流程：

1. **定义任务。** 模型到底应该做什么？要精确。"回答问题"太模糊。"给定客户投诉邮件，提取产品名称、问题类别和情感"是一个你可以评估的任务。

2. **创建测试用例。** 原型评估至少 50 个，生产 200+。每个测试用例是一个（输入，期望输出）对。包含边缘情况：空输入、对抗性输入、模糊输入、其他语言的输入。

3. **定义评分。** 结构化输出用精确匹配。文本相似度用 BLEU/ROUGE。开放式质量用 LLM-as-judge。提取任务用 F1。组合多个指标并加权。

4. **自动化。** 每次评估用一条命令运行。无手动步骤。以支持随时间比较的格式存储结果。

5. **随时间追踪。** 评估分数在孤立状态下毫无意义。你需要趋势线。上次提示更改后分数是否提高？切换模型后是否回退？将评估与提示一起版本化。

| 评估类型 | 每次判断成本 | 与人类一致性 | 最适用于 |
|-----------|------------------|----------------------|----------|
| 精确匹配 | ~$0 | 100%（适用时） | 结构化输出、分类 |
| BLEU/ROUGE | ~$0 | ~60% | 翻译、摘要 |
| LLM-as-judge | ~$0.01 | ~80% | 开放式生成 |
| 人类评估 | $0.10-$2.00 | N/A（即真值） | 模糊、高风险任务 |

```figure
perplexity-loss
```

## 构建它

### 步骤1：最小评估框架

定义核心抽象。一个评估用例有输入、期望输出和可选的元数据字典。一个评分器接收预测和参考，返回 0 到 1 之间的分数。

```python
import json
from collections import Counter

class EvalCase:
    def __init__(self, input_text, expected, metadata=None):
        self.input_text = input_text
        self.expected = expected
        self.metadata = metadata or {}

class EvalSuite:
    def __init__(self, name, cases, scorers):
        self.name = name
        self.cases = cases
        self.scorers = scorers

    def run(self, model_fn):
        results = []
        for case in self.cases:
            prediction = model_fn(case.input_text)
            scores = {}
            for scorer_name, scorer_fn in self.scorers.items():
                scores[scorer_name] = scorer_fn(prediction, case.expected)
            results.append({
                "input": case.input_text,
                "expected": case.expected,
                "prediction": prediction,
                "scores": scores,
            })
        return results
```

### 步骤2：评分函数

构建精确匹配、token F1 和模拟的 LLM-as-judge 评分器。

```python
def exact_match(prediction, expected):
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0

def token_f1(prediction, expected):
    pred_tokens = set(prediction.lower().split())
    exp_tokens = set(expected.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def llm_judge_simulated(prediction, expected):
    pred_words = set(prediction.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    overlap = len(pred_words & exp_words) / len(exp_words)
    length_penalty = min(1.0, len(prediction) / max(len(expected), 1))
    return round(overlap * 0.7 + length_penalty * 0.3, 3)
```

### 步骤3：ELO 评分系统

实现带 ELO 更新的成对比较。这正是 Chatbot Arena 用来排名模型的系统。

```python
class ELOTracker:
    def __init__(self, k=32, initial_rating=1500):
        self.ratings = {}
        self.k = k
        self.initial_rating = initial_rating
        self.history = []

    def _ensure_player(self, name):
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def record_match(self, player_a, player_b, outcome):
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        ea = self.expected_score(self.ratings[player_a], self.ratings[player_b])
        eb = 1 - ea

        if outcome == "a":
            sa, sb = 1.0, 0.0
        elif outcome == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] += self.k * (sa - ea)
        self.ratings[player_b] += self.k * (sb - eb)

        self.history.append({
            "a": player_a, "b": player_b,
            "outcome": outcome,
            "rating_a": round(self.ratings[player_a], 1),
            "rating_b": round(self.ratings[player_b], 1),
        })

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])
```

### 步骤4：困惑度计算

使用 token 概率计算困惑度。实践中你会从模型的 logits 获取这些。这里我们用概率分布模拟。

```python
import numpy as np

def perplexity(log_probs):
    if not log_probs:
        return float("inf")
    avg_neg_log_prob = -np.mean(log_probs)
    return float(np.exp(avg_neg_log_prob))

def token_log_probs_simulated(text, model_quality=0.8):
    np.random.seed(hash(text) % 2**31)
    tokens = text.split()
    log_probs = []
    for i, token in enumerate(tokens):
        base_prob = model_quality
        if len(token) > 8:
            base_prob *= 0.6
        if i == 0:
            base_prob *= 0.7
        prob = np.clip(base_prob + np.random.normal(0, 0.1), 0.01, 0.99)
        log_probs.append(float(np.log(prob)))
    return log_probs
```

### 步骤5：汇总结果

计算评估运行的汇总统计：均值、中位数、阈值通过率和按指标细分。

```python
def summarize_results(results, threshold=0.8):
    all_scores = {}
    for r in results:
        for metric, score in r["scores"].items():
            all_scores.setdefault(metric, []).append(score)

    summary = {}
    for metric, scores in all_scores.items():
        arr = np.array(scores)
        summary[metric] = {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "pass_rate": round(float(np.mean(arr >= threshold)), 3),
            "n": len(scores),
        }
    return summary

def print_summary(summary, suite_name="Eval"):
    print(f"\n{'=' * 60}")
    print(f" {suite_name} Summary")
    print(f"{'=' * 60}")
    for metric, stats in summary.items():
        print(f"\n {metric}:")
        print(f" Mean: {stats['mean']:.3f}")
        print(f" Median: {stats['median']:.3f}")
        print(f" Std: {stats['std']:.3f}")
        print(f" Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f" Pass rate: {stats['pass_rate']:.1%} (threshold >= 0.8)")
        print(f" N: {stats['n']}")
```

### 步骤6：运行完整流程

将一切串联起来。定义任务、创建测试用例、模拟两个模型、运行评估、从成对比较计算 ELO，并打印排行榜。

```python
def demo_model_good(prompt):
    responses = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Hamlet?": "William Shakespeare",
        "What language is PyTorch written in?": "Python and C++",
        "What is the boiling point of water?": "100 degrees Celsius",
    }
    return responses.get(prompt, "I don't know")

def demo_model_bad(prompt):
    responses = {
        "What is the capital of France?": "Paris is the capital city of France",
        "What is 2 + 2?": "The answer is four",
        "Who wrote Hamlet?": "Shakespeare",
        "What language is PyTorch written in?": "Python",
        "What is the boiling point of water?": "212 Fahrenheit",
    }
    return responses.get(prompt, "Unknown")

cases = [
    EvalCase("What is the capital of France?", "Paris"),
    EvalCase("What is 2 + 2?", "4"),
    EvalCase("Who wrote Hamlet?", "William Shakespeare"),
    EvalCase("What language is PyTorch written in?", "Python and C++"),
    EvalCase("What is the boiling point of water?", "100 degrees Celsius"),
]

suite = EvalSuite(
    name="General Knowledge",
    cases=cases,
    scorers={
        "exact_match": exact_match,
        "token_f1": token_f1,
        "llm_judge": llm_judge_simulated,
    },
)

results_good = suite.run(demo_model_good)
results_bad = suite.run(demo_model_bad)

print_summary(summarize_results(results_good), "Model A (concise)")
print_summary(summarize_results(results_bad), "Model B (verbose)")
```

"好"模型给出精确答案。"差"模型给出冗长释义。精确匹配严厉惩罚冗长模型。Token F1 和 LLM-as-judge 更宽容。这说明了为什么指标选择很重要：同一个模型根据评分方式看起来可以很好或很糟。

### 步骤7：ELO 锦标赛

在多轮中运行模型间的成对比较。

```python
elo = ELOTracker(k=32)

for case in cases:
    pred_a = demo_model_good(case.input_text)
    pred_b = demo_model_bad(case.input_text)

    score_a = token_f1(pred_a, case.expected)
    score_b = token_f1(pred_b, case.expected)

    if score_a > score_b:
        outcome = "a"
    elif score_b > score_a:
        outcome = "b"
    else:
        outcome = "tie"

    elo.record_match("model_a_concise", "model_b_verbose", outcome)

print("\nELO Leaderboard:")
for name, rating in elo.leaderboard():
    print(f" {name}: {rating:.0f}")
```

### 步骤8：困惑度比较

比较不同质量级别"模型"的困惑度。

```python
test_text = "The quick brown fox jumps over the lazy dog in the garden"

for quality, label in [(0.9, "Strong model"), (0.7, "Medium model"), (0.4, "Weak model")]:
    log_probs = token_log_probs_simulated(test_text, model_quality=quality)
    ppl = perplexity(log_probs)
    print(f" {label} (quality={quality}): perplexity = {ppl:.2f}")
```

## 使用它

### lm-evaluation-harness (EleutherAI)

在任何模型上运行基准测试的标准工具。

```python
# pip install lm-eval
# Command line:
# lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu --batch_size 8

# Python API:
# import lm_eval
# results = lm_eval.simple_evaluate(
#     model="hf",
#     model_args="pretrained=meta-llama/Llama-3.1-8B",
#     tasks=["mmlu", "hellaswag", "arc_easy"],
#     batch_size=8,
# )
# print(results["results"])
```

### promptfoo

配置驱动的提示工程评估。在 YAML 中定义测试，对多个提供商运行。

```yaml
# promptfoo.yaml
providers:
  - openai:gpt-4o-mini
  - anthropic:claude-3-haiku

prompts:
  - "Answer in one word: {{question}}"

tests:
  - vars:
      question: "What is the capital of France?"
    assert:
      - type: contains
        value: "Paris"
  - vars:
      question: "What is 2 + 2?"
    assert:
      - type: equals
        value: "4"
```

### RAGAS 用于 RAG 评估

```python
# pip install ragas
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
#
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
# )
# print(result)
```

RAGAS 衡量通用评估遗漏的东西：模型的答案是否基于检索到的上下文，而非答案在抽象意义上是否"正确"。

## 发布它

本课产出 `outputs/prompt-eval-designer.md`——一个可复用的提示，为任何任务设计自定义评估套件。给它任务描述，它生成测试用例、评分函数和通过/失败阈值建议。

它还产出 `outputs/skill-llm-evaluation.md`——一个根据任务类型、预算和延迟要求选择正确评估策略的决策框架。

## 练习

1. 添加一个"一致性"评分器，将同一输入通过模型运行 5 次并衡量输出匹配的频率。确定性输入上的不一致答案暴露了脆弱的提示或高温设置。

2. 扩展 ELO 追踪器以支持多个评判函数（精确匹配、F1、LLM-as-judge）并加权。比较重度加权精确匹配与重度加权 F1 时排行榜如何变化。

3. 为特定任务构建评估套件：将邮件分类为 5 个类别。创建 100 个包含多样化示例的测试用例，包括边缘情况（可能属于多个类别的邮件、空邮件、其他语言的邮件）。衡量不同"模型"（基于规则、关键词匹配、模拟 LLM）的表现。

4. 实现污染检测：给定一组评估问题和训练语料，检查评估问题（或近似释义）在训练数据中出现的百分比。这是研究人员审计基准有效性的方式。

5. 构建"模型差异"工具。给定两个模型版本的评估结果，高亮显示哪些具体测试用例改善了、哪些回退了、哪些保持不变。这是评估版的代码差异——对理解一个更改是帮助还是伤害至关重要。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| MMLU | "那个基准" | 大规模多任务语言理解——57 个学科的 15,908 道多选题，2025 年饱和度超过 88% |
| HumanEval | "代码评估" | OpenAI 的 164 道 Python 函数补全题，仅测试孤立函数生成 |
| SWE-bench | "真实编码评估" | 来自 12 个 Python 仓库的 2,294 个 GitHub issue，衡量包括测试生成在内的端到端 bug 修复 |
| Perplexity | "模型有多困惑" | exp(-avg(log P(token_i given context)))——越低意味着模型对实际 token 赋予越高概率 |
| ELO rating | "模型的象棋排名" | 从成对胜负记录计算的相对技能评分，Chatbot Arena 用它排名 100+ 个模型 |
| LLM-as-judge | "用 AI 给 AI 打分" | 强模型根据评分标准对弱模型输出评分，与人类评判约 80% 一致，约 $0.01/判断 |
| Data contamination | "模型见过测试题" | 训练数据包含基准问题，在不提高真实能力的情况下抬高分数 |
| Eval suite | "一堆测试" | 一个版本化的（输入，期望输出，评分器）三元组集合，衡量特定能力 |
| Pass rate | "正确率" | 评估用例中分数超过阈值的比例——比均值分数更具可操作性，因为它衡量可靠性 |
| Chatbot Arena | "模型排名网站" | LMSYS 平台，拥有 200 万+ 人类偏好投票，通过 ELO 评分产生最可信的 LLM 排行榜 |

## 延伸阅读

- [Hendrycks et al., 2021 -- "Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300) -- MMLU 论文，尽管已饱和，仍是引用最多的 LLM 基准
- [Chen et al., 2021 -- "Evaluating Large Language Models Trained on Code"](https://arxiv.org/abs/2107.03374) -- OpenAI 的 HumanEval 论文，建立了代码生成评估方法论
- [Zheng et al., 2023 -- "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685) -- 使用 LLM 评估 LLM 的系统分析，包括位置偏差和冗长偏差发现
- [LMSYS Chatbot Arena](https://chat.lmsys.org/) -- 众包模型比较平台，拥有 200 万+ 投票，最可信的真实世界 LLM 排名
