# 结构化输出与语法 — 将 NLP 解析为约束

> 当模型输出 JSON 时，不要相信它的承诺。用语法约束它。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 18（分词）
**时间：** 约 50 分钟

## 问题

LLM 输出文本。你通常需要其他东西：JSON、SQL、代码、分类标签或表格。要求 LLM 输出 JSON 然后使用正则表达式解析是脆弱的。它偶尔会输出 `}{` 这样的错误内容，破坏了解析器。

解决方案是约束解码——在生成过程中只允许语法上有效的 token。如果下一个 token 必须以 `"` 开头才能成为有效的 JSON，那么所有其他 token 的概率就会被归零。模型永远不会提议一个语法上无效的 token，因为它根本没有这个选项。

## 概念

**约束解码** 在推理时修改 token 概率。模型输出 logits，但只有符合语法的 token 才被允许接受 softmax 处理。如果不允许逗号，`P(",")` 被设为 0，概率在允许的 token 之间重新分布。

**语法引导的生成** 约束解码是更普遍的约束类型。使用上下文无关语法（CFG）或正则表达式定义允许的 token 序列。解码器在每一步都要检查语法：在当前状态下，哪些 token 是合法的下一个 token？

**分步解码** 将输出拆分为多个部分：首先是分类标签，然后是理由，最后是 JSON。每一步都受到不同的约束？每一步通常有不同的约束——但这超出了基本的约束解码。

```figure
bpe-merge
```

## 构建

### 步骤 1：使用 logit 屏蔽进行约束解码

```python
import math

def constrained_decode(logits, allowed_token_ids):
    """将 logits 限制为仅允许的 token IDs。"""
    masked = [-float("inf")] * len(logits)
    for tid in allowed_token_ids:
        masked[tid] = logits[tid]
    # 在屏蔽的 logits 上计算 softmax
    max_l = max(masked)
    exps = [math.exp(l - max_l) if l > -float("inf") else 0 for l in masked]
    total = sum(exps)
    probs = [e / total for e in exps]
    return probs
```

如果 `allowed_token_ids` 是一个仅包含数字 0-9 的集合，则模型被限制为只能生成数字。屏蔽函数的抽象实现可以处理 CFG 符号、JSON 模式和正则表达式。

### 步骤 2：正则表达式约束

```python
import re

class RegexConstraint:
    def __init__(self, pattern):
        self.pattern = re.compile(pattern)

    def allowed_next(self, partial_output, tokenizer):
        """返回在下文中合法的 token ID 集合。"""
        allowed = []
        for tid, token in enumerate(tokenizer.get_vocab()):
            candidate = partial_output + token
            if self.pattern.fullmatch(candidate) or self.pattern.match(candidate):
                allowed.append(tid)
        return allowed
```

如果已经生成的部分是 `{"name": "J`，正则表达式约束 `\{.*\}` 允许 "o"（继续字符串）但不允许 "{"（在字符串内部不能嵌套开花括号）。

## 使用

### 大纲（Outlines）——结构化生成

```python
import outlines

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

# JSON 模式生成
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str

generator = outlines.generate.json(model, Person)
result = generator("Generate a person from New York")
print(result)
```

```python
Person(name="John Doe", age=34, email="john.doe@email.com")
```

Outlines 在库级别处理 JSON 模式和其他受限生成。它在每一步检查模式约束，允许指定类型，并屏蔽非法 token。如果 schema 说 `age` 应该是一个整数，则只生成数字 token。

### JSON 模式验证

```python
import json
from jsonschema import validate

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age", "email"]
}

def parse_llm_output(text):
    try:
        data = json.loads(text)
        validate(data, schema)
        return data
    except (json.JSONDecodeError, Exception) as e:
        return {"error": str(e)}
```

如果 JSON 损坏，则 try-catch 解析无效。如果 JSON 有效但与模式不匹配，则验证会捕获它。"required" 强制执行必需字段，"minimum" 捕获负年龄，"format" 检查电子邮件。

### 用于分类的 JSON 模式

```python
import json

def extract_json(text):
    """从 LLM 输出中提取第一个 JSON 对象。"""
    brace_depth = 0
    start = None
    for i, char in enumerate(text):
        if char == "{":
            if start is None:
                start = i
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None
```

提取 JSON 的故障安全方法。查找第一个 `{`，跟踪大括号深度，并在深度归零时终止。易出错，因为字符串中的 `{` 括号很容易被错误地识别为包装器。约束解码避免了整个解析步骤。

## 发布

LLM 结构化输出工作流。

保存为 `outputs/skill-structured-generation.md`：

```markdown
---
name: structured-generation
description: 确保 LLM 输出的格式符合预期。
phase: 5
lesson: 19
---

确保 LLM 输出是结构化的。

1. 在提示中定义模式。包含 JSON 示例（"输出必须完全匹配此模式："）。
2. 使用库（Outlines、Jsonformer）在解码时进行约束。
3. 验证：在 try-catch 中进行 JSON 解析。如果失败，重试（最多重试 3 次）。
4. 重试失败：返回一个默认的"解析失败"响应，不要返回原始字符串。

如果模式很复杂，在提示中提供 JSON 模式，并展示一个正反示例。"这是正确的 JSON：{...}。这是错误的：{...}"。

尽量避免深层嵌套的模式——LLM 在深约 3 层后 JSON 生成质量会下降。
```

## 练习

1. **简单。** 运行 `outlines` 以 JSON 模式输出。尝试输出一个 User 对象（name、email、age）。验证输出是否符合模式。
2. **中等。** 实现一个简单的基于正则表达式的约束解码器，用于生成有效的电子邮件地址。屏蔽不符合格式的 token。
3. **困难。** 将正则表达式约束解码器与约束解码相结合，构建一个仅输出有效 SQL 的数据库查询生成器。测试约束是否阻止了 SQL 注入。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 约束解码 | 结构化输出 | 仅允许从语法上有效的 token 生成。 |
| JSON 模式 | 数据契约 | 定义输出结构的 JSON 格式。 |
| CFG | 上下文无关文法 | 定义嵌套结构的递归语法规则。 |
| Logit 屏蔽 | Token 过滤 | 将非法 token 的概率设为零。 |
| 结构化生成 | 确保输出结构 | 确保 LLM 输出能够被可靠解析的技术。 |
