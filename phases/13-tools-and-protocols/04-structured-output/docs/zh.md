# 结构化输出——JSON Schema、Pydantic、Zod、受限解码

> "礼貌地让模型返回 JSON"即使在最前沿的模型上也有 5% 到 15% 的失败率。结构化输出通过受限解码弥补了这一差距：模型在字面意义上被阻止发出违反 schema 的 token。OpenAI 的 strict 模式、Anthropic 的 schema 类型工具使用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 和 Zod 的 `.parse` 都是同一个想法的五种表现形式。本课程构建了 schema 验证器和严格模式契约，学习者在每个生产提取管道中都会用到。

**类型：** 构建
**语言：** Python（标准库，JSON Schema 2020-12 子集）
**前置要求：** Phase 13 · 02（函数调用深入）
**时间：** ~75 分钟

## 学习目标

- 为提取目标编写 JSON Schema 2020-12，使用正确的约束（enum、min/max、required、pattern）。
- 解释为什么 strict 模式和受限解码提供的保证与"生成后验证"不同。
- 区分三种失败模式：解析错误、schema 违规、模型拒绝。
- 部署带有类型化修复和类型化拒绝处理的提取管道。

## 问题

一个读取采购订单邮件的智能体需要将自由文本转换为 `{customer, line_items, total_usd}`。三种方法。

**方法一：提示要求 JSON。** "用 JSON 回复，字段为 customer、line_items、total_usd。"在前沿模型上 85% 到 95% 有效。以六种方式失败：缺少花括号、尾随逗号、类型错误、幻觉字段、在 token 限制处截断、泄漏如"这是你的 JSON："之类的散文。

**方法二：生成后验证。** 自由生成，解析，对照 schema 验证，失败时重试。可靠但昂贵——每次重试都要付费，截断错误每次发生都要额外花费一次回复。

**方法三：受限解码。** 提供商在解码时强制 schema。无效 token 从采样分布中被屏蔽。输出保证可解析且保证通过验证。失败收敛为一种模式：拒绝（模型决定输入不适合 schema）。

每个 2026 年的前沿提供商都提供某种形式的方法三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}` 加上如果模型拒绝则在响应中的 `refusal`。
- **Anthropic。** `tool_use` 输入的 schema 强制；`stop_reason: "refusal"` 不存在，但 `end_turn` 且无工具调用是信号。
- **Gemini。** 请求级别的 `responseSchema`；2026 年 Gemini 对选定类型提供 token 级别语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 发出类型化为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 运行时解析器，根据 Zod schema 验证提供商输出；与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

共同点：声明一次 schema，端到端强制执行。

## 概念

### JSON Schema 2020-12——通用语言

每个提供商都接受 JSON Schema 2020-12。你最常用的构造：

- `type`: `object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`: 字段名到子 schema 的映射。
- `required`: 必须出现的字段名列表。
- `enum`: 允许值的封闭集合。
- `minimum` / `maximum`（数字）、`minLength` / `maxLength` / `pattern`（字符串）。
- `items`: 应用于每个数组元素的子 schema。
- `additionalProperties`: `false` 禁止额外字段（默认值因模式而异）。

OpenAI strict 模式增加了三个要求：每个属性必须在 `required` 中列出，所有地方都要 `additionalProperties: false`，并且不能有未解析的 `$ref`。如果你违反这些，API 在请求时返回 400。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从数据类形状的模型生成 JSON Schema。Pydantic AI 将其包装，因此你编写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

智能体框架将 schema 转换为 OpenAI strict 模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型的输出作为类型化的 `Invoice` 实例返回。验证错误抛出 `ValidationError`，带有类型化的错误路径。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TS 中的等价物。OpenAI 的 Node SDK 暴露了 `zodResponseFormat(Invoice)`，它转换为 API 的 JSON Schema 载荷。

### 拒绝

严格模式不能强迫模型回答。如果输入不适合 schema（"邮件是一首诗，不是发票"），模型发出一个包含原因的 `refusal` 字段。你的代码必须将此视为一等结果，而不是失败。拒绝也可以作为安全信号：模型被要求从受保护内容的邮件中提取信用卡号时，会返回一个带有安全原因的拒绝。

### 开源的受限解码

开源权重实现使用三种技术：

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建一个确定性有限自动机；在每一步，屏蔽会违反 FSM 的 token 的 logits。
2. **带 JSON 解析器的 logit 屏蔽**：与模型步调一致地运行流式 JSON 解析器；在每一步，计算有效下一个 token 集合。
3. **带验证器的推测解码**：便宜的草案模型提议 token，验证器强制 schema。

商业提供商在幕后选择其中之一。2026 年的技术现状是：对于短结构化输出比普通生成更快，对于长输出大致相同。

### 三种失败模式

1. **解析错误。** 输出不是有效 JSON。在 strict 模式下不可能发生。在非 strict 提供商上仍可能发生。
2. **Schema 违规。** 输出可解析但违反 schema。在 strict 模式下不可能发生。在其外部很常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当你在 strict 模式之外时（Anthropic 工具使用、非 strict OpenAI、旧版 Gemini），恢复模式是：

```
生成 -> 解析 -> 验证 -> 如果失败，注入错误并重试，最多 3 次
```

一次重试通常就足够了。三次重试可以捕获弱模型的偶发失败。超过三次是坏 schema 的标志：模型对某些输入无法满足，提示或 schema 需要修复。

### 小模型支持

受限解码在小模型上有效。带有语法强制的 3B 参数开源模型在结构化任务上优于带有原始提示的 70B 参数模型。这是结构化输出对生产重要的主要原因：它将可靠性与模型大小解耦。

## 使用它

`code/main.py` 提供了一个标准库的最小 JSON Schema 2020-12 验证器（类型、required、enum、min/max、pattern、items、additionalProperties）。它包装了一个 `Invoice` schema 并通过验证器运行一个假的 LLM 输出，演示了解析错误、schema 违规和拒绝路径。在生产中，将假的输出替换为任何提供商的真实响应。

关注点：

- 验证器返回一个带有路径和消息的类型化 `[ValidationError]` 列表。这就是你想要暴露给重试提示的形状。
- 拒绝分支不会重试。它记录并返回类型化的拒绝。Phase 14 · 09 将拒绝用作安全信号。
- `additionalProperties: false` 检查在对抗性测试输入上触发，显示了为什么 strict 模式关闭了幻觉字段的大门。

## 交付物

本课程产生 `outputs/skill-structured-output-designer.md`。给定一个自由文本提取目标（发票、支持工单、简历等），该技能产生一个与 strict 模式兼容的 JSON Schema 2020-12 和一个镜像它的 Pydantic 模型，并嵌入了类型化拒绝和重试处理。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 为负数。确认验证器使用 `minimum` 约束路径拒绝它。

2. 扩展验证器以支持带判别器的 `oneOf`。常见情况：`line_item` 可以是产品或服务，由 `kind` 标记。严格模式对此有微妙的规则；查看 OpenAI 的结构化输出指南。

3. 将相同的 Invoice schema 编写为 Pydantic BaseModel，并将 `model_json_schema()` 输出与你的手写 schema 进行比较。确定 Pydantic 默认设置而手写版本省略的一个字段。

4. 测量拒绝率。构造十个不应被提取的输入（一首歌词、一个数学证明、一封空白邮件），并通过真实的提供商在 strict 模式下运行它们。计数拒绝与幻觉输出。这是你拒绝感知重试的基础事实。

5. 从头到尾阅读 OpenAI 的结构化输出指南。确定它在 strict 模式下明确禁止的、而普通 JSON Schema 允许的一个构造。然后设计一个非必要使用该禁止构造的 schema，并将其重构为与 strict 模式兼容。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| JSON Schema 2020-12 | "schema 规范" | 每个现代提供商使用的 IETF 草案 schema 方言 |
| 严格模式 | "保证的 schema" | OpenAI 标志，通过受限解码强制 schema |
| 受限解码 | "Logit 屏蔽" | 解码时强制屏蔽无效下一个 token 的执行 |
| 拒绝 | "模型拒绝" | 当输入不适合 schema 时的类型化结果 |
| 解析错误 | "无效 JSON" | 输出不能解析为 JSON；严格模式下不可能 |
| Schema 违规 | "形状错误" | 解析但违反类型 / required / enum / 范围 |
| `additionalProperties: false` | "不允许额外" | 禁止未知字段；OpenAI 严格模式下必需 |
| Pydantic BaseModel | "类型化输出" | 发出并验证 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript 输出类型" | 用于提供商输出验证的 TS 运行时 schema |
| 语法强制 | "开源受限解码" | 基于 FSM 的 logit 屏蔽，如 outlines / guidance 中 |

## 延伸阅读

- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — strict 模式、拒绝和 schema 要求
- [OpenAI — 介绍结构化输出](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布文章，解释解码保证
- [Pydantic AI — 输出](https://ai.pydantic.dev/output/) — 序列化到每个提供商的类型化 output_type 绑定
- [JSON Schema — 2020-12 发布说明](https://json-schema.org/draft/2020-12/release-notes) — 规范参考
- [Microsoft — Azure OpenAI 中的结构化输出](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明和严格模式注意事项
