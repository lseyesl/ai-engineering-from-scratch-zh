# 审核系统——OpenAI、Perspective、Llama Guard

> 生产审核系统将第 12-16 课定义的安全策略操作化。OpenAI 审核 API：`omni-moderation-latest`（2024）基于 GPT-4o，在一次调用中分类文本+图像；在多语言测试集上比先前版本好 42%；响应模式返回 13 个类别布尔值——骚扰、骚扰/威胁、仇恨、仇恨/威胁、非法、非法/暴力、自残、自残/意图、自残/指令、性、性/未成年人、暴力、暴力/图像；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用隐藏延迟；标记时使用占位响应。Llama Guard 3/4（第 16 课）：14 个 MLCommons 危害类别、代码解释器滥用、8 种语言（v3）、多图像（v4）。Perspective API（Google Jigsaw）：在 LLM 作为审核者浪潮之前的毒性评分；主要是带严重毒性/侮辱/亵渎变体的单维度毒性；是内容审核研究的基线。弃用：Azure 内容审核器于 2024 年 2 月弃用，2027 年 2 月退役，由 Azure AI 内容安全替代。

**Type:** Build
**Languages:** Python（stdlib，三层审核框架）
**Prerequisites:** Phase 18 · 16（Llama Guard / Garak / PyRIT）
**Time:** ~60 分钟

## 学习目标

- 描述 OpenAI 审核 API 的类别分类法及其与 Llama Guard 3 的 MLCommons 集的区别。
- 描述三层审核模式（输入、输出、自定义）并说出每种模式的一种失败模式。
- 描述 Perspective API 作为前 LLM 时代基线的位置以及它为什么仍在研究中被使用。
- 陈述 Azure 弃用时间线。

## 问题

第 12-16 课描述了攻击和防御工具。第 29 课涵盖了在用户接触产品的表面对防御进行操作的已部署审核系统。三层模式是 2026 年的默认配置。

## 概念

### OpenAI 审核 API

`omni-moderation-latest`（2024）。基于 GPT-4o。在一次调用中分类文本+图像。对大多数开发者免费。

类别（响应模式中的 13 个布尔值）：
- 骚扰、骚扰/威胁
- 仇恨、仇恨/威胁
- 自残、自残/意图、自残/指令
- 性、性/未成年人
- 暴力、暴力/图像
- 非法、非法/暴力

多模态支持适用于`暴力`、`自残`和`性`但不适用于`性/未成年人`；其余仅为文本。

对于 `code/main.py` 中的代码框架，为教学简洁起见，我们将 `/威胁`、`/意图`、`/指令`和 `/图像`子类别折叠到它们的顶层父类别中。生产代码应使用完整的 13 类别模式。

在多语言测试集上比前一代审核端点好 42%。按类别分数；应用设置阈值。

### Llama Guard 3/4

在第 16 课中已涵盖。14 个 MLCommons 危害类别（与 OpenAI 的 13 个响应模式布尔值组织方式不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）是原生多模态的，12B。

OpenAI 和 Llama Guard 的分类法重叠但不同。OpenAI 将"非法"作为一个宽泛类别；Llama Guard 有"暴力犯罪"和"非暴力犯罪"分开设置。部署根据其策略分类法适合度进行选择。

### Perspective API（Google Jigsaw）

在 LLM 作为审核者浪潮（2020 年前）之前的毒性评分系统。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。单维度主要分数（TOXICITY）带子维度变体。

因 API 稳定、有文档记录且有多年校准数据，被广泛用作内容审核研究基线。对于现代 LLM 相邻用例，Llama Guard 或 OpenAI 审核通常更合适。

### 三层模式

1. **输入审核。** 在生成前对用户的提示进行分类。如果标记则拒绝。延迟：一次分类器调用。
2. **输出审核。** 在发送前对模型的输出进行分类。如果标记则用拒绝代替。延迟：生成后一次分类器调用。
3. **自定义审核。** 领域特定规则（正则表达式、白名单、业务策略）。在输入或输出时运行。

三个层按设计是顺序的：输入审核必须在生成前完成，输出审核在生成后运行。在单个层内可应用并行性——同时对同一文本运行多个分类器（例如，OpenAI 审核 + Llama Guard + Perspective）以隐藏每个分类器的延迟。作为可选优化，在输入审核完成时可以显示占位响应（"稍等，正在检查..."），延迟第一个 token 的流式输出。标记行为是可配置的：拒绝、消毒、升级至人工审核。

### 失败模式

- **仅输入。** 无法捕获输出幻觉（第 12-14 课的编码攻击绕过了输入分类器）。
- **仅输出。** 允许任何输入到达模型；增加成本；将内部推理暴露给攻击者。
- **仅自定义。** 跨类别不鲁棒；正则表达式是脆弱的。

分层是默认配置。双重保险。

### Azure 弃用

Azure 内容审核器：2024 年 2 月弃用，2027 年 2 月退役。由 Azure AI 内容安全替代，后者基于 LLM 并与 Azure OpenAI 集成。迁移是 2024-2027 年 Azure 部署的领域级项目。

### 在 Phase 18 中的位置

第 16 课在红队背景下涵盖了审核工具。第 29 课涵盖了部署的审核。第 30 课以当前的���重用途能力证据结束。

## 使用它

`code/main.py` 构建了一个三层审核框架：输入审核员（关键词 + 类别分数）、输出审核员（对输出的相同分类器）、自定义审核员（领域规则）。你可以通过框架运行输入并观察每个层捕获了什么。

## 交付物

本课程产出 `outputs/skill-moderation-stack.md`。给定一个部署，它会推荐一个审核栈配置：输入使用哪个分类器、输出使用哪个、哪些自定义规则、以及边缘情况使用什么评判者。

## 练习

1. 运行 `code/main.py`。将良性、边缘和有害输入通过所有三层运行。报告每个输入被哪层触发。
2. 用 Perspective API 风格的毒性评分扩展框架，针对特定类别。将其阈值行为与类别分数进行比较。
3. 阅读 OpenAI 审核 API 文档和 Llama Guard 3 类别列表。将每个 OpenAI 类别映射到最相近的 Llama Guard 类别。识别三个无法干净映射的类别。
4. 为代码助手部署（例如，GitHub Copilot）设计一个审核栈。确定最相关和最不相关的类别，并提出自定义规则。
5. Azure 内容审核器于 2027 年 2 月退役。规划迁移至 Azure AI 内容安全。确定���移中风险最高的要素。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OpenAI Moderation | "omni-moderation-latest" | 基于 GPT-4o 的 13 类别（文本）分类器，具有部分多模态支持 |
| Perspective API | "Google Jigsaw 毒性" | 前 LLM 时代的毒性评分基线 |
| Llama Guard | "MLCommons 14 类别" | Meta 的危害分类器（v3：8B 文本，8 语言；v4：12B 多模态） |
| Input moderation | "生成前过滤器" | 模型调用前对用户提示的分类器 |
| Output moderation | "生成后过滤器" | 发送前对模型输出的分类器 |
| Custom moderation | "领域规则" | 部署特定规则（正则表达式、白名单、策略） |
| Layered moderation | "所有三层" | 标准生产部署模式 |

## 延伸阅读

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations)——omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama)——Llama Guard 仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/)——毒性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/)——Azure 替代方案
