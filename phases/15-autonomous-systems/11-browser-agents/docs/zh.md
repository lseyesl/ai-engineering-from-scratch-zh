# 浏览器代理与长周期 Web 任务

> ChatGPT agent（2025 年 7 月）将 Operator 和 deep research 合并为一个浏览器/终端代理，并将 BrowseComp SOTA 设定在 68.9%。OpenAI 于 2025 年 8 月 31 日关闭了 Operator —— 产品层面的整合。Anthropic 收购 Vercept 将 Claude Sonnet 在 OSWorld 上的得分从低于 15% 提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的假阴性率，并发布了 258 个任务的 Hard 子集。这些数字是真实的。攻击面同样真实：OpenAI 的安全负责人公开表示，对浏览器代理的间接提示注入"不是一个可以完全修补的 bug"。2025-2026 年有记录的攻击包括：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks），以及 Perplexity Comet 中的一键劫持。

**类型：** 学习
**语言：** Python（stdlib，间接提示注入攻击面模型）
**前置要求：** 第 15 阶段 · 10（权限模式），第 15 阶段 · 01（长周期代理）
**时间：** ~45 分钟

## 问题

浏览器代理是一个读取不受信任内容并采取有后果行动的长周期代理。代理访问的每个页面都是用户未编写的输入。页面上的每个表单都是一个潜在的命令通道。2025-2026 年的攻击语料库显示这并非假设：Tainted Memories 让攻击者通过精心构造的页面将恶意指令绑定到代理的记忆中；HashJack 将命令隐藏在代理访问的 URL 片段中；Perplexity Comet 劫持只需一次点击即可完成。

防御图景令人不安。OpenAI 的安全负责人直言不讳：间接提示注入"不是一个可以完全修补的 bug"。这是因为攻击存在于代理的读取-行动边界上，而这个边界在架构上是模糊的 —— 模型读取的每个 token 原则上都可以被解读为一条指令。

本课命名了攻击面，命名了基准测试全景（BrowseComp、OSWorld、WebArena-Verified），并建模了一个最小的间接提示注入场景，以便你在第 14 课和第 18 课中推理真正的防御措施。

## 概念

### 2026 年全景，每个系统一段话

**ChatGPT agent（OpenAI）。** 2025 年 7 月发布。统一了 Operator（浏览）和 Deep Research（多小时研究）。2025 年 8 月 31 日关闭了独立的 Operator。BrowseComp 上 SOTA 为 68.9%；在 OSWorld 和 WebArena-Verified 上表现强劲。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept 专注于计算机使用能力。将 Claude Sonnet 在 OSWorld 上的得分从 <15% 提升到 72.5%。Claude Computer Use 以工具 API 形式提供。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use 集成提供了计算机使用控制；FSF v3（2026 年 4 月，第 20 课）专门追踪 ML 研发领域的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个有据可查的问题：原始 WebArena 的假阴性率约为 11.3%（标记为失败但实际上已解决的任务）。Verified 版本使用人工筛选的成功标准重新评分，并增加了一个 258 个任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准测试 | 衡量什么 | 时间跨度 |
|---|---|---|
| BrowseComp | 在时间压力下在开放网络上查找特定事实 | 分钟级 |
| OSWorld | 代理操作完整桌面（鼠标、键盘、shell） | 十分钟级 |
| WebArena-Verified | 模拟站点中的事务性 Web 任务 | 分钟级 |
| Hard 子集 | WebArena-Verified 中需要多页面状态转换的任务 | 十分钟级 |

不同的维度。高 BrowseComp 分数说明代理能查找事实；并不说明代理能预订航班。OSWorld 分数更接近"它能在我的桌面上工作吗"。WebArena-Verified 更接近"它能完成一个流程吗"。任何生产决策都需要与任务分布匹配的基准测试。

### 攻击面，逐一命名

1. **间接提示注入（Indirect prompt injection）。** 不受信任的页面内容包含指令。代理读取它们。代理执行它们。公开示例：2024 年 Kai Greshake 等人，2025 年 Tainted Memories 论文，2026 年 HashJack（Cato Networks）。
2. **URL 片段/查询注入。** 被爬取 URL 的 `#fragment` 或查询字符串包含命令。从未被可视化渲染；仍然在代理的上下文中。
3. **记忆绑定攻击（Memory-binding attacks）。** 页面指示代理写入持久记忆（第 12 课涵盖持久状态）。下一个会话中，记忆在没有可见触发器的情况下触发载荷。
4. **针对已认证会话的 CSRF 式攻击。** Tainted Memories 类别：代理在某处已登录；攻击者的页面发出状态变更请求，代理使用用户的 cookie 执行。
5. **一键劫持（One-click hijack）。** 一个视觉上无害的按钮携带代理跟随的载荷。Comet 类别。
6. **代理宿主表面中的内容安全策略（CSP）漏洞。** 渲染层和工具层本身可能成为攻击向量；浏览器-在-浏览器-代理的堆栈很宽。

### 为什么"不能完全修补"

攻击与代理的能力同构。代理必须读取不受信任的内容才能完成其工作。代理读取的任何内容都可能包含指令。代理遵循的任何指令都可能与用户的实际请求不一致。防御措施（信任边界、分类器、工具允许列表、对后果性动作的人工介入）提高了攻击的成本并缩小了其爆炸半径。它们不会关闭这个类别。

这与 Lob 定理（第 8 课）的推理模式相同：代理无法证明下一个 token 是安全的；它只能建立一个系统，使不安全的 token 更可检测。

### 实际投入使用的防御姿态

- **读/写边界（Read/write boundary）。** 读取从不产生后果。写入（提交表单、发布内容、调用有副作用的工具）需要全新的人工审批，如果发起内容来自信任边界之外。
- **每个任务的工具允许列表。** 代理可以浏览；除非该工具被显式启用用于该任务，否则不能发起电汇。第 13 课涵盖预算。
- **会话隔离。** 浏览器代理会话仅使用限定范围的凭证运行。无生产认证，无个人邮箱。每个 HTTP 请求的日志保留用于审计。
- **内容清理器（Content sanitizer）。** 获取的 HTML 在拼接到模型上下文之前，会剥离已知的危险模式。（减少简单攻击；不能阻止复杂的载荷。）
- **对后果性动作的人工介入（HITL）。** 提议-然后-提交模式（第 15 课）。
- **记忆上的蜜罐令牌（Canary tokens）。** 如果记忆条目被触发，用户会看到它（第 14 课）。

## 使用它

`code/main.py` 模拟一个微型浏览器代理对三个合成页面的运行。一个页面是良性的，一个在可见文本中有直接的提示注入块，一个在 URL 片段中有注入（不可见但在代理的上下文中）。脚本展示了 (a) 一个幼稚的代理会做什么，(b) 读/写边界能捕获什么，(c) 清理器能捕获什么，(d) 两者都未捕获什么。

## 交付它

`outputs/skill-browser-agent-trust-boundary.md` 限定一个提议的浏览器代理部署的范围：它触及哪些信任区域、它被授权写入什么，以及在首次运行前必须就位的防御措施。

## 练习

1. 运行 `code/main.py`。识别清理器能捕获但读/写边界不能捕获的攻击，以及只有读/写边界能捕获的攻击。

2. 扩展清理器以检测一类 HashJack 风格的 URL 片段注入。在带有合法片段的良性 URL 上测量误报率。

3. 选择一个你知道的真实浏览器代理工作流（例如"预订航班"）。列出每次读取和每次写入。标记哪些写入需要人工介入并说明原因。

4. 阅读 WebArena-Verified ICLR 2026 论文。找出原始 WebArena 评分不可靠的一类任务，并解释 Verified 子集如何解决该问题。

5. 为浏览器代理场景设计一个记忆蜜罐。你会存储什么、存储在哪里，以及什么会触发警报？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|---|---|---|
| 间接提示注入 (Indirect prompt injection) | "糟糕的页面文本" | 代理读取的页面中不受信任的内容包含代理执行的指令 |
| Tainted Memories | "记忆攻击" | 代理将攻击者提供的指令写入持久记忆；下一个会话触发 |
| HashJack | "URL 片段攻击" | 载荷隐藏在 URL 片段/查询字符串中，在代理上下文中但未可视化渲染 |
| 一键劫持 (One-click hijack) | "坏按钮" | 可见的交互元素携带代理执行的后续载荷 |
| BrowseComp | "Web 搜索基准测试" | 在开放网络上查找特定事实；分钟级时间跨度 |
| OSWorld | "桌面基准测试" | 完整的操作系统控制；多步骤 GUI 任务 |
| WebArena-Verified | "修复的 Web 任务基准测试" | ServiceNow 重新评分的 WebArena，带有 Hard 子集 |
| 读/写边界 (Read/write boundary) | "副作用门" | 读取从不产生后果；如果内容来自信任边界外，写入需要全新审批 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) —— Operator 和 deep research 的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) —— Operator 谱系及成为 ChatGPT agent 的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) —— 原始基准测试。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) —— ICLR 2026 修复子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) —— 包含计算机使用代理的攻击面讨论。