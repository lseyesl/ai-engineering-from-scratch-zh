# Agno 与 Mastra：生产运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年的生产运行时配对。Agno 瞄准微秒级智能体实例化和无状态 FastAPI 后端。Mastra 在 Vercel AI SDK 基础上提供智能体、工具、工作流、统一模型路由和复合存储。

**类型：** Learn
**语言：** Python、TypeScript
**前置知识：** Phase 14 · 01（智能体循环），Phase 14 · 13（LangGraph）
**时间：** ~45 分钟

## 学习目标

- 识别 Agno 的性能目标以及它们何时重要。
- 说出 Mastra 的三个原语——Agents、Tools、Workflows——以及支持的服务器适配器。
- 解释为什么无状态会话范围的 FastAPI 后端是推荐的 Agno 生产路径。
- 为给定的技术栈（Python 优先 vs TypeScript 优先）选择 Agno 或 Mastra。

## 问题

LangGraph、AutoGen、CrewAI 是框架密集型。想要"只有智能体循环，快速，在我的运行时中"的团队选择 Agno（Python）或 Mastra（TypeScript）。两者都以一些框架拥有的原语换取原始速度和与周围技术栈更紧密的配合。

## 概念

### Agno

- Python 运行时，前身为 Phi-data。
- "没有图、链或复杂的模式——只有纯 Python。"
- 文档中的性能目标：~2μs 智能体实例化、~3.75 KiB 每个智能体的内存、~23 个模型提供商。
- 生产路径：无状态会话范围的 FastAPI 后端。每个请求启动一个新的智能体；会话状态存在于数据库中。
- 原生多模态（文本、图像、音频、视频、文件）和智能体 RAG。

速度目标在你每秒有数千个短生命周期智能体时很重要（聊天扇入、评估管道）。当一个智能体运行 10 分钟时，它们不那么重要。

### Mastra

- TypeScript，基于 Vercel AI SDK 构建。
- 三个原语：**Agents**、**Tools**（Zod 类型化）、**Workflows**。
- 统一模型路由器——跨 94 个提供商的 3,300+ 模型（2026 年 3 月）。
- 复合存储：记忆、工作流、可观测性到不同的后端；大规模可观测性推荐 ClickHouse。
- Apache 2.0 带有源可用的企业许可 `ee/` 目录。
- Express、Hono、Fastify、Koa 的服务器适配器；一流的 Next.js 和 Astro 集成。
- 提供 Mastra Studio（localhost:4111）用于调试。
- 22k+ GitHub 星标，1.0 版本（2026 年 1 月）每周 300k+ npm 下载。

### 定位

两者都不是试图成为 LangGraph。它们在以下方面竞争：

- **语言适配。** Agno 面向 Python 优先团队；Mastra 面向 TypeScript 优先团队。
- **运行时人体工程学。** Agno = 接近零开销；Mastra = 与 Vercel 生态系统集成。
- **可观测性。** 两者都与 Langfuse/Phoenix/Opik（第 24 课）集成，但 Mastra Studio 是第一方的。

### 何时选择哪个

- **Agno**——Python 后端、许多短生命周期智能体、强性能要求、FastAPI 团队。
- **Mastra**——TypeScript 后端、Next.js / Vercel 部署、统一多提供商模型路由、Zod 类型化工具。
- **LangGraph**（第 13 课）——当持久状态和显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK**——当你想要提供商的产品化形状时（第 16-17 课）。

### 这种模式出错的地方

- **为性能而性能。** 因为"2μs"听起来好而选择 Agno，但工作负载是每个请求一个慢速智能体调用。开销不是瓶颈。
- **生态系统锁定。** Mastra 的 Vercel 风格集成在 Vercel 上是加分项，在其他地方是减分项。
- **企业许可混淆。** Mastra 的 `ee/` 目录是源可用的，不是 Apache 2.0。如果你计划分叉，请阅读许可。

## 构建

本课主要是比较性的——没有一个代码工件能公正对待两个框架。参见 `code/main.py` 的并排玩具：一个最小的"运行智能体、流式输出、持久化会话"流程实现两次（一次 Agno 风格，一次 Mastra 风格）。

运行：

```
python3 code/main.py
```

两个结构不同但功能等价的轨迹。

## 使用

- **Agno**——需要速度和 FastAPI 形状的 Python 后端。
- **Mastra**——具有许多提供商和工作流原语的 TypeScript 后端。
- 两者都提供第一方可观测性钩子。两者都与 Langfuse 集成。

## 交付

`outputs/skill-runtime-picker.md` 根据技术栈、延迟预算和操作形态选择 Agno、Mastra、LangGraph 或提供商 SDK。

## 练习

1. 阅读 Agno 文档。将标准库 ReAct 循环（第 01 课）移植到 Agno。什么消失了？什么保留了？
2. 阅读 Mastra 文档。将相同循环移植到 Mastra。工具类型化（Zod vs 无）有什么变化？
3. 基准测试：在你的技术栈上测量智能体实例化延迟。Agno 的 2μs 对你的工作负载重要吗？
4. 设计迁移：如果你一直在 Python 中运行 CrewAI，迁移到 Agno 会破坏什么？
5. 阅读 Mastra 的 `ee/` 许可条款。哪些限制会影响开源分叉？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Agno | "快速 Python 智能体" | 无状态会话范围的智能体运行时 |
| Mastra | "Vercel AI SDK 上的 TypeScript 智能体" | Agents + Tools + Workflows + Model Router |
| 统一模型路由器 | "多提供商访问" | 跨 94 个提供商的 3,300+ 模型的单一客户端 |
| 复合存储 | "多个后端" | 记忆/工作流/可观测性各自到一个不同存储 |
| Mastra Studio | "本地调试器" | 用于检查智能体的 localhost:4111 UI |
| 源可用 | "非开源" | 许可允许阅读源代码但限制商业使用 |

## 延伸阅读

- [Agno Agent Framework 文档](https://www.agno.com/agent-framework) — 性能目标、FastAPI 集成
- [Mastra 文档](https://mastra.ai/docs) — 原语、服务器适配器、模型路由器
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 集成引用的可观测性比较
