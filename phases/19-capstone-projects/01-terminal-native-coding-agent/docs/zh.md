# 顶点项目 01——终端原生编码智能体

> 到 2026 年，编码智能体的形态已经定型。一个 TUI 工具集、一个状态化计划、一个沙箱化的工具面、一个规划-行动-观察-恢复的循环。Claude Code、Cursor 3 和 OpenCode 从 50 英尺外看都一样。这个顶点项目要求你端到端地构建一个——CLI 输入，PR 输出——并在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 进行比较。你将学到为什么困难的部分不是模型调用，而是工具循环、沙箱和 50 轮运行的成本上限。

**类型:** Capstone
**语言:** TypeScript / Bun（工具集）、Python（评估脚本）
**前置要求:** Phase 11（LLM 工程）、Phase 13（工具与协议）、Phase 14（智能体）、Phase 15（自主系统）、Phase 17（基础设施）
**涉及阶段:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**时间:** 35 小时

## 问题

2026 年，编码智能体成为主导的 AI 应用类别。Claude Code（Anthropic）、Cursor 3（含 Composer 2 和 Agent Tabs）、Amp（Sourcegraph）、OpenCode（11.2 万星）、Factory Droids 和 Google Jules 都提供了相同架构的变体：一个终端工具集、一个带权限的工具面、一个沙箱，以及围绕前沿模型构建的计划-行动-观察循环。前沿是狭窄的——Live-SWE-agent 在 SWE-bench Verified 上用 Opus 4.5 达到了 79.2%——但工程技艺是广阔的。大多数故障模式不是模型错误。它们是工具循环不稳定、上下文中毒、失控的 token 成本和破坏性的文件系统操作。

你无法从外部推断这些智能体。你必须构建一个，看着循环在第 47 轮因 ripgrep 返回 8MB 匹配结果而崩溃，然后重建截断层。这就是这个顶点项目的意义所在。

## 概念

工具集有四个面。**Plan** 维护一个 TodoWrite 风格的状态对象，模型每轮重写它。**Act** 分派工具调用（读取、编辑、运行、搜索、git）。**Observe** 捕获标准输出/标准错误/退出代码，截断，并将摘要反馈回去。**Recover** 处理工具错误，而不会撑爆上下文窗口或无限循环。2026 年的形态增加了一个东西：**hooks**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`——可配置的扩展点，操作员在此注入策略、遥测和护栏。

沙箱是 E2B 或 Daytona。每个任务在全新的 devcontainer 中运行，带有一个读写挂载的 git worktree。工具集从不接触主机文件系统。worktree 在成功或失败时被拆除。成本控制在三个层面实施：每轮的 token 上限、每会话的美元预算和硬性的轮数限制（通常为 50）。可观测层是带有 GenAI 语义约定的 OpenTelemetry spans，发送到自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                   |
                   v
            plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                   |                          (via OpenRouter, 模型无关)
                   v
            tool dispatcher (MCP StreamableHTTP client)
                   |
      +------------+------------+----------+
      v            v            v          v
   read/edit    ripgrep     tree-sitter   git/run
      |            |            |          |
      +------------+------------+----------+
                   |
                   v
            E2B / Daytona sandbox  (worktree 隔离)
                   |
                   v
            hooks: Pre/Post, Session, Prompt, Compact
                   |
                   v
            OpenTelemetry -> Langfuse (spans, tokens, $)
                   |
                   v
            PR via GitHub app
```

## 技术栈

- 工具集运行时：Bun 1.2 + Ink 5（终端内 React）
- 模型访问：OpenRouter 统一 API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（针对最困难的任务）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱：E2B 沙箱（JS SDK）或 Daytona devcontainers
- 代码搜索：ripgrep 子进程、17 种语言的 tree-sitter 解析器（预编译）
- 隔离：每个任务 `git worktree add`，成功/失败时清理
- 评估工具：SWE-bench Pro（已验证子集）+ Terminal-Bench 2.0 + 你自己的 30 个任务留出集
- 可观测性：OpenTelemetry SDK 及 `gen_ai.*` semconv → 自托管 Langfuse
- PR 提交：GitHub App，细粒度令牌，范围限定到目标仓库

## 构建它

1. **TUI 和命令循环。** 用 Bun 搭建项目骨架。接受 `agent run <repo> "<task>"`。打印一个分屏视图：计划面板（顶部）、工具调用流（中间）、token 预算（底部）。添加 Ctrl-C 取消，在退出前触发 `SessionEnd` 钩子。

2. **计划状态。** 定义一个类型化的 TodoWrite 模式（带备注的待办/进行中/已完成项）。模型每轮通过工具调用重写完整状态——不允许增量修改。将计划持久化到 `.agent/state.json`，以便崩溃后可以恢复。

3. **工具面。** 定义六个工具：`read_file`、`edit_file`（带差异预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（状态/差异/提交/推送）。通过 MCP StreamableHTTP 暴露，使工具集与传输无关。每个工具返回截断的输出（每次调用上限 4k tokens）。

4. **沙箱包装。** 每个任务生成一个 E2B 沙箱。`git worktree add -b agent/$TASK_ID` 一个全新分支。所有工具调用在沙箱内执行。主机文件系统不可访问。

5. **钩子。** 实现所有八种 2026 钩子类型。至少接入四个用户编写的钩子：(a) `PreToolUse` 破坏性命令守卫，阻止 worktree 外的 `rm -rf`，(b) `PostToolUse` token 记账，(c) `SessionStart` 预算初始化，(d) `Stop` 写入最终跟踪包。

6. **评估循环。** 克隆 SWE-bench Pro Python 的 30 个问题子集。针对每个问题运行你的工具集。与 mini-swe-agent（最小基线）在 pass@1、每任务轮数和每任务美元成本上进行比较。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性截断：50 轮、200k 上下文、每任务 5 美元。`PreCompact` 钩子在 150k 标记处将较旧的轮次总结为先前状态块，为新观察腾出空间而不丢失计划。

8. **PR 提交。** 成功时，最后一步是 `git push` + GitHub API 调用，在正文中附带计划和差异摘要创建 PR。

## 使用它

```
$ agent run ./my-repo "修复 worker.rs 中的竞态条件"
[plan]  1 定位 worker.rs 并列举互斥锁使用
        2 识别争议下的共享状态
        3 提出修复方案，验证测试
[tool]  ripgrep mutex.*lock -t rust           (44 个匹配，已截断)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (通过)
[plan]  1 done · 2 done · 3 done
[done]  PR 已开启: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付物

交付的技能文件位于 `outputs/skill-terminal-coding-agent.md`。给定一个仓库路径和任务描述，它在沙箱中运行完整的计划-行动-观察循环，并返回 PR URL 和跟踪包。此顶点项目的评分标准：

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | SWE-bench Pro pass@1 与基线对比 | 你的工具集 vs mini-swe-agent 在 30 个匹配的 Python 任务上 |
| 20 | 架构清晰度 | Plan/Act/Observe 分离、钩子面、工具模式——与 Live-SWE-agent 布局对照审查 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令守卫通过红队测试 |
| 20 | 可观测性 | 跟踪完整性（100% 工具调用有 span）、每轮 token 记账 |
| 15 | 开发者体验 | 冷启动 < 2s、崩溃恢复续接计划、Ctrl-C 干净地取消进行中的工具 |
| **100** | | |

## 练习

1. 将底层模型从 Claude Sonnet 4.7 替换为在 vLLM 上服务的 Qwen3-Coder-30B。比较 pass@1 和每任务美元成本。报告开源模型在哪些方面表现不佳。

2. 添加一个 `reviewer` 子智能体，在 PR 提交前读取差异并可以请求修订循环。测量误报性评审是否使 SWE-bench 通过率降至单智能体基线以下（提示：通常为是）。

3. 压力测试沙箱：编写一个尝试 `curl` 外部 URL 的任务和一个在 worktree 外写入的任务。确认两者都被 PreToolUse 钩子阻止。记录这些尝试。

4. 使用较小的模型（Haiku 4.5）实现 `PreCompact` 摘要。测量在 3 倍压缩下丢失了多少计划保真度。

5. 将 MCP StreamableHTTP 传输替换为 stdio。基准测试冷启动和每次调用延迟。为仅本地使用选出一个胜者。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Harness | "智能体循环" | 围绕模型的代码，分派工具、维护计划状态并执行预算 |
| Hook | "智能体事件监听器" | 由工具集在八个生命周期事件之一触发的用户编写脚本 |
| Worktree | "Git 沙箱" | 分离路径上的链接 git 检出；可丢弃而不影响主克隆 |
| TodoWrite | "计划状态" | 模型每轮重写的待办/进行中/已完成项的类型化列表 |
| StreamableHTTP | "MCP 传输" | 2026 MCP 修订版：长寿命 HTTP 连接，带双向流；取代 SSE |
| Token ceiling | "上下文预算" | 每轮或每会话的输入+输出 token 上限；触发压缩或终止 |
| pass@1 | "单次尝试通过率" | 在 SWE-bench 任务上首次运行即解决的比例，无重试或测试集窥探 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code)——来自 Anthropic 的参考工具集
- [Cursor 3 变更日志](https://cursor.com/changelog)——Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)——SWE-bench 工具集比较的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent)——在 SWE-bench Verified 上用 Opus 4.5 达到 79.2%
- [OpenCode](https://opencode.ai)——开源工具集，11.2 万星
- [SWE-bench Pro 排行榜](https://www.swebench.com)——此顶点项目目标的评估
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/)——StreamableHTTP、能力元数据
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——工具调用和 token 使用的 span schema
