# 智能体指令作为可执行约束（Agent Instructions as Executable Constraints）

> 以散文形式编写的指令只是愿望。以约束形式编写的指令则是测试。工作台将每条规则转化为智能体在运行时可以检查、审查者在事后可以验证的东西。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 32 (Minimal Workbench)
**Time:** ~50 分钟

## 学习目标

- 将路由散文与操作规则分离。
- 将启动规则、禁止行为、完成定义、不确定性处理和审批边界表达为机器可检查的约束。
- 实现一个规则检查器，根据规则集对一次运行进行评分。
- 使规则集对 diff 友好，以便审查可以看到哪些内容发生了变化。

## 问题

典型的 `AGENTS.md` 读起来像入职文档。它告诉智能体"要小心"、"全面测试"、"不确定就问"。三天后，智能体提交了一个没有测试的变更，写入了一个禁止目录，而且从未询问过，因为它根本不知道界限在哪里。

指令在可操作时是强大的，在空泛时是软弱的。解决方案是编写工作台可以解释、审查者可以评分的规则。

## 概念

规则应放在 `docs/agent-rules.md` 中，远离简短的路由根文件。每条规则都有一个名称、一个类别和一个检查函数。

```mermaid
flowchart LR
  Router[AGENTS.md] --> Rules[docs/agent-rules.md]
  Rules --> Checker[rule_checker.py]
  Checker --> Report[rule_report.json]
  Report --> Reviewer[Reviewer]
```

### 覆盖大多数规则的五种类别

| 类别 | 规则回答的问题 | 示例 |
|----------|---------------------------|---------|
| 启动（Startup） | 工作开始前必须满足什么条件？ | "状态文件存在且未过期" |
| 禁止（Forbidden） | 什么绝对不能发生？ | "不要编辑 `scripts/release.sh`" |
| 完成定义（Definition of done） | 什么证明任务已完成？ | "pytest 退出码为 0 且验收线通过" |
| 不确定性（Uncertainty） | 智能体不确定时该怎么做？ | "创建问题笔记，而不是猜测" |
| 审批（Approval） | 什么需要人工批准？ | "任何新依赖、任何生产环境写入" |

一条规则如果无法归入这五类中的一类，通常意味着它应该被拆分成两条规则。强制拆分。

### 规则是机器可读的

每条规则都有一个 slug、一个类别、一行描述和一个 `check` 字段，该字段引用 `rule_checker.py` 中的一个函数。添加一条规则意味着添加一个检查函数；检查器随工作台一起增长。

### 规则对 diff 友好

规则以每条一个标题的形式存放在单个 markdown 文件中。重命名在 diff 中可见。新规则放在其类别的顶部。过时的规则被删除，而不是注释掉，因为工作台是事实来源，而不是团队上个季度感受的聊天记录。

### 规则与框架护栏

框架护栏（OpenAI Agents SDK guardrails、LangGraph interrupts）在运行时层面执行规则。本课中的规则集是人类可读、可审查的契约，这些护栏正是实现该契约的手段。两者都需要：运行时在单轮交互中捕获违规，规则集证明运行时在做正确的事情。

### 渐进披露：地图，而非百科全书

`AGENTS.md` 不断膨胀的原因是每次事故都会添加一条规则，而没有任何事故会删除一条规则。一年后，文件有两千行，智能体读了第一屏就用完了注意力预算，只执行了被告知内容的一小部分。巨大的指令文件失败的原因与四十页的入职文档相同：读者浏览一次后再也不会回到重要的部分。

解决方案不是更短的文件，而是分层的文件。根路由器保持足够小以在每次会话中阅读，只包含指针。深度内容放在智能体仅在任务涉及时才加载的主题文件中。给智能体一张地图，而不是整本百科全书，让它走到需要的那一页。

```
AGENTS.md                  # 路由器，< 50 行：仓库是什么、去哪里看、5 条硬性规则
docs/
  agent-rules.md           # 完整规则集（本课）
  architecture.md          # 任务涉及模块边界时加载
  testing.md               # 任务编写或运行测试时加载
  deploy.md                # 仅在发布工作时加载，由审批规则门控
feature_list.json          # 待办列表（Phase 14 · 36）
```

| 层级 | 所在位置 | 何时阅读 | 大小预算 |
|------|----------|-----------|----------|
| 路由器 | `AGENTS.md` | 每次会话，始终 | 50 行以内 |
| 规则 | `docs/agent-rules.md` | 每次会话，启动时 | 每个类别一屏 |
| 主题文档 | `docs/<topic>.md` | 仅当任务涉及该主题时 | 需要多深就多深 |

两个测试保持分层的诚实。可达性测试：智能体从路由器出发最多两跳就能到达任何规则，因此路由器必须按路径链接每个主题文档，而不是用散文描述它。新鲜度测试：路由器足够短，审查者在每个 PR 上都会重新阅读它，这是阻止它悄悄膨胀回它所取代的百科全书的唯一手段。一个不再解析的指针比缺失的规则是更严重的失败，因此路由器中的断链本身就是启动检查违规。

## 构建它

`code/main.py` 提供：

- `agent-rules.md` 解析器，将规则加载到数据类中。
- `rule_checker.py` 风格的检查函数，每个 `check` 引用对应一个。
- 一个违反两条规则的演示智能体运行，以及一个捕获这些违规的检查过程。

运行：

```
python3 code/main.py
```

输出：解析后的规则集、运行轨迹、每条规则的通过/失败状态，以及保存在脚本旁边的 `rule_report.json`。

## 生产环境中的模式

三种模式能让一个规则集持续一个季度，而不是在一周内退化。

**编写时标记严重级别。** 每条规则携带 `severity`：`block`、`warn` 或 `info`。检查器报告所有三种级别；运行时仅在 `block` 级别拒绝执行。大多数团队早期会过度标注严重级别，然后在截止日期压力下悄悄降低；编写时标记迫使你在前期就校准好。与验证门（Phase 14 · 38）配合使用，后者将对 `block` 规则的任何覆盖签署到 `overrides.jsonl` 审计日志中。

**规则过期作为强制函数。** 每条规则携带一个 `expires_at` 日期（默认为编写后 90 天）。当一条未过期的规则连续 60 天没有违规时，检查器会发出警告；下一次季度审查要么证明保留它的理由，要么将其降级为 `info`，要么删除它。Cloudflare 的生产 AI 代码审查数据（2026 年 4 月，30 天内 5,169 个仓库上的 131,246 次审查运行）显示，具有显式过期机制的规则集每个仓库保持在 30 条规则以下；而没有过期机制的规则集会增长到 80 条以上，且大多数从未触发过。

**Markdown 作为源码，JSON 作为缓存。** `agent-rules.md` 是编写的文件；`agent-rules.lock.json` 是检查器在热路径中读取的缓存。锁文件由 pre-commit 钩子重新生成。Markdown diff 是可审查的；JSON 解析不会拖慢每一轮交互。与 `package.json` / `package-lock.json` 和 `Cargo.toml` / `Cargo.lock` 的模式相同。

## 使用它

在生产环境中：

- Claude Code、Codex、Cursor 在会话启动时读取规则，并在拒绝操作时引用这些规则。检查器在 CI 中重新运行它们以捕获静默漂移。
- OpenAI Agents SDK guardrails 将相同的检查注册为输入和输出护栏。Markdown 是文档层；SDK 是运行时层。
- LangGraph interrupts 在运行中的节点违反规则时触发。中断处理器读取规则，询问人类，然后恢复执行。

规则集在所有三种环境中都是可移植的，因为它只是 markdown 加上函数名。

## 交付它

`outputs/skill-rule-set-builder.md` 采访项目负责人，将其现有的散文式指令分类到五个类别中，并输出一个带版本的 `agent-rules.md` 和一个检查器存根。

## 练习

1. 如果你的产品确实需要，添加第六个类别。论证为什么它不能归入现有的五个类别之一。
2. 扩展检查器，使规则可以携带严重级别（`block`、`warn`、`info`），并且报告相应地聚合。
3. 将检查器接入 CI：如果最新智能体运行中有一条 block 级别的规则失败，则构建失败。
4. 为每条规则添加一个"过期"字段。如果 90 天内没有检查失败，该规则进入审查状态。
5. 找一个真实的 `AGENTS.md`，将其重写为五类规则。其中有多少行是可操作的？有多少行是空泛的？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 可操作规则（Operational rule） | "真正的指令" | 工作台可以在运行时检查的规则 |
| 空泛规则（Aspirational rule） | "要小心" | 没有检查的规则；要么删除，要么升级 |
| 完成定义（Definition of done） | "验收标准" | 一个客观的、基于文件的证明，表明任务已完成 |
| 阻塞级别（Block severity） | "硬性规则" | 违规会中止运行；没有操作员干预无法静默 |
| 规则过期（Rule expiry） | "过时规则清理" | 在 N 天内没有失败的规则进入退役审查 |

## 延伸阅读

- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [LangGraph interrupts](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Rick Hightower, Agent RuleZ: A Deterministic Policy Engine](https://medium.com/@richardhightower/agent-rulez-a-deterministic-policy-engine-for-ai-coding-agents-9489e0561edf) — 生产环境中的 block/warn/info 严重级别
- [Cloudflare, Orchestrating AI Code Review at Scale](https://blog.cloudflare.com/ai-code-review/) — 131k 次审查运行，规则组合经验
- [microservices.io, GenAI development platform — part 1: guardrails](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html) — 规则与 CI 之间的纵深防御
- [Type-Checked Compliance: Deterministic Guardrails (arXiv 2604.01483)](https://arxiv.org/pdf/2604.01483) — Lean 4 作为规则即检查的上限
- [logi-cmd/agent-guardrails](https://github.com/logi-cmd/agent-guardrails) — 合并门实现：范围、变异测试、违规预算
- Phase 14 · 32 — 本规则集所嵌入的最小工作台
- Phase 14 · 38 — 消费规则报告的验证门
- Phase 14 · 39 — 对规则合规性进行评分的审查者智能体