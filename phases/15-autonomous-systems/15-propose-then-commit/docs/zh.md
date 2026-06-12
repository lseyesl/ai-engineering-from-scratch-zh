# 人在回路中：提议后提交 (Human-in-the-Loop: Propose-Then-Commit)

> 2026 年关于 HITL 的共识是具体的。它不是"智能体询问，用户点击批准"。它是提议后提交 (propose-then-commit)：提议的操作持久化到带有幂等键的持久化存储中；向审查者展示意图、数据血缘、所触及的权限、爆炸半径和回滚计划；仅在收到肯定确认后提交；执行后验证以确认副作用确实发生了。LangGraph 的 `interrupt()` 加 PostgreSQL 检查点、Microsoft Agent Framework 的 `RequestInfoEvent`、以及 Cloudflare 的 `waitForApproval()` 都实现了相同的形态。典型的故障模式是橡皮图章式批准："批准？"被点击而没有审查。有记录的缓解措施是带有明确检查清单的挑战-响应机制。

**Type:** 学习 (Learn)
**Languages:** Python（stdlib，带幂等性的提议后提交状态机）
**Prerequisites:** Phase 15 · 12（持久化执行），Phase 15 · 14（绊网）
**Time:** ~60 分钟

## 问题 (The Problem)

智能体执行一个操作。用户必须决定：批准还是不批准。如果决定是即时的，那可能不是真正的审查。如果决定是结构化的，它很慢但值得信赖。工程问题是如何让结构化审查成为阻力最小的路径。

2023 年时代的 HITL 模式是一个同步提示："智能体想向 X 发送邮件，正文为 Y —— 批准？"用户点击批准。每个人都觉得系统是安全的。实际上，这种表面被严重橡皮图章化：用户批准很快，批准预测不了什么，当智能体出错时，审计追踪显示一长串用户无法回忆的批准历史。

2026 年的模式 —— 提议后提交 —— 将 HITL 置于持久化基础之上，附加结构化元数据，并要求肯定的提交。每个托管智能体 SDK 都提供了一种版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名称不同，形态不变。

## 概念 (The Concept)

### 提议后提交状态机 (The propose-then-commit state machine)

1. **提议 (Propose)。** 智能体生成一个提议的操作。持久化到持久化存储（PostgreSQL、Redis、Durable Object）。包括：
   - 意图 (intent) —— 智能体为什么这样做
   - 数据血缘 (data lineage) —— 什么来源导致了此提议
   - 所触及的权限 (permissions touched) —— 哪些作用域/文件/端点
   - 爆炸半径 (blast radius) —— 最坏情况是什么
   - 回滚计划 (rollback plan) —— 如果提交了，如何撤销
   - 幂等键 (idempotency key) —— 每个提议唯一；重新提交返回相同记录
2. **展示 (Surface)。** 审查者看到带有所有元数据的提议。审查者是人（而不是智能体自我审查）。
3. **提交 (Commit)。** 肯定确认。操作执行。
4. **验证 (Verify)。** 执行后，读取并确认副作用。如果验证步骤失败，系统处于已知的坏状态，告警机制启动。

### 幂等键 (The idempotency key)

没有幂等键，临时失败后的重试可能会重复执行一个已批准的操作。具体例子：用户批准"从 A 转账 100 美元到 B。"网络抖动。工作流重试。用户批准了一次，但转账执行了两次。幂等键将批准绑定到一个唯一的副作用上；第二次执行是无操作。

这与 Stripe 和 AWS API 使用的幂等模式相同。将其用于智能体批准在 Microsoft Agent Framework 文档中有明确说明。

### 持久性：为什么批准比进程更持久 (Durability: why approvals outlast processes)

批准等待室是智能体不拥有的一块状态。工作流被暂停（第 12 课）。当批准到达时，工作流从那个确切点恢复。这就是为什么 LangGraph 将 `interrupt()` 与 PostgreSQL 检查点配对，而不仅仅是内存状态 —— 两天后到达的批准仍然能找到完整的工作流。

### 橡皮图章式批准与挑战-响应缓解措施 (Rubber-stamp approvals and the challenge-and-response mitigation)

HITL 的默认 UI（"批准"/"拒绝"按钮）产生快速批准，没有真正的审查。有记录的缓解措施：一个挑战-响应检查清单，要求在批准按钮启用之前对特定问题给出肯定回答。具体形态：

- "你理解这触及了什么资源吗？[ ]"
- "你确认爆炸半径是可接受的吗？[ ]"
- "如果失败，你有回滚计划吗？[ ]"

这不是为了官僚主义而官僚主义 —— 而是一个强制函数。无法勾选方框的审查者要么要求澄清（升级），要么拒绝（安全默认）。Anthropic 的智能体安全研究明确引用检查清单驱动的 HITL 作为橡皮图章式批准模式的缓解措施。

### 什么算关键操作 (What counts as consequential)

并非每个操作都需要提议后提交。2026 年的指导：

- **关键操作 (Consequential actions)**（始终 HITL）：不可逆写入、金融交易、对外通信、生产数据库更改、破坏性文件系统操作。
- **可逆操作 (Reversible actions)**（有时 HITL）：对本地文件的编辑、暂存环境更改、带有明确回滚的可逆写入。
- **读取和检查 (Reads and inspections)**（从不 HITL）：读取文件、列出资源、调用只读 API。

### 操作后验证 (Post-action verification)

"提交已运行"不等于"副作用已发生"。网络分区和竞态条件可能导致工作流认为它成功了，而后端并未持久化。验证步骤在提交后重新读取目标资源以确认。这与数据库事务中的 `RETURNING` 子句或 AWS 的 `GetObject` 后跟 `PutObject` 是相同的模式。

### EU AI 法案第 14 条 (EU AI Act Article 14)

第 14 条要求对欧盟高风险 AI 系统进行有效的人类监督。"有效"不是装饰性的。监管语言明确排除了橡皮图章模式。带有挑战-响应的提议后提交是在 Microsoft Agent Governance Toolkit 合规文档中能够经受第 14 条审查的形态。

## 使用它 (Use It)

`code/main.py` 在 stdlib Python 中实现了一个提议后提交状态机。持久化存储是一个 JSON 文件。幂等键是 (thread_id, action_signature) 的哈希值。驱动程序模拟了三种情况：一个干净的批准流程、临时失败后的重试（不能重复执行）、以及橡皮图章默认与挑战-响应流程的对比。

## 交付物 (Ship It)

`outputs/skill-hitl-design.md` 审查一个拟议的 HITL 工作流是否符合提议后提交形态，并标记缺失的元数据、幂等性、验证或挑战-响应层级。

## 练习 (Exercises)

1. 运行 `code/main.py`。确认已批准提议的重试使用持久化记录且不重新执行。现在将幂等键改为包含时间戳，显示重试会重复执行。

2. 扩展提议记录，添加一个 `rollback` 字段。模拟一个验证步骤失败的执行。显示回滚自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。找出 API 包含但玩具引擎缺少的一个元数据字段。添加它并解释它防范什么。

4. 为特定操作（例如，"发布到公共 Twitter 账户"）设计一个挑战-响应检查清单。审查者必须回答哪三个问题？为什么是这三个？

5. 选择一个同步"批准？"提示就足够的情况（不需要持久化存储）。解释原因，并说明你接受的风险类别。

## 关键术语 (Key Terms)

| Term | What people say | What it actually means |
|---|---|---|
| Propose-then-commit | "两阶段批准" | 持久化的提议 + 肯定提交 + 验证 |
| Idempotency key | "重试安全令牌" | 每个提议唯一；第二次执行无操作 |
| Data lineage | "来自哪里" | 导致提议的特定源内容 |
| Blast radius | "最坏情况" | 如果操作出错的影响范围 |
| Rubber-stamp | "快速批准" | 没有真正审查就点击"批准" |
| Challenge-and-response | "强制检查清单" | 审查者必须肯定回答特定问题 |
| RequestInfoEvent | "MS Agent Framework 原语" | 带有结构化元数据的持久化 HITL 请求 |
| `interrupt()` / `waitForApproval()` | "框架原语" | LangGraph / Cloudflare 的相同形态等价物 |

## 延伸阅读 (Further Reading)

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) —— `RequestInfoEvent`、持久化批准。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) —— `waitForApproval()` 和 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) —— HITL 作为长周期风险的缓解措施。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) —— 高风险系统的监管基线。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) —— 围绕监督的宪法框架。