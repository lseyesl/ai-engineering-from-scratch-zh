# 检查点与回滚 (Checkpoints and Rollback)

> 每个图状态转换都会持久化。当工作进程崩溃时，其租约过期，另一个工作进程从最新检查点接替。Cloudflare Durable Objects 跨数小时或数周保持状态。提议后提交（第 15 课）为每个操作定义了一个回滚计划。操作后验证闭环。EU AI 法案第 14 条要求对高风险系统进行有效的人类监督 —— 在实践中这意味着检查点必须可查询、回滚必须经过演练、审计追踪必须在部署后仍然存活。尖锐的故障模式：没有幂等键和前置条件检查，临时失败后的重试可能会重复执行一个已批准的操作。操作后验证正是捕获这一问题的机制。

**Type:** 学习 (Learn)
**Languages:** Python（stdlib，检查点和回滚状态机）
**Prerequisites:** Phase 15 · 12（持久化执行），Phase 15 · 15（提议后提交）
**Time:** ~60 分钟

## 问题 (The Problem)

持久化执行（第 12 课）使崩溃的智能体可恢复。提议后提交（第 15 课）使已批准的操作可审计。本课将它们结合起来：当一个已批准的操作部分执行、崩溃然后恢复时会发生什么？回滚何时运行，针对什么状态运行？

真实系统以不同方式实现这一点：

- **LangGraph** 将每个图状态转换检查点到 PostgreSQL。工作进程崩溃时，租约释放，另一个工作进程从最新检查点恢复。工作流在 `interrupt()` 上暂停，而 `interrupt()` 本身也会持久化。
- **Cloudflare Durable Objects** 跨数小时或数周保持每个键的状态。将计算与已批准操作的存储放在一起。
- **Microsoft Agent Framework** 在工作流 API 中暴露 `Checkpoint` 原语；重放加幂等性覆盖了重试。

在每种情况下，真正有效的组合是：幂等键（防止重复执行）+ 前置条件检查（状态仍然是我们批准时的状态）+ 操作后验证（副作用确实发生了）+ 验证失败时回滚。

## 概念 (The Concept)

### 每个转换都持久化 (Every transition persists)

图状态转换是将工作流从一个命名状态移动到另一个命名状态的任何步骤。朴素实现仅在特定提交点持久化；生产实现持久化每个转换。成本（几次额外的写入）相对于可靠性增益（重放可以落在任何位置，租约恢复精确）来说是小的。

### 租约恢复 (Lease recovery)

当工作进程崩溃时，工作流不会丢失；租约（一个短期的声明，表明此工作进程正在执行此运行）只是过期。另一个工作进程获取最新检查点并恢复。租约机制是生产系统能够在滚动部署中存活而不丢失进行中工作的原因。

### 幂等性加前置条件 (Idempotency plus preconditions)

仅有幂等性是不够的。考虑：一个工作流被批准"在余额 > 1000 美元时从 A 转账 100 美元到 B。"工作流已提交，中途崩溃，然后恢复。如果只检查幂等键，并且执行恢复，转账运行一次（正确）。但考虑在崩溃和恢复之间，A 的余额通过另一个工作流降至 500 美元。幂等检查仍然通过；前置条件不通过。没有前置条件检查，我们就会产生透支。

每个关键操作都需要两者：

- **幂等键 (Idempotency key)**：防止重复执行。
- **前置条件检查 (Precondition check)**：确认状态仍然与批准时一致。

### 操作后验证 (Post-action verification)

"工具返回了 200"不是验证。真正的验证重新读取目标状态并确认副作用确实发生了。模式：

- 数据库更新：`UPDATE ... RETURNING *` 然后断言返回的行与预期状态匹配。
- 邮件发送：提交后在已发送文件夹中检查消息 ID。
- 文件写入：读回文件并计算哈希值。
- API 调用：对目标资源进行后续 `GET`。

如果验证失败，工作流处于已知的坏状态。回滚启动。

### 回滚计划 (Rollback plans)

提议后提交（第 15 课）中的每个关键操作都带有一个回滚计划。类型：

- **带内回滚 (In-band rollback)**：直接逆转副作用（`INSERT` 后 `DELETE`，发送后 `发送更正邮件`）。
- **补偿事务 (Compensating transaction)**：一个中和原始操作的新操作（标准 SAGA 模式）。
- **带外回滚 (Out-of-band rollback)**：告警人类、暂停工作流、将坏状态留待调查。

无操作回滚（"我们无法撤销这个"）必须在提议中明确命名。没有回滚的操作在提交时需要更强的 HITL（第 15 课挑战-响应）。

### EU AI 法案第 14 条的操作性解读 (EU AI Act Article 14 operational reading)

第 14 条要求对高风险系统进行"有效的人类监督"。在操作层面，实施者将其解读为：

- 检查点可供审计员查询。
- 回滚经过演练（至少端到端测试一次）。
- 审计追踪在部署后仍然存活（检查点后端不是临时的）。
- 验证失败会触发告警，而不是静默记录。

一个在提交中途崩溃、恢复并完成副作用而没有验证+回滚路径的工作流，无法通过第 14 条测试。

### 尖锐的故障模式：重复执行 (The sharp failure mode: the double-execute)

这个领域最常见的生产事故：

1. 操作已批准，幂等键 k。
2. 提交开始，执行，返回 200。
3. 工作流在持久化"已提交"状态之前崩溃。
4. 工作流恢复；看到"已批准但未提交"；重新执行。
5. 副作用触发两次。

缓解措施：在执行前持久化一个"进行中"意图，使用幂等键执行，然后仅在操作后验证成功后标记为"已提交"。如果操作触发但状态写入失败，你知道要验证并（如有必要）重新触发。如果状态写入成功但操作失败，你验证并通过恢复路径精确触发一次。

## 使用它 (Use It)

`code/main.py` 实现了一个带检查点的工作流，包含幂等性、前置条件、验证和回滚。驱动程序模拟了四种场景：干净运行、崩溃后重试（幂等性捕获）、前置条件失败（工作流中止而不触发）、验证失败（回滚触发）。

## 交付物 (Ship It)

`outputs/skill-rollback-rehearsal.md` 为拟议的工作流设计一个回滚演练测试，并审计检查点后端的审计追踪持久性。

## 练习 (Exercises)

1. 运行 `code/main.py`。验证四种场景。对于提交期间崩溃的情况，确认操作在重试中精确触发一次。

2. 修改"先标记完成，再执行"的模式，使状态写入在操作之后触发。重新运行崩溃场景。测量有多少重复操作被触发。

3. 为特定的生产操作（例如，"发布到 Slack 频道"）设计一个回滚计划。分类为带内、补偿或带外。证明选择的合理性。

4. 选择一个你熟悉的工作流。识别每个状态转换。为每个标记持久性要求（持久化/不持久化）。统计你当前没有持久化的那些。

5. 演练回滚测试：设计一个端到端测试，运行一个真实工作流，使其崩溃，并确认回滚路径触发。测试断言什么？

## 关键术语 (Key Terms)

| Term | What people say | What it actually means |
|---|---|---|
| Checkpoint | "保存点" | 每个图状态转换持久化到持久化存储 |
| Lease | "工作进程声明" | 工作进程正在执行运行的短期声明；崩溃时过期 |
| Precondition | "状态关卡" | 断言状态仍然与已批准的操作一致 |
| Post-action verify | "重新读取检查" | 确认副作用在目标系统中确实发生了 |
| In-band rollback | "直接撤销" | 用逆操作逆转副作用 |
| Compensating transaction | "SAGA 撤销" | 一个中和原始操作的新操作 |
| Mark-as-done-first | "状态写入顺序" | 在从提交返回之前持久化已提交状态 |
| Article 14 | "EU AI 法案人类监督" | 操作性要求：可查询的检查点、演练过的回滚、可审计的追踪 |

## 延伸阅读 (Further Reading)

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) —— 检查点原语和租约恢复。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) —— Durable Objects 作为状态基础。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) —— 监管基线。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) —— 长周期工作流的可靠性框架。
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) —— Claude Code Routines 的工作流形态。