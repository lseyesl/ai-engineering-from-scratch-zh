# 评估驱动的 Agent 开发 (Eval-Driven Agent Development)

> Anthropic 的指导原则："从简单的 prompt 开始，用全面的评估来优化它们，只有在必要时才添加多步骤 Agent 系统。"评估不是最后一步。它是驱动 Phase 14 中所有其他选择的外层循环。

**类型：** Learn + Build
**语言：** Python (stdlib)
**前置知识：** Phase 14 全部内容
**时间：** ~60 分钟

## 学习目标

- 说出三个评估层次——静态基准、自定义离线、在线生产——以及各自的用途。
- 解释评估器-优化器 (Evaluator-Optimizer) 的紧密循环。
- 描述 2026 年的最佳实践：评估与代码共存、在 CI 中运行、门控 PR 合并。
- 将 Phase 14 的每一课连接到它所产生的评估用例。

## 问题

Agent 能通过演示。但它们在生产环境中会以演示无法预测的方式失败。基准测试回答的是"这个模型在广泛意义上是否具备能力？"，而不是"这个 Agent 是否为我的产品交付了正确的补丁？"答案是：三个层次的评估，持续运行，每个护栏和学到的规则都映射到一个评估用例。

## 概念

### 三个评估层次

1. **静态基准 (Static benchmarks)** — 代码领域的 SWE-bench Verified（第 19 课）、浏览器/桌面领域的 WebArena/OSWorld（第 20 课）、通用领域的 GAIA（第 19 课）、工具使用领域的 BFCL V4（第 06 课）。用于跨模型比较和回归门控。数据污染是真实存在的：SWE-bench+ 发现了 32.67% 的解决方案泄露。始终报告 Verified / +-audited 分数。

2. **自定义离线评估 (Custom offline evals)** — 你的产品形态：
   - LLM 作为裁判（Langfuse、Phoenix、Opik — 第 24 课）。
   - 基于执行（运行补丁，检查测试）。
   - 基于轨迹（将动作序列与黄金标准对比；OSWorld-Human 显示顶级 Agent 是黄金标准的 1.4-2.7 倍）。

3. **在线评估 (Online evals)** — 生产环境：
   - 会话回放（Langfuse）。
   - 护栏触发的告警（第 16、21 课）。
   - 每步成本/延迟追踪（第 23 课 OTel spans）。

### 评估器-优化器 (Evaluator-optimizer) — Anthropic

紧密循环：

1. 生成器 (Proposer) 产生输出。
2. 评估器 (Evaluator) 评判。
3. 优化直到评估器通过。

这是 Self-Refine（第 05 课）的泛化形式。任何你关心的 Agent 流程都可以包装在评估器-优化器中以提高可靠性。

### 2026 年最佳实践

- 评估与代码共存。
- 在每个 PR 的 CI 中运行。
- 门控合并基于评估分数（例如"与主分支相比无超过 5% 的回归"）。
- 每个护栏都映射到一个评估用例。
- 每个学到的规则（Reflexion、pro-workflow learn-rule）都映射到一个失败用例。

### 串联 Phase 14

Phase 14 的每一课都产生评估用例：

| 课程 | 产生的评估用例 |
|--------|----------------|
| 01 Agent 循环 | 预算耗尽、无限循环防护 |
| 02 ReWOO | 工具失败时规划器正确重新规划 |
| 03 Reflexion | 学到的反思在重试时生效 |
| 05 Self-Refine/CRITIC | 裁判通过优化后的输出 |
| 06 工具使用 | 参数强制转换正常；未知工具被拒绝 |
| 07-10 记忆 | 检索引用匹配来源；过期事实失效 |
| 12 工作流模式 | 每种模式产生正确输出 |
| 13 LangGraph | 恢复时精确重现状态 |
| 14 AutoGen Actors | DLQ 捕获崩溃的处理器 |
| 16 OpenAI Agents SDK | 护栏在正确的输入上触发 |
| 17 Claude Agent SDK | 子 Agent 结果返回给编排器 |
| 19-20 基准测试 | SWE-bench Verified 分数、WebArena 成功率、OSWorld 效率 |
| 21 计算机使用 | 每步安全检查捕获注入的 DOM |
| 23 OTel | Spans 发出所需属性 |
| 26 失败模式 | 检测器标记已知失败 |
| 27 Prompt 注入 | PVE 拒绝被污染的检索结果 |
| 28 编排 | 监督者路由到正确的专家 |
| 29 运行时形态 | DLQ 处理 N% 的失败率 |

如果你的评估套件为每个用例都覆盖了，你就覆盖了整个 Phase 14。

### 评估驱动开发常见的失败点

- **没有基线。** 没有上次已知良好值的评估是不可读的。存储基线。
- **LLM 裁判没有依据。** 裁判也会产生幻觉。CRITIC 模式（第 05 课）——裁判基于外部工具进行判断。
- **过度拟合评估。** 针对评估进行优化会偏离生产实用性。轮换用例。
- **不稳定的评估。** 非确定性用例导致误报。固定随机种子，快照状态。

## 动手构建

`code/main.py` 是一个基于 stdlib 的评估框架：

- 用例注册表，带分类（基准、自定义、在线）。
- 一个脚本化的待测 Agent。
- 评估器-优化器循环：生成、评判、优化直到通过或达到最大轮次。
- CI 门控：聚合通过率 + 与基线的回归检测。

运行方式：

```
python3 code/main.py
```

输出：每个用例的通过/失败、回归标志、CI 门控裁决。

## 使用场景

- 在与 Agent 代码相同的仓库中编写评估用例。
- 通过 CI 在每个 PR 上运行它们。
- 在回归时让构建失败。
- 追踪通过率随时间的变化。
- 将每个生产故障关联到一个新的评估用例。

## 交付物

`outputs/skill-eval-suite.md` 为一个 Agent 产品构建三层评估套件，包含 CI 门控和回归追踪。

## 练习

1. 找一个你的生产故障。编写一个能复现它的评估用例。你的 Agent 现在能通过吗？
2. 为你的领域构建一个 LLM 裁判评分标准，包含三个维度（事实性、语气、范围）。评分 50 个会话。
3. 将评估套件接入 CI。在 >=5% 回归时让构建失败。
4. 添加一个轨迹效率指标：Agent 走了多少步，与黄金轨迹相比如何？
5. 将 Phase 14 的每一课映射到你套件中的一个评估用例。有缺失吗？那就是需要填补的空白。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| Static benchmark | "现成的评估" | SWE-bench、GAIA、AgentBench、WebArena、OSWorld |
| Custom offline eval | "领域评估" | 在你的产品形态上使用 LLM 裁判 / 执行 / 轨迹评估 |
| Online eval | "生产评估" | 会话回放、护栏告警、成本/延迟追踪 |
| Evaluator-optimizer | "生成-评判-优化" | 迭代直到裁判通过 |
| CI gate | "合并阻塞器" | 在评估回归时让构建失败 |
| Baseline | "上次已知良好值" | 用于检测回归的参考分数 |
| Trajectory efficiency | "步数/黄金标准" | Agent 步数除以人类专家最小步数 |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — "从简单开始，用评估优化"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 精选基准测试
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — 工具使用基准测试
- [Langfuse docs](https://langfuse.com/) — 评估 + 会话回放实践