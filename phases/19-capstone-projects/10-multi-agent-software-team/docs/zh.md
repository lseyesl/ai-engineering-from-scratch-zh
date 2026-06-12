# 顶点项目 10——多智能体软件工程团队

> SWE-AF 的工厂架构、MetaGPT 的基于角色的提示、AutoGen 0.4 的类型化角色图、Cognition 的 Devin 和 Factory 的 Droids 都收敛到 2026 年的相同形态：架构师规划，N 个编码者在并行工作树中工作，审阅者把关，测试者验证。并行工作树将墙上时钟转化为吞吐量。共享状态和交接协议成为故障面。顶点项目是构建这个团队，在 SWE-bench Pro 上评估，并报告哪些交接出错以及频率。

**类型:** Capstone
**语言:** Python / TypeScript（智能体）、Shell（工作树脚本）
**前置要求:** Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 15（自主系统）、Phase 16（多智能体）、Phase 17（基础设施）
**涉及阶段:** P11 · P13 · P14 · P15 · P16 · P17
**时间:** 40 小时

## 问题

单智能体编码工具在大型任务上遇到了天花板。不是因为任何单个智能体弱，而是因为 200k token 的上下文无法同时容纳架构计划、四个并行代码库切片、审阅者评论和测试输出。多智能体工厂分解了问题：架构师拥有计划，编码者在并行工作树中拥有实现，审阅者把关，测试者验证。SWE-AF 的"工厂"架构、MetaGPT 的角色、AutoGen 的类型化角色图——所有三种框架都描述了相同的形态。

故障面是交接。架构师规划了编码者无法实现的东西。编码者产生冲突的差异。审阅者批准了幻觉修复。测试者与仍在编写的编码者竞争。你将构建其中一个团队，在 50 个 SWE-bench Pro 问题上运行它，跟踪每个交接，并发布事后分析。

## 概念

角色是类型化的智能体。**架构师**（Claude Opus 4.7）阅读问题，编写计划，并将其分解为带有显式接口的子任务。**编码者**（Claude Sonnet 4.7，N 个并行实例，每个在 `git worktree` + Daytona 沙箱中）独立实现子任务。**审阅者**（GPT-5.4）读取合并后的差异并批准或请求特定更改。**测试者**（Gemini 2.5 Pro）在隔离环境中运行测试套件并报告通过/失败及产物。

通信通过共享任务板（文件后端或 Redis）进行。每个角色消费其被允许处理的任务。交接是 A2A 协议类型化的消息。协调关注点：合并冲突解决（协调者角色或自动三方合并）、共享状态同步（编码者开始后计划冻结；重新计划是独立事件）和审阅者把关（审阅者不能批准自己的更改或它提出的更改）。

Token 放大是隐藏成本。每个角色边界增加了摘要提示和交接上下文。一个 40 轮的单智能体运行变为跨四个角色的总共 160 轮。评分标准特别权衡 token 效率与单智能体基线，因为问题不是"多智能体是否有效"，而是"它每美元是否胜出"。

## 架构

```
GitHub issue URL
      |
      v
架构师 (Opus 4.7)
   读取 issue，生成含子任务和接口的计划
      |
      v
任务板 (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 并行)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
          合并协调者  (三方合并 + 冲突解决)
               |
               v
          审阅者 (GPT-5.4)
               |
               v
          测试者 (Gemini 2.5 Pro)  -> 通过? -> 打开 PR
                                    -> 失败? -> 路由回编码者
```

## 技术栈

- 编排：LangGraph，带共享状态 + 每智能体子图
- 消息传递：A2A 协议（Google 2025）用于类型化的智能体间消息
- 模型：Opus 4.7（架构师）、Sonnet 4.7（编码者）、GPT-5.4（审阅者）、Gemini 2.5 Pro（测试者）
- 工作树隔离：每个编码者 `git worktree add` + Daytona 沙箱
- 合并协调者：自定义三方合并 + LLM 中介的冲突解决
- 评估：SWE-bench Pro（50 个问题）、SWE-AF 场景、HumanEval++（单元测试）
- 可观测性：Langfuse，带角色标记的 spans，每智能体 token 记账
- 部署：K8s，每个角色作为独立 Deployment + 基于积压的 HPA

## 构建它

1. **任务板。** 文件后端的 JSONL，带类型化消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体订阅标签。

2. **架构师。** 阅读 GitHub issue，使用 Opus 4.7 和计划模板运行，要求显式子任务接口（接触的文件、公共函数、测试影响）。发出一个带子任务 DAG 的 `plan_request`。

3. **编码者。** N 个并行工作线程，每个从板上认领一个子任务。每个生成一个全新的 `git worktree add` 分支加 Daytona 沙箱。实现子任务。发出带补丁和测试差异的 `diff_ready`。

4. **合并协调者。** 所有编码者完成后，将 N 个分支三方合并到一个 staging 分支。仅在存在文件级重叠时进行 LLM 中介的冲突解决。

5. **审阅者。** GPT-5.4 读取合并后的差异。不能批准自己编写的差异。发出 `approved`（无操作）或带特定更改请求的 `review_feedback`，路由回相关编码者。

6. **测试者。** Gemini 2.5 Pro 在干净的沙箱中运行测试套件。捕获产物。发出 `test_passed` 或带堆栈跟踪的 `test_failed`。失败的测试循环回拥有失败子任务的编码者。

7. **交接记账。** 跨越角色边界的每条消息在 Langfuse 中获取一个 span，包含负载大小和使用的模型。计算每子任务 token 放大（编码者 token + 审阅者 token + 测试者 token + 架构师份额 / 编码者 token）。

8. **评估。** 在 50 个 SWE-bench Pro 问题上运行。比较 pass@1 和每个已解决问题的美元成本与单智能体基线（单个 Sonnet 4.7 在单个工作树中）。

9. **事后分析。** 对每个失败的 issue，识别出错的交接（计划太模糊、合并冲突、审阅者误批准、测试不稳定）。生成一个交接失败直方图。

## 使用它

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 个子任务 (parser, cache, api, migration)
[board]     分派给 4 个编码者在并行工作树中
[coder-A]   subtask parser  -> 42 行, 测试本地通过
[coder-B]   subtask cache   -> 88 行, 测试本地通过
[coder-C]   subtask api     -> 31 行, 测试本地通过
[coder-D]   subtask migration -> 19 行, 测试本地通过
[merge]     三方合并: 0 冲突
[reviewer]  对 cache 发表评论 (线程池大小); 路由到 coder-B
[coder-B]   修订: 92 行; 提交
[reviewer]  批准
[tester]    全部 412 个测试通过
[pr]        已开启 #3382   4 个编码者, 1 次修订, $4.90, 18m
```

## 交付物

`outputs/skill-multi-agent-team.md` 是交付物。给定一个 issue URL 和并行度级别，团队生成一个可合并的 PR，带每角色 token 记账。

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配的 50 问题子集上的 pass@1 |
| 20 | 并行加速 | 与单智能体基线的墙上时钟对比 |
| 20 | 审阅质量 | 注入错误探测上的误批准率 |
| 20 | Token 效率 | 每个已解决问题的总 token vs 单智能体 |
| 15 | 协调工程 | 合并冲突解决、交接失败直方图 |
| **100** | | |

## 练习

1. 在差异中间注入一个明显的错误（主主体前多了一个 `return None`）。测量审阅者的误批准率。调整审阅者提示直到误批准低于 5%。

2. 减少到两个编码者（架构师 + 编码者 + 审阅者 + 测试者，编码者顺序运行两个子任务）。比较墙上时钟和通过率。

3. 将合并协调者替换为单写入者约束（子任务接触不相交的文件集）。测量架构师的规划负担。

4. 将审阅者从 GPT-5.4 替换为 Claude Opus 4.7。测量误批准率和 token 成本差异。

5. 添加第五个角色：文档编写者（Haiku 4.5）。审阅后，它生成一个变更日志条目。测量文档质量是否值得额外的 token 开销。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Parallel worktree | "隔离分支" | `git worktree add` 为每个编码者生成一个全新的工作树 |
| Task board | "共享消息总线" | 智能体订阅的类型化消息的文件或 Redis 存储 |
| Handoff | "角色边界" | 从一个角色的上下文跨越到另一个角色的任何消息 |
| Token amplification | "多智能体开销" | 跨角色的总 token / 相同任务的单智能体 token |
| A2A protocol | "智能体到智能体" | Google 2025 年用于类型化智能体间消息的规范 |
| Merge coordinator | "集成者" | 运行三方合并和中介冲突的组件 |
| False approval | "审阅者幻觉" | 审阅者批准了包含已知错误的差异 |

## 延伸阅读

- [SWE-AF 工厂架构](https://github.com/Agent-Field/SWE-AF)——2026 年多智能体工厂参考
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT)——基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen)——Microsoft 的类型化角色框架
- [Cognition AI (Devin)](https://cognition.ai)——参考产品
- [Factory Droids](https://www.factory.ai)——备选参考产品
- [Google A2A 协议](https://developers.google.com/agent-to-agent)——智能体间消息规范
- [git worktree 文档](https://git-scm.com/docs/git-worktree)——隔离基础
- [SWE-bench Pro](https://www.swebench.com)——评估目标
