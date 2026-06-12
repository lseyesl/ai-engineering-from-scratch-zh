# 顶点项目 16——GitHub Issue 到 PR 的自主智能体

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 在 2026 年都实现了相同的产品形态：标记一个 issue，得到一个 PR。在云端沙箱中运行智能体，验证测试通过，并发布带理由的、可供审查的 PR。难点在于自动复现仓库的构建环境、防止凭证泄露、强制执行每仓库预算，以及确保智能体不能强制推送。这个顶点项目构建自托管版本，并在成本和通过率上与托管替代方案进行比较。

**类型:** Capstone
**语言:** Python（智能体）、TypeScript（GitHub App）、YAML（Actions）
**前置要求:** Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 15（自主系统）、Phase 17（基础设施）
**涉及阶段:** P11 · P13 · P14 · P15 · P17
**时间:** 30 小时

## 问题

异步云端编码智能体是与交互式编码智能体（顶点项目 01）不同的独立产品类别。其用户体验是一个 GitHub 标签。你给一个 issue 标记 `@agent fix this`，一个 worker 在云端沙箱中启动，克隆仓库，运行测试，编辑文件，验证，并打开一个 PR，在正文中包含智能体的理由。没有交互式循环，没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都收敛到这个模式。

工程挑战是具体的：环境复现（智能体必须在没有缓存开发镜像的情况下从零开始构建仓库）、不稳定测试（必须重新运行或隔离）、凭证范围界定（具有最小细粒度权限的 GitHub App）、每个仓库每天的预算执行，以及禁止强制推送策略。该顶点项目测量通过率、成本和安全性，并与托管替代方案进行比较。

## 概念

触发器是一个 GitHub webhook（issue 标签或 PR 评论）。调度器将工作入队到 ECS Fargate 或 Lambda。Worker 将仓库拉入 Daytona 或 E2B 沙箱，并附带从仓库推断出的通用 Dockerfile（语言、框架）。智能体使用 mini-swe-agent 或 SWE-agent v2 循环，针对 Claude Opus 4.7 或 GPT-5.4-Codex 运行。它迭代：阅读代码，提出修复，应用补丁，运行测试。

验证是门控步骤。在 PR 打开之前，完整的 CI 必须在沙箱中通过。计算覆盖率差异；如果低于阈值，PR 仍然打开但带有 `needs-review` 标签。智能体将理由发布为 PR 描述，加上一个 `@agent` 线程，审查者可以在其中 ping 以进行后续操作。

安全性通过两个不同的 GitHub 表面来界定：App 提供一个短期安装令牌，具有 `workflows: read` 和狭窄的仓库内容/PR 范围；分支保护（而非应用权限）强制执行"不直接写入 `main`"和"不强制推送"——该应用从未被添加到绕过列表。对 `.github/workflows` 的路径范围只读访问不是真正的 GitHub App 原语，因此智能体对文件编辑的允许列表必须在 worker 级别强制执行。每个仓库每天的预算上限在调度器层面执行（例如，每个仓库每天最多 5 个 PR，每个 PR 最多 $20）。

## 架构

```
GitHub issue 标记为 `@agent fix` 或 PR 评论
            |
            v
    GitHub App webhook -> AWS Lambda 调度器
            |
            v
    ECS Fargate 任务（或 GitHub Actions 自托管运行器）
       - 拉取仓库
       - 推断 Dockerfile（语言、包管理器）
       - Daytona / E2B 沙箱，带目标运行时
       - clone -> git worktree -> 智能体分支
            |
            v
    mini-swe-agent / SWE-agent v2 循环
       Claude Opus 4.7 或 GPT-5.4-Codex
       工具: ripgrep、tree-sitter、read/edit、run_tests、git
            |
            v
    验证 CI 在沙箱中通过 + 覆盖率差异检查
            |
            v (已验证)
    git push + 通过 GitHub App 打开 PR
       PR 正文 = 理由 + diff 摘要 + trace URL
       标签: needs-review
            |
            v
    操作者审查；可以通过 @-mention 智能体进行后续操作
```

## 技术栈

- 触发器：带细粒度令牌的 GitHub App；通过 Lambda 或 Fly.io 的 webhook 接收器
- Worker：ECS Fargate 任务（或 GitHub Actions 自托管运行器）
- 沙箱：每个任务的 Daytona devcontainer 或 E2B 沙箱
- 智能体循环：mini-swe-agent 基线或 SWE-agent v2，基于 Claude Opus 4.7 / GPT-5.4-Codex
- 检索：tree-sitter repo-map + ripgrep
- 验证：沙箱内完整 CI + 覆盖率差异门控
- 可观测性：Langfuse，每个 PR 的跟踪存档链接到 PR 正文
- 预算：每个仓库每日美元上限；每个仓库每天最大 PR 数

## 构建它

1. **GitHub App。** 细粒度安装令牌：issues 读写、pull_requests 写、contents 读写、workflows 读。分支保护（唯一能做到这一点的表面）强制执行"不直接推送到 `main`"和"不强制推送"；该应用不在绕过列表中。Worker 在提议的 diff 上强制执行"不写入 `.github/workflows`"作为允许列表检查，因为 GitHub App 权限不是路径范围的。

2. **Webhook 接收器。** Lambda 函数接受 issue 标签 / PR 评论 webhook。按标签 `@agent fix this` 过滤。入队到 SQS。

3. **调度器。** 从 SQS 弹出任务。强制执行每个仓库每天的预算。使用仓库 URL、issue 正文和新的 Daytona 沙箱启动 ECS Fargate 任务。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。如果不存在，则动态生成 Dockerfile。

5. **智能体循环。** mini-swe-agent 或 SWE-agent v2，使用 Claude Opus 4.7。工具：ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。硬限制：$20 成本、30 分钟墙上时钟、30 个智能体轮次。

6. **验证。** 循环结束后，在沙箱中运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率差异。如果 CI 为红色：停止，不打开 PR。如果覆盖率下降超过 2%：打开带有 `needs-review` 标签的 PR。

7. **PR 发布。** 推送智能体分支。通过 GitHub API 打开 PR，包含：标题、理由、diff 摘要、trace URL、成本、轮次。

8. **凭证卫生。** Worker 使用短期 GitHub App 安装令牌运行。日志在归档前清除密钥。

9. **评估。** 30 个不同难度的内部种子 issue。测量通过率、PR 质量（diff 大小、风格、覆盖率）、成本、延迟。与 Cursor Background Agents 和 AWS Remote SWE Agents 在相同 issue 上进行比较。

## 使用它

```
# 在 github.com 上
  - 用户标记 issue #842 为 `@agent fix this`
  - 14 分钟后 PR #1903 出现
  - 正文：
    > 修复了由空比较器条目引起的 widget.dedupe() 中的 NPE。
    > 添加了回归测试 widget_test.go::TestDedupeNullComparator。
    > 覆盖率差异：+0.12%
    > 轮次：7  成本：$1.80  跟踪：langfuse:...
    > 标签：needs-review
```

## 交付物

`outputs/skill-issue-to-pr.md` 是交付物。一个 GitHub App + 异步云端 worker，将标记的 issue 转换为可审查的 PR，具有可控的成本和范围界定的凭证。

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | 30 个 issue 的通过率 | 端到端成功（CI 通过 + 覆盖率 OK）|
| 20 | PR 质量 | Diff 大小、覆盖率差异、风格一致性 |
| 20 | 每个解决 issue 的成本和延迟 | 每个 PR 的 $ 和墙上时钟时间 |
| 20 | 安全性 | 范围界定令牌、每仓库预算、无强制推送、凭证卫生 |
| 15 | 操作者 UX | 理由评论、重试机制、@-mention 后续操作 |

## 练习

1. 添加"修复不稳定测试"模式：标记 `@agent stabilize-flake TestX` 在沙箱中运行测试 50 次，并提出最小化更改以稳定它。

2. 在三个共享 issue 上与 Cursor Background Agents 比较成本。报告哪些工具在哪些情况下胜出。

3. 实现预算仪表板：每个仓库每天成本、每个用户成本。异常时发出告警。

4. 构建"dry-run"模式，在不运行 CI 的情况下打开草稿 PR，以便审查者可以低成本检查计划。

5. 添加保留策略：超过 7 天未合并的 PR 分支自动删除。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| GitHub App | "范围界定的机器人身份" | 具有细粒度权限 + 短期安装令牌的 App |
| 异步云端智能体 | "后台智能体" | 在云端沙箱中运行的非交互式 worker，而非终端 |
| 环境推断 | "Dockerfile 合成" | 检测语言 + 包管理器，如果缺失则生成 Dockerfile |
| 验证 | "沙箱内 CI" | 在打开 PR 之前在 worker 内运行完整测试套件 |
| 覆盖率差异 | "覆盖率保持" | 从基础分支到智能体分支的测试覆盖率百分比变化 |
| 每仓库预算 | "每日上限" | 在调度器级别强制执行的美元和 PR 数量上限 |
| 理由 | "PR 正文解释" | 智能体对更改内容和原因的总结；PR 正文中必需 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents)——规范的异步云端智能体参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent)——CLI 参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent)——商业替代方案
- [OpenAI Codex (cloud)](https://openai.com/codex)——托管竞争者
- [Google Jules](https://jules.google)——Google 的托管版本
- [Factory Droids](https://www.factory.ai)——备选商业参考
- [GitHub App 文档](https://docs.github.com/en/apps)——范围界定的机器人身份
- [Daytona 云端沙箱](https://daytona.io)——参考沙箱
