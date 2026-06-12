# 顶点项目 09——代码迁移智能体（仓库级语言/运行时升级）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 迁移工具设定了 2026 年的标准。Moderne 的 OpenRewrite 大规模执行确定性 AST 重写。Grit 用 codemod 风格 DSL 针对同一问题。生产模式结合了两者：一个用于安全重写的确定性基础，加上一个用于模糊情况的智能体层、一个用于每个分支构建的沙箱，以及在 PR 打开前变绿的测试工具。顶点项目是迁移 50 个真实仓库并发布一个带有失败分类的通过率。

**类型:** Capstone
**语言:** Python（智能体）、Java / Python（目标）、TypeScript（仪表板）
**前置要求:** Phase 5（NLP）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 15（自主系统）、Phase 17（基础设施）
**涉及阶段:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**时间:** 30 小时

## 问题

大规模代码迁移是 2026 年编码智能体最清晰的生产应用之一。真理来源很明显（迁移后测试套件是否通过？），奖励是真实的（Java 8 项目群迁移是一个人月级别的项目），并且基准是公开的（MigrationBench 50 仓库子集）。Moderne 的 OpenRewrite 处理确定性方面。智能体层处理 OpenRewrite 配方无法处理的一切：模糊重写、构建系统漂移、长尾语法、传递性依赖破坏。

你将构建一个智能体，它接收 Java 8 仓库（或 Python 2 仓库）并生成一个 CI 为绿的已迁移分支。你将测量通过率、测试覆盖率保持、每仓库成本，并构建一个失败分类。与纯确定性基线的并排比较告诉你智能体的价值实际上在哪里。

## 概念

管道有两层。**确定性基础**（Java 的 OpenRewrite，Python 的 libcst）安全地运行大部分机械重写：导入、方法签名、空安全编辑、try-with-resources、废弃 API 替换。它速度快且产生可审计的差异。**智能体层**（OpenAI Agents SDK 或基于 Claude Opus 4.7 和 GPT-5.4-Codex 的 LangGraph）处理配方无法处理的案例：构建文件升级（Maven/Gradle/pyproject）、传递性依赖冲突、测试不稳定、自定义注解。

每个仓库获得一个预先安装了目标运行时的 Daytona 沙箱。智能体迭代：运行构建，分类失败，应用修复，重新运行。硬限制：每仓库 30 分钟、每仓库 8 美元、20 个智能体轮次。如果所有测试通过且覆盖率差异不为负，分支打开 PR。如果不能，仓库被归档到带有证据的失败类下。

失败分类是交付物。在 50 个仓库中，什么出错了？传递性依赖？自定义注解？构建工具版本？与迁移无关的测试不稳定？每个类获得一个计数和一个示例差异。未来的配方作者可以针对前三名。

## 架构

```
目标仓库
      |
      v
OpenRewrite / libcst 确定性配方
   (安全、快速、可审计，约 70-80% 的修复)
      |
      v
每个分支的 Daytona 沙箱
      |
      v
智能体循环 (Claude Opus 4.7 / GPT-5.4-Codex):
   - 运行构建 -> 捕获失败
   - 分类失败（构建、测试、lint）
   - 应用修复（补丁或重试配方）
   - 重新运行
   - 预算：30 分钟、8 美元、20 轮
      |
      v
测试 + 覆盖率差异门
      |
      v (通过)
打开 PR
      |
      v (失败)
归档到失败类下 + 附上复现方法
```

## 技术栈

- 确定性基础：OpenRewrite（Java）或 libcst（Python）
- 智能体：OpenAI Agents SDK 或基于 Claude Opus 4.7 + GPT-5.4-Codex 的 LangGraph
- 沙箱：每分支的 Daytona devcontainers，预安装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准：Amazon MigrationBench 50 仓库子集（Java 8 到 17）、Google App Engine Py2-to-Py3 仓库
- 测试工具：并行运行器，Java 用 Jacoco 或 Python 用 coverage.py 做覆盖率
- 可观测性：Langfuse + 每仓库的跟踪包，含每个差异块
- 仪表板：失败分类仪表板，含每类计数和示例差异

## 构建它

1. **配方扫描。** 首先运行 OpenRewrite（Java）或 libcst（Python）配方。捕获 70-80% 的机械性迁移。提交为"recipe"提交。

2. **构建尝试。** Daytona 沙箱：安装目标运行时，运行构建。如果为绿，跳过到测试。如果为红，交给智能体。

3. **智能体循环。** 带工具的 LangGraph：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。智能体分类失败（依赖、语法、测试、构建工具）并应用目标修复。重新运行。

4. **预算上限。** 每仓库 30 分钟墙上时钟、8 美元成本、20 个智能体轮次。任何突破都会停止并归档到"budget_exhausted"下，附上当前差异。

5. **测试 + 覆盖率门。** 构建变绿后，运行测试套件。与基础仓库比较覆盖率。如果覆盖率下降超过 2%，归档到"coverage_regression"下。

6. **打开 PR。** 成功时，推送分支，打开 PR，附上差异和已应用哪些配方及智能体编写的哪些提交的摘要。

7. **失败分类。** 对每个失败的仓库，打上类标签：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建一个仪表板。

8. **50 仓库运行。** 在 MigrationBench 子集上执行。报告每类通过率、每仓库成本、覆盖率保持情况，以及与仅确定性基线的比较。

## 使用它

```
$ migrate legacy-java-service --target java17
[recipe]   应用了 27 个重写 (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: 找不到符号 sun.misc.BASE64Encoder
[agent]    第 1 轮分类: removed_jdk_api
[agent]    第 2 轮应用: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 通过; 覆盖率 84.1% -> 84.3%
[pr]       已开启 #1841  成本=$3.20  轮次=4
```

## 交付物

`outputs/skill-migration-agent.md` 是交付物。给定一个仓库，它执行确定性配方然后智能体循环，以产生一个已迁移的绿色分支，或者将仓库归档到分类类下。

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | MigrationBench 通过率 | 50 仓库子集的 pass@1 |
| 20 | 测试覆盖率保持 | 相对基线的平均覆盖率差异 |
| 20 | 每迁移仓库成本 | 通过的运行上的美元/仓库 |
| 20 | 智能体/确定性工具集成 | OpenRewrite 处理的修复与智能体编写的修复比例 |
| 15 | 失败分析报告 | 带示例的分类完备性 |
| **100** | | |

## 练习

1. 仅用 OpenRewrite（无智能体）运行迁移管道。将通过率与完整管道比较。识别智能体独自带来差异的案例。

2. 实现"lint-clean"检查：迁移后，运行风格 linter（Java 用 spotless，Python 用 ruff）。如果出现新的 lint 错误则失败 PR。测量覆盖率保持但样式退步的比率。

3. 添加"最小差异"优化器：在智能体的分支通过测试后，用第二轮清除不必要的更改。报告差异大小缩减。

4. 扩展到第三个迁移：Node 18 到 Node 22。重用沙箱包装；将配方层替换为自定义 codemod。

5. 测量首次构建通过时间（TTFGB）作为 UX 指标。目标：p50 不超过 10 分钟。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Deterministic substrate | "配方引擎" | OpenRewrite / libcst：声明式 AST 重写，具有安全保证 |
| Codemod | "代码修改程序" | 机械地更改源代码的重写规则 |
| Build drift | "工具版本偏差" | 主要版本之间微妙的 Maven/Gradle/uv 行为变化 |
| Failure class | "分类桶" | 仓库未迁移的标记原因：依赖、语法、测试、构建工具、预算 |
| Coverage delta | "覆盖率保持" | 从基础到迁移分支的测试覆盖率百分比变化 |
| Agent turn | "工具调用轮次" | 智能体循环中的一个计划 -> 行动 -> 观察周期 |
| Budget exhaustion | "达到上限" | 仓库消耗了其 30 分钟/8 美元/20 轮限制而未通过 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/)——2026 年标准基准
- [Moderne.io OpenRewrite 平台](https://www.moderne.io)——确定性基础参考
- [OpenRewrite 文档](https://docs.openrewrite.org)——配方编写
- [Grit.io](https://www.grit.io)——备选 codemod DSL
- [OpenAI 沙箱化迁移 cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent)——Agents SDK 参考
- [Google App Engine Py2 到 Py3 迁移工具](https://cloud.google.com/appengine)——备选迁移基准
- [libcst](https://github.com/Instagram/LibCST)——Python 确定性基础
- [Daytona sandboxes](https://daytona.io)——参考每分支沙箱
