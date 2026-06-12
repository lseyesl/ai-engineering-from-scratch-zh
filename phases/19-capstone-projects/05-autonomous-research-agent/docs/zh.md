# 顶点项目 05——自主研究智能体（AI 科学家类）

> Sakana 的 AI-Scientist-v2 发表了完整论文。Agent Laboratory 运行了实验。Allen AI 分享了跟踪信息。2026 年的形态是计划-执行-验证树搜索，在实验上进行，有预算限制、沙箱化代码执行、带视觉反馈的 LaTeX 编写器以及自动化的 NeurIPS 风格审阅者集成。这个顶点项目是构建一个，在每篇论文 30 美元预算内端到端运行，并经受住 Sakana 记录的沙箱逃逸红队测试。

**类型:** Capstone
**语言:** Python（智能体 + 沙箱）、LaTeX（输出）
**前置要求:** Phase 2（ML）、Phase 3（深度学习）、Phase 7（Transformer）、Phase 10（LLM 从零开始）、Phase 14（智能体）、Phase 15（自主系统）、Phase 16（多智能体）、Phase 18（安全）
**涉及阶段:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**时间:** 40 小时

## 问题

自主研究智能体在 2026 年跨越了一个门槛。Sakana AI 的 AI-Scientist-v2 在 Nature 上发表，生成的论文通过了 workshop 同行评审。ShinkaEvolve（ICLR 2026）将这条线扩展到进化假设。AMD 的 Agent Laboratory 发布了可重现的跟踪信息。这些智能体并不神奇——它们是一个计划-执行-验证循环，在候选实验树上运行，带有成本上限、种子绑定的沙箱和自动审阅。技艺在于循环、预算和安全故事。

你通过在狭窄领域内针对种子想法实现该循环来学习它（例如，100M 参数 transformer 上的注意力稀疏性消融）。价值不在于在第一次运行时发现新东西。价值在于基础设施：树搜索、实验沙箱、编写器-审阅者循环、红队报告。Sakana 团队记录了沙箱逃逸失败；你的智能体必须通过相同的红队测试。

## 概念

智能体是一个最佳优先树搜索。节点是实验规格：（假设、配置、代码、预期结果）。扩展步骤以小的编辑（切换优化器、移动批量大小、消融一个组件）提出子节点。每个子节点在具有硬资源上限的全新沙箱中运行。结果反馈到一个评分函数，该函数按（新颖性 × 质量 × 剩余预算）对节点排序。树增长直到预算耗尽，然后最佳分支被写成论文。

编写器是多模态的。它生成 LaTeX 草稿，编译它，渲染图形，并将渲染后的 PDF 反馈到 Claude Opus 4.7 的视觉模式，以对布局、图形可读性和声明-证据对齐进行批评。由五个 LLM 评判者组成的审阅集成发出 NeurIPS 风格的分数（新颖性、严谨性、清晰度、可重现性、影响力）；如果平均分低于阈值，论文带着批评意见返回给编写器。

安全是承重结构。每个实验在 E2B 或 Daytona 沙箱中运行，无网络出口，墙上时钟受限，资源限制固定。智能体的代码生成步骤通过一个策略层，该层阻止逃逸沙箱的系统调用。红队报告重现了 Sakana 记录的受攻击面（fork 炸弹、文件系统逃逸、LLM 编写的网络调用）。

## 架构

```
种子想法 + 领域
      |
      v
  文献检索 (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph 计划-执行-验证树
      |
      v
  +--- expand node ----+      每节点沙箱
  |                    |      (E2B / Daytona)
  v                    v      资源上限
  child_1           child_k   无网络出口
  |                    |      确定性种子
  v                    v
  运行实验         运行实验
  |                    |
  v                    v
  按得分排序 (新颖性, 质量, 预算)
      |
      v
  最佳分支 -> LaTeX 编写器
      |
      v
  编译 + 视觉批评 (Opus 4.7 vision)
      |
      v
  审阅集成 (5 个 LLM 评判者, NeurIPS 评分规则)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排：带检查点和人工审批门的 LangGraph
- 树搜索：实验节点上的自定义最佳优先（类似 Sakana v2 的 AB-MCTS）
- 沙箱：每次实验的 E2B，Docker-in-Docker 备用；通过 cgroups 进行资源限制
- 文献：Semantic Scholar Graph API + OpenAlex + 本地 FAISS 摘要缓存
- 编写器：LaTeX 模板 + Claude Opus 4.7（视觉模式）用于图形批评和布局
- 审阅：5 位评判者的集成（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），加权聚合
- 实验框架：PyTorch 2.5 用于实际实验，W&B 用于日志记录
- 可观测性：Langfuse 用于智能体跟踪，每篇论文硬性预算 30 美元

## 构建它

1. **种子和领域范围确定。** 获取一个种子想法（例如，"研究低于 1B 的 transformer 的注意力图中稀疏性模式"）。定义搜索空间：模型、数据集、计算预算。

2. **文献扫描。** 查询 Semantic Scholar + OpenAlex，获取 50 篇最高引用的相关论文；在本地缓存摘要；生成 1 页领域摘要。

3. **树脚手架。** 用种子假设初始化根节点。实现 `expand(node) -> children`，使用小编辑建议（每个子节点一个配置更改）。实现 `score(node)` 作为加权新颖性 × 质量 × 预算项。

4. **沙箱包装。** 每个实验运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等效的 E2B 策略）。种子写入沙箱；输出以只读方式挂载出来。

5. **计划-执行-验证循环。** `plan` 提出子节点。`execute` 运行沙箱，捕获日志和指标。`verify` 对指标运行单元检查（损失是否下降？消融是否隔离了效果？）。失败的节点将失败原因存储在树上。

6. **编写器。** 预算用完后，选择最佳分支。用 matplotlib 渲染图形。通过 Claude Opus 4.7 生成 LaTeX 草稿，分支跟踪信息放在上下文中。编译。将编译后的 PDF 反馈给 Opus 4.7 视觉进行批评。迭代。

7. **审阅者集成。** 五位评判者对草稿进行评分（新颖性、严谨性、清晰度、可重现性、影响力），使用 NeurIPS 风格的评分规则。如果均值 < 4.0/5，带着批评意见返回给编写器。3 次重写后硬性停止。

8. **红队。** 构建或集成一组针对沙箱的对抗性任务：fork 炸弹、网络泄露尝试、文件系统逃逸、LLM 编写的 shell 元字符。确认所有都被阻止。编写调查结果。

9. **可重现性。** 每篇论文附带其树搜索跟踪信息 JSON、种子、W&B 运行链接、沙箱配置和端到端重现它的 README。

## 使用它

```
$ ai-scientist run --seed "低于 1B transformer 中的注意力稀疏性" --budget 30
[lit]    50 篇论文, 12s 内摘要
[tree]   扩展了 8 个节点, 预算 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (目前最佳)
[exec]   node #6 sparsity=top-4, loss=3.12 (更差)
[exec]   ...
[tree]   选择了以 node #3 为根的分支 (新颖性 0.62, 质量 0.81)
[write]  LaTeX 草稿 v1 完成
[vision] 批评: 图 2 图例太小, 声明-证据 OK
[write]  3 次编辑后的草稿 v2
[review] 平均 4.2/5 (新颖性 3.9, 严谨性 4.3, 清晰度 4.1, 可重现性 4.5, 影响力 4.2)
[done]   paper.pdf + review.md + trace.json    花费 $28.40
```

## 交付物

`outputs/skill-ai-scientist.md` 是交付物。给定一个种子想法 + 领域 + 30 美元预算，它运行完整管道并输出可审阅的论文和可重现性包。

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | 论文质量 | 根据已发表的 workshop 论文进行盲审评分 |
| 20 | 实验严谨性 | 基线、种子、消融；每个声明由结果表中的单元格支持 |
| 20 | 成本和计算纪律 | 30 美元/论文上限强制实施，Langfuse 跟踪 |
| 20 | 安全性 | 沙箱红队测试通过；网络策略和终止开关已验证 |
| 15 | 可重现性 | 单命令重新运行，使用相同的种子重现论文 |
| **100** | | |

## 练习

1. 在同一领域中针对三个不同的种子想法运行管道。比较树搜索的哪些部分重叠。识别重复浪费的计算。

2. 在估计超过 5 美元的节点执行前添加一个人工介入门。测量总成本下降多少。

3. 将审阅者集成替换为单一评判者。在已知质量差的论文的留出集上测量误接受率。

4. 引入网络泄露红队测试：智能体编写尝试 `curl` 外部地址的代码。确认 `--network=none` 策略阻止了它。记录尝试。

5. 将你的树搜索与平面随机基线（相同预算，无扩展策略）进行比较。报告新颖性 × 质量的增益。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Tree search | "AB-MCTS 风格扩展" | 在实验节点上的最佳优先探索，使用新颖性×质量×预算评分 |
| Sandbox | "实验隔离" | 无网络、CPU/内存受限、种子固定、输入只读的容器 |
| Vision critique | "渲染后读取" | 将论文编译为 PDF，将 PDF 反馈给 VLM 进行布局和声明-证据批评 |
| Reviewer ensemble | "自动化同行评审" | 多个 LLM 评判者使用 NeurIPS 评分规则对论文评分；加权聚合控制管道 |
| Novelty score | "这是新的吗？" | 启发式方法，惩罚与 50 篇论文文献缓存的接近程度 |
| Cost ceiling | "$ 预算" | 每篇论文总花费的硬性上限；Langfuse 计数器 + 运行前估算 |
| Red team | "沙箱逃逸审计" | 如果策略有误将会逃逸沙箱的对抗性任务 |

## 延伸阅读

- [Sakana AI-Scientist-v2 仓库](https://github.com/SakanaAI/AI-Scientist-v2)——参考生产研究智能体
- [Sakana AI-Scientist-v1 论文 (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292)——原始方法论
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai)——进化扩展
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory)——多角色研究实验室框架
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)——参考编排层
- [Semantic Scholar Graph API](https://api.semanticscholar.org/)——文献检索
- [E2B sandboxes](https://e2b.dev)——参考实验隔离
- [NeurIPS 审阅者指南](https://neurips.cc/Conferences/2026/Reviewer-Guidelines)——审阅集成的评分规则
