# 顶点项目 06——面向 Kubernetes 的 DevOps 故障排除智能体

> AWS 的 DevOps Agent 已正式发布，Resolve AI 发布了 K8s 手册，NeuBird 演示了语义监控，Metoro 将 AI SRE 与每服务 SLO 关联。生产形态已确定：告警 webhook 触发，智能体读取遥测，遍历 K8s 对象图，对根因假设排序，并发布带有批准按钮的 Slack 简报。默认为只读。每个修复措施都需要人工审批。这个顶点项目就是那个智能体，在 20 个合成事件上评估，并在三个共享案例上与 AWS 的 Agent 进行比较。

**类型:** Capstone
**语言:** Python（智能体）、TypeScript（Slack 集成）
**前置要求:** Phase 11（LLM 工程）、Phase 13（工具与 MCP）、Phase 14（智能体）、Phase 15（自主系统）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段:** P11 · P13 · P14 · P15 · P17 · P18
**时间:** 30 小时

## 问题

2025-2026 年的 SRE 叙事变成了："AI 智能体分类事件，人类批准修复措施。"AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产中提供这种形态。智能体读取 Prometheus 指标、Loki 日志、Tempo 追踪、kube-state-metrics 和 K8s 对象知识图。它在五分钟内生成带有遥测引用的排名的根因假设。未经人类通过 Slack 明确批准，它从不执行破坏性命令。

大部分困难工作是范围确定和安全性，而非推理。智能体需要一个默认只读的 RBAC 面、一个加固的 MCP 工具服务器，以及每个考虑中和已执行命令的审计日志。它需要知道何时超出自身能力范围并上报。而且它必须运行得足够便宜，以免 OOM-kill 级联产生 5000 美元的智能体账单。

## 概念

智能体在知识图上操作。节点是 K8s 对象（Pods、Deployments、Services、Nodes、HPAs、PVCs）加上遥测源（Prometheus 序列、Loki 流、Tempo 追踪）。边编码所有权（Pod -> ReplicaSet -> Deployment）、调度（Pod -> Node）和观测（Pod -> Prometheus 序列）。图由 kube-state-metrics 同步保持新鲜，并在每次告警时重新采样。

当告警触发时，智能体从受影响的对象开始根因分析。它遍历边，拉取相关的遥测切片（最后 15 分钟），并起草假设。假设按证据排序：有多少遥测引用支持它、多近、多具体。top-3 假设带着图路径可视化和修复操作的批准按钮发送到 Slack。

修复是受门控的。允许的默认操作是只读的。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 批准；ArgoCD 回滚钩子需要智能体从未持有的认证令牌。审计日志记录智能体*考虑过*的每个命令——不仅仅是已执行的——因此评审过程可以捕获千钧一发的情况。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- 只读 MCP 工具 ----+
           |                        |
           v                        v
   K8s knowledge graph        遥测切片
     (Neo4j / kuzu)          Prometheus, Loki, Tempo
   所有权 + 调度               最后 15 分钟，限定范围
           |
           v
   假设排序 (证据权重)
           |
           v
   Slack 简报 + 批准按钮
           |
           v (已批准)
   ArgoCD 回滚钩子 / PagerDuty 上报
           |
           v
   审计日志: 考虑 vs 执行，每个命令
```

## 技术栈

- 可观测性源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图：Neo4j（托管）或 kuzu（嵌入式），K8s 对象 + 遥测边
- 智能体：LangGraph，每个工具有允许列表，默认只读
- 工具传输：FastMCP over StreamableHTTP；破坏性工具在批准门后的独立服务器上
- 模型：Claude Sonnet 4.7 用于根因推理，Gemini 2.5 Flash 用于日志摘要
- 修复：ArgoCD 回滚 webhook、PagerDuty 上报、Slack 批准卡片
- 审计：仅追加的结构化日志（考虑、执行、批准、结果）
- 部署：K8s 部署，带有自己的狭窄 RBAC 角色；独立 namespace

## 构建它

1. **图摄取。** 每 30 秒将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测叠加边：OBSERVED_BY（Pod 被 Prometheus 序列观测）。

2. **告警接收器。** FastAPI 端点，接受 PagerDuty 或 Alertmanager webhook。提取受影响的对象和 SLO 违规。

3. **只读工具面。** 通过 FastMCP 包装 kubectl、Prometheus 查询、Loki logql、Tempo traceql。每个工具都有狭窄的 RBAC 动词（"get"、"list"、"describe"）。默认服务器中没有"delete"、"exec"、"scale"。

4. **根因智能体。** 带有三个节点的 LangGraph：`sample` 拉取最后 15 分钟的遥测切片，`walk` 查询图的相邻对象，`hypothesize` 起草带遥测引用的排名的根因候选。

5. **证据评分。** 每个假设的分数 = 近因性 × 特异性 × 图路径长度倒数 × 引用计数。返回 top-3。

6. **Slack 简报。** 发布带有假设、图路径可视化（服务器端渲染的子图图像）和最多一个修复操作的批准按钮的附件。

7. **修复门。** 破坏性工具（缩容、回滚、删除）位于第二个 MCP 服务器上，后端有批准 token。只有在 Slack 卡片被人批准后，智能体才能调用它们。

8. **审计日志。** 仅追加的 JSONL：对每个候选命令，记录它是否被考虑过、是否被执行、谁批准的。每日发送到 S3。

9. **合成事件套件。** 构建 20 个场景：OOMKill 级联、DNS 抖动、HPA 震荡、PVC 填满、吵闹的邻居、故障 sidecar、错误的 ConfigMap 发布、证书轮换、镜像拉取回退等。对智能体进行根因准确性和假设时间评分。

## 使用它

```
webhook: alert.pagerduty.com -> checkout-api SLO 违规, 错误率 14%
[graph]   受影响: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    邻居: ReplicaSet checkout-api-abc, Service checkout-api,
           最近的滚动更新在 14m 前
[sample]  prometheus error_rate 14%, 上升趋势; loki /api/v2/pay 上 500s
[hypo]    #1 错误发布: 最新镜像 checkout-api:v2.41 /healthz 失败
          引用: deploy.yaml (rev 42), prometheus errorRate, loki 500 栈
[slack]   [回滚到 v2.40]  [上报]  [忽略]
          (需批准; 智能体不会单方面回滚)
```

## 交付物

`outputs/skill-devops-agent.md` 是交付物。给定 K8s 集群和告警源，智能体生成排名的根因假设和 Slack 门控的修复流程。

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | 场景套件的 RCA 准确性 | 在 20 个合成事件中 ≥80% 正确根因 |
| 20 | 安全性 | 破坏性操作守卫在没有 Slack 批准的情况下从未在审计日志中触发 |
| 20 | 假设时间 | 从告警到 Slack 简报的 p50 低于 5 分钟 |
| 20 | 可解释性 | 每个假设都有图路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端工作 |
| **100** | | |

## 练习

1. 在 AWS 的 DevOps Agent 演示过的三个相同事件上运行你的智能体。发布并排比较。报告智能体在哪些方面出现分歧。

2. 添加一个"千钧一发"审计，标记智能体*考虑过*但未经批准会是破坏性的任何命令。测量一周内的千钧一发率。

3. 将假设模型从 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量 RCA 准确性差异和每事件美元成本。

4. 构建一个因果过滤器：区分相关的遥测尖峰与真正的根因。在 20 个场景标签上训练一个小型分类器。

5. 添加回滚干跑：针对具有相同清单的 staging 集群进行 ArgoCD 回滚。在 Slack 批准按钮之前，在实时集群中验证回滚计划。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| K8s knowledge graph | "集群图" | 节点 = K8s 对象 + 遥测序列；边 = 所有权、调度、观测 |
| Read-only-by-default | "作用域 RBAC" | 智能体的服务账户只有 get/list/describe 动词；破坏性动词位于批准后的独立服务器中 |
| Audit log | "考虑 vs 执行" | 每个候选命令的仅追加记录，是否运行，谁批准的 |
| Hypothesis ranking | "证据分数" | 近因性 × 特异性 × 图路径长度倒数 × 引用计数 |
| Slack approval card | "HITL 门" | 带修复按钮的交互式 Slack 消息；智能体在人类点击前不能继续 |
| Telemetry citation | "证据指针" | 支持声明的 Prometheus 查询、Loki 选择器或 Tempo 追踪 URL |
| MTTR | "解决时间" | 从告警触发到 SLO 恢复的墙上时钟时间 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/)——2026 年的权威参考
- [Resolve AI K8s 故障排除](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai)——竞品参考
- [NeuBird 语义监控](https://www.neubird.ai)——语义图方法
- [Metoro AI SRE](https://metoro.io)——SLO 优先的生产框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics)——集群状态源
- [LangGraph](https://langchain-ai.github.io/langgraph/)——参考智能体编排器
- [FastMCP](https://github.com/jlowin/fastmcp)——Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/)——门控修复目标
