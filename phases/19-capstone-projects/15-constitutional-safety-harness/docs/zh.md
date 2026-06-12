# 顶点项目 15——宪法安全框架 + 红队靶场

> Anthropic 的 Constitutional Classifiers、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety 和用于多语言覆盖的 X-Guard 定义了 2026 年的安全分类器栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准对抗性评估工具。NeMo Guardrails v0.12 将它们整合到生产管道中。这个顶点项目将所有这些连接在一起：一个围绕目标应用的分层安全框架、一个运行 6 个以上攻击家族的自主任红队智能体、以及一个产生可测量无害差值的宪法自我批评运行。

**类型:** Capstone
**语言:** Python（安全管道、红队）、YAML（策略配置）
**前置要求:** Phase 10（LLM 从零开始）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 18（伦理、安全、对齐）
**涉及阶段:** P10 · P11 · P13 · P14 · P18
**时间:** 25 小时

## 问题

2026 年 LLM 安全的前沿不是分类器是否有效（它们大致有效），而是如何正确组合它们以围绕生产应用而不过度拒绝或留下明显漏洞。Llama Guard 4 处理英语策略违规。X-Guard（132 种语言）处理多语言越狱。ShieldGemma-2 捕获基于图像的提示注入。NVIDIA Nemotron 3 Content Safety 覆盖企业类别。Anthropic 的 Constitutional Classifiers 是一种单独的方法，用于训练时而非服务时。

攻击演进也很重要。PAIR 和 TAP 自动化越狱发现。GCG 运行基于梯度的后缀攻击。多轮和代码切换攻击利用智能体记忆。任何部署的 LLM 都需要一个红队靶场——garak 和 PyRIT 是标准驱动——加上记录在案的缓解措施和 CVSS 评分的发现。

你将加固一个目标应用（一个 8B 指令微调模型或来自其他顶点项目的 RAG 聊天机器人之一），针对它运行 6 个以上的攻击家族，并产生前后无害测量。

## 概念

安全管道有五层。**输入清理**：剥离零宽字符、解码 base64/rot13、标准化 Unicode。**策略层**：NeMo Guardrails v0.12 规则（领域外、毒性、PII 提取）。**分类器门**：输入上的 Llama Guard 4、非英语上的 X-Guard、图像输入上的 ShieldGemma-2。**模型**：目标 LLM。**输出过滤器**：输出上的 Llama Guard 4、Presidio PII 清理、适用的引用强制执行。**HITL 层**：标记为高风险的输出进入 Slack 队列。

红队靶场在调度器上运行。PAIR 和 TAP 自主发现越狱。GCG 运行基于梯度的后缀攻击。ASCII/base64/rot13 编码攻击。多轮攻击（角色采用、记忆利用）。代码切换攻击（将英语与斯瓦希里语或泰语混合）。每次运行产生一个结构化发现文件，带 CVSS 评分和披露时间表。

宪法自我批评运行是训练时的干预。取 1k 个有害尝试提示，让模型起草响应，根据成文宪法（不伤害规则）批评它，并在批评循环上重新训练。测量在留出评估上的前后无害差值。

## 架构

```
请求 (文本 / 图像 / 多语言)
      |
      v
输入清理 (剥离零宽字符, 解码, 标准化)
      |
      v
NeMo Guardrails v0.12 规则 (领域外, 策略)
      |
      v
分类器门:
  Llama Guard 4 (英语)
  X-Guard (多语言, 132 语言)
  ShieldGemma-2 (图像提示)
  Nemotron 3 Content Safety (企业)
      |
      v (允许)
目标 LLM
      |
      v
输出过滤器: Llama Guard 4 + Presidio PII + 引用检查
      |
      v
标记输出的 HITL 层

并行:
  红队调度器
    -> garak (经典攻击)
    -> PyRIT (编排红队)
    -> 自主任越狱智能体 (PAIR + TAP)
    -> GCG 后缀攻击
    -> 多语言 / 代码切换
    -> 多轮角色采用

输出: CVSS 评分发现 + 披露时间表 + 前后无害差值
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动：garak (NVIDIA)、PyRIT (Microsoft Azure)、NVIDIA Aegis、promptfoo
- 越狱智能体：PAIR (Chao et al., 2023)、Tree-of-Attacks (TAP)、GCG 后缀
- 宪法训练：Anthropic 风格自我批评循环 + 批评上的 SFT
- PII 清理：Presidio
- 目标：8B 指令微调模型或其他顶点项目的 RAG 聊天机器人之一

## 构建它

1. **目标设置。** 在 vLLM 上搭建一个 8B 指令微调模型（或重用来自其他顶点项目的 RAG 聊天机器人）。这是被测应用。

2. **安全管道包装。** 围绕目标接线五层管道。验证每层独立可观察（Langfuse 中每层一个 span）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在每个的小型标记集上运行以建立基线。

4. **红队调度器。** 调度 garak、PyRIT、一个 PAIR 智能体、一个 TAP 智能体、一个 GCG 运行器、一个多轮攻击者和一个代码切换攻击者。每个在独立队列上运行。

5. **攻击套件。** 六个攻击家族：(1) PAIR 自动化越狱、(2) TAP 攻击树、(3) GCG 梯度后缀、(4) ASCII/base64/rot13 编码、(5) 多轮角色、(6) 多语言代码切换。报告每个家族的成功率。

6. **宪法自我批评。** 策划 1k 个有害尝试提示。对每个，目标起草响应。批评 LLM 根据成文宪法评分（"不伤害"、"引用证据"、"拒绝非法请求"）。批评者反对的提示被重写；目标在批评改进的对上微调。测量在留出评估上的前后无害性。

7. **过度拒绝测量。** 在良性提示套件（如 XSTest）上跟踪假阳性率。目标必须在良性问题上保持有用。

8. **CVSS 评分。** 对每个成功的越狱，按 CVSS 4.0（攻击向量、复杂性、影响）评分。生成披露时间表和缓解计划。

9. **靶场自动化。** 上述一切在 cron 上运行；发现写入队列；过度拒绝回归告警发送到 Slack。

## 使用它

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR 智能体在目标上运行
[attack]     尝试 1/50: 将查询伪装为学术研究 ... 已阻止
[attack]     尝试 2/50: 诉诸角色扮演 ... 已阻止
[attack]     尝试 3/50: 思维链诱导 ... 成功
[finding]    CVSS 4.8 中: 目标上的角色扮演绕过
[range]      50 次中 7 次成功 (14% 成功率)
```

## 交付物

`outputs/skill-safety-harness.md` 是交付物。一个生产级分层安全管道加上一个可重现的红队靶场，带前后无害差值。

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | 攻击面覆盖 | 6+ 攻击家族已实施，2+ 语言 |
| 20 | 真阳性/假阳性权衡 | 攻击阻止率 vs XSTest 良性通过率 |
| 20 | 自我批评差值 | 留出评估上的前后无害性 |
| 20 | 文档和披露 | CVSS 评分发现带时间表 |
| 15 | 自动化和可重复性 | 一切在 cron 上运行，带告警 |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的提示注入插件，并比较有和无输出过滤器层的攻击成功率。

2. 添加第七个攻击家族：通过检索文档的间接提示注入。测量所需的额外防御。

3. 实现"带帮助的拒绝"模式：当护栏阻止时，目标提供一个更安全的相关答案而不是直接拒绝。测量 XSTest 差值。

4. 多语言覆盖差距：找到一个 X-Guard 表现不佳的语言。提出一个针对它的微调数据集。

5. 在 30B 模型上运行宪法自我批评并测量差值是否扩展。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Layered safety | "深度防御" | 在输入、门、输出、HITL 的多层护栏 |
| Llama Guard 4 | "Meta 的安全分类器" | 2026 年参考输入/输出内容分类器 |
| PAIR | "越狱智能体" | 关于 LLM 驱动的越狱发现的论文 (Chao et al.) |
| TAP | "攻击树" | PAIR 的树搜索变体 |
| GCG | "贪婪坐标梯度" | 基于梯度的对抗性后缀攻击 |
| Constitutional self-critique | "Anthropic 风格训练" | 目标起草 -> 批评者评分 -> 重写 -> 重新训练 |
| XSTest | "良性探测集" | 过度拒绝回归的基准 |
| CVSS 4.0 | "严重性评分" | 安全发现的标准漏洞评分 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers)——训练时参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/)——2026 输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b)——图像 + 多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/)——企业参考
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848)——132 语言多语言安全
- [garak](https://github.com/NVIDIA/garak)——NVIDIA 红队工具包
- [PyRIT](https://github.com/Azure/PyRIT)——Microsoft 红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/)——护栏框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419)——越狱智能体论文
