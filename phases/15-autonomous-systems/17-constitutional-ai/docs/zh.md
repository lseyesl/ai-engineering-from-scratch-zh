# 宪法 AI 与规则覆盖 (Constitutional AI and Rule Overrides)

> Anthropic 于 2026 年 1 月 22 日发布的 Claude 宪法长达 79 页，采用 CC0 许可。它将对齐方式从基于规则转向基于推理，并建立了四级优先级层级：(1) 安全与支持人类监督，(2) 伦理，(3) Anthropic 指南，(4) 有用性。行为分为硬编码禁令（生物武器提升、CSAM），操作者和用户均无法覆盖；以及软编码默认值，操作者可在规定范围内调整。2022 年的原始版本（Bai 等人）通过自我批评和基于宪法的人工智能反馈强化学习（RLAIF）训练无害性。诚实的说明：基于推理的对齐依赖于模型将原则泛化到未预见的情况。Anthropic 自己在 2023 年的参与式实验显示，公众来源与企业原则之间存在约 50% 的分歧；2026 年的版本并未纳入这些发现。

**类型：** 学习 (Learn)
**语言：** Python（标准库，四级优先级解析器）
**前置要求：** 阶段 15 · 06（自动化对齐研究），阶段 15 · 10（权限模式）
**时间：** ~60 分钟

## 问题 (The Problem)

一个部署在外的智能体（agent）会遇到其设计者从未见过的输入。没有哪条规则列表能长到覆盖所有情况。也没有哪条规则列表能短到在计算压力下快速应用。实际问题：如何让智能体对齐到那些既能应对长尾情况、又能在快速推理中存活下来的原则？

基于规则的对齐（Rule-based alignment, RBA）：列出所有禁止的事项。检查速度快，审计容易，但无法保持最新，且经常对未预见的近似情况过度拒绝。基于推理的对齐（2026 年 Claude 宪法）：编码原则，让模型进行推理。能扩展到未见情况，但审计更困难，失败模式是原则误用而非遗漏规则。

2026 年宪法采取了一个明确的中间立场。硬编码禁令——那些错误不依赖于上下文的事项（生物武器提升、CSAM）——属于 RBA：无论操作者或用户如何指示，永远禁止。其他所有事项都在四级层级内基于推理处理：安全与支持人类监督优先；伦理其次；Anthropic 声明的指南再次；有用性最后。操作者可以在软编码区域内调整默认值，但不能触及硬编码禁令。

## 概念 (The Concept)

### 四级优先级层级 (The four-tier priority hierarchy)

1. **安全与支持人类监督 (Safety and supporting human oversight)。** 最高优先级。模型优先考虑不削弱人类和 Anthropic 监督及纠正 AI 的能力。这不是"谨慎行事"，而是特指"不要以让人类监督变得更困难的方式行事。"
2. **伦理 (Ethics)。** 诚实、避免伤害他人、不欺骗、不操纵。当与 Anthropic 指南冲突时，伦理优先。
3. **Anthropic 指南 (Anthropic guidelines)。** Anthropic 认为重要的操作规范：产品范围、交互模式、何时使用何种工具。
4. **有用性 (Helpfulness)。** 最低优先级。在更高优先级的约束下尽可能有用。

当层级冲突时，更高优先级获胜。这与 Unix 优先级或网络 QoS 的形状相同——这种框架旨在产生可预测的解决方案，而不一定是在任何单一轴上的最佳行为。

### 硬编码禁令 vs 软编码默认值 (Hardcoded prohibitions vs soft-coded defaults)

**硬编码 (Hardcoded)：**
- 生物武器 / CBRN 提升
- CSAM
- 对关键基础设施的攻击
- 在直接询问时欺骗用户关于模型身份

操作者无法覆盖这些。用户也无法覆盖这些。它们在模型权重层面（通过 RLHF / 宪法 AI 训练）尽可能强制执行，在推理层面对无法强制执行的部分进行补充。

**软编码默认值（操作者可调整）(Soft-coded defaults (operator-adjustable))：**
- 响应长度默认值
- 主题范围（模型可以拒绝操作者部署范围之外的主题）
- 风格（正式 vs 随意）
- 工具使用模式

操作者的调整在声明的边界内进行。操作者不能通过重命名来移除硬编码禁令。

### 2022 年 CAI 训练 (The 2022 CAI training)

最初的宪法 AI（Constitutional AI, CAI）（Bai 等人，2022 年）训练无害性：

1. 对一组提示生成响应。
2. 要求模型根据宪法（明确的原则）对每个响应进行批评。
3. 基于批评修改响应。
4. 在修改后的配对上进行 RLAIF（基于人工智能反馈的强化学习）。

结果：一个能够以有原则的解释拒绝有害请求的模型，而非全面拒绝。2026 年宪法使用了这种训练的后继版本，并在明确的层级优先级上进行了额外的后训练。

### 基于推理的对齐能捕捉到什么、遗漏什么 (What reason-based alignment catches and misses)

**能捕捉 (Catches)：**
- 允许原语的未预见组合，其中原则明确适用。
- 与被禁止请求高度相似的新颖请求。
- 依赖"你没有说 X 是被禁止的"的社会工程攻击。

**遗漏 (Misses)：**
- 利用原则模糊性的攻击（"用户要求这个，所以有用性说可以"）。
- 两个原则以未预见方式冲突且层级顺序不明确的场景。
- 在训练周期中原则解释的缓慢漂移（重新解释）。

### 2023 年参与式实验 (The 2023 participatory experiment)

Anthropic 在 2023 年进行了一项实验，比较了企业编写的宪法与通过公众意见（约 1000 名美国受访者）生成的宪法。两个版本在大约 50% 的原则上达成一致。在分歧的地方，公众来源的版本在某些问题上（政治内容处理）更严格，在其他问题上（AI 身份的自我披露）则不那么严格。2026 年宪法并未纳入公众来源的发现。这是该方法中一个已记录在案的紧张关系。

### 为什么硬编码禁令是必要的 (Why hardcoded prohibitions are necessary)

仅靠基于推理的对齐无法覆盖长尾情况。一个能让模型接受某个前提的攻击者（例如，"我们是一家持牌生物武器研究实验室"）通常可以绕过那些依赖于案例推理的原则。硬编码禁令不会因前提框架而改变。它们是对齐层的"硬宪法限制"（第 14 课）。

### 宪法在堆栈中的位置 (Where the Constitution sits in the stack)

宪法不是第 14 课的终止开关（kill switch）。它位于模型层：模型的权重被训练成偏好什么。终止开关和警示令牌（canary tokens）位于运行时层：运行时允许什么。两者都是必需的。一个因为模型权重过于宽松而执行了所有错误操作的运行时，是运行时问题。一个因为运行时过于严格而拒绝了所有正确操作的模型，也是运行时问题。各层覆盖不同类别的问题。

## 使用它 (Use It)

`code/main.py` 实现了一个最小的四级优先级解析器。该解析器接收一个提议的动作和一组原则评估（安全、伦理、指南、有用性），并返回动作、拒绝或修改后的动作。驱动程序运行一个小型案例集：明确允许、明确禁止、硬编码禁令、跨层级的模糊案例。

## 交付它 (Ship It)

`outputs/skill-constitution-review.md` 审计部署的宪法层：什么是硬编码的，什么是软编码的，操作者可以在哪里调整，以及四级优先级层级是否确实是解决方案顺序。

## 练习 (Exercises)

1. 运行 `code/main.py`。确认即使有用性很高时，硬编码禁令也会触发。修改解析器，使有用性权重高于伦理；观察失败模式。

2. 阅读 Claude 宪法（公开，79 页，CC0 许可）。找出你认为一个规定不够明确的原则。写两段话解释具体的模糊性，并提出更严谨的表述。

3. 为客服智能体设计一套软编码默认值。操作者可以调整什么？操作者不能触碰什么？证明每个边界的合理性。

4. 阅读 Bai 等人 2022 年的 CAI 论文。描述一个宪法 AI 的批评-修改循环会产生比全面规则更差结果的案例。确定该案例的类别。

5. Anthropic 2023 年的参与式实验发现公众与企业原则之间存在约 50% 的分歧。选择一个对生产部署重要的类别（例如，政治中立性）。提出一个设计方案，让操作者能够表达自己的价值观，同时保持硬编码禁令不变。

## 关键术语 (Key Terms)

| 术语 (Term) | 人们说的 (What people say) | 实际含义 (What it actually means) |
|---|---|---|
| 宪法 AI (Constitutional AI) | "Anthropic 的对齐方法" | 基于书面宪法的自我批评 + RLAIF |
| 基于推理的对齐 (Reason-based alignment) | "原则，而非规则" | 模型在原则上进行推理以处理未见情况 |
| 硬编码禁令 (Hardcoded prohibition) | "永远不做 X" | 任何操作者或用户都无法覆盖的基于规则的禁令 |
| 软编码默认值 (Soft-coded default) | "操作者可调整" | 在声明边界内的行为，由操作者控制 |
| 四级优先级层级 (Four-tier hierarchy) | "优先级顺序" | 安全 > 伦理 > 指南 > 有用性 |
| RLAIF | "AI 反馈强化学习" | 奖励来自模型生成批评的强化学习 |
| 参与式宪法 (Participatory constitution) | "公众来源的原则" | 2023 年 Anthropic 实验；与企业原则约 50% 分歧 |
| 原则漂移 (Principle drift) | "解释偏差" | 模型对固定原则文本的解读缓慢变化 |

## 延伸阅读 (Further Reading)

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 79 页 CC0 文档。
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — 2022 年原始论文。
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — 参与式实验。
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 宪法在 RSP 堆栈中的位置。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 宪法在长周期部署中的作用。