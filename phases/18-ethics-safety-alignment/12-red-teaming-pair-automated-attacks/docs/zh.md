# 红队测试——PAIR 与自动化攻击

> Chao、Robey、Dobriban、Hassani、Pappas、Wong（NeurIPS 2023，arXiv:2310.08419）。PAIR——Prompt Automatic Iterative Refinement——是规范的自动化黑盒越狱。一个带有红队系统提示的攻击者 LLM 迭代地为目标 LLM 提出越狱方案，在其自身的聊天历史中积累尝试和响应作为上下文反馈。PAIR 通常在 20 次查询内成功，比 GCG（Zou 等人的 token 级梯度搜索）高效数个数量级，且无需白盒访问。PAIR 现在是 JailbreakBench（arXiv:2404.01318）和 HarmBench 中的标准基线，与 GCG、AutoDAN、TAP 和 Persuasive Adversarial Prompt 并列。

**Type:** Build
**Languages:** Python（stdlib，针对玩具目标的模拟 PAIR 循环）
**Prerequisites:** Phase 18 · 01（指令遵循）、Phase 14（代理工程）
**Time:** ~75 分钟

## 学习目标

- 描述 PAIR 算法：攻击者系统提示、迭代精炼、上下文反馈。
- 解释为什么 PAIR 在目标为黑盒时严格优于 GCG。
- 说出其他四种自动化攻击基线（GCG、AutoDAN、TAP、PAP）并陈述每种攻击的一个区分特征。
- 描述 JailbreakBench 和 HarmBench 的评估协议以及每种协议中"攻击成功率"的含义。

## 问题

红队测试曾经是一项手动活动。少量专家测试人员构建对抗性提示并跟踪哪些有效。这无法扩展：攻击成功率需要统计样本，而目标是随每个模型版本变化的目标。PAIR 将红队测试操作化为一个对黑盒目标的优化问题。

## 概念

### PAIR 算法

输入：
- 目标 LLM T（我们正在攻击的模型）。
- 评判 LLM J（对响应是否为越狱进行评分）。
- 攻击者 LLM A（红队优化器）。
- 目标字符串 G："以 [有害指令] 响应。"
- 预算 K（通常为 20 次查询）。

循环，对于 k 从 1 到 K：
1. A 被提示目标 G 和到目前为止的（提示，响应）对历史。
2. A 生成一个新的提示 p_k。
3. 将 p_k 提交给 T；获取响应 r_k。
4. J 对 (p_k, r_k) 在目标 G 上的表现进行评分。
5. 如果评分 >= 阈值，停止——找到越狱。
6. 否则，将 (p_k, r_k) 附加到 A 的历史中；继续。

实证结果（NeurIPS 2023）：针对 GPT-3.5-turbo、Llama-2-7B-chat 的攻击成功率 >50%；平均成功查询次数在 10-20 之间。

### 为什么 PAIR 高效

GCG（Zou 等人 2023）通过梯度搜索对抗性 token 后缀；需要白盒模型访问，产生不可读的后缀。PAIR 是黑盒的，产生可跨模型迁移的自然语言攻击。PAIR 的上下文反馈让攻击者能从每次拒绝中学习；GCG 没有等价物（每次新的 token 更新都得重新发现先前的进展）。

### 相关的自动化攻击

- **GCG（Zou 等人 2023，arXiv:2307.15043）。** token 级梯度搜索对抗性后缀。白盒、可迁移、产生不可读字符串。
- **AutoDAN（Liu 等人 2023）。** 对提示进行进化搜索，由层次化目标引导。
- **TAP（Mehrotra 等人 2024）。** 带剪枝的攻击树——分支多个 PAIR 风格的 rollout。
- **PAP（Zeng 等人 2024）。** 说服性对抗提示——将人类说服技巧编码为提示模板。

### JailbreakBench 和 HarmBench

两者（2024）标准化了评估：

- JailbreakBench（arXiv:2404.01318）。跨越 10 个 OpenAI 政策类别的 100 种有害行为。攻击成功率（ASR）为主要指标。需要评判模型（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika 等人 2024）。7 个类别的 510 种行为，包含语义和功能危害测试。比较 18 种攻击对 33 个模型。

ASR 通常在固定的查询预算下报告。比较攻击需要匹配预算；200 次查询 90% ASR 与 20 次查询 85% ASR 不可比。

### 为什么这对 2026 年部署很重要

每个前沿实验室现在都在发布前对生产模型运行 PAIR 和 TAP。ASR 轨迹出现在模型卡（第 26 课）和安全案例附录（第 18 课）中。这种攻击并不特殊——它是标准的基础设施。

### 在 Phase 18 中的位置

第 12 课是自动化攻击基础。第 13 课（多轮次越狱）是一种互补的长度利用式攻击。第 14 课（ASCII 艺术/视觉）是一种编码攻击。第 15 课（间接提示注入）是 2026 年的生产环境攻击面。第 16 课介绍了对应的防御工具（Llama Guard、Garak、PyRIT）。

## 使用它

`code/main.py` 构建了一个玩具 PAIR 循环。目标是一个模拟分类器，拒绝"明显"有害的提示（关键词过滤）。攻击者是一个基于规则的优化器，尝试改写、角色扮演框架和编码。评判者对响应进行评分。你可以观察到攻击者在约 5-15 次迭代后突破关键词过滤器，但在语义过滤器面前失败。

## 交付物

本课程产出 `outputs/skill-attack-audit.md`。给定一份红队评估报告，它会审计：运行了哪些攻击（PAIR、GCG、TAP、AutoDAN、PAP），每种攻击的预算，使用了哪个评判模型，在哪个有害行为集上（JailbreakBench、HarmBench、内部集）。

## 练习

1. 运行 `code/main.py`。测量三种内置攻击者策略的平均成功查询次数。解释每种策略利用了哪种目标防御假设。
2. 实现第四种攻击者策略（例如，翻译成另一种语言、base64 编码）。报告针对关键词过滤目标和语义过滤目标的新平均成功查询次数。
3. 阅读 Chao 等人 2023 年图 5（PAIR 与 GCG 比较）。描述尽管 PAIR 有效率优势，但仍然偏好 GCG 的两种场景。
4. JailbreakBench 报告针对固定目标集的 ASR。设计一个测量攻击多样性（成功提示的方差）的额外指标。解释为什么多样性对防御评估很重要。
5. TAP（Mehrotra 2024）通过分支 + 剪枝扩展了 PAIR。草拟一个针对 `code/main.py` 的 TAP 风格扩展，并描述计算成本与成功率之间的权衡。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| PAIR | "自动化越狱" | 提示自动迭代精炼；攻击者 LLM + 评判 LLM 循环 |
| GCG | "梯度越狱" | 白盒 token 级梯度搜索对抗性后缀 |
| Attack success rate (ASR) | "k 次查询的越狱率" | 主要指标；必须附带查询预算和评判模型身份 |
| Judge LLM | "评分者" | 对响应是否满足有害目标进行评分的 LLM |
| JailbreakBench | "评估标准" | 标准化有害行为集，带标签类别 |
| HarmBench | "更广泛的基准" | 510 种行为，功能 + 语义危害测试 |
| TAP | "攻击树" | 带分支 + 剪枝的 PAIR；在更高计算量下提供更好的 ASR |

## 延伸阅读

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419)——PAIR 论文，NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043)——GCG 论文
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318)——标准化评估
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249)——更广泛的评估
