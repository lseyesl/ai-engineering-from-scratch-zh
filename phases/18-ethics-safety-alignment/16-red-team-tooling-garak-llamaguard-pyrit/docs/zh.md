# 红队工具——Garak、Llama Guard、PyRIT

> 三种生产工具构成了 2026 年的红队工具栈。Llama Guard（Meta）——在 14 个 MLCommons 危害类别上微调的 Llama-3.1-8B 分类器；2025 年的 Llama Guard 4 是从 Llama 4 Scout 剪枝而来的 12B 原生多模态分类器。Garak（NVIDIA）——开源 LLM 漏洞扫描器，配备用于幻觉、数据泄漏、提示注入、毒性和越狱的静态、动态和自适应探针。PyRIT（微软）——多轮红队活动，带有 Crescendo、TAP 和用于深度利用的自定义转换器链。Llama Guard 3 记录在 Meta 的"Llama 3 Herd of Models"（arXiv:2407.21783）中；Llama Guard 3-1B-INT4 在 arXiv:2411.17713 中；Garak 的探针架构在 github.com/NVIDIA/garak 中。这些工具是红队研究（第 12-15 课）与部署（第 17 课以上）之间的 2026 年生产接口。

**Type:** Build
**Languages:** Python（stdlib，工具架构模拟器和 Llama Guard 风格分类器模拟）
**Prerequisites:** Phase 18 · 12-15（越狱和 IPI）
**Time:** ~75 分钟

## 学习目标

- 描述 Llama Guard 3/4 在安全栈中的位置：输入分类器、输出分类器或两者兼有。
- 说出 14 个 MLCommons 危害类别，并陈述一个不太明显的类别（代码解释器滥用）。
- 描述 Garak 的探针架构：探针、检测器、框架。
- 描述 PyRIT 的多轮活动结构及其如何与 Garak 探针组合。

## 问题

第 12-15 课展示了攻击面。生产部署需要可重复、可扩展的评估。2026 年有三种工具主导：Llama Guard（防御分类器）、Garak（扫描器）、PyRIT（活动编排器）。每种工具针对红队生命周期的不同层面。

## 概念

### Llama Guard（Meta）

Llama Guard 3 是一个 Llama-3.1-8B 模型，针对 MLCommons AILuminate 14 个类别的输入/输出分类进行了微调：
- 暴力犯罪、非暴力犯罪、性相关、CSAM、诽谤
- 专业建议、隐私、知识产权、无差别武器、仇恨言论
- 自杀/自残、色情内容、选举、代码解释器滥用

支持 8 种语言。用法：放在 LLM 之前（输入审核）、之后（输出审核）或两者皆用。两种用途产生不同的训练分布——Llama Guard 3 以一个模型同时处理两者。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，移动 CPU 上约 30 tokens/s）是量化的边缘变体。

Llama Guard 4（2025 年 4 月）是 12B，原生多模态，从 Llama 4 Scout 剪枝而来。它用一个接收文本+图像的分类器取代了之前的 8B 文本和 11B 视觉分类器。

### Garak（NVIDIA）

开源漏洞扫描器。架构：
- **探针（Probes）。** 用于幻觉、数据泄漏、提示注入、毒性、越狱的攻击生成器。静态（固定提示）、动态（生成提示）、自适应（响应目标输出）。
- **检测器（Detectors）。** 根据预期的失败模式对输出进行评分——有毒、泄漏、越狱成功。
- **框架（Harnesses）。** 管���探针-检测器对，运行活动，生成报告。

TrustyAI 将 Garak 与 Llama-Stack 防护（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器）集成，用于端到端的防护目标评估。基于层级评分（TBSA）取代了二进制的通过/失败——模型可以在同一探针上通过严重级别 3 而在严重级别 5 上失败。

### PyRIT（微软）

Python 风险识别工具包。多轮红队活动。围绕以下组件构建：
- **转换器（Converters）。** 转换种子提示——改写、编码、翻译、角色扮演。
- **编排器（Orchestrators）。** 运行活动：Crescendo（升级）、TAP（分支）、RedTeaming（自定义循环）。
- **评分（Scoring）。** LLM 作为评判或分类器作为评判。

PyRIT 是 Garak 的更重量级表亲。Garak 运行数千个单轮探针；PyRIT 运行旨在突破特定失败模式的深度多轮活动。

### 工具栈

在模型的两侧都放置 Llama Guard。每晚运行 Garak 进行回归测试。为发布前活动运行 PyRIT。这是 2026 年大多数生产部署的默认配置。

### 评估陷阱

- **评判者身份。** 这三种工具都可以使用 LLM 评判者；评判者的校准驱动报告的 ASR（第 12 课）。在工具旁边指定评判者。
- **探针过时。** 随着模型针对探针进行修补，Garak 探针会老化。自适应探针（PAIR 风格的）比静态探针老化得更慢。
- **Llama Guard 对良性内容的误报。** 早期 Llama Guard 版本过度标记了政治和 LGBTQ+ 内容；Llama Guard 3/4 的校准有所改进，但未按部署进行校准。

### 在 Phase 18 中的位置

第 12-15 课是攻击家族。第 16 课是生产工具。第 17 课（WMDP）是双重用途能力的评估。第 18 课是将这些工具包裹在政策结构中的前沿安全框架。

## 使用它

`code/main.py` 构建了一个玩具型的 Llama Guard 风格分类器（14 个类别上的关键词 + 语义特征）、一个玩具型的 Garak 框架（探针-检测器循环）和一个 PyRIT 风格的多轮转换器链。你可以对模拟目标运行这三种工具并观察不同的覆盖特征。

## 交付物

本课程产出 `outputs/skill-red-team-stack.md`。给定一个部署描述，它会指出三种工具中哪些适用、每种工具配置什么、以及运行何种回归节奏。

## 练习

1. 运行 `code/main.py`。比较 Llama Guard 风格分类器在单轮与多轮攻击上的检测率。
2. 实现一个新的 Garak 探针：一个 base64 编码的有害请求。测量 Llama Guard 风格分类器对其的检测率。
3. 用"翻译成法语，然后改写"转换器扩展 PyRIT 风格的转换器链。重新测量攻击成功率。
4. 阅读 Llama Guard 3 的危害类别列表。识别两个类别，其训练数据在合法的开发者内容上可能产生高误报率。
5. 比较 Garak 和 PyRIT 的设计原则。论证一个各自作为合适工具的部署场景。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Llama Guard | "分类器" | 在 14 个危害类别上微调的 Llama-3.1-8B/4-12B 安全分类器 |
| Garak | "扫描器" | NVIDIA 开源漏洞扫描器；探针、检测器、框架 |
| PyRIT | "活动工具" | 微软多轮红队编排器；转换器、编排器、评分 |
| Prompt-Guard | "小分类器" | Meta 的 86M 提示注入分类器，与 Llama Guard 配对 |
| TBSA | "基于层级评分" | Garak 的基于层级通过/失败取代二进制结果 |
| Converter chain | "改写 + 编码 + ..." | PyRIT 用于构建多步攻击的组合原语 |
| MLCommons hazard categories | "14 种分类法" | Llama Guard 针对的行业标准分类法 |

## 延伸阅读

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783)——8B 分类器
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713)——量化移动分类器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak)——扫描器仓库和文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT)——活动工具包
