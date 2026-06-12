# 水印——SynthID、Stable Signature、C2PA

> 三种技术构成了 2026 年 AI 生成内容出处（provenance）的结构。SynthID（Google DeepMind）——2023 年 8 月推出图像水印，2024 年 5 月推出文本+视频（Gemini + Veo），2024 年 10 月通过 Responsible GenAI Toolkit 开源文本水印，2025 年 11 月随 Gemini 3 Pro 推出统一多模态检测器。Stable Signature（Fernandez 等人，ICCV 2023，arXiv:2303.15435）——微调潜扩散解码器，使每个输出包含一个固定消息；裁剪（至 10% 内容）生成图像的检测率 >90%，FPR<1e-6。后续"Stable Signature is Unstable"（arXiv:2405.07145，2024 年 5 月）——微调在保持质量的同时移除了水印。C2PA——加密签名、篡改显迹的元数据标准（C2PA 2.2 Explainer 2025）。水印和 C2PA 是互补的：元数据可以去除但携带更丰富的出处信息；水印在转码后仍持续存在但携带较少的信息。

**Type:** Build
**Languages:** Python（stdlib，token 水印嵌入 + 检测）
**Prerequisites:** Phase 10 · 04（采样）、Phase 01 · 09（信息论）
**Time:** ~75 分钟

## 学习目标

- 描述 token 级水印（SynthID-text 风格）及其可检测的机制。
- 描述 Stable Signature 以及 2024 年打破它的移除攻击。
- 陈述 C2PA 的角色以及为什么它与水印互补。
- 描述关键的局限性：模型特定信号、在改写下的鲁棒性和保持含义的攻击（arXiv:2508.20228）。

## 问题

2023-2024 年，深度伪造和 AI 生成内容大规模进入政治和消费者语境。水印是提议的技术性出处信号：在创建时标记生成内容，稍后再检测。2025 年的证据：没有水印是无条件鲁棒的，但与 C2PA 元数据分层后，该组合提供了一个可用的出处方案。

## 概念

### 文本水印（SynthID-text 风格）

Kirchenbauer 等人 2023 年的机制，由 Google 生产化：

1. 在每个解码步骤，对前 K 个 token 进行哈希，产生词汇表的伪随机划分，分为"绿色"和"红色"集合。
2. 通过向绿色对数添加 δ 来使采样偏向绿色集合。
3. 生成结果中包含比随机预期更多的绿色 token。

检测：重新哈希每个前缀，计算生成结果中的绿色 token 数量，计算 z 分数。水印文本的 z 分数 >0，人类文本约为 0。

属性：
- 对读者不可感知（δ 足够小，质量损失很小）。
- 在可访问词汇表划分函数时可检测。
- 对改写不鲁棒——重写文本会破坏信号。

SynthID-text 于 2024 年 10 月通过 Google 的 Responsible GenAI Toolkit 开源。

### Stable Signature（图像）

Fernandez 等人 ICCV 2023。微调潜扩散解码器，使每个生成的图像在潜表示中包含一个固定的二进制消息。检测通过神经解码器从潜变量中解码。裁剪（至 10% 内容）的生成图像检测率 >90%，FPR<1e-6。

2024 年 5 月"Stable Signature is Unstable"（arXiv:2405.07145）：微调解码器在保持图像质量的同时移除了水印。对抗性的生成后微调成本低廉；水印的对抗鲁棒性有限。

### SynthID 统一检测器（2025 年 11 月）

随 Gemini 3 Pro 一起发布：一个多模态检测器，在一个 API 中读取来自文本、图像、音频和视频的 SynthID 信号。统一了 Google 的出处技术栈。

### C2PA

内容出处与真实性联盟（Coalition for Content Provenance and Authenticity）。加密签名、篡改显迹的元数据标准。C2PA 2.2 Explainer（2025）。C2PA 清单记录由创建者密钥签名的出处声明（谁创建、何时创建、进行了什么转换）。

与水印互补：
- 元数据可以去除；水印不能（轻易去除）。
- 元数据丰富（完整的出处链）；水印携带比特信息。
- C2PA 依赖于平台采用；水印自动嵌入。

Google 在搜索、广告和"关于此图像"中集成了两者。

### 局限性

- **模型特定。** SynthID 水印来自启用 SynthID 的模型的生成内容。来自没有 SynthID 的模型的生成内容没有水印，因此"无 SynthID 信号"不是真实性的证明。
- **改写。** 文本水印在保持含义的改写下无法存活。
- **转换攻击。** arXiv:2508.20228（2025）展示了破坏文本水印和许多图像水印的保持含义攻击。
- **微调移除。** 根据"Stable Signature is Unstable"，生成后微调会移除嵌入的水印。

### 欧盟 AI 法案第 50 条

AI 生成内容标注透明度守则（2025 年 12 月第一稿，2026 年 3 月第二稿，预计最终版 2026 年 6 月）。这一监管层要求技术层。深度伪造必须被标注。

### 在 Phase 18 中的位置

第 22-23 课是关于模型发射了什么（私密数据、出处信号）。第 27 课涵盖训练数据治理。第 24 课是要求这些技术措施的监管框架。

## 使用它

`code/main.py` 构建了一个玩具文本水印。Token 是整数 0..N-1；带水印的采样偏向哈希定义的绿色集合。检测器计算绿色 token 的 z 分数。你可以观察在 1000 token 生成中的检测、观察改写如何破坏信号、并在人类文本上测量误报率。

## 交付物

本课程产出 `outputs/skill-provenance-audit.md`。给定一个带有出处声明的���容部署，它会审计：水印机制（如有）、C2PA 签名链（如有）、每种机制的对抗鲁棒性、以及按模态的覆盖范围。

## 练习

1. 运行 `code/main.py`。报告带水印的 1000 token 生成与人类撰写文本的 z 分数。在 95% 置信阈值下确定误报率。
2. 实现一个用同义词替换 30% token 的改写攻击。重新测量 z 分数。
3. 阅读 Kirchenbauer 等人 2023 年第 6 节关于鲁棒性的内容。为什么文本水印在改写面前失败而图像水印在裁剪面前存活？
4. 设计一个同时使用 SynthID-text + C2PA 元数据的部署。描述消费者看到的出处链。指出每个组件的一种失败模式。
5. 2024 年的"Stable Signature is Unstable"结果表明微调移除了图像水印。设计一个限制此攻击的部署控制——例如，要求对微调检查点进行签名发布。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| SynthID | "Google 的水印" | 跨模态出处信号；文本、图像、音频、视频 |
| Token watermark | "Kirchenbauer 风格" | 可通过绿色 token z 分数检测的有偏采样文本水印 |
| Stable Signature | "图像水印" | 微调解码器水印；ICCV 2023 |
| C2PA | "元数据标准" | 加密签名、篡改显迹的出处元数据 |
| Paraphrase robustness | "改写是否会破坏它" | 文本水印属性；目前有限 |
| Fine-tune removal | "对抗性去水印" | 通过解码器微调移除图像水印的攻击 |
| Cross-modal detector | "统一 SynthID" | 2025 年 11 月跨模态统一 API |

## 延伸阅读

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226)——token 水印机制
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435)——图像水印论文
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145)——移除攻击
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/)——跨模态水印
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html)——元数据标准
