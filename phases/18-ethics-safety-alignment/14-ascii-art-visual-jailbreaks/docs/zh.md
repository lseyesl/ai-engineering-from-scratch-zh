# ASCII 艺术与视觉越狱

> Jiang、Xu、Niu、Xiang、Ramasubramanian、Li、Poovendran，"ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs"（ACL 2024，arXiv:2402.11753）。将有害请求中的安全相关 token 遮盖起来，替换为相同字母的 ASCII 艺术渲染，并发送遮盖后的提示。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 均未能稳健地识别 ASCII 艺术 token。该攻击绕过了 PPL（困惑度过滤器）、改写防御和重分词（retokenization）。相关成果：ViTC 基准测试测量了对非语义视觉提示的识别能力；StructuralSleight 将其推广到不常见文本编码结构（树、图、嵌套 JSON）作为一个编码攻击家族。

**Type:** Build
**Languages:** Python（stdlib，ArtPrompt token 遮盖框架）
**Prerequisites:** Phase 18 · 12（PAIR）、Phase 18 · 13（MSJ）
**Time:** ~60 分钟

## 学习目标

- 描述 ArtPrompt 攻击：单词识别步骤、ASCII 艺术替换、最终的遮盖提示。
- 解释为什么标准防御（PPL、改写、重分词）在 ArtPrompt 面前失败。
- 定义 ViTC 并描述其测量的内容。
- 描述 StructuralSleight 作为对任意不常见文本编码结构（UTES）的推广。

## 问题

通过改写和角色扮演（第 12 课）以及通过长上下文（第 13 课）的攻击作用于文本层面的模式。ArtPrompt 作用于识别层面：模型没有解析被禁止的 token。它解析一个用字符渲染的图像。安全过滤器看到无害的标点符号。模型看到一个单词。

## 概念

### ArtPrompt，两步

第 1 步：单词识别。给定一个有害请求，攻击者使用 LLM 识别安全相关的单词（例如，"bomb"在"how to make a bomb"中）。

第 2 步：遮盖提示生成。将每个识别的单词替换为其 ASCII 艺术渲染（一个 7x5 或 7x7 的字符块构成字母形状）。模型接收到一个由标点和空格组成的网格，一个足够有能力的模型可以识别出该单词；安全过滤器只看到这个网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 全部失败。攻击成功率在其基准子集上超过 75%。

### 为什么标准防御失败

- **PPL（困惑度过滤器）。** ASCII 艺术具有高困惑度——但所有新颖的输入都是如此。阻止 ArtPrompt 的阈值选择也会阻止合法的结构化输入。
- **改写。** 改写提示会破坏 ASCII 艺术。在实践中，改写 LLM 通常会保留或重建该艺术。
- **重分词。** 以不同方式分割 token 不会改变模型的视觉正在识别字母形状这一事实。

底层问题是安全过滤器是 token 级或语义级的；ArtPrompt 作用于视觉识别层面。

### ViTC 基准测试

非语义视觉提示的识别能力。衡量模型读取 ASCII 艺术、webdings 和其他非文本语义的视觉内容的能力。ArtPrompt 的有效性与 ViTC 准确率相关：模型阅读视觉文本的能力越强，ArtPrompt 在其上的效果就越好。这是一种能力-安全性权衡。

### StructuralSleight

推广了 ArtPrompt：不常见文本编码结构（UTES）。树、图、嵌套 JSON、JSON 中的 CSV、差异风格的代码块。如果某个结构在训练安全数据中罕见但模型可以解析，它可以隐藏有害内容。

防御的意义：安全性必须泛化到模型可以解析的结构化表示上。这个集合很大且在增长。

### 图像模态类比

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。使用实际图像的 ArtPrompt 风格攻击比 ASCII 艺术类比更强，因为图像编码器产生更丰富的信号。

### 在 Phase 18 中的位置

第 12-14 课描述了三种正交的攻击向量：迭代精炼（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第 15 课从以模型为中心的攻击转向系统边界攻击（间接提示注入）。第 16 课描述了防御工具集的响应。

## 使用它

`code/main.py` 构建了一个玩具版 ArtPrompt。你可以将有害查询中的特定单词用 ASCII 艺术字形遮盖起来，验证遮盖后的字符串能通过关键词过滤器，并（可选）使用一个简单的识别器将遮盖后的字符串解码回原文。

## 交付物

本课程产出 `outputs/skill-encoding-audit.md`。给定一份越狱防御报告，它会枚举所涵盖的编码攻击家族（ASCII 艺术、base64、leet 语、UTF-8 同形字、UTES）以及拦截每种编码的防御层。

## 练习

1. 运行 `code/main.py`。验证遮盖后的字符串能通过一个简单的关键词过滤器。报告所需的字符级改变。
2. 实现第二种编码：针对同一目标单词的 base64。比较对 ArtPrompt 的过滤器绕过率和恢复难度。
3. 阅读 Jiang 等人 2024 年第 4.3 节（五个模型的结果）。提出一个原因，解释为什么 Claude 的 ArtPrompt 抗性在相同基准上高于 Gemini。
4. 设计一个检测提示中 ASCII 艺术形状区域的生成前防御。测量对合法代码、表格和数学符号的误报率。
5. StructuralSleight 列出了 10 种编码结构。草拟一个能处理所有 10 种的通用防御，并估算每个受防御提示的计算成本。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| ArtPrompt | "ASCII 艺术攻击" | 两步越狱，用 ASCII 艺术渲染遮盖安全词汇 |
| Cloaking | "隐藏单词" | 将禁止的 token 替换为模型能读取但过滤器不能读取的视觉表示 |
| UTES | "不常见结构" | 不常见文本编码结构——树、图、嵌套 JSON 等，用于携带内容 |
| ViTC | "视觉-文本能力" | 衡量模型读取非语义视觉编码能力的基准 |
| Perplexity filter | "PPL 防御" | 拒绝高困惑度提示；因合法结构化输入的困惑度也高而失败 |
| Retokenization | "分词器移位防御" | 用不同的分词器预处理提示；因识别是视觉的而失败 |
| Homoglyph | "形似字符" | 看起来与拉丁字母相同的 Unicode 字符；绕过子串检查 |

## 延伸阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753)——ASCII 艺术越狱论文
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754)——UTES 推广
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419)——互补的迭代攻击
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking)——互补的长度攻击
