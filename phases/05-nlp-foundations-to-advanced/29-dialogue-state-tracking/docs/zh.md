# 对话状态跟踪

> "我想要一家北部的便宜餐厅……实际上改成中档……再加意大利菜。"三轮对话，三次状态更新。DST 保持槽值字典同步，确保预订正常工作。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 17（聊天机器人），阶段 5 · 20（结构化输出）
**时间：** 约 75 分钟

## 问题

在面向任务的对话系统中，用户的目标被编码为一组槽值对：`{cuisine: italian, area: north, price: moderate}`。每一轮用户对话都可以添加、更改或删除一个槽位。系统必须读取整个对话并正确输出当前状态。

任何一个槽位出错，系统就会订错餐厅、安排错误的航班或扣错银行卡。DST 是用户所说的内容与后端执行操作之间的关键环节。

即使到了 2026 年，尽管有 LLM，DST 仍然重要的原因：

- 合规敏感领域（银行、医疗、机票预订）需要确定性的槽值，而非自由格式的生成。
- 工具使用智能体在调用 API 之前仍然需要槽位解析。
- 多轮纠错比看起来更难："实际上不，改成周四。"

现代流程：经典 DST 概念 + LLM 提取器 + 结构化输出护栏。

## 概念

**任务结构。** 模式定义了领域（餐厅、酒店、出租车）及其槽位（菜系、区域、价格、人数）。每个槽位可以为空、填充封闭集中的值（价格：{便宜、中档、昂贵}），或自由格式的值（名称："铜壶餐厅"）。

**两种 DST 形式：**

- **分类。** 对于每个（槽位、候选值）对，预测是/否。适用于封闭词汇的槽位。2020 年之前的标准做法。
- **生成。** 给定对话，以自由文本形式生成槽值。适用于开放词汇的槽位。现代的默认做法。

**评估指标。** 联合目标准确率（JGA）——每个轮次中*每个*槽位都正确的比例。全对或全错。MultiWOZ 2.4 排行榜在 2026 年的最佳成绩约为 83%。

**架构：**

1. **基于规则（槽位正则表达式 + 关键词）。** 针对狭窄领域的强基线。可调试。
2. **TripPy / BERT-DST。** 基于 BERT 编码的复制式生成。LLM 之前的标准做法。
3. **LDST（LLaMA + LoRA）。** 使用领域槽位提示进行指令微调的 LLM。在 MultiWOZ 2.4 上达到 ChatGPT 级别的质量。
4. **免本体（2024-2026）。** 跳过模式；直接生成槽名称和值。处理开放领域。
5. **提示 + 结构化输出（2024-2026）。** 使用 Pydantic 模式 + 约束解码的 LLM。5 行代码，生产就绪。

### 经典的失败模式

- **跨轮次共指。** "我们选第一个选项。"需要解析是哪个选项。
- **覆盖 vs 追加。** 用户说"加意大利菜。"是替换菜系还是追加？
- **隐式确认。** "好的，可以。"——这是在接受提供的预订吗？
- **纠错。** "实际上改成 7 点。"必须更新时间而不清除其他槽位。
- **指代上一轮系统话语。** "是的，那个。"哪个"那个"？

## 构建

### 步骤 1：基于规则的槽位提取器

正则表达式 + 同义词字典覆盖狭窄领域中 70% 的标准话语：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}

def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在标准词汇之外表现脆弱。适用于确定性的槽位确认。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变约束：

- 绝不要重置用户没有触及的槽位。
- 显式否定（"菜系无所谓"）必须清除。
- 用户纠错（"实际上……"）必须覆盖，而非追加。

### 步骤 3：使用结构化输出的 LLM 驱动的 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None

def llm_dst(history, llm):
    prompt = f"""你负责跟踪跨轮次餐厅预订的槽值。
到目前为止的对话：
{render(history)}

根据最新的用户轮次更新状态。只输出 JSON 状态。"""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证输出有效的状态对象。无需正则表达式、无需模式不匹配、不会产生幻觉槽位。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在所有槽位上完全正确的轮次比例是多少？对于 MultiWOZ 2.4，2026 年的顶级系统达到 80-83%。如果你的领域词汇集很小且 LLM 基线的表现超过了你的系统，那么你的领域内系统应该超过这个数字。

### 步骤 5：处理纠错

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}

def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到纠错时，覆盖最后更新的槽位而非追加。在没有 LLM 帮助的情况下很难做对。现代模式：总是让 LLM 从历史记录中重新生成整个状态，而非增量更新——这自然处理了纠错。

## 陷阱

- **全历史重新生成的成本。** 让 LLM 在每个轮次重新生成状态的总 token 成本是 O(n²)。对历史进行裁剪或总结较早的轮次。
- **模式漂移。** 事后添加新槽位会破坏旧的训练数据。对你的模式进行版本管理。
- **大小写敏感性。** "Italian" vs "italian" vs "ITALIAN"——在所有地方进行归一化。
- **隐式继承。** 如果用户之前指定了"4 个人"，则新的不同时间请求不应清除人数。总是传递完整的历史记录。
- **自由格式 vs 封闭集。** 名称、时间和地址需要自由格式的槽位；菜系和区域是封闭的。在模式中混合使用这两种类型。

## 使用

2026 年的技术栈：

| 情况 | 方法 |
|-----------|----------|
| 狭窄领域（一两个意图） | 基于规则 + 正则表达式 |
| 广泛领域，有标注数据 | LDST（在 MultiWOZ 风格数据上使用 LLaMA + LoRA） |
| 广泛领域，无标注，生产就绪 | LLM + Instructor + Pydantic 模式 |
| 语音 | ASR + 归一化器 + LLM-DST |
| 多领域预订流程 | 基于模式的 LLM，每个领域使用 Pydantic 模型 |
| 合规敏感 | 基于规则为主，LLM 后备配合确认流程 |

## 发布

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: 设计对话状态跟踪器——模式、提取器、更新策略、评估。
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

给定用例（领域、语言、词汇开放性、合规需求），输出：

1. 模式。领域列表、每个领域的槽位、每个槽位的开放 vs 封闭词汇。
2. 提取器。基于规则 / seq2seq / 使用 Pydantic 的 LLM。说明原因。
3. 更新策略。重新生成完整状态 / 增量更新；纠错处理；否定处理。
4. 评估。在保留的对话集上的联合目标准确率，槽位级别的精确率/召回率，最难搞的槽位的混淆情况。
5. 确认流程。何时显式询问用户确认（破坏性操作、低置信度提取）。

对于合规敏感的槽位，拒绝仅依赖 LLM 的 DST，要求有基于规则的辅助检查。拒绝任何无法在用户纠错时回滚槽位的 DST。标记没有版本标签的模式。
```

## 练习

1. **简单。** 为 3 个槽位（菜系、区域、价格）构建基于规则的状态跟踪器。在 10 个人工构建的对话上进行测试。测量 JGA。
2. **中等。** 使用 Instructor + Pydantic + 小型 LLM 在相同数据集上测试。比较 JGA。检查最难的轮次。
3. **困难。** 实现两者并进行路由：基于规则为主，当基于规则以低于 2 个槽位且置信度不高时，LLM 后备。测量组合的 JGA 和每个轮次的推理成本。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| DST | 对话状态跟踪 | 在对话轮次中维护槽值字典。 |
| 槽位 | 用户意图单元 | 后端需要的命名参数（菜系、日期）。 |
| 领域 | 任务区域 | 餐厅、酒店、出租车——槽位的集合。 |
| JGA | 联合目标准确率 | 每个轮次所有槽位完全正确的比例。全对或全错。 |
| MultiWOZ | 基准测试 | 多领域 WOZ 数据集；标准 DST 评估。 |
| 免本体 DST | 无模式 | 直接生成槽名称和值，无固定列表。 |
| 纠错 | "实际上……" | 覆盖先前已填充槽位的轮次。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 标准基准。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — 针对 DST 的 LLaMA + LoRA 指令微调。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 基于复制的 DST 主力模型。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于 EM 的无监督 TOD。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — 标准 DST 结果。
