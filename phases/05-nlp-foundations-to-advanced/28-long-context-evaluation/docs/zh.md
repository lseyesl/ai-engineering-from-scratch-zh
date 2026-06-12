# 对话状态跟踪 — 记忆是一切

> 用户可以说"把它改成周二"。对话状态要知道"它"是"预订"和"周二"是"2026-06-16"。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 16（聊天机器人）
**时间：** 约 50 分钟

## 问题

对话状态跟踪（DST）是任务型对话系统的核心。它跟踪对话的当前状态——收集到的、缺失的和已确认的信息。

任务是：用户说"我想订一张去纽约的航班"。系统需要知道：`destination = "纽约"`、`intent = "book_flight"`。然后用户说"周二出发"。系统更新：`date = "周二"`。然后系统需要查询航班信息。如果没有 DST，每次对话后系统都会遗忘之前的信息。

## 概念

**槽位填充 DST：** 跟踪预定义槽位的值："目的地"、"日期"、"人数"。在每一步更新填写或覆盖槽位。适用于结构化的任务型对话。

**基于帧的 DST：** 每个"帧"跟踪一个独立的用户目标。如果用户同时想要订航班和订酒店，就会有两个帧。

**神经 DST：** 使用 BERT 对槽位值进行分类，或使用生成式模型预测状态变化。泛化能力更强——可以处理未见过得槽位值（如"目的地：特古西加尔巴"）。

## 构建

### 步骤 1：基于规则的状态跟踪器

```python
import re
from datetime import datetime, timedelta

class RuleDST:
    def __init__(self, slots):
        self.slots = slots  # 槽位模式
        self.state = {s: None for s in slots}

    def update(self, utterance):
        """从用户输入中更新槽位状态。"""
        # 日期提取
        date_match = re.search(r"(\w+)day", utterance, re.IGNORECASE)
        if date_match:
            self.state["date"] = date_match.group(0)

        # 城市提取
        cities = ["New York", "London", "Tokyo", "Paris", "Berlin",
                  "San Francisco", "Beijing", "Sydney"]
        for city in cities:
            if city.lower() in utterance.lower():
                if self.state.get("origin") is None:
                    self.state["origin"] = city
                elif self.state.get("destination") is None:
                    self.state["destination"] = city

        # 人数
        people_match = re.search(r"(\d+) (people|persons|passengers|tickets)",
                                 utterance, re.IGNORECASE)
        if people_match:
            self.state["passengers"] = int(people_match.group(1))

    def is_complete(self):
        return all(v is not None for v in self.state.values())

    def missing_slots(self):
        return [s for s, v in self.state.items() if v is None]
```

基于规则的状态更新优先使用正则表达式模式。日期提取需要处理"下周二"（相对日期）与"6 月 15 日"（绝对日期）的区别。

### 步骤 2：基于 BERT 的 DST

```python
class BertDST:
    def __init__(self, model, slot_types):
        self.model = model
        self.slot_types = slot_types
        self.state = {}

    def update_state(self, utterance, history=""):
        """使用 BERT 从话语中分类槽位值。"""
        context = history + " " + utterance if history else utterance

        for slot_name, slot_type in self.slot_types.items():
            if slot_type == "categorical":
                # 分类槽位：预定义值的分类
                candidate = self._classify_slot(context, slot_name)
                if candidate and candidate != "none":
                    self.state[slot_name] = candidate

            elif slot_type == "extractive":
                # 抽取式槽位：从上下文中提取（去程日期）
                value = self._extract_slot(context, slot_name)
                if value:
                    self.state[slot_name] = value

            elif slot_type == "generative":
                # 生成式槽位：自由形式的值
                value = self._generate_slot(context, slot_name)
                if value:
                    self.state[slot_name] = value

        return self.state
```

基于 BERT 的 DST 在上下文中对对话进行分类。它将对话的当前状态编码为 BERT 的输入，并输出每个槽位的预测。MultiWOZ 和 SGD（模式引导对话）数据集是为 DST 范围提供支持的标准基准。

### 步骤 3：请求槽位与确认

```python
class DialogueManager:
    def __init__(self, dst, actions):
        self.dst = dst
        self.actions = actions  # 可用操作

    def process(self, utterance):
        self.dst.update(utterance)
        missing = self.dst.missing_slots()

        if missing:
            # 请求缺失的信息
            return f"I still need: {', '.join(missing)}."
        else:
            # 确认信息然后执行操作
            confirm = "您想确认以下信息吗？\n"
            for slot, value in self.dst.state.items():
                confirm += f"- {slot}: {value}\n"
            return confirm + "（是 / 修改）"
```

有确认功能的对话管理器会先向用户展示收集到的信息，然后再执行操作。如果信息有误，用户可以逐一修改。这对基于对话的预订任务至关重要——一次性确认比逐步确认更高效。

## 使用

### schema-guided DST

```python
from datasets import load_dataset

dataset = load_dataset("schema_guided_dialog")

def evaluate_dst(model, dataset_split):
    correct = 0
    total = 0
    for dialog in dataset_split:
        state = {}
        for turn in dialog["turns"]:
            utterance = turn["utterance"]
            state = model.update_state(utterance, state)
            if turn["active_intent"] == model.predict_intent(utterance):
                correct += 1
            total += 1
    return correct / total
```

Schema-Guided DST 由 Google 发布，涵盖了超过 20 种服务（餐厅、酒店、电影、购物等），每种服务都有自己的一套槽位定义。对于评估 DST 系统的泛化能力非常有用。

### 用于 DST 评估的 MultiWOZ

MultiWOZ 是一个包含超过 10000 个多领域对话的数据集。对话跨多个领域进行切换（例如，先订餐厅，然后订出租车）。常用的 DST 评估指标包括联合目标准确率（其中 JGA 测量每个对话回合中所有槽位的正确预测），以及请求成功率。

## 发布

DST 系统验证提示。

保存为 `outputs/prompt-dst-verification.md`：

```markdown
---
name: dst-verification
description: 提示：验证对话状态跟踪（DST）系统。
phase: 5
lesson: 28
---

验证对话状态跟踪（DST）系统：

1. 状态更新准确性：在对话中更新状态时是否存在值丢失或覆盖错误的情况？
2. 槽位删除处理：当用户取消某个值时（"我不需要酒店了"），系统是否删除了相关槽位？
3. 多槽位同步：当用户一次性提供多个值时，系统是否同时捕获了所有值？
4. 否定处理：用户说"周二不行，周三"时，系统是否更新为周三（而不是同时保留两者）？
5. 跨对话重置：当新对话开始时，状态是否重置？

计算联合目标准确率和槽位准确率。
```

## 练习

1. **简单。** 实现一个基于规则的 DST，带"目的地"和"日期"两个槽位。测试以下序列："我想去纽约"→"周三出发"。状态是否正确更新？
2. **中等。** 在 MultiWOZ 的 100 个对话上评估规则 DST。计算联合目标准确率。模型表现如何？
3. **困难。** 实现基于 BERT 的 DST（将状态编码为 BERT 输入）。在 MultiWOZ 数据集上训练和评估。将神经 DST 的表现与规则基准进行比较。（提示：使用 `bert-base-uncased` 作为起始，并添加一个槽位分类层。）

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| DST | 对话状态跟踪 | 跟踪从对话开始收集到的所有信息的系统。 |
| 槽位 | 信息字段 | 对话中需要填充值的单位（目的地、日期）。 |
| 意图 | 用户目标 | 用户想要完成的操作（预订航班、搜索酒店）。 |
| JGA | 联合目标准确率 | 在一个对话回合中所有槽位正确预测的百分比。严格的指标。 |
| MultiWOZ | 多领域对话 | 标准的 DST 评估数据集，包含 10000+ 个多领域对话。 |
