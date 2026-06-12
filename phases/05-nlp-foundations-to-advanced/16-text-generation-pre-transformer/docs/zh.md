# 聊天机器人与对话式 AI — 对话即系统设计

> "你好"的回应方式有 1000 种。聊天机器人的挑战在于知道如何选择合适的。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 15（文本生成）
**时间：** 约 45 分钟

## 问题

QA 系统回答一个问题。聊天机器人维持对话。区别在于状态——每次交换都会改变对话的状态。之前说过什么？今天晚些时候或整周，用户的需求可能会发生变化。聊天机器人必须在多轮对话中跟踪状态，同时保持个性一致且有帮助。

## 概念

**基于检索的聊天机器人** 从预定义的回应库中选择回应。它们不会生成新的文本，而是选择最合适的回应。更安全（不会产生幻觉），但缺乏灵活性。当回应数量有限且任务明确时（常见问题解答、客服脚本），效果最好。

**基于生成的聊天机器人** 根据对话历史生成回应的序列到序列模型。它们更灵活，但需要更多的训练数据，并且可能产生不合逻辑或不安全的回应。

**混合型聊天机器人** 使用基于检索的方法来处理常见情况（问候、常见问题解答），使用生成式 AI 来应对更复杂的情况。大多数生产聊天机器人使用混合模式。

```figure
ngram-backoff
```

## 概念（续）

**对话管理** 跟踪三件事：当前状态（槽位填充 - 用户提供了航班号了吗？）、对话历史（他们几分钟前提过退款吗？）以及外部上下文（退货窗口是多久？）。

**个性化** 通过保持一致的说话风格、记住用户偏好（"我是素食主义者"）以及适应用户的详细程度，使聊天机器人感觉像是一个连贯的个体。没有个人一致性，聊天机器人就无法通过图灵测试——它看起来像一个没有记忆的智能体。

## 构建

### 步骤 1：基于检索的回应选择器

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RetrievalBot:
    def __init__(self):
        self.intents = {}
        self.vectorizer = TfidfVectorizer()

    def add_intent(self, intent, patterns, response):
        self.intents[intent] = {"patterns": patterns, "response": response}

    def respond(self, message):
        all_patterns = []
        intent_map = []
        for intent, data in self.intents.items():
            for p in data["patterns"]:
                all_patterns.append(p)
                intent_map.append(intent)

        vec = self.vectorizer.fit_transform(all_patterns + [message])
        sims = cosine_similarity(vec[-1:], vec[:-1])[0]
        best_idx = sims.argmax()

        if sims[best_idx] > 0.3:
            best_intent = intent_map[best_idx]
            return self.intents[best_intent]["response"]
        return "I'm not sure how to respond to that."
```

TF-IDF 向量化器将用户消息与所有已知模式进行比较。最佳匹配的意图被选择用于回复。`threshold=0.3` 的匹配阈值确保聊天机器人在不确定时会优雅地降级，而不是自信地输出无关的回复。

### 步骤 2：基于槽位填充的对话状态跟踪

```python
class SlotFillingState:
    def __init__(self, slots, required_slots):
        self.slots = slots  # slot 名 → 默认值
        self.required = required_slots  # 必须填写的 slot 列表

    def fill(self, slot, value):
        self.slots[slot] = value

    def is_complete(self):
        return all(self.slots[s] is not None for s in self.required)

    def next_question(self):
        for s in self.required:
            if self.slots[s] is None:
                return f"What is the {s}?"
        return None
```

```python
# 航班预订流程
state = SlotFillingState(
    slots={"origin": None, "destination": None, "date": None},
    required=["origin", "destination"]
)

# 多轮对话
state.fill("origin", "SFO")       # "我想从 SFO 出发"
print(state.next_question())       # "What is the destination?"
state.fill("destination", "JFK")  # "去 JFK"
print(state.is_complete())         # True → 触发预订动作
```

槽位填充聊天机器人执行"结构化对话"——它们从一个流程中规定的显式定义插槽开始。如果用户提供了三个槽位，它们就会一次性处理所有信息。"从 SFO 到 JFK 的航班"——这用一个输入就填充了 `origin` 和 `destination`。

### 步骤 3：生成式聊天机器人的对话历史记忆

```python
class ConversationMemory:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history

    def add_turn(self, user, bot):
        self.history.append({"user": user, "bot": bot})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def format_for_model(self):
        lines = []
        for turn in self.history:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Bot: {turn['bot']}")
        return "\n".join(lines)
```

历史记录按时间顺序存储在列表中。当长度超过 `max_history` 时，最早的对话回合被弹出，以保持上下文大小固定。格式化函数将历史记录转换为模型提示输入。

## 使用

### LangChain 中的对话链

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
chain = ConversationChain(memory=memory)
response = chain.predict(input="Hello, I need help with an order")
```

LangChain 的 `ConversationBufferMemory` 在内存中维护对话历史，并在每次 LLM 调用之前自动将其插入提示中。`ConversationSummaryMemory` 使用 LLM 生成对话摘要，以避免在长时间的对话中填满上下文窗口。

## 发布

聊天机器人评估框架。

保存为 `outputs/prompt-chatbot-eval.md`：

```markdown
---
name: chatbot-eval
description: 提示：多轮对话机器人的评估维度。
phase: 5
lesson: 16
---

评估聊天机器人。针对多轮对话：

1. 一致性：聊天机器人是否自我矛盾？（例如：之前说支持退款，后来却说不能）。
2. 恢复能力：当聊天机器人犯错时，它能否优雅地恢复？（"对不起，让我再确认一下"）。
3. 语境保持：5 次交换后，聊天机器人是否仍然记得对话的最初目的？
4. 降级：当聊天机器人不确定时，它会说"我不知道"还是随机猜测？
5. 安全：聊天机器人会拒绝有害请求吗？它是否被提示注入绕过？

评估最好在真实的用户交互中进行，而非固定的数据集。
```

## 练习

1. **简单。** 在本地运行一个会话式 pipeline。针对一个主题进行多轮对话（5+ 个回合）——模型能否保持语境？
2. **中等。** 为披萨店实现一个基于检索的聊天机器人。定义 5 个意图及其对应的模式。添加一个"我不知道"的后备回复。测试它能否从用户的意外输入中恢复。
3. **困难。** 构建一个混合聊天机器人，将基于检索的槽位填充（用于预订）与生成式应答（用于开放域聊天）相结合。使用对话管理器在两种模式之间进行路由。评估你系统的任务完成率和用户满意度。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 槽位填充 | 信息收集 | 在完成任务前从用户那里强制收集特定信息片段。 |
| 对话管理 | 状态跟踪 | 跟踪到目前为止收集到的信息以及下一步需要的决策。 |
| 基于检索的 | 预定义的回应 | 从预定义的回应库中选择回应。 |
| 基于生成的 | 合成回应 | 使用 Seq2Seq 或 LLM 生成回应的文本。 |
| 混合模式 | 两者结合 | 检索用于常见场景，生成用于处理复杂情况。 |
