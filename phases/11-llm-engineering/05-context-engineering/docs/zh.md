# 上下文工程：窗口、预算、记忆与检索

> 提示词工程是一个子集。上下文工程才是全局。提示词是你输入的一个字符串。上下文是进入模型窗口的所有内容：系统指令、检索到的文档、工具定义、对话历史、few-shot 示例以及提示词本身。2026 年最优秀的 AI 工程师都是上下文工程师。他们决定什么进去、什么不进去，以及以什么顺序。

**类型：** Build
**语言：** Python
**前置要求：** 阶段 10（LLMs from Scratch），阶段 11 课程 01-02
**预计时间：** ~90 分钟
**关联：** 阶段 11 · 15（提示词缓存）——缓存友好的布局是上下文工程的延伸。阶段 5 · 28（长上下文评估）了解如何使用 NIAH/RULER 衡量"中间迷失"。

## 学习目标

- 计算所有上下文窗口组件的 token 预算（系统提示词、工具、历史记录、检索文档、生成空间）
- 实现上下文窗口管理策略：截断、摘要和对话历史的滑动窗口
- 对上下文组件进行优先级排序和排序，以最大化模型对最相关信息的注意力
- 构建一个上下文组装器，根据查询类型和可用窗口空间动态分配 token

## 问题

Claude Opus 4.7 有 200K token 的窗口（beta 版 1M）。GPT-5 有 400K。Gemini 3 Pro 有 2M。Llama 4 声称有 10M。这些数字听起来很大，直到你填满它们。

以下是编码助手的实际分解。系统提示词：500 tokens。50 个工具的定义：8,000 tokens。检索到的文档：4,000 tokens。对话历史（10 轮）：6,000 tokens。当前用户查询：200 tokens。生成预算（最大输出）：4,000 tokens。总计：22,700 tokens。这仅占 128K 窗口的 18%。

但注意力并不随上下文长度线性扩展。具有 128K token 上下文的模型付出二次注意力成本（在标准 transformer 中为 O(n²)，尽管大多数生产模型使用高效注意力变体）。更重要的是，检索准确率会下降。"大海捞针"测试表明，模型难以找到放置在长上下文中间的信息。Liu 等人（2023）的研究表明，LLM 在长上下文的开头和结尾以接近完美的准确率检索信息，但对于放置在中间（上下文位置的 40-70%）的信息，准确率下降 10-20%。这种"中间迷失"效应因模型而异，但影响所有当前架构。

实际的教训：拥有 200K tokens 不意味着使用 200K tokens 是有效的。精心策划的 10K token 上下文通常胜过倾倒的 100K token 上下文。上下文工程是在上下文窗口内最大化信噪比的学科。

你放入窗口的每个 token 都会挤掉一个可能携带更相关信息的 token。每个无关的工具定义、每轮过期的对话、每个不能回答问题的检索文本块——每一个都使模型在该任务上稍差一些。

## 概念

### 上下文窗口是稀缺资源

将上下文窗口视为内存，而非磁盘。它快速且可直接访问，但有限。你无法容纳所有内容。你必须做出选择。

```mermaid
graph TD
    subgraph Window["上下文窗口（128K tokens）"]
        direction TB
        S["系统提示词\n~500 tokens"] --> T["工具定义\n~2K-8K tokens"]
        T --> R["检索上下文\n~2K-10K tokens"]
        R --> H["对话历史\n~2K-20K tokens"]
        H --> F["Few-shot 示例\n~1K-3K tokens"]
        F --> Q["用户查询\n~100-500 tokens"]
        Q --> G["生成预算\n~2K-8K tokens"]
    end

    style S fill:#1a1a2e,stroke:#e94560,color:#fff
    style T fill:#1a1a2e,stroke:#0f3460,color:#fff
    style R fill:#1a1a2e,stroke:#ffa500,color:#fff
    style H fill:#1a1a2e,stroke:#51cf66,color:#fff
    style F fill:#1a1a2e,stroke:#9b59b6,color:#fff
    style Q fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#0f3460,color:#fff
```

每个组件都在争夺空间。添加更多工具定义意味着对话历史的可用空间减少。添加更多检索上下文意味着 few-shot 示例的空间减少。上下文工程是分配这个预算以最大化任务性能的艺术。

### 中间迷失

上下文工程中最重要的经验发现。模型对上下文开头和结尾的信息关注更好。中间的信息获得较低的注意力分数，更可能被忽略。

Liu 等人（2023）系统地测试了这一点。他们将相关文档放置在 20 个不相关文档中的不同位置，并衡量回答准确率。当相关文档在第一个或最后一个时，准确率为 85-90%。当它在中间时（20 个中的第 10 个），准确率下降到 60-70%。

这对工程有直接的影响：

- 将最重要的信息放在开头（系统提示词、关键指令）
- 将当前查询和最相关的上下文放在最后（近因偏差有帮助）
- 将上下文的中间视为最低优先级区域
- 如果必须在中间包含信息，在结尾重复关键点

```mermaid
graph LR
    subgraph Attention["跨上下文的注意力分布"]
        direction LR
        P1["位置 0-20%\n高注意力\n（系统提示词）"]
        P2["位置 20-40%\n中等"]
        P3["位置 40-70%\n低注意力\n（中间迷失）"]
        P4["位置 70-90%\n中等"]
        P5["位置 90-100%\n高注意力\n（当前查询）"]
    end

    style P1 fill:#51cf66,color:#000
    style P2 fill:#ffa500,color:#000
    style P3 fill:#ff6b6b,color:#fff
    style P4 fill:#ffa500,color:#000
    style P5 fill:#51cf66,color:#000
```

### 上下文组件

**系统提示词**：设定身份、约束和行为规则。这放在开头，并且在对话轮次中保持不变。Claude Code 的系统提示词大约使用 6,000 tokens，包括工具定义和行为指令。保持紧凑。系统提示词中的每个词都在每次 API 调用中重复出现。

**工具定义**：每个工具增加 50-200 tokens（名称、描述、参数 schema）。50 个工具每个 150 tokens 就是 7,500 tokens，在开始任何对话之前。动态工具选择——只包含与当前查询相关的工具——可以将此减少 60-80%。

**检索上下文**：来自向量数据库的文档、搜索结果、文件内容。检索质量直接决定响应质量。糟糕的检索比没有检索更糟糕——它用噪声填充窗口并主动误导模型。

**对话历史**：每个先前的用户消息和助手响应。随对话长度线性增长。50 轮的对话，每轮 200 tokens，就是 10,000 tokens 的历史记录。其中大部分与当前查询无关。

**Few-shot 示例**：展示期望行为的输入/输出对。两到三个精心挑选的示例往往比数千 token 的指令更能提高输出质量。但它们占用空间。

**生成预算**：为模型响应保留的 tokens。如果你将窗口填满到容量，模型就没有空间回答。至少保留 2,000-4,000 tokens 用于生成。

### 上下文压缩策略

**历史摘要**：不是逐字保留所有先前的对话轮次，而是定期总结对话。"我们讨论了 X，决定了 Y，用户想要 Z"用 100 tokens 替换了原本需要 2,000 tokens 的 10 轮对话。当历史记录超过阈值（例如 5,000 tokens）时运行摘要。

**相关性过滤**：对每个检索到的文档根据当前查询评分，并丢弃低于阈值的文档。如果你检索了 10 个块但只有 3 个相关，丢弃其他 7 个。拥有 3 个高度相关的块比 10 个平庸的块更好。

**工具修剪**：对用户的查询意图进行分类，只包含与该意图相关的工具。代码问题不需要日历工具。日程安排问题不需要文件系统工具。这可以将工具定义从 8,000 tokens 减少到 1,000。

**递归摘要**：对于非常长的文档，分阶段摘要。先总结每个部分，然后总结摘要。一份 50 页的文档变成一个 500 token 的摘要，捕捉关键点。

### 记忆系统

上下文工程跨越三个时间跨度。

**短期记忆**：当前对话。直接存储在上下文窗口中。随每轮对话增长。通过摘要和截断进行管理。

**长期记忆**：跨对话持续存在的事实和偏好。"用户偏好 TypeScript。""项目使用 PostgreSQL。"存储在数据库中，在会话开始时检索。Claude Code 将其存储在 CLAUDE.md 文件中。ChatGPT 将其存储在其记忆功能中。

**情景记忆**：可能相关的特定过往交互。"上周二，我们在 auth 模块中调试了类似问题。"作为嵌入存储，在当前对话匹配过去情景时检索。

```mermaid
graph TD
    subgraph Memory["记忆架构"]
        direction TB
        STM["短期记忆\n（当前对话）\n直接在上下文窗口中"]
        LTM["长期记忆\n（事实、偏好）\n数据库 -> 会话开始时检索"]
        EM["情景记忆\n（过往交互）\n嵌入 -> 按相似度检索"]
    end

    Q["当前查询"] --> STM
    Q --> LTM
    Q --> EM

    STM --> CW["上下文窗口"]
    LTM --> CW
    EM --> CW

    style STM fill:#1a1a2e,stroke:#51cf66,color:#fff
    style LTM fill:#1a1a2e,stroke:#0f3460,color:#fff
    style EM fill:#1a1a2e,stroke:#e94560,color:#fff
    style CW fill:#1a1a2e,stroke:#ffa500,color:#fff
```

### 动态上下文组装

关键洞见：不同的查询需要不同的上下文。静态系统提示词 + 静态工具 + 静态历史是浪费的。最好的系统根据查询动态组装上下文。

1. 分类查询意图
2. 选择相关工具（不是所有工具）
3. 检索相关文档（不是固定集合）
4. 包含相关的历史轮次（不是全部历史）
5. 添加匹配任务类型的 few-shot 示例
6. 按重要性排序：关键的在开头，重要的在结尾，可选的放在中间

这就是优秀 AI 应用与伟大 AI 应用的区别。模型是相同的。上下文是差异化因素。

## 构建它

### 步骤 1：Token 计数器

你无法预算你无法衡量的东西。构建一个简单的 token 计数器（使用空格分割的近似值，因为精确计数取决于 tokenizer）。

```python
import json
import numpy as np
from collections import OrderedDict

def count_tokens(text):
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

def count_tokens_json(obj):
    return count_tokens(json.dumps(obj))
```

### 步骤 2：上下文预算管理器

核心抽象。预算管理器跟踪每个组件使用多少 token 并强制执行限制。

```python
class ContextBudget:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.max_tokens = max_tokens
        self.generation_reserve = generation_reserve
        self.available = max_tokens - generation_reserve
        self.allocations = OrderedDict()

    def allocate(self, component, content, max_tokens=None):
        tokens = count_tokens(content)
        if max_tokens and tokens > max_tokens:
            words = content.split()
            target_words = int(max_tokens / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        used = sum(self.allocations.values())
        if used + tokens > self.available:
            allowed = self.available - used
            if allowed <= 0:
                return None, 0
            words = content.split()
            target_words = int(allowed / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        self.allocations[component] = tokens
        return content, tokens

    def remaining(self):
        used = sum(self.allocations.values())
        return self.available - used

    def utilization(self):
        used = sum(self.allocations.values())
        return used / self.max_tokens

    def report(self):
        total_used = sum(self.allocations.values())
        lines = []
        lines.append(f"上下文预算报告（{self.max_tokens:,} token 窗口）")
        lines.append("-" * 50)
        for component, tokens in self.allocations.items():
            pct = tokens / self.max_tokens * 100
            bar = "#" * int(pct / 2)
            lines.append(f"  {component:<25} {tokens:>6} tokens ({pct:>5.1f}%) {bar}")
        lines.append("-" * 50)
        lines.append(f"  {'已使用':<25} {total_used:>6} tokens ({total_used/self.max_tokens*100:.1f}%)")
        lines.append(f"  {'生成预留':<25} {self.generation_reserve:>6} tokens")
        lines.append(f"  {'剩余':<25} {self.remaining():>6} tokens")
        return "\n".join(lines)
```

### 步骤 3：中间迷失重排序

实现重排序策略：最重要的项目放在开头和结尾，最不重要的放在中间。

```python
def reorder_lost_in_middle(items, scores):
    paired = sorted(zip(scores, items), reverse=True)
    sorted_items = [item for _, item in paired]

    if len(sorted_items) <= 2:
        return sorted_items

    first_half = sorted_items[::2]
    second_half = sorted_items[1::2]
    second_half.reverse()

    return first_half + second_half

def score_relevance(query, documents):
    query_words = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        if not query_words:
            scores.append(0.0)
            continue
        overlap = len(query_words & doc_words) / len(query_words)
        scores.append(round(overlap, 3))
    return scores
```

### 步骤 4：对话历史压缩器

摘要旧的对话轮次以回收 token 预算。

```python
class ConversationManager:
    def __init__(self, max_history_tokens=5000):
        self.turns = []
        self.summaries = []
        self.max_history_tokens = max_history_tokens

    def add_turn(self, role, content):
        self.turns.append({"role": role, "content": content})
        self._compress_if_needed()

    def _compress_if_needed(self):
        total = sum(count_tokens(t["content"]) for t in self.turns)
        if total <= self.max_history_tokens:
            return

        while total > self.max_history_tokens and len(self.turns) > 4:
            old_turns = self.turns[:2]
            summary = self._summarize_turns(old_turns)
            self.summaries.append(summary)
            self.turns = self.turns[2:]
            total = sum(count_tokens(t["content"]) for t in self.turns)

    def _summarize_turns(self, turns):
        parts = []
        for t in turns:
            content = t["content"]
            if len(content) > 100:
                content = content[:100] + "..."
            parts.append(f"{t['role']}：{content}")
        return "之前： " + " | ".join(parts)

    def get_context(self):
        parts = []
        if self.summaries:
            parts.append("[对话摘要]")
            for s in self.summaries:
                parts.append(s)
        parts.append("[最近对话]")
        for t in self.turns:
            parts.append(f"{t['role']}：{t['content']}")
        return "\n".join(parts)

    def token_count(self):
        return count_tokens(self.get_context())
```

### 步骤 5：动态工具选择器

只包含与当前查询相关的工具。分类意图，然后过滤。

```python
TOOL_REGISTRY = {
    "read_file": {
        "description": "读取文件内容",
        "tokens": 120,
        "categories": ["code", "files"],
    },
    "write_file": {
        "description": "写入内容到文件",
        "tokens": 150,
        "categories": ["code", "files"],
    },
    "search_code": {
        "description": "在代码库中搜索模式",
        "tokens": 130,
        "categories": ["code"],
    },
    "run_command": {
        "description": "执行 shell 命令",
        "tokens": 140,
        "categories": ["code", "system"],
    },
    "create_calendar_event": {
        "description": "创建新的日历事件",
        "tokens": 180,
        "categories": ["calendar"],
    },
    "list_emails": {
        "description": "列出最近的邮件",
        "tokens": 160,
        "categories": ["email"],
    },
    "send_email": {
        "description": "发送电子邮件",
        "tokens": 200,
        "categories": ["email"],
    },
    "web_search": {
        "description": "搜索网络信息",
        "tokens": 140,
        "categories": ["research"],
    },
    "query_database": {
        "description": "在数据库上运行 SQL 查询",
        "tokens": 170,
        "categories": ["code", "data"],
    },
    "generate_chart": {
        "description": "根据数据生成图表",
        "tokens": 190,
        "categories": ["data", "visualization"],
    },
}

def classify_intent(query):
    query_lower = query.lower()

    intent_keywords = {
        "code": ["code", "function", "bug", "error", "file", "implement", "refactor", "debug", "test"],
        "calendar": ["meeting", "schedule", "calendar", "appointment", "event"],
        "email": ["email", "mail", "send", "inbox", "message"],
        "research": ["search", "find", "what is", "how does", "explain", "look up"],
        "data": ["data", "query", "database", "chart", "graph", "analytics", "sql"],
    }

    scores = {}
    for intent, keywords in intent_keywords.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return ["code"]

    max_score = max(scores.values())
    return [intent for intent, score in scores.items() if score >= max_score * 0.5]

def select_tools(query, token_budget=2000):
    intents = classify_intent(query)
    relevant = {}
    total_tokens = 0

    for name, tool in TOOL_REGISTRY.items():
        if any(cat in intents for cat in tool["categories"]):
            if total_tokens + tool["tokens"] <= token_budget:
                relevant[name] = tool
                total_tokens += tool["tokens"]

    return relevant, total_tokens
```

### 步骤 6：完整上下文组装管道

将所有组件连接在一起。给定一个查询，动态组装最优上下文。

```python
class ContextEngine:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.budget = ContextBudget(max_tokens, generation_reserve)
        self.conversation = ConversationManager(max_history_tokens=5000)
        self.system_prompt = (
            "你是一个有用的 AI 助手。你可以使用代码编辑、文件管理、"
            "网络搜索和数据分析工具。"
            "对每个任务使用适当的工具。要简洁和准确。"
        )
        self.knowledge_base = [
            "Python 3.12 引入了使用括号表示法的泛型类类型参数语法。",
            "项目使用 PostgreSQL 16 和 pgvector 进行嵌入存储。",
            "身份验证由 Supabase Auth 处理，使用 JWT token。",
            "前端使用 Next.js 15 构建，采用 App Router。",
            "API 速率限制设置为每个用户每分钟 100 个请求。",
            "部署管道使用 GitHub Actions 和 Docker 多阶段构建。",
            "所有新模块的测试覆盖率必须超过 80%。",
            "代码库遵循数据访问的仓库模式。",
        ]

    def assemble(self, query):
        self.budget = ContextBudget(self.budget.max_tokens, self.budget.generation_reserve)

        system_content, _ = self.budget.allocate("system_prompt", self.system_prompt, max_tokens=1000)

        tools, tool_tokens = select_tools(query, token_budget=2000)
        tool_text = json.dumps(list(tools.keys()))
        tool_content, _ = self.budget.allocate("tools", tool_text, max_tokens=2000)

        relevance = score_relevance(query, self.knowledge_base)
        threshold = 0.1
        relevant_docs = [
            doc for doc, score in zip(self.knowledge_base, relevance)
            if score >= threshold
        ]

        if relevant_docs:
            doc_scores = [s for s in relevance if s >= threshold]
            reordered = reorder_lost_in_middle(relevant_docs, doc_scores)
            doc_text = "\n".join(reordered)
            doc_content, _ = self.budget.allocate("retrieved_context", doc_text, max_tokens=3000)

        history_text = self.conversation.get_context()
        if history_text.strip():
            history_content, _ = self.budget.allocate("conversation_history", history_text, max_tokens=5000)

        query_content, _ = self.budget.allocate("user_query", query, max_tokens=500)

        return self.budget

    def chat(self, query):
        self.conversation.add_turn("user", query)
        budget = self.assemble(query)
        response = f"[对以下内容的响应：{query[:50]}...]"
        self.conversation.add_turn("assistant", response)
        return budget


def run_demo():
    print("=" * 60)
    print("  上下文工程管道演示")
    print("=" * 60)

    engine = ContextEngine(max_tokens=128000, generation_reserve=4000)

    print("\n--- 查询 1：代码任务 ---")
    budget = engine.chat("修复认证模块中 JWT token 过期太早的 bug")
    print(budget.report())

    print("\n--- 查询 2：研究任务 ---")
    budget = engine.chat("在 PostgreSQL 中实现向量搜索的最佳方法是什么？")
    print(budget.report())

    print("\n--- 查询 3：对话历史累积后 ---")
    for i in range(8):
        engine.conversation.add_turn("user", f"关于系统实现细节的跟进问题 {i+1}")
        engine.conversation.add_turn("assistant", f"这是对跟进 {i+1} 的响应，包含架构的技术细节")

    budget = engine.chat("现在实现我们讨论的更改")
    print(budget.report())

    print("\n--- 工具选择示例 ---")
    test_queries = [
        "修复 auth.py 中的 bug",
        "安排周二与团队的会议",
        "显示数据库查询性能统计",
        "搜索关于错误处理的最佳实践",
    ]

    for q in test_queries:
        tools, tokens = select_tools(q)
        intents = classify_intent(q)
        print(f"\n  查询：{q}")
        print(f"  意图：{intents}")
        print(f"  工具：{list(tools.keys())}（{tokens} tokens）")

    print("\n--- 中间迷失重排序 ---")
    docs = ["文档 A（最相关）", "文档 B（有些相关）", "文档 C（最不相关）",
            "文档 D（相关）", "文档 E（中等相关）"]
    scores = [0.95, 0.60, 0.20, 0.80, 0.50]
    reordered = reorder_lost_in_middle(docs, scores)
    print(f"  原始顺序：{docs}")
    print(f"  分数：    {scores}")
    print(f"  重排序：  {reordered}")
    print(f"  （最相关在开头和结尾，最不相关在中间）")
```

## 使用它

### Claude Code 的上下文策略

Claude Code 采用分层方法管理上下文。系统提示词包括行为规则和工具定义（~6K tokens）。当你打开文件时，其内容作为上下文注入。当你搜索时，结果被添加。旧的对话轮次被摘要。CLAUDE.md 提供跨会话持续的长期记忆。

关键的工程决策：Claude Code 不会将整个代码库倾倒到上下文中。它按需检索相关文件。这就是上下文工程的实际应用。

### Cursor 的动态上下文加载

Cursor 将整个代码库索引为嵌入。当你输入查询时，它使用向量相似度检索最相关的文件和代码块。只有这些片段进入上下文窗口。一个 500K 行的代码库被压缩为 5-10 个最相关的代码块。

这就是模式：嵌入所有内容，按需检索，只包含重要的部分。

### ChatGPT 记忆

ChatGPT 将用户偏好和事实存储为长期记忆。在每个对话开始时，检索相关记忆并包含在系统提示词中。"用户偏好 Python"花费 5 个 token，但节省了跨对话的数百个重复指令 token。

### RAG 作为上下文工程

检索增强生成是形式化的上下文工程。你不是将知识塞进模型权重（训练）或系统提示词（静态上下文），而是在查询时检索相关文档并将其注入到上下文窗口中。整个 RAG 管道——分块、嵌入、检索、重排序——存在的意义就是解决一个问题：把正确的信息放入上下文窗口。

## 交付物

本课程产出 `outputs/prompt-context-optimizer.md` —— 一个可重用的提示词，审计上下文组装策略并推荐优化。输入你的系统提示词、工具数量、平均历史长度和检索策略，它会识别 token 浪费并提出改进建议。

它还产出 `outputs/skill-context-engineering.md` —— 一个决策框架，根据任务类型、上下文窗口大小和延迟预算设计上下文组装管道。

## 练习

1. 向 ContextBudget 类添加一个"token 浪费检测器"。它应标记使用超过 30% 预算的组件，并针对每种组件类型建议压缩策略（摘要历史、修剪工具、重排序文档）。

2. 为检索上下文实现语义去重。如果两个检索到的文档超过 80% 相似（按词重叠或嵌入余弦相似度），只保留分数较高的那个。衡量这能回收多少 token 预算。

3. 构建一个"上下文重放"工具。给定一个对话记录，通过 ContextEngine 重放它，并可视化预算分配如何逐轮变化。绘制每个组件的 token 使用随时间变化图。识别上下文开始被压缩的那一轮。

4. 实现一个基于优先级的工具选择器。不是二元的包含/排除，而是为每个工具分配一个相对于当前查询的相关性分数。按相关性降序包含工具，直到工具预算耗尽。比较包含 5、10、20 和 50 个工具时的任务表现。

5. 构建一个多策略上下文压缩器。实现三种压缩策略（截断、摘要、关键句提取）并在 20 个文档集上对它们进行基准测试。衡量压缩比与信息保留之间的权衡（压缩版本是否仍包含对查询的答案？）。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| 上下文窗口 | "模型能读多少" | 模型在单次前向传递中处理的最大 token 数（输入 + 输出）——GPT-5 为 400K，Claude Opus 4.7 为 200K（1M beta），Gemini 3 Pro 为 2M |
| 上下文工程 | "高级提示词工程" | 决定什么内容进入上下文窗口、以什么顺序、以什么优先级的学科——涵盖检索、压缩、工具选择和记忆管理 |
| 中间迷失 | "模型忘记中间的东西" | 经验发现，LLM 对上下文开头和结尾的关注更好，中间信息的准确率下降 10-20% |
| Token 预算 | "你还剩多少 token" | 跨组件（系统提示词、工具、历史、检索、生成）显式分配上下文窗口容量，并设置每个组件的限制 |
| 动态上下文 | "即时加载内容" | 根据意图分类、相关工具选择和检索结果为每个查询不同地组装上下文窗口 |
| 历史摘要 | "压缩对话" | 用简洁的摘要替换逐字的旧对话轮次，降低 token 成本的同时保留关键信息 |
| 工具修剪 | "只包含相关工具" | 分类查询意图，只包含匹配的工具定义，将工具 token 成本降低 60-80% |
| 长期记忆 | "跨会话记住" | 存储在数据库中并在会话开始检索的事实和偏好——CLAUDE.md、ChatGPT Memory 等系统 |
| 情景记忆 | "记住特定过去事件" | 作为嵌入存储的过往交互，在当前查询与过去对话相似时检索 |
| 生成预算 | "答案的空间" | 为模型输出保留的 token——如果上下文完全填满窗口，模型就没有空间回应 |

## 延伸阅读

- [Liu 等，2023 —— "中间迷失：语言模型如何使用长上下文"](https://arxiv.org/abs/2307.03172) —— 关于位置依赖性注意力的决定性研究，展示模型在长上下文中间的信息上表现不佳
- [Anthropic 的上下文检索博客文章](https://www.anthropic.com/news/contextual-retrieval) —— Anthropic 如何处理上下文感知的块检索，将检索失败减少 49%
- [Simon Willison 的"上下文工程"](https://simonwillison.net/2025/Jun/27/context-engineering/) —— 命名该学科并将其与提示词工程区分开来的博客文章
- [LangChain 关于 RAG 的文档](https://python.langchain.com/docs/tutorials/rag/) —— 作为上下文工程模式的检索增强生成的实际实现
- [Greg Kamradt 的大海捞针测试](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) —— 揭示所有主要模型中位置依赖检索失败的基准
- [Pope 等，"高效扩展 Transformer 推理"（2022）](https://arxiv.org/abs/2211.05102) —— 为什么上下文长度驱动内存和延迟，以及 KV 缓存、MQA 和 GQA 如何改变预算计算。
- [Agrawal 等，"SARATHI：通过分块预填充搭载解码实现高效 LLM 推理"（2023）](https://arxiv.org/abs/2308.16369) —— 使长提示词在 TTFT 中昂贵但在 TPOT 中便宜的推理两个阶段；上下文打包权衡背后的事实依据。
- [Ainslie 等，"GQA：从多头检查点训练广义多查询 Transformer 模型"（EMNLP 2023）](https://arxiv.org/abs/2305.13245) —— 将生产解码器中的 KV 内存减少 8 倍而不损失质量的分组查询注意力论文。
