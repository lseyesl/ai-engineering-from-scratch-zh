# 语言模型评估框架

> 一个在你无法定义的任务上表现良好的模型，是一个偶然表现良好的模型。评估框架就是任务定义、指标、运行器和排行榜，全部封装在一个简短、可替换的形状中。

**类型:** Build
**语言:** Python
**前置知识:** 第19阶段第42至45课
**时间:** ~90分钟

## 学习目标

- 将任务定义为 JSONL 文件，每个示例包含 `prompt`、`targets`、`metric` 和可选的 `extras`。
- 实现五个指标：精确匹配、rouge-l F1、可执行检查、多项选择和子串包含。
- 构建一个运行器，按任务批量处理示例并分派到可替换的模型适配器。
- 输出一个排行榜 JSON，包含每个任务的分数、延迟和可复现的总体平均值。

## 问题

每周都有新的语言模型发布。营销声称它表现良好。诚实的问题是：在什么上表现良好？诚实的答案是你自己编写的排行榜，因为供应商的排行榜是他们调优过的那个。

没有评估框架在你的仓库中，你通过感觉比较两个模型。有了评估框架，你通过固定任务集和固定指标上的分数来比较它们，输出 JSON 你可以做差异比较。评估框架是昨天运行和今天运行之间的契约。没有它，回归就会上线。

陷阱是将评估框架过度拟合到单个模型。修复方法反过来也是同一个陷阱：评估框架足够小，十五分钟就能读完；任务足够小，可以随仓库一起发布；指标从头编写，以便同事可以审计；适配器是唯一存放模型特定代码的地方。更换适配器，排行榜移动；更换任务，排行榜移动。其他任何东西都不应该移动。

## 概念

```mermaid
flowchart TD
  tasks[task JSONLs: prompt, targets, metric, extras] --> loader[load_all_tasks]
  loader --> runner[run_leaderboard]
  runner --> adapter[ModelAdapter.generate batch]
  adapter --> metrics[METRIC_FNS dispatch by name]
  metrics --> scores[per example score]
  scores --> board[Leaderboard: per task + overall]
  board --> out[leaderboard.json]
```

### 任务规范

每个示例是一行 JSONL：

```json
{"id": "arith-00", "prompt": "compute: 2 + 2", "targets": ["4"], "metric": "exact_match"}
```

对于需要评分辅助函数的指标，`extras` 携带附加负载：

```json
{
  "id": "code-00",
  "prompt": "python: write a function f that doubles its input",
  "targets": ["ok"],
  "metric": "code_exec",
  "extras": {"io_pairs": [[1, 2], [3, 6]]}
}
```

一个任务是一个 `outputs/tasks/` 下的 `.jsonl` 文件。文件名就是任务名。文件中的所有示例共享一个指标。

### 五个固定任务

| 任务 | 指标 | 测试内容 |
|------|------|----------|
| arithmetic | exact_match | 确定性答案上的词元级正确性 |
| summary | rouge_l | 针对一行参考摘要的最长公共子序列 F1 |
| code-exec | code_exec | 可执行测试：预测的函数必须满足一组输入输出对 |
| multiple-choice | multiple_choice | 预测的第一个字母必须匹配允许的字母 |
| generation | substring_contains | 自由形式文本必须包含至少一个目标子串 |

### 指标契约

每个指标是一个从 `(prediction, targets, extras) -> float in [0.0, 1.0]` 的函数。评估框架平均每个示例的分数得到任务分数，然后平均任务分数得到总体分数。指标函数很小：

- `exact_match`：小写、折叠空白、相等性。
- `substring_contains`：相同的规范化、子串测试。
- `multiple_choice`：首字母大写。
- `rouge_l`：LCS 长度除以预测和参考的长度，精确率和召回率的 F1。
- `code_exec`：在受限命名空间中执行预测，对每个输入输出对调用 `f(x)`，计数匹配。

code_exec 指标在剥离的内置命名空间中运行预测。本课的测试断言 `import os` 会失败，因为 `os` 不在命名空间中；你无法从代码预测中访问文件系统。

### 模型适配器

```python
class ModelAdapter(Protocol):
    def generate(self, prompts: Sequence[str]) -> List[str]: ...
    @property
    def name(self) -> str: ...
```

适配器是接缝。本课提供 `ToyAdapter`，一个确定性模式匹配器，为五个固定任务中的每个提示返回正确答案。真正的适配器调用模型并返回其输出。评估框架不关心是哪一个。

### 运行器

`run_task` 每次批量处理 `batch_size` 个提示，并分派到指标函数。`run_leaderboard` 遍历每个任务并平均。`write_leaderboard` 输出带有 schema 字符串的 JSON，以便未来的格式更改不会静默破坏仪表盘。

```mermaid
flowchart LR
  examples[N examples] --> batches[B-sized batches]
  batches --> adapter[adapter.generate]
  adapter --> per[per example score 0..1]
  per --> avg[task score]
  avg --> over[overall = mean of task scores]
```

```figure
eval-harness-matrix
```

## 构建

`code/main.py` 是可运行的工件。

### 第1步：生成固定任务

`seed_fixture_tasks(target_dir)` 写入五个 `.jsonl` 文件。`main.py` 的首次运行在目录为空时生成它们。

### 第2步：加载任务

`load_all_tasks(task_dir)` 读取每个 `.jsonl` 并返回从任务名到 `Example` 记录列表的字典。以 `#` 开头的注释行和空行被跳过，以便贡献者可以注释文件。

### 第3步：实现指标

每个指标是一个带有单元测试的小函数。本课的测试套件包含13个用例，涵盖规范化、部分重叠、代码执行和不安全代码拒绝。

### 第4步：编写运行器

`run_task` 迭代批次并生成一个包含分数、正确计数、总计数和延迟的 `TaskResult`。`run_leaderboard` 遍历所有任务并生成一个包含总体平均值的 `Leaderboard`。

### 第5步：输出 JSON

`write_leaderboard` 序列化排行榜。`--include-per-example` 标志转储每个示例的记录，以便在分数变化时你可以将预测与上次运行进行差异比较。

运行：

```bash
python3 code/main.py
```

脚本在首次运行时生成固定任务，使用玩具适配器（它能正确回答每个固定任务）对它们评分，并写入 `outputs/leaderboard.json`。使用玩具适配器时总体分数为 1.0；`test_main.py` 中的存根适配器测试显示，当适配器无法回答时，相同的评估框架产生 0.0。

## 使用

要接入真实模型，编写一个适配器。形状如下：

```python
class HttpAdapter:
    name = "vendor.v1"

    def __init__(self, endpoint, api_key):
        self.endpoint = endpoint
        self.api_key = api_key

    def generate(self, prompts):
        out = []
        for prompt in prompts:
            response = http_post(self.endpoint, prompt, self.api_key)
            out.append(response["text"])
        return out
```

在 `main()` 顶部将 `ToyAdapter` 替换为 `HttpAdapter`。评估框架、任务、指标和排行榜保持不变。

在真实项目中交付评估框架时要强制执行的三种模式：

- **固定任务文件。** leaderboard.json 要么携带哈希固定的任务内容，要么与 JSONL 文件一起提供；否则当任务文件变化时分数会移动，你无法分辨。
- **对预测做差异比较，而不仅仅是分数。** `--include-per-example` 标志让你看到分数下降那天模型说了什么。
- **限制批次大小。** 真实适配器有速率限制。较小的批次大小使评估框架跨供应商保持兼容。

## 交付

`outputs/skill-lm-eval-harness.md` 包含配方：JSONL 任务规范、五个指标、可替换适配器、批量运行器、带有 schema 字符串的排行榜 JSON。`outputs/tasks/` 中的任务文件是固定任务；将它们复制到真实项目中作为起点。

## 练习

1. 添加第六个任务，使用你从头编写的自定义指标（BLEU 式重叠、BLEURT 式参考评分，任何有清晰契约的指标）。
2. 扩展 `code_exec` 以捕获 stdout 并接受预期的 stdout 列表作为目标。
3. 添加排行榜差异比较命令：给定两个 `leaderboard.json` 文件，打印哪些任务发生了变化以及变化幅度。
4. 限制每个示例的延迟。将适配器调用包裹在超时中；在排行榜中显示单独的 `timeouts` 列。
5. 在排行榜中使用 sha256 固定任务内容，以便未来的读者可以验证他们评分的是相同的任务。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 任务规范 | "评估格式" | JSONL 文件，每个示例包含 prompt、targets、metric 和可选的 extras |
| 指标 | "你如何评分" | 从 (prediction, targets, extras) 到 [0, 1] 范围内浮点数的函数 |
| 适配器 | "模型客户端" | 具有 generate(prompts) -> list[str] 方法的对象；唯一的模型特定代码 |
| 排行榜 | "记分板" | 包含每个任务分数、总计数、延迟和总体平均值的 JSON |
| 代码执行指标 | "运行并检查" | 在受限命名空间中执行预测，与输入输出对进行比较 |

## 延伸阅读

- 原始的 lm-evaluation-harness，作为生产参考，更大但形状相同。
- HuggingFace 的 lighteval，作为同一契约的另一种实现。
- 第19阶段第46课涵盖了评估框架评分的训练栈中使用的梯度累积模式。
- 第19阶段第47课涵盖了你要评分的检查点格式；在排行榜中固定检查点哈希。
- 第19阶段第48课涵盖了产生被测模型的分布式训练栈。