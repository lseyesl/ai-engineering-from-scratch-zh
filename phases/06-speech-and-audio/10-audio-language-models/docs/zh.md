# 音频语言模型 — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的音频语言模型可以对语音 + 环境声音 + 音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上与 GPT-4o Audio 持平。Audio Flamingo Next 在 LongAudioBench 上超越 Gemini 2.5 Pro。开放与闭源之间的差距基本消失 — 除了多音频任务，在那上面所有人都接近随机水平。

**类型：** 学习
**语言：** Python
**前置条件：** 阶段 6 · 04（ASR），阶段 12 · 03（视觉语言模型），阶段 7 · 10（音频 Transformer）
**时间：** ~45 分钟

## 问题

你有 5 秒的音频：狗叫、有人喊"停！"、然后安静。有用的问题涵盖多个维度：

- **转录。** "说了什么？" — ASR 领域。
- **语义推理。** "这个人有危险吗？" — 需要联合理解狗叫 + 喊叫 + 安静。
- **音乐推理。** "什么乐器演奏了旋律？"
- **长音频检索。** "在这 90 分钟的讲座中，讲师在哪里解释了梯度下降？"

一个能用一个提示回答所有这些问题的单一模型就是**音频语言模型**（LALM / ALM）。与纯 ASR 不同：LALM 产生自由形式的自然语言答案，而不仅仅是转录。

## 概念

![音频语言模型：音频编码器 + 投影器 + LLM 解码器](../assets/alm-architecture.svg)

### 三组件模板

每个 2026 年的 LALM 都有相同的骨架：

1. **音频编码器。** Whisper 编码器 · BEATs · CLAP · WavLM · 或每个模型的自定义编码器。
2. **投影器。** 线性层或 MLP，将音频编码器特征桥接到 LLM 的 token 嵌入空间。
3. **LLM。** 基于 Llama / Qwen / Gemma 的解码器。接收交错的文本 + 音频 token；生成文本。

训练：

- **阶段 1。** 冻结编码器 + LLM；仅在 ASR / 字幕数据上训练投影器。
- **阶段 2。** 在指令跟随音频任务（QA、推理、音乐理解）上进行全参数 / LoRA 微调。
- **阶段 3（可选）。** 语音输入 / 语音输出增加一个语音解码器。Qwen2.5-Omni 和 AF3-Chat 做到了这一点。

### 2026 年模型地图

| 模型 | 骨干网络 | 音频编码器 | 输出模态 | 访问方式 |
|------|---------|----------|---------|---------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（闭源） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（闭源） | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准测试现实检查（2026 年）

**MMAU-Pro。** 1800 个 QA 对，涵盖语音/声音/音乐/混合。包含多音频子集。

| 模型 | 整体 | 语音 | 声音 | 音乐 | 多音频 |
|------|------|------|------|------|--------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench SOTA | — | — | — | — |

**多音频列对所有人都是致命的。** 4 选 1 多项选择题的随机概率 = 25%；大多数模型在这个水平附近。LALM 仍然难以比较两个片段。

### 2026 年 LALM 的实用场景

- **呼叫中心录音合规审计。** "座席是否提到了必须的披露信息？"
- **无障碍。** 向聋人用户描述声音事件（不仅仅是转录）。
- **内容审核。** 检测暴力语言 + 威胁语气 + 背景上下文。
- **播客 / 会议章节划分。** 语义摘要，不仅仅是说话人轮换。
- **音乐目录分析。** "查找所有带有 B 段转调的音轨。"

### 它们还不太有用的地方

- 精细音乐理论（低于和弦级别）。
- 长对话的说话人归因推理（超过 10 分钟后性能下降）。
- 多音频比较（22-26% 仅略高于随机）。
- 实时流式推理（大多数是离线批量推理）。

## 动手构建

### 第 1 步：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "你听到了什么声音，发生了什么？"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 第 2 步：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就是这样。投影器通常是 1-3 个线性层。在 ASR 对（音频 → 转录）上训练它是阶段 1 的预文本任务。

### 第 3 步：基准测试 MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"准确率: {correct / len(mmau['test']):.3f}")
```

按类别（语音/声音/音乐/多音频）分别报告。聚合数字会隐藏模型失败的地方。

## 使用它

| 任务 | 2026 年选择 |
|------|-----------|
| 自由形式音频 QA（开源） | Qwen2.5-Omni-7B |
| 长音频最佳开源方案 | Audio Flamingo Next |
| 最佳闭源方案 | Gemini 2.5 Pro |
| 语音输入/语音输出代理 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（音乐专用 AF-CLAP） |
| 呼叫中心审计 | 通过 API 的 Gemini 2.5 Pro，配合策略文档的 RAG |

## 陷阱

- **过度信任多音频。** 如果你的任务需要"哪个片段有 X"的判定，随机水平的表现是真实存在的。
- **长音频退化。** 超过 10 分钟后，大多数模型的说话人归因会失效。先做说话人日志（课程 6），然后做摘要。
- **静音上的幻觉。** 继承自使用 Whisper 编码器的 LALM 的 Whisper 风格问题。使用 VAD 门控。
- **基准测试选择偏差。** 供应商博客文章突出最佳案例类别。你自己运行 MMAU-Pro 多音频子集。

## 输出

保存为 `outputs/skill-alm-picker.md`。针对给定的音频理解任务选择 LALM + 基准测试子集 + 输出模态（文本 vs 语音）。

## 练习

1. **简单。** 运行 `code/main.py` 以查看玩具投影器模式 + 假 LALM 路由（音频嵌入、文本 token）→ 输出 token。
2. **中等。** 在 100 个 MMAU-Pro 语音项目上评分 Qwen2.5-Omni-7B。与论文报告的数字进行比较。
3. **困难。** 构建一个最小的音频字幕基线：BEATs 编码器 + 2 层投影器 + 冻结的 Llama-3.2-1B。仅在 AudioCaps 上微调投影器。在 Clotho-AQA 上与 SALMONN 比较。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| LALM | 音频 ChatGPT | 音频编码器 + 投影器 + LLM 解码器。 |
| 投影器 | 适配器 | 将音频特征映射到 LLM 嵌入空间的小型 MLP。 |
| MMAU | 基准测试 | 10k 音频-QA 对，涵盖语音、声音、音乐。 |
| MMAU-Pro | 更难的 MMAU | 1800 个多音频 / 推理密集型问题。 |
| LongAudioBench | 长格式评估 | 带语义查询的几分钟长片段。 |
| 语音输入/语音输出 | 语音原生 | 模型输入语音并输出语音，无需文本中转。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音输入语音输出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开放长音频领导者。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱。
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — 实时 2026 年排名。
