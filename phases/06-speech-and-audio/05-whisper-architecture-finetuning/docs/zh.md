# Whisper — 架构与微调

> Whisper 是一个 30 秒窗口的 Transformer 编码器-解码器，在 68 万小时的多语言弱监督音频-文本对上训练。一个架构、多个任务、对 99 种语言鲁棒。2026 年的参考 ASR。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 6 · 04（ASR），阶段 5 · 10（注意力），阶段 7 · 05（完整 Transformer）
**时间：** ~75 分钟

## 问题

Whisper 由 OpenAI 于 2022 年 9 月发布，是第一个作为商品提供的 ASR 模型：粘贴音频、获取文本、支持 99 种语言、对噪音鲁棒、可在笔记本上运行。到 2024 年 OpenAI 已发布 Large-v3 和 Turbo 变体；到 2026 年，Whisper 是从播客转录到语音助手再到 YouTube 字幕的默认基线。

但 Whisper 不是一个可以一直当作黑盒处理的流水线。领域偏移会将其击倒 — 技术术语、说话人口音、专有名词、短视频、静音。你需要了解：

1. 它内部究竟是什么。
2. 如何正确地为其提供分块、流式或长音频。
3. 何时以及如何微调。

## 概念

![Whisper 编码器-解码器、任务、分块推理、微调](../assets/whisper.svg)

**架构。** 标准的 Transformer 编码器-解码器。

- 输入：30 秒的对数梅尔频谱图，80 个梅尔，10 ms 步长 → 3000 帧。更短的片段补零，更长的片段分块。
- 编码器：卷积下采样（步长 2）+ `N` 个 Transformer 块。Large-v3：32 层，1280 维，20 个注意力头。
- 解码器：`N` 个带因果自注意力的 Transformer 块 + 对编码器输出的交叉注意力。大小与编码器相同。
- 输出：51,865 token 词汇表上的 BPE token。

Large-v3 有 15.5 亿参数。Turbo 使用 4 层解码器（原先为 32 层），延迟降低 8 倍且 WER 损失小于 1%。

**提示格式。** Whisper 是一个多任务模型，通过解码器提示中的特殊 token 控制：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 语言标签；强制转录 vs 翻译行为。
- `<|transcribe|>` 或 `<|translate|>` — 从任何语言输入翻译为英语输出，或逐字转录。
- `<|notimestamps|>` — 跳过词级时间戳（更快）。

这个提示让一个模型可以做多个任务。将 `<|en|>` 改为 `<|fr|>` 就能转录法语。

**30 秒窗口。** 一切都固定在 30 秒。更长的片段需要分块；更短的片段被填充。窗口不是原生流式的 — 这就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**对数梅尔归一化。** `(log_mel - mean) / std`，其中统计量来自 Whisper 自己的训练语料。你*必须*使用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年的变体

| 变体 | 参数量 | 延迟（A100） | WER（LibriSpeech-clean） |
|------|--------|-------------|------------------------|
| Tiny | 39M | 1× 实时 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | 流式 | 2.0% |

### 微调

2026 年的标准流程：

1. 收集 10–100 小时目标领域的音频及对齐的转录文本。
2. 使用带 `generate_with_loss` 回调的 `transformers.Seq2SeqTrainer` 运行。
3. 参数高效方式：对注意力层的 `q_proj`、`k_proj`、`v_proj` 使用 LoRA，将 GPU 内存减少 4 倍且 WER 成本低于 0.3。
4. 如果数据少于 10 小时，冻结编码器。只调整解码器。
5. 使用 Whisper 自己的 tokenizer 和提示格式；永远不要更换 tokenizer。

社区结果：在 20 小时医疗口述数据上微调 Medium，WER 从 12% 降至 4.5%（针对医学术语）。在 4 小时冰岛语数据上微调 Turbo，WER 从 18% 降至 6%。

## 动手构建

### 第 1 步：开箱即用运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 防止重复失控
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应该始终覆盖的关键默认值：`temperature=0.0`（采样默认从 0.0 → 0.2 → 0.4 … 的降级链），`condition_on_previous_text=False`（防止级联幻觉问题），以及 `no_speech_threshold=0.6`（静音检测）。

### 第 2 步：分块长音频

```python
# whisperx 是 2026 年带词级时间戳的长音频参考实现
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 增加了 (1) Silero VAD 门控、(2) 通过 wav2vec 2.0 的词级对齐、(3) 通过 `pyannote.audio` 的说话人日志。2026 年生产转录的主力工具。

### 第 3 步：使用 LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M 可训练 / 809M 总计
```

然后使用标准 Trainer 循环。每 1000 步保存一次检查点。使用 WER 在保留集上评估。

### 第 4 步：检查每层学到的内容

```python
# 在解码过程中获取交叉注意力权重，查看解码器关注什么
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

使用热力图可视化 — 你会看到解码器步骤扫描编码器帧时呈现对角线对齐。这条对角线就是 Whisper 的"词时间戳"概念。

## 使用它

2026 年的技术栈：

| 情况 | 选择 |
|------|------|
| 通用英语、离线 | 通过 `whisperx` 使用 Large-v3-turbo |
| 移动端 / 边缘设备 | 量化的 Whisper-Tiny (int8) 或 Moonshine |
| 多语言长音频 | 通过 `whisperx` 使用 Large-v3 + 说话人日志 |
| 低资源语言 | 使用 LoRA 微调 Medium 或 Turbo |
| 流式（2 秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时 — 比原版快 4 倍，输出相同。

## 2026 年仍在犯的陷阱

- **在静音上产生幻觉文本。** Whisper 在字幕数据上训练，包含"Thanks for watching!"、"Subscribe!"、歌词。始终在调用前使用 VAD 门控。
- **`condition_on_previous_text` 级联。** 一个幻觉污染后面的窗口。除非需要跨片段的流畅性，否则设为 `False`。
- **短视频填充。** 一段 2 秒的音频填充到 30 秒可能会在尾部的静音中产生幻觉。使用 `pad=False` 或 VAD 门控。
- **错误的梅尔统计量。** 使用 librosa 的梅尔而非 Whisper 的会产生几乎随机的输出。使用 `whisper.audio.log_mel_spectrogram`。

## 输出

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计 Whisper 微调或推理流水线。

## 练习

1. **简单。** 运行 `code/main.py`。它对 Whisper 风格的提示进行 tokenize、计算解码后的形状预算，并打印一段 10 分钟音频的分块计划。
2. **中等。** 安装 `faster-whisper`，转录一段 10 分钟的播客，与人工转录比较 WER。尝试 `language="auto"` vs 强制 `language="en"`。
3. **困难。** 使用 HF `datasets`，选择一个 Whisper 难以处理的语言（例如乌尔都语），使用 LoRA 在 2 小时数据上微调 Medium 2 个 epoch，并报告 WER 差值。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 30 秒窗口 | Whisper 的限制 | 硬性输入上限；更长的音频需要分块。 |
| SOT | 转录起始 | `<\|startoftranscript\|>` 启动解码器提示。 |
| 时间戳 token | 时间对齐 | 每 0.02 秒的偏移是 5.1 万词汇表中的一个特殊 token。 |
| Turbo | 快速变体 | 4 层解码器，速度提升 8 倍，WER 退化 <1%。 |
| WhisperX | 长音频封装 | VAD + Whisper + wav2vec 对齐 + 说话人日志。 |
| LoRA 微调 | 高效调整 | 向注意力层添加低秩适配器；训练约 0.3% 的参数。 |
| 幻觉 | 静默失败 | Whisper 从噪音/静音中产生流畅的英语。 |

## 延伸阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 原始架构和训练方法。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4 层解码器，8 倍加速。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 长音频、词级对齐、说话人日志。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — 基于 CTranslate2，快 4 倍。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — 权威的 LoRA / 全参数微调指南。
