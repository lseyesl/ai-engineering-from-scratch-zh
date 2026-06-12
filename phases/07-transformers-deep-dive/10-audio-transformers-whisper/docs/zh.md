# 音频 Transformer — Whisper 架构

> 音频是频率随时间变化的图像。Whisper 是一个处理梅尔频谱图并用语音回应的 ViT。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 7 · 05（完整 Transformer），阶段 7 · 08（编码器-解码器），阶段 7 · 09（ViT）
**时间：** ~45 分钟

## 问题

在 Whisper（OpenAI，Radford 等人，2022）之前，最先进的自动语音识别（ASR）是指 wav2vec 2.0 和 HuBERT——自监督特征提取器加上微调头部。高质量、昂贵的数据流水线、领域脆弱。多语言语音识别需要每种语言族使用单独的模型。

Whisper 做了三个赌注：

1. **在所有数据上训练。** 从互联网上抓取的 680,000 小时弱标注音频，覆盖 97 种语言。没有干净的学术语料库。没有音素标签。
2. **多任务单一模型。** 一个解码器联合训练转录、翻译、语音活动检测、语言 ID 和时间戳——通过任务 token 实现。
3. **标准编码器-解码器 transformer。** 编码器消费 log-mel 频谱图。解码器自回归生成文本 token。没有声码器、没有 CTC、没有 HMM。

结果：Whisper large-v3 对各种口音、噪声和没有干净标注数据的语言都非常鲁棒。它是 2026 年每个开源语音助手和大多数商业助手的默认语音前端。

## 概念

![Whisper 流水线：音频 → mel → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 步骤 1 — 重采样 + 加窗

16 kHz 的音频。剪切/填充到 30 秒。计算 log-mel 频谱图：80 个 mel 频带，10 ms 步长 → 约 3,000 帧 × 80 个特征。这就是 Whisper 看到的"输入图像"。

### 步骤 2 — 卷积主干

两个 Conv1D 层，核大小为 3、步长为 2，将 3,000 帧减少到 1,500。在不增加大量参数的情况下将序列长度减半。

### 步骤 3 — 编码器

在 1,500 个时间步上的 24 层（large 版本）transformer 编码器。正弦位置编码、自注意力、GELU FFN。产生 1,500 × 1,280 的隐藏状态。

### 步骤 4 — 解码器

24 层 transformer 解码器。它从 BPE 词表中自回归生成 token，该词表是 GPT-2 的超集，并带有一些音频特定的特殊 token。

### 步骤 5 — 任务 token

解码器提示以控制 token 开头，告诉模型要做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或者

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型是在这个约定下训练的。你通过前缀控制任务。这相当于 2026 年的指令微调，但应用于语音。

### 步骤 6 — 输出

带对数概率阈值的束搜索（宽度 5）。当 `<|notimestamps|>` token 不存在时，每 0.02 秒音频预测一次时间戳。

### Whisper 规模

| 模型 | 参数量 | 层数 | d_model | 头数 | VRAM（fp16） |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4 层解码器） |

Large-v3-turbo（2024 年）将解码器从 32 层减少到 4 层。解码速度提升 8 倍，WER 退化 <1 个点。正是这种解码速度的解锁使得 Whisper-turbo 成为 2026 年实时语音代理的默认选择。

### Whisper 不做的事

- 无说话人日记化（谁在说话）。配合 pyannote 使用。
- 原生不支持实时流式传输——30 秒窗口是固定的。现代封装（`faster-whisper`、`WhisperX`）通过 VAD + 重叠实现了流式传输。
- 没有外部分块的情况下不支持超过 30 秒的长形式上下文。在实践中效果很好，因为人类语音很少需要长距离上下文来进行转录。

### 2026 年格局

| 任务 | 模型 | 说明 |
|------|-------|-------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在边缘设备上快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可实现 150 ms 延迟目标 |
| TTS | Piper、XTTS-v2、Kokoro | 编码器-解码器模式，但 Whisper 形状 |
| 音频 + 语言 | AudioLM、SeamlessM4T | 一个 transformer 中的文本 token + 音频 token |

## 动手实现

参见 `code/main.py`。我们不训练 Whisper——我们构建 log-mel 频谱图流水线 + 任务 token 提示格式化器。这些是你实际会在生产中触及的部分。

### 步骤 1：合成音频

生成一个 440 Hz 的 1 秒正弦波，以 16 kHz 采样。16,000 个样本。

### 步骤 2：log-mel 频谱图（简化版）

完整的 mel 频谱图需要 FFT。我们做一个简化的分帧 + 每帧能量版本，显示流水线而不需要 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧 = 25 ms，步长 = 10 ms。与 Whisper 的加窗一致。出于教学目的，每帧能量代表 mel 频带。

### 步骤 3：填充到 30 秒

Whisper 总是处理 30 秒的块。将频谱图填充（或裁剪）到 3,000 帧。

### 步骤 4：构建提示 token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制界面。一个 4 token 的前缀。

## 使用

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快、与 OpenAI 兼容的版本：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 使用一个模型进行多语言 ASR。
- 对嘈杂、多样化的音频进行鲁棒转录。
- 研究 / 原型 ASR——最快的起点。

**何时选择其他方案：**

- 边缘设备上的超低延迟流式传输——Moonshine 在同等质量下击败 Whisper。
- 需要 <200 ms 的实时对话 AI——专用流式 ASR。
- 说话人日记化——Whisper 不做这个；配合 pyannote 使用。

## 产出

参见 `outputs/skill-asr-configurator.md`。该技能为新的语音应用选择 ASR 模型、解码参数和预处理流水线。

## 练习

1. **简单。** 运行 `code/main.py`。确认在 16 kHz、10 ms 步长下，1 秒信号的帧数约为 100 帧。30 秒：约 3,000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整的 log-mel 频谱图。验证 80 个 mel 频带在数值误差范围内与 `librosa.feature.melspectrogram(n_mels=80)` 匹配。
3. **困难。** 实现流式推理：将音频分块为 10 秒窗口（2 秒重叠），在每个块上运行 Whisper，合并转录文本。在 5 分钟播客样本上测量字符错误率与单次传递的对比。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| Mel 频谱图 | "音频图像" | 二维表示：一个轴上是频率带，另一个轴上是时间帧；每个格子的对数缩放能量。 |
| Log-mel | "Whisper 看到的" | 经过对数处理的 mel 频谱图；近似人类对响度的感知。 |
| 帧（Frame） | "一个时间片" | 25 ms 的样本窗口；以 10 ms 步长重叠。 |
| 任务 token（Task token） | "语音的提示前缀" | 解码器提示中的特殊 token，如 `<\|transcribe\|>` / `<\|translate\|>`。 |
| 语音活动检测（VAD） | "找到语音" | 在 ASR 前移除静音的门控；大幅降低成本。 |
| CTC | "连接主义时序分类" | 经典 ASR 损失，用于无对齐训练；Whisper 不使用它。 |
| Whisper-turbo | "小解码器，完整编码器" | large-v3 编码器 + 4 层解码器；解码速度快 8 倍。 |
| Faster-whisper | "生产级封装" | CTranslate2 重新实现；int8 量化；比 OpenAI 参考实现快 4 倍。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 参考代码 + 模型权重。阅读 `whisper/model.py` 查看 Conv1D 主干 + 编码器 + 解码器的完整实现，约 400 行。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 步骤 5-6 中描述的束搜索 + 任务 token 逻辑就在这里；500 行，完全可读。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前身；在某些设置中仍然是 SOTA 特征。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产封装，比参考实现快 4 倍。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年边缘友好型 ASR，Whisper 形状但更小。
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 规范微调配方，包括 mel 频谱图预处理和时间戳处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（编码器、解码器、交叉注意力、生成），与课程架构图对应的完整实现。
