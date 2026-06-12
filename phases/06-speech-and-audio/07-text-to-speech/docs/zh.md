# 文本转语音 (TTS) — 从 Tacotron 到 F5 和 Kokoro

> ASR 将语音反转为文本；TTS 将文本反转为语音。2026 年的技术栈包含三部分：文本 → token、token → 梅尔、梅尔 → 波形。每个部分都有一个适合在笔记本上运行的默认模型。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 6 · 02（频谱图与梅尔），阶段 5 · 09（Seq2Seq），阶段 7 · 05（完整 Transformer）
**时间：** ~75 分钟

## 问题

你有一个字符串："请提醒我下午 6 点给植物浇水。"你需要一段 3 秒的音频片段，听起来自然、有正确的韵律（停顿、重音）、用正确的元音发音"plants"，并在实时语音助手的 CPU 上在 300 ms 内运行。你还需要更换声音、处理混合语言输入（"remind me at 6 pm, 大丈夫？"），并且不想在人名上出丑。

现代 TTS 流水线如下：

1. **文本前端。** 归一化文本（日期、数字、电子邮件），转换为音素或子词 token，预测韵律特征。
2. **声学模型。** 文本 → 梅尔频谱图。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **声码器。** 梅尔 → 波形。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022）、2024+ 的神经编解码声码器。

到 2026 年，声学 + 声码器的界限因端到端扩散和流匹配模型而变得模糊。但三部分的心智模型对于调试仍然有效。

## 概念

![Tacotron、FastSpeech、VITS、F5/Kokoro 并排比较](../assets/tts.svg)

**Tacotron 2（2017）。** Seq2seq：字符嵌入 → BiLSTM 编码器 → 位置敏感注意力 → 自回归 LSTM 解码器发射梅尔帧。慢（自回归），长文本不稳定。仍作为基线被引用。

**FastSpeech 2（2020）。** 非自回归。持续时间预测器输出每个音素得到多少梅尔帧。单次通过，比 Tacotron 快 10 倍。损失了一些自然度（单调对齐）但随处可部署。

**VITS（2021）。** 联合训练编码器 + 基于流的持续时间 + HiFi-GAN 声码器，端到端，使用变分推理。高质量、单一模型。2022–2024 年主导的开源 TTS。变体：YourTTS（多说话人零样本）、XTTS v2（2024，Coqui）。

**F5-TTS（2024）。** 基于流匹配的扩散 Transformer。自然的韵律、零样本语音克隆只需 5 秒参考音频。2026 年开源 TTS 排行榜第一名。3.35 亿参数。

**Kokoro（2024）。** 小型（8200 万参数）、可在 CPU 上运行、同类最佳的实时英语 TTS。封闭词汇表仅英语、Apache-2.0 许可证。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业状态最先进。ElevenLabs v2.5 的情感标签（"[耳语]"、"[笑]"）和角色声音在 2026 年主导有声书制作。

### 声码器演进

| 时代 | 声码器 | 延迟 | 质量 |
|------|--------|------|------|
| 2016 | WaveNet | 仅离线 | 发布时 SOTA |
| 2018 | WaveRNN | ~实时 | 好 |
| 2020 | HiFi-GAN | 100× 实时 | 接近人类 |
| 2022 | BigVGAN | 50× 实时 | 跨说话人/语言泛化 |
| 2024 | SNAC、DAC（神经编解码器） | 与 AR 模型集成 | 离散 token，比特高效 |

到 2026 年，大多数"TTS"模型是从文本到波形的端到端模型；梅尔频谱图成为内部表示。

### 评估

- **MOS（平均意见分）。** 1–5 分制，众包。仍然是黄金标准；慢得令人痛苦。
- **CMOS（比较 MOS）。** A-vs-B 偏好。每次标注的置信区间更窄。
- **UTMOS、DNSMOS。** 无参考的神经 MOS 预测器。用于排行榜。
- **CER（字符错误率）通过 ASR。** 将 TTS 输出通过 Whisper 运行，根据输入文本计算 CER。作为可懂度的代理指标。
- **SECS（说话人嵌入余弦相似度）。** 语音克隆质量。

2026 年 LibriTTS test-clean 上的数据：

| 模型 | UTMOS | CER（通过 Whisper） | 参数量 |
|------|-------|--------------------|--------|
| 真实音频 | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 动手构建

### 第 1 步：音素化输入

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用桥梁。避免将原始文本输入到任何低于 VITS 级别质量的模型中。

### 第 2 步：运行 Kokoro（2026 年 CPU 默认方案）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = 美式英语
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单个文件，8200 万参数。

### 第 3 步：使用 F5-TTS 进行语音克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入一个 5 秒的参考片段及其转录文本；F5 克隆韵律和音色。

### 第 4 步：从头实现 HiFi-GAN 声码器

太大无法放入教程脚本，但结构如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 个上采样块，总计 256 倍从梅尔速率到音频速率
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> 波形
```

训练：对抗性（短窗口上的判别器）+ 梅尔频谱图重建损失 + 特征匹配损失。已商品化 — 使用来自 `hifi-gan` 仓库或 nvidia-NeMo 的预训练检查点。

### 第 5 步：完整流水线（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 使用它

2026 年的技术栈：

| 情况 | 选择 |
|------|------|
| 实时英语语音助手 | Kokoro（CPU）或 XTTS v2（GPU） |
| 从 5 秒参考进行语音克隆 | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声书旁白 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 在 5–20 小时目标语言数据上训练 VITS |
| 表情 / 情感标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

截至 2026 年的开源领头羊：**F5-TTS 在质量上，Kokoro 在效率上**。除非你是研究者，否则不要接触 Tacotron。

## 陷阱

- **没有文本归一化器。** "Dr. Smith"读作"Doctor"还是"Drive"？"2026"读作"twenty twenty six"还是"two zero two six"？在音素化*之前*进行归一化。
- **词表外专有名词。** "Ghumare" → "ghyu-mair"？为未知 token 提供一个备用的字形到音素模型。
- **削波。** 声码器输出很少削波，但推理时梅尔缩放不匹配可能导致超出 ±1.0。始终 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24 kHz；你的下游流水线期望 16 kHz → 重采样或接受混叠。

## 输出

保存为 `outputs/skill-tts-designer.md`。为给定的声音、延迟和语言目标设计 TTS 流水线。

## 练习

1. **简单。** 运行 `code/main.py`。从玩具词汇表构建一个音素字典，估计每个音素的持续时间，并打印一个假的"梅尔"计划。
2. **中等。** 安装 Kokoro，用 `af_bella` 和 `am_adam` 声音合成同一句话。比较音频时长和主观质量。
3. **困难。** 录制一段 5 秒的参考片段。使用 F5-TTS 克隆它。报告参考音频和克隆输出之间的 SECS。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 音素 | 声音单元 | 抽象声音类别；英语有 39 个（ARPABet）。 |
| 持续时间预测器 | 每个音素持续多久 | 非 AR 模型输出；每个音素的整数帧数。 |
| 声码器 | 梅尔 → 波形 | 将梅尔谱映射到原始样本的神经网络。 |
| HiFi-GAN | 标准声码器 | 基于 GAN；2020–2024 年主导。 |
| MOS | 主观质量 | 来自人类评分者的 1-5 分平均意见分。 |
| SECS | 语音克隆指标 | 目标与输出说话人嵌入之间的余弦相似度。 |
| F5-TTS | 2024 年开源 SOTA | 流匹配扩散；零样本克隆。 |
| Kokoro | CPU 英语领导者 | 8200 万参数模型，Apache 2.0。 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq 基线。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 端到端基于流。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前开源 SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026 年仍在使用的声码器。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 年 CPU 友好的英语 TTS。
