# 语音活动检测与轮次判断 — Silero、Cobra 和 Flush 技巧

> 每个语音代理的生死取决于两个判断：用户现在在说话吗，用户说完了吗？VAD 回答第一个。轮次检测（VAD + 静音挂起 + 语义端点模型）回答第二个。任何一个出了错，你的助手要么打断用户，要么永远说个不停。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 6 · 11（实时音频），阶段 6 · 12（语音助手）
**时间：** ~45 分钟

## 问题

语音代理在每 20 ms 块上做出三个不同的判断：

1. **这一帧是语音吗？** — VAD。二值，逐帧。
2. **用户开始新的发言了吗？** — 起始检测。
3. **用户说完了吗？** — 端点检测（轮次结束）。

朴素的答案（能量阈值）在有任何噪音时就会失败 — 交通、键盘、人群嘈杂声。2026 年的答案：Silero VAD（开源、深度学习）+ 轮次检测模型（语义端点检测）+ VAD 校准的静音挂起。

## 概念

![VAD 级联：能量 → Silero → 轮次检测器 → flush 技巧](../assets/vad-turn-taking.svg)

### 三层 VAD 级联

**第 1 层：能量门控。** 最便宜。RMS 阈值 -40 dBFS。过滤明显的静音，但对高于阈值的任何噪音都会触发。

**第 2 层：Silero VAD**（2020-2026，MIT）。100 万个参数。在 6000+ 种语言上训练。在单个 CPU 线程上每 30 ms 块运行约 1 ms。87.7% TPR @ 5% FPR。开源默认方案。

**第 3 层：语义轮次检测器。** LiveKit 的轮次检测模型（2024-2026）或你自己的小型分类器。区分"句中停顿"和"说完了"。使用语言上下文（语调和最近词汇），而不仅仅是静音。

### 关键参数及其默认值

- **阈值。** Silero 输出概率；在 > 0.5（默认）或 > 0.3（敏感）时分类为语音。较低的阈值 = 更少的首词截断，更多的误报。
- **最小语音时长。** 拒绝短于 250 ms 的语音 — 通常是咳嗽或椅子噪音。
- **静音挂起（端点检测）。** VAD 回到 0 后，等待 500-800 ms 再宣布轮次结束。太短 → 打断用户。太长 → 感觉迟钝。
- **预卷缓冲区。** 在 VAD 触发前保留 300-500 ms 的音频。防止"嘿"被截断。

### Flush 技巧（Kyutai 2025）

流式 STT 模型有一个前视延迟（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 秒）。通常你需要在此之后等待转录。Flush 技巧：当 VAD 触发语音结束时，**向 STT 发送一个 flush 信号** 强制立即输出。STT 以约 4 倍实时速度处理，因此 500 ms 缓冲在约 125 ms 内完成。

端到端：125 ms VAD + flush STT = 对话延迟。

### 2026 年 VAD 比较

| VAD | TPR @ 5% FPR | 延迟 | 许可证 |
|-----|--------------|------|--------|
| WebRTC VAD（Google，2013） | 50.0% | 30 ms | BSD |
| Silero VAD（2020-2026） | 87.7% | ~1 ms | MIT |
| Cobra VAD（Picovoice） | 98.9% | ~1 ms | 商业 |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero 是正确的默认方案。Cobra 是合规/准确性的升级。仅能量 VAD 在 2026 年的生产中没有位置。

## 动手构建

### 第 1 步：能量门控

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 第 2 步：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 第 3 步：轮次结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 第 4 步：flush 技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能工作。Whisper streaming 不支持 — 它基于块且始终等待块。

## 使用它

| 情况 | VAD 选择 |
|------|---------|
| 开源、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究 / 说话人日志 | pyannote segmentation |
| 零依赖回退 | WebRTC VAD（遗留） |
| 需要轮次结束质量 | Silero + LiveKit 轮次检测器叠加 |

经验法则：除非真的没有其他选择，否则绝不交付仅能量的 VAD。

## 陷阱

- **固定阈值。** 在安静环境下有效，在嘈杂环境下失效。要么在设备上校准，要么切换到 Silero。
- **静音挂起太短。** 代理在句中打断。500-800 ms 是对话语音的最佳点。
- **挂起时间太长。** 感觉迟钝。与目标用户进行 A/B 测试。
- **无预卷缓冲区。** 用户音频的前 200-300 ms 丢失。始终保持滚动预卷。
- **忽略语义端点检测。** "嗯，让我想想..."包含长停顿。用户讨厌在思考时被打断。使用 LiveKit 的轮次检测器或类似方案。

## 输出

保存为 `outputs/skill-vad-tuner.md`。为特定工作负载选择 VAD 模型、阈值、挂起时间、预卷和轮次检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。它模拟一个语音 + 静音 + 语音 + 咳嗽序列，并测试三个 VAD 层级。
2. **中等。** 安装 `silero-vad`，处理一段 5 分钟的录音，调整阈值以最小化首词截断和误触发。报告精确率/召回率。
3. **困难。** 构建一个小型轮次检测器：Silero VAD + 在最后 10 个词嵌入上的 3 层 MLP（使用 sentence-transformers）。在手工标注的轮次结束数据集上训练。比纯 Silero 提高 10% F1。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| VAD | 语音检测器 | 二值逐帧：这是语音吗？ |
| 轮次检测 | 端点检测 | VAD + 静音挂起 + 语义端点。 |
| 静音挂起 | 说话后等待 | 宣布轮次结束前的等待时间；500-800 ms。 |
| 预卷 | 语音前缓冲区 | 保留 VAD 触发前 300-500 ms 的音频。 |
| Flush 技巧 | Kyutai 技巧 | VAD → flush-STT → 125 ms 而非 500 ms 延迟。 |
| 语义端点 | "他们打算停了吗？" | 看词而不只是静音的 ML 分类器。 |
| TPR @ FPR 5% | ROC 点 | 标准 VAD 基准；Silero 87.7%，WebRTC 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考开源 VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业准确率领导者。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 亚 200 ms 工程技巧。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 生产中的语义端点检测。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 遗留基线。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 说话人日志级分割。
