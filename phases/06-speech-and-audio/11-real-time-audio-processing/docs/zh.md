# 实时音频处理

> 批处理流水线处理一个文件。实时流水线处理接下来的 20 毫秒，而下一个 20 毫秒到达之前。每个对话式 AI、广播工作室和电话机器人都取决于这个延迟预算。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 6 · 02（频谱图），阶段 6 · 04（ASR），阶段 6 · 07（TTS）
**时间：** ~75 分钟

## 问题

你希望一个语音助手感觉是活的。人类对话的轮换延迟约为 230 毫秒（无声到响应）。超过 500 毫秒会感觉像机器人；超过 1500 毫秒会感觉损坏。2026 年完整 **听 → 理解 → 响应 → 说** 循环的预算：

| 阶段 | 预算 |
|------|------|
| 麦克风 → 缓冲区 | 20 ms |
| VAD | 10 ms |
| ASR（流式） | 150 ms |
| LLM（第一个 token） | 100 ms |
| TTS（第一个块） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi（Kyutai，2024）实现了 200 ms 的全双工。GPT-4o-realtime（2024）约 320 ms。2022 年的级联流水线达到 2500 ms。10 倍的改进来自三项技术：(1) 处处流式、(2) 带部分结果的异步流水线、(3) 可中断的生成。

## 概念

![带环形缓冲区、VAD 门控、中断的流式音频流水线](../assets/real-time.svg)

**帧 / 块 / 窗口。** 实时音频以固定大小的块流式传输。常见选择：20 ms（16 kHz 下 320 个样本）。所有下游组件必须跟上这个节奏。

**环形缓冲区。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。防止在热路径上分配内存。大小 ≈ 最大延迟 × 采样率；一个 2 秒的 16 kHz 环形缓冲区 = 32,000 个样本。

**VAD（语音活动检测）。** 在没有人说话时对下游工作进行门控。Silero VAD 4.0（2024）在 CPU 上每 30 ms 帧运行不到 1 ms。`webrtcvad` 是老牌替代方案。

**流式 ASR。** 在音频到达时发射部分转录的模型。流式模式下的 Parakeet-CTC-0.6B（NeMo，2024）在 320 ms 延迟下实现 2–5% WER。Whisper-Streaming（Macháček et al.，2023）将 Whisper 分块，在约 2 秒延迟下实现接近流式的效果。

**中断。** 当助手说话时用户也说话，你必须 (a) 检测闯入、(b) 停止 TTS、(c) 丢弃剩余的 LLM 输出。所有这些必须在 100 ms 内完成，否则用户会感觉助手失聪。

**WebRTC Opus 传输。** 20 ms 帧，48 kHz，自适应比特率 8–128 kbps。浏览器和移动端标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的技术栈。

**抖动缓冲区。** 网络数据包乱序/延迟到达。抖动缓冲区重新排序和平滑；太小 → 可听见的间隙，太大 → 延迟。典型 60–80 ms。

### 常见陷阱

- **线程争用。** Python 的 GIL + 重型模型可能使音频线程饥饿。使用 C 回调音频库（sounddevice、PortAudio）并让 Python 远离热路径。
- **采样率转换延迟。** 在流水线内重采样增加 5–20 ms。要么预先重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热。** 即使是像 Kokoro 这样快速的 TTS，在第一次请求时也有 100–200 ms 的预热。缓存模型 + 在第一次真实轮次之前用虚拟运行预热。
- **回声消除。** 如果没有 AEC，TTS 输出会重新进入麦克风并触发 ASR 识别机器人自己的声音。WebRTC AEC3 是开源默认方案。

```figure
nyquist-aliasing
```

## 动手构建

### 第 1 步：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16 kHz 下 32,000 个样本 = 2 秒。

### 第 2 步：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境中替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 第 3 步：流式 ASR

```python
# Parakeet-CTC-0.6B 通过 NeMo 进行流式处理
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 第 4 步：中断处理

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # 闯入
        # 然后送入流式 ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

依赖于异步 I/O 和可取消的 TTS 流式传输。在音频轨道上调用 WebRTC peerconnection.stop() 是规范方式。

## 使用它

2026 年的技术栈：

| 层 | 选择 |
|----|------|
| 传输 | LiveKit（WebRTC）或 Pion（Go） |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 第一个 token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API 或 Moshi |

## 陷阱

- **为了安全缓冲 500 ms。** 缓冲区*就是*你的延迟地板。缩小它。
- **未固定线程。** 在比 UI 线程优先级低的情况下运行音频回调 = 负载下出现毛刺。
- **TTS 块太小。** 小于 200 ms 的块会使声码器伪影变得可听。320 ms 块是最佳点。
- **无抖动缓冲区。** 真实网络存在抖动；没有平滑处理会出现爆破音。
- **单次错误处理。** 音频流水线必须抗崩溃。一个异常会杀死整个会话。

## 输出

保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频流水线，包含每阶段的具体延迟预算。

## 练习

1. **简单。** 运行 `code/main.py`。模拟环形缓冲区 + 能量 VAD；为一个假的 10 秒流打印各阶段延迟。
2. **中等。** 使用 `sounddevice`，构建一个直通循环，以 20 ms 帧处理你的麦克风输入，并在每帧打印 VAD 状态。
3. **困难。** 使用 `aiortc` 构建一个全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。使用 1 kHz 脉冲测量端到端延迟。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 环形缓冲区 | 循环队列 | 固定大小、无锁（或 SPSC 锁定）的音频帧 FIFO。 |
| VAD | 静音门控 | 标记语音与非语音的模型或启发式方法。 |
| 流式 ASR | 实时 STT | 在音频到达时发射部分文本；有界前视。 |
| 抖动缓冲区 | 网络平滑器 | 重新排序乱序数据包的队列；60–80 ms 典型值。 |
| AEC | 回声消除 | 减去扬声器到麦克风的反馈路径。 |
| 闯入 | 用户中断 | 系统在 TTS 过程中检测到用户语音；必须取消播放。 |
| 全双工 | 同时双向 | 用户和机器人可以同时说话；Moshi 是全双工的。 |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 全双工 200 ms 延迟。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 生产音频代理编排。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 亚 1 ms VAD，Apache 2.0。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源回声消除。
