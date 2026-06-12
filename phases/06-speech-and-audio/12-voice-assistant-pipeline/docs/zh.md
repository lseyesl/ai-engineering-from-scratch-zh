# 构建语音助手流水线 — 阶段 6 的顶点项目

> 将课程 01-11 的所有内容串联起来。构建一个能听、能推理、能回话的语音助手。2026 年这是一个已解决的工程问题，而不是研究问题 — 但集成细节决定它能否真正交付。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 6 · 04、05、06、07、11；阶段 11 · 09（函数调用）；阶段 14 · 01（代理循环）
**时间：** ~120 分钟

## 问题

构建一个端到端的助手：

1. 捕捉麦克风输入（16 kHz 单声道）。
2. 检测用户语音的开始/结束。
3. 流式转录。
4. 将转录传递给可以调用工具的 LLM（计时器、天气、日历）。
5. 将 LLM 文本流式传输到 TTS。
6. 向用户播放音频。
7. 如果用户在响应过程中打断，则停止。

延迟目标：在笔记本电脑 CPU 上，用户说完话后的第一个 TTS 音频字节在 800 ms 内。质量目标：没有漏掉的词、没有在静音上产生幻觉的字幕、没有语音克隆泄漏、没有成功的提示注入。

## 概念

![语音助手流水线：麦克风 → VAD → STT → LLM+工具 → TTS → 扬声器](../assets/voice-assistant.svg)

### 七个组件

1. **音频捕获。** 麦克风 → 16 kHz 单声道 → 20 ms 块。通常使用 Python 的 `sounddevice` 或在生产中使用原生 AudioUnit/ALSA/WASAPI。
2. **VAD（课程 11）。** Silero VAD @ 阈值 0.5，最小语音 250 ms，静音挂起 500 ms。发出"开始"和"结束"信号。
3. **流式 STT（课程 4-5）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。部分 + 最终转录。
4. **带工具调用的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。工具的 JSON schema。流式输出 token。
5. **流式 TTS（课程 7）。** Kokoro-82M（最快的开源）或 Cartesia Sonic（商业）。在获取 20 个 LLM token 后开始 TTS。
6. **播放。** 扬声器输出；opus 编码用于低带宽网络。
7. **中断处理。** 如果在 TTS 播放期间 VAD 触发，停止播放、取消 LLM、重新启动 STT。

### 你会遇到的三类失败

1. **首词截断。** VAD 晚了一拍启动。用户的"嘿"丢失了。起始阈值设为 0.3 而不是 0.5。
2. **响应中间的中断混乱。** 用户打断后 LLM 继续生成；助手与用户同时说话。连接 VAD → 取消 LLM。
3. **静音幻觉。** Whisper 在静音的预热帧上输出"Thanks for watching"。始终使用 VAD 门控。

### 2026 年生产参考技术栈

| 技术栈 | 延迟 | 许可证 | 备注 |
|--------|------|--------|------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 商业 API | 2026 年行业默认 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | 基本开源 | 适合 DIY |
| Moshi（全双工） | 200-300 ms | CC-BY 4.0 | 单模型；不同架构，课程 15 |
| Vapi / Retell（托管） | 300-500 ms | 商业 | 启动最快；定制有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私 / 边缘设备 |

## 动手构建

### 第 1 步：带分块的麦克风捕获（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 第 2 步：VAD 门控的轮次捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 第 3 步：流式 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 第 4 步：LLM 循环内的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 第 5 步：中断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用它

参见 `code/main.py` 中的可运行模拟，它使用桩模型连接所有七个组件，让你可以在没有硬件的情况下看到流水线的形状。对于真实实现，将桩替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 用于 I/O

## 陷阱

- **永远记录 PII。** 完整轮次的音频在大多数司法管辖区属于 PII。30 天保留期，加密存储。
- **无闯入机制。** 用户会打断。你的助手必须停止说话。
- **阻塞的 TTS。** 同步 TTS 会阻塞事件循环。使用异步或单独线程。
- **无工具调用错误处理。** 工具会失败。LLM 必须收到错误 + 重试一次，然后优雅降级。
- **过度积极的幻觉过滤。** 过度过滤会使助手重复"我无法帮助您。"过滤不足则什么都说。在保留集上校准。
- **无唤醒词选项。** 始终监听是隐私责任。添加唤醒词门控（Porcupine 或 openWakeWord）。

## 输出

保存为 `outputs/skill-voice-assistant-architect.md`。在给定的预算 + 规模 + 语言 + 合规约束下，生成完整的技术栈规范。

## 练习

1. **简单。** 运行 `code/main.py`。它用桩模块端到端模拟一个完整的轮次，并打印每阶段延迟。
2. **中等。** 在预录的 `.wav` 文件上，将 STT 桩替换为真实的 Whisper 模型。测量 WER 和端到端延迟。
3. **困难。** 添加工具调用：实现 `get_weather`（任意 API）和 `set_timer`。将 LLM 通过工具路由，并验证当用户说"设置一个 5 分钟计时器"时，正确的函数被触发，口语回复确认了它。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 轮次 | 用户 + 助手的一次往返 | 一次 VAD 边界内的用户语音 + 一次 LLM-TTS 响应。 |
| 闯入 | 中断 | 助手说话时用户说话；助手停止。 |
| 唤醒词 | "嘿助手" | 短关键词检测器；Porcupine、Snowboy、openWakeWord。 |
| 端点检测 | 轮次结束 | VAD + 最小静音决定用户已完成。 |
| 预卷 | 语音前缓冲区 | 保留 VAD 触发前 200-400 ms 的音频，避免首词截断。 |
| 工具调用 | 函数调用 | LLM 发射 JSON；运行时调度；结果反馈到循环中。 |

## 延伸阅读

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — 生产级参考。
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — 适合 DIY 的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 托管式语音原生路径。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工参考（课程 15）。
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — 唤醒词门控。
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM 函数调用。
