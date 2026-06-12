# 语音智能体：Pipecat 与 LiveKit

> 语音智能体在 2026 年是一个一等生产类别。Pipecat 为你提供 Python 基于帧的管道（VAD → STT → LLM → TTS → 传输）。LiveKit Agents 通过 WebRTC 将 AI 模型桥接到用户。生产延迟目标为高级技术栈 450-600ms 端到端。

**类型：** Learn
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（智能体循环），Phase 14 · 12（工作流模式）
**时间：** ~60 分钟

## 学习目标

- 描述 Pipecat 的基于帧的管道：DOWNSTREAM（源→宿）和 UPSTREAM（控制）。
- 说出典型语音管道阶段以及 Pipecat 支持的传输方式。
- 解释 LiveKit Agents 的两个语音智能体类（MultimodalAgent、VoicePipelineAgent）以及各自何时适合。
- 总结 2026 年生产延迟预期以及它们如何驱动架构选择。

## 问题

语音智能体不是带有 TTS 附加的文本循环。延迟预算非常紧张（约 600ms），部分音频是默认情况，转折检测是一个模型，传输范围从电话 SIP 到 WebRTC。要么你构建基于帧的管道（Pipecat），要么你依赖平台（LiveKit）。

## 概念

### Pipecat（pipecat-ai/pipecat）

- Python 基于帧的管道框架。
- `Frame` → `FrameProcessor` 链。
- 两个流方向：
  - **DOWNSTREAM** — 源 → 宿（音频输入，TTS 输出）。
  - **UPSTREAM** — 反馈和控制（取消、指标、打断）。
- `PipelineTask` 使用事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）和观察者管理生命周期，用于指标/追踪/RTVI。

典型管道：

```
VAD（Silero）→ STT → LLM（上下文交替用户/助手）→ TTS → 传输
```

传输方式：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 增加了结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 将 AI 模型桥接到用户。
- 关键概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两个语音智能体类：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或等效的直接音频。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本级控制。
- 通过 Transformer 模型进行语义转折检测。
- 原生 MCP 集成。
- 通过 SIP 的电话功能。
- 通过 LiveKit Inference 提供 50+ 模型无需 API 密钥；通过插件提供 200+ 模型。

### 商业平台

Vapi（在优化的高级技术栈上约 450-600ms）和 Retell（跨 180 次测试调用约 600ms 端到端）在此基础上构建。当你想要托管语音技术栈而没有 WebRTC 团队时，选择平台。

### 这种模式出错的地方

- **没有打断处理。** 用户打断；智能体继续说话。需要 Pipecat 中的 UPSTREAM 取消帧，LiveKit 中的等效功能。
- **STT 置信度被忽略。** 低置信度记录作为真理输入 LLM。对置信度设门或请求确认。
- **TTS 句子中间截断。** 当管道在说话中间取消时，TTS 需要知道或剪切音频。
- **延迟预算被忽略。** 每个组件增加 50-200ms。在交付前求和你的链。

### 2026 年典型延迟

- VAD：20-60ms
- STT 部分：100-250ms
- LLM 首个 token：150-400ms
- TTS 首个音频：100-200ms
- 传输 RTT：30-80ms

端到端 450-600ms 是高级。800-1200ms 是常见的。任何超过 1500ms 的感觉像坏了。

## 构建

`code/main.py` 是一个基于帧的玩具管道，包含：

- `Frame` 类型（audio、transcript、text、tts_audio、control）。
- 带有 `process(frame)` 的 `Processor` 接口。
- 一个五阶段管道（VAD → STT → LLM → TTS → 传输）作为脚本化处理器。
- 一个 UPSTREAM 取消帧来演示打断。

运行：

```
python3 code/main.py
```

轨迹显示正常流程和一个打断取消，在 TTS 说话中间停止。

## 使用

- **Pipecat** 用于完全控制——自定义处理器、Python 优先、可插拔提供商。
- **LiveKit Agents** 用于 WebRTC 优先的部署和电话功能。
- **Vapi / Retell** 用于托管的语音智能体，无需 WebRTC 团队。
- **OpenAI Realtime / Gemini Live** 用于直接音频输入/音频输出（MultimodalAgent）。

## 交付

`outputs/skill-voice-pipeline.md` 搭建一个 Pipecat 形状的语音管道，包含 VAD + STT + LLM + TTS + 传输以及打断处理。

## 练习

1. 为你的玩具管道添加指标观察者：计算每秒每阶段的帧数。延迟在哪里累积？
2. 实现置信度门控 STT：低于阈值，请求"你能重复一遍吗？"
3. 添加语义转折检测：简单规则——如果记录以"？"结尾，转折结束。
4. 阅读 Pipecat 的传输文档。将标准库传输替换为 SmallWebRTCTransport 配置（存根）。
5. 在相同查询上测量 OpenAI Realtime 与 STT+LLM+TTS 级联的对比。文本级控制带来了什么延迟成本？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 帧 | "事件" | 管道中的类型化数据单元（音频、记录、文本、控制） |
| 处理器 | "管道阶段" | 带有 process(frame) 的处理程序 |
| DOWNSTREAM | "正向流" | 源到宿：音频输入，语音输出 |
| UPSTREAM | "反馈流" | 控制：取消、指标、打断 |
| VAD | "语音活动检测" | 检测用户何时在说话 |
| 语义转折检测 | "智能结束转折" | 基于模型的用户说完决策 |
| MultimodalAgent | "直接音频智能体" | 音频输入，音频输出；中间无文本 |
| VoicePipelineAgent | "级联智能体" | STT + LLM + TTS；文本级控制 |

## 延伸阅读

- [Pipecat 文档](https://docs.pipecat.ai/getting-started/introduction) — 基于帧的管道、处理器、传输
- [LiveKit Agents 文档](https://docs.livekit.io/agents/) — WebRTC + 语音原语
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音，延迟基准测试
