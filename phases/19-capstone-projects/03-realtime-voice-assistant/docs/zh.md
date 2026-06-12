# 顶点项目 03——实时语音助手（ASR 到 LLM 到 TTS）

> 感觉对的语音智能体具有端到端延迟低于 800ms、知道你何时停止说话、处理打断、并在不卡顿的情况下调用工具的能力。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都达到了这个标准。它们用相同的形态做到了：流式 ASR、话轮检测器、流式 LLM 和流式 TTS，全部通过 WebRTC 连接，每个跳点都有激进的延迟预算。构建一个，测量 WER、MOS 和误切断率，并在丢包条件下运行。

**类型:** Capstone
**语言:** Python（智能体 + 管道）、TypeScript（Web 客户端）
**前置要求:** Phase 6（语音与音频）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 17（基础设施）
**涉及阶段:** P6 · P7 · P11 · P13 · P14 · P17
**时间:** 30 小时

## 问题

语音一直是 2025-2026 年发展最快的 AI UX 类别。技术门槛每个季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都将低于 800ms 的首个音频输出变为可实现。标准不仅仅是延迟。还有交互感受：不打断用户、不被用户打断、从说话中途的打断中恢复、在对话中调用工具而不卡顿音频、在不稳定的移动网络上存活。

你无法通过串接三个 REST 调用来实现这一点。架构是端到端的管道化流式传输。构建它，故障模式就会变得可见：针对电话音频调优的 VAD 在背景电视噪音上触法、等待永不到来的标点符号的话轮检测器、缓冲 400ms 才发射的 TTS。顶点项目就是在负载下逐一修复这些问题，并发布延迟与质量报告。

## 概念

管道有五个流式阶段：**audio in**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**turn detection**（VAD 加上一个读取部分转录以寻找完成提示的小型话轮检测器模型）、**LLM**（一旦话轮被判定完成就开始流式 token）、**TTS**（在第一个 LLM token 出现后约 200ms 内开始流式音频输出）。

三个交叉关注点。**Barge-in**：当用户在智能体说话时开始说话，TTS 立即取消，ASR 立即接管。**Tool use**：对话中的函数调用（天气、日历）必须在侧通道上运行，不卡顿音频；如果延迟超过 300ms，智能体预填一个确认 token（"请稍等……"）。**Backpressure**：在丢包情况下，部分转录被暂存，VAD 提高语音门限阈值，智能体避免在未确认的消息上说话。

测量标准是量化的。WER 在汉明 VAD 基准上低于 8%（信噪比 15 dB）。在 100 次测量通话中，首个音频输出 p50 低于 800ms。误切断率低于 3%。TTS MOS 高于 4.2。单个 g5.xlarge 上支持 50 个并发通话。这些数字就是交付物。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (或 Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (天气,
 Whisper-v3)      每 20ms         基于部分文稿      日历)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     级联 Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- 传输：LiveKit Agents 1.0（WebRTC）加 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR：Deepgram Nova-3（流式，首次部分结果低于 300ms）或自托管 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 加上 LiveKit 话轮检测器（读取部分转录的小型 transformer）
- LLM：OpenAI GPT-4o-realtime 用于紧密集成，Gemini 2.5 Flash Live，或级联 Claude Haiku 4.5（流式补全，独立音频路径）
- TTS：Cartesia Sonic-2（最低首字节延迟）、ElevenLabs Flash v3 或开源 Orpheus（自托管）
- 工具：FastMCP 侧通道用于天气/日历/预订；如果工具耗时超过 300ms，智能体预发射填充语
- 可观测性：OpenTelemetry 语音 spans、带音频回放的 Langfuse 语音跟踪
- 部署：单个 g5.xlarge（24GB VRAM）用于自托管 Whisper + Orpheus；托管 API 用于最低延迟

## 构建它

1. **WebRTC 会话。** 搭建一个 LiveKit 房间和一个流式传输麦克风音频的 Web 客户端。在服务器上，附加一个加入房间的智能体 worker。

2. **ASR 流式传输。** 将 20ms PCM 帧馈送到 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录。记录每次部分结果的延迟。

3. **VAD 和话轮检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件上，针对最新的部分转录触发 LiveKit 话轮检测器。仅当 VAD 表示静音持续 500ms 且话轮检测器完成度评分超过 0.6 时，才确认为"话轮完成"。

4. **LLM 流。** 话轮完成后，启动 LLM 调用，传入进行中的对话和最终转录。流式输出 token。在第一个 token 处，交给 TTS。

5. **TTS 流。** Cartesia Sonic-2 流式返回音频块。第一个块必须在第一个 LLM token 出现后 200ms 内离开服务器。将块发射到 LiveKit 房间；客户端通过 WebRTC 抖动缓冲区播放。

6. **Barge-in。** 当 VAD 在 TTS 播放期间检测到新的用户语音时，立即取消 TTS 流，丢弃剩余的 LLM 输出，并重新启动 ASR。发布一个 `tts_canceled` span。

7. **工具侧通道。** 将天气和日历注册为函数调用工具。调用时，并发触发调用；如果在 300ms 内未解决，让 LLM 发出"请稍等，让我查一下"作为填充语；一旦工具返回则继续。

8. **评估工具。** 录制 100 次通话。计算 WER（对照留出的转录）、误切断率（用户说话中途 TTS 被取消）、首个音频输出 p50、TTS MOS（人工或 NISQA）和抖动丢包测试（丢弃 3% 的数据包）。

9. **负载测试。** 使用合成呼叫者在单个 g5.xlarge 上驱动 50 个并发通话。测量稳定的首个音频输出 p95。

## 使用它

```
caller: "明天东京的天气怎么样"
[asr  ] partial @280ms: "明天"
[asr  ] partial @540ms: "明天东京的天气"
[turn ] completion score 0.82 @820ms; 提交
[llm  ] 第一个 token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 多云 @1140ms
[tts  ] 首个音频输出 @1040ms: "明天东京多云..."
话轮延迟: 1040ms 用户停止 -> 音频输出
```

## 交付物

`outputs/skill-voice-agent.md` 是交付物。给定一个领域（客户支持、日程安排或自助终端），它会架设一个 LiveKit 智能体，包含根据测量标准调整的 ASR/VAD/LLM/TTS 管道。评分标准：

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | 端到端延迟 | 跨 100 次录制通话的 p50 首个音频输出低于 800ms |
| 20 | 话轮切换质量 | 汉明 VAD 基准上的误切断率低于 3% |
| 20 | 工具使用正确性 | 对话中的工具调用在返回正确数据时不卡顿音频 |
| 15 | 评估工具完整性 | 可重现的测量，带公共配置 |
| **100** | | |

## 练习

1. 将 Deepgram Nova-3 替换为 g5.xlarge 上的 faster-whisper v3 turbo。测量延迟和 WER 差距。确定 CPU 与 GPU 决策在哪些方面重要。

2. 添加一个打断仲裁策略：当用户在工具调用期间打断时，智能体做什么？比较三种策略（硬取消、完成工具然后停止、排队下一个话轮）。

3. 运行对抗性话轮检测器测试：让用户在句子中间长时间停顿。调优 VAD 静音阈值和话轮检测器评分阈值，以在不超过 900ms 的情况下获得最低误切断率。

4. 通过 Twilio 在 PSTN 上部署相同的智能体。比较 PSTN 首个音频输出与 WebRTC。解释抖动缓冲区和编码器差异。

5. 添加非英语语言（日语、西班牙语）的语音活动检测。测量 Silero VAD v5 的误触发率与语言特定微调模型的对比。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Turn detection | "话轮结束" | 分类器，根据 VAD 静音和部分转录决定用户已完成说话 |
| Barge-in | "打断处理" | 当 VAD 检测到新用户语音时取消 TTS 播放 |
| First-audio-out | "延迟" | 从用户停止说话到第一个音频包离开服务器的时间 |
| VAD | "语音门限" | 将音频帧分类为语音或静音的模型；Silero VAD v5 是 2026 年的默认选择 |
| Jitter buffer | "音频平滑" | 客户端缓冲区，短暂持有数据包以吸收网络波动 |
| Filler | "确认 token" | 智能体在工具较慢时发出的短语，以避免沉默 |
| MOS | "平均意见评分" | 感知语音质量评级；NISQA 是自动化代理 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents)——参考 WebRTC 智能体框架
- [Pipecat](https://github.com/pipecat-ai/pipecat)——备用的 Python 优先流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)——集成语音模型的参考
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs)——流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad)——VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai)——低延迟 TTS 参考
- [Retell AI 架构](https://docs.retellai.com)——生产语音智能体架构
- [Vapi.ai 生产栈](https://docs.vapi.ai)——备选生产参考
