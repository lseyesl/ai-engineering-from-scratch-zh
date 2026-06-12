# 音频生成

> 音频是一个 16-48 kHz 的一维信号。一段五秒的片段是 80-240k 个样本。没有 Transformer 能直接处理这么长的序列。2026 年每个生产级音频模型的解决方案都是一样的：一个神经编解码器（Encodec、SoundStream、DAC）将音频压缩为 50-75 Hz 的离散 token，然后一个 Transformer 或扩散模型生成 token。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 6 · 02（音频特征），阶段 6 · 04（ASR），阶段 8 · 06（DDPM）
**时间：** ~45 分钟

## 问题

三个音频生成任务：

1. **文本到语音。** 给定文本，生成语音。干净的语音是窄带的，有很强的语音结构——使用 Transformer-over-token 的方法解决得很好。VALL-E（微软）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示（文本、旋律、和弦进行、风格），生成音乐。分布要广泛得多。MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效/声音设计。** 给定提示，生成环境声或拟音。AudioGen、AudioLDM 2、Stable Audio Open。

所有三个都在相同的基础上运行：神经音频编解码器 + token-AR 或扩散生成器。

## 概念

![音频生成：编解码器 token + Transformer 或扩散](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。一个卷积编码器将波形压缩为每时间步的向量；残差向量量化（RVQ）将每个向量转换为 K 个码本条目的级联。解码器反转这个过程。24 kHz 音频以 2 kbps 使用 8 个 RVQ 码本在 75 Hz 下 = 600 token/s。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 之上的两种生成范式

**Token 自回归。** 将 RVQ token 展平为序列，运行一个仅解码器的 Transformer。MusicGen 使用"延迟并行"以每流偏移并行发射 K 个码本流。VALL-E 从文本提示 + 3 秒语音样本生成语音 token。

**潜在扩散。** 将编解码器 token 打包为连续潜在变量或用分类扩散对其进行建模。Stable Audio 2.5 在连续的音频潜在变量上使用流匹配。AudioLDM 2 使用文本到梅尔频谱再到音频的扩散。

2024-2026 年趋势：流匹配在音乐领域胜出（推理更快，样本更干净），而 token-AR 仍然主导语音领域，因为它天然是因果的并且流式表现良好。

## 生产格局

| 系统 | 任务 | 骨干 | 延迟 |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + 神经声码器 | 约 300ms 第一个 token |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态 AR | 约 200ms |
| NaturalSpeech 3 | TTS | 潜在流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐 / 音效 | DiT + 音频潜在变量上的流匹配 | 约 10s 每 1 分钟片段 |
| Suno v4 | 完整歌曲 | 未公开；怀疑是 token-AR | 约 30s 每首歌 |
| Udio v1.5 | 完整歌曲 | 未公开 | 约 30s 每首歌 |
| MusicGen 3.3B | 音乐 | Token-AR on Encodec 32kHz | 实时 |
| AudioCraft 2 | 音乐 + 音效 | 流匹配 | 约 5s 每 5s 片段 |
| Riffusion v2 | 音乐 | 频谱扩散 | 约 10s |

## 动手实现

`code/main.py` 模拟核心思想：在一个从两种不同"风格"（风格 A 是交替的低和高 token，风格 B 是单调递增）生成的合成"音频 token"序列上训练一个微型下一个 token 预测 Transformer。以风格为条件并采样。

### 步骤 1：合成音频 token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "类似语音"：交替
        return [i % vocab_size for i in range(length)]
    # "类似音乐"：递增
    return [(i * 3) % vocab_size for i in range(length)]
```

### 步骤 2：训练微型 token 预测器

一个以风格为条件的二元模型风格预测器。重点是模式：编解码器 token → 交叉熵训练 → 自回归采样。

### 步骤 3：条件采样

给定风格 token 和一个起始 token，从预测分布中采样下一个 token。继续 20-40 个 token。

## 陷阱

- **编解码器质量上限。** 如果编解码器不能忠实地表示声音，无论生成器质量多好都没用。DAC 是目前公开的最佳编解码器。
- **RVQ 误差累积。** 每个 RVQ 层对前一层的残差进行建模。第 1 层的误差会传播。在高层使用温度为 0 的采样有所帮助。
- **音乐结构。** 30 秒的音频率 token 在 75 Hz 下是 20k+ 个 token。对 Transformer 来说很难。MusicGen 使用滑动窗口 + 提示延续；Stable Audio 使用更短的片段 + 交叉淡入淡出。
- **边界处的伪影。** 生成片段之间的交叉淡入淡出需要仔细的重叠相加。
- **干净数据的胃口。** 音乐生成器需要数万小时的授权音乐。Suno / Udio 的 RIAA 诉讼（2024）将这一问题浮出水面。
- **语音克隆伦理。** 3 秒的样本加上一个文本提示就足以让 VALL-E / XTTS / ElevenLabs 克隆一个声音。每个生产模型都需要滥用检测 + 退出名单。

## 使用

| 任务 | 2026 年技术栈 |
|------|------------|
| 商业 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 语音克隆（经同意验证） | XTTS v2（开放）或 ElevenLabs Pro |
| 背景音乐，快速 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / 拟音 | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音代理 | GPT-4o realtime 或 Gemini Live |
| 开放权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 产出

保存 `outputs/skill-audio-brief.md`。技能接受音频简报（任务、时长、风格、声音、许可）并输出：模型 + 托管、提示格式（风格标签、风格描述符、结构标记）、编解码器 + 生成器 + 声码器链、种子协议和评估计划（MOS / CLAP 分数 / TTS 的 CER / 用户 A/B 测试）。

## 练习

1. **简单。** 运行 `code/main.py` 并显式设置风格。验证生成的序列是否匹配该风格的模式。
2. **中等。** 添加延迟并行解码：模拟 2 个 token 流，它们必须保持偏移 1 步。训练一个联合预测器。
3. **困难。** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三个不同的提示生成 10 秒片段；A/B 测试风格遵循度。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 编解码器 | "神经压缩" | 音频的编码器/解码器；典型输出是 50-75 Hz 的 token。 |
| RVQ | "残差 VQ" | K 个量化器的级联；每个对前一个的残差进行建模。 |
| Token | "一个编解码器符号" | 码本中的离散索引；通常为 1024 或 2048。 |
| 延迟并行 | "偏移码本" | 以交错偏移发射 K 个 token 流以减少序列长度。 |
| 流匹配 | "2024 年音频的胜利" | 比扩散路径更直的替代方案；采样更快。 |
| 语音提示 | "3 秒样本" | 引导克隆语音的说话者嵌入或 token 前缀。 |
| 梅尔频谱 | "可视化" | 对数幅度感知频谱；许多 TTS 系统使用。 |
| 声码器 | "梅尔到波形" | 将梅尔频谱转换回音频的神经组件。 |

## 生产说明：音频是一个流式问题

音频是用户期望*在生成的同时*到达的输出模态，而不是一次性全部到达。用生产的术语来说，这意味着 TPOT（每个输出 token 的时间）很重要，因为用户的听觉速度就是目标吞吐量——而不是他们的阅读速度。对于在约 75 tokens/s（Encodec）下 token 化的 16kHz 音频，服务器必须为用户生成 ≥75 tokens/s 以保持播放流畅。

两个架构上的后果：

- **流匹配音频模型不能简单地流式传输。** Stable Audio 2.5 和 AudioCraft 2 一次渲染固定长度的片段。要流式传输，你需要分块片段并重叠边界——想想滑动窗口扩散——与编解码器 AR 模型相比增加 100-300ms 的延迟开销。

如果产品是"实时语音聊天"或"实时音乐延续"，选择编解码器 AR 路径。如果是"提交后渲染 30 秒片段"，流匹配在质量和总延迟上胜出。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 编解码器标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 第一个广泛使用的神经音频编解码器。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025 年通过流匹配实现的文本到音乐。
