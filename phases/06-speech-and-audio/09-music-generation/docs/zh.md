# 音乐生成 — MusicGen、Stable Audio、Suno 与许可地震

> 2026 年音乐生成：Suno v5 和 Udio v4 主导商业；MusicGen、Stable Audio Open 和 ACE-Step 领跑开源。技术问题基本解决。法律问题（华纳音乐 5 亿美元和解、UMG 和解）在 2025-2026 年重塑了该领域。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 6 · 02（频谱图），阶段 4 · 10（扩散模型）
**时间：** ~75 分钟

## 问题

文本 → 一段 30 秒到 4 分钟的音乐片段，带歌词、人声和结构。三个子问题：

1. **器乐生成。** 文本如"lo-fi 嘻哈鼓配温暖键盘" → 音频。MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（带人声+歌词）。** "关于雨夜德州的乡村歌曲" → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **条件化/可控。** 扩展现有片段、重新生成桥段、更换流派、音轨分离或修补。Udio 的修补 + 音轨分离是 2026 年的标杆功能。

## 概念

![音乐生成：token-LM vs 扩散，2026 年模型地图](../assets/music-generation.svg)

### 神经编解码器 token 上的 Token LM

Meta 的 **MusicGen**（2023，MIT）及其衍生品：在文本/旋律嵌入的条件下，自回归预测 EnCodec token（32 kHz，4 个码本），用 EnCodec 解码。3 亿 - 33 亿参数。强大的基线；超过 30 秒后会遇到困难。

**ACE-Step**（开源，2026 年 4 月发布 4B XL）将这一点扩展到带歌词条件的完整歌曲生成。开源社区最接近 Suno 的作品。

### 梅尔或潜在空间上的扩散

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：在压缩音频上的潜在扩散。擅长循环、声音设计、环境纹理。不太擅长结构化的完整歌曲。

**AudioLDM / AudioLDM2**：通过 T2I 风格的潜在扩散实现文本到音频，泛化到音乐、音效、语音。

### 混合（生产）— Suno、Udio、Lyria

闭源权重。可能是 AR 编解码器 LM + 基于扩散的声码器，带有专门的声乐/鼓/旋律头。Suno v5（2026）是 ELO 1293 质量领导者。Udio v4 增加了修补 + 音轨分离（贝斯、鼓、人声可单独下载）。

### 评估

- **FAD（弗雷歇音频距离）。** 使用 VGGish 或 PANNs 特征的生成 vs 真实音频分布之间的嵌入级距离。越低越好。MusicGen small 在 MusicCaps 上：4.5 FAD；SOTA 约 3.0。
- **音乐性（主观）。** 人类偏好。Suno v5 ELO 1293 领先。
- **文本-音频对齐。** 提示和输出之间的 CLAP 分数。
- **音乐性伪影。** 不合拍的过渡、声乐短语漂移、超过 30 秒后结构丢失。

## 2026 年模型地图

| 模型 | 参数量 | 时长 | 人声 | 许可证 |
|------|--------|------|------|--------|
| MusicGen-large | 3.3B | 30 秒 | 否 | MIT |
| Stable Audio Open | 1.2B | 47 秒 | 否 | Stability 非商业 |
| ACE-Step XL（2026 年 4 月） | 4B | > 2 分钟 | 是 | Apache-2.0 |
| YuE | 7B | > 2 分钟 | 是，多语言 | Apache-2.0 |
| Suno v5（闭源） | ? | 4 分钟 | 是，ELO 1293 | 商业 |
| Udio v4（闭源） | ? | 4 分钟 | 是 + 音轨 | 商业 |
| Google Lyria 3（闭源） | ? | 实时 | 是 | 商业 |
| MiniMax Music 2.5 | ? | 4 分钟 | 是 | 商业 API |

## 法律格局（2025-2026）

- **华纳音乐 vs Suno 和解。** 5 亿美元。WMG 现在对 Suno 上的 AI 相似性、音乐版权和用户生成曲目拥有监督权。Udio 上的类似 UMG 和解。
- **欧盟 AI 法案** + **加州 SB 942**：AI 生成的音乐必须被披露。
- **MIT 许可的 Riffusion / MusicGen** 没有合规负担，但也没有商业人声。

安全可发货的模式：

1. 仅生成器乐（MusicGen、Stable Audio Open、MIT/CC0 输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music）配合每次生成许可证。
3. 在自有或许可的目录上训练（大多数企业最终选择此方案）。
4. 用标记水印 + 元数据标记生成内容。

## 动手构建

### 第 1 步：使用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种规格：`small`（3 亿，快速）、`medium`（15 亿）、`large`（33 亿）。Small 足以判断"这个想法是否成立。"

### 第 2 步：旋律条件化

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接受一个色度图，在保留旋律的同时更换音色。适用于"把这个旋律变成弦乐四重奏。"

### 第 3 步：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish 嵌入距离。适用于流派级回归测试；不能替代人工试听。

### 第 4 步：集成到 LLM-音乐工作流中

结合课程 7-8 的思路：

```python
prompt = "写一段 30 秒的爵士循环。描述鼓、贝斯和钢琴编排。"
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使用它

| 目标 | 技术栈 |
|------|--------|
| 器乐声音设计 | Stable Audio Open |
| 游戏 / 自适应音乐 | Google Lyria RealTime（闭源） |
| 带人声的完整歌曲（商业） | Suno v5 或 Udio v4，带有明确许可 |
| 带人声的完整歌曲（开源） | ACE-Step XL 或 YuE |
| 短广告曲 | MusicGen 旋律条件化（基于哼唱的参考） |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍在犯的陷阱

- **版权洗白提示。** "Taylor Swift 风格的歌曲" — Suno/Udio 现已过滤此类提示，开放模型不会。添加你自己的过滤列表。
- **超过 30 秒的重复/漂移。** AR 模型会循环。交叉淡化多个生成片段，或使用 ACE-Step 获得结构连贯性。
- **速度漂移。** 模型会偏离 BPM。在提示中使用 BPM 标签，并在后期使用 librosa 的 `beat_track` 过滤。
- **人声可懂度。** Suno 表现出色；开放模型在歌词上往往模糊。如果歌词重要，使用商业 API 或微调。
- **单声道输出。** 开放模型生成单声道或假立体声。使用适当的立体声重建升级（ezst、Cartesia 的立体声扩散）。

## 输出

保存为 `outputs/skill-music-designer.md`。选择模型、许可策略、时长/结构计划和音乐生成部署的披露元数据。

## 练习

1. **简单。** 运行 `code/main.py`。它产生一个"生成式"和弦进行 + 鼓模式作为 ASCII 符号 — 一个音乐生成的卡通版。如果需要，可以通过任何 MIDI 渲染器回放。
2. **中等。** 安装 `audiocraft`，用 MusicGen-small 在 4 个流派提示上生成 10 秒片段，测量与参考流派集的 FAD。
3. **困难。** 使用 ACE-Step（或 MusicGen-melody）生成同一旋律的三种变体，使用不同的音色提示。计算与提示的 CLAP 相似度以验证对齐。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| FAD | 音频 FID | 真实与生成音频嵌入分布之间的弗雷歇距离。 |
| 色度图 | 作为音高的旋律 | 每帧 12 维向量；旋律条件化的输入。 |
| 音轨 | 乐器轨道 | 分离的贝斯/鼓/人声/旋律作为 WAV。 |
| 修补 | 重新生成某个部分 | 屏蔽一个时间窗口；模型仅重新生成该部分。 |
| CLAP | 文本-音频 CLIP | 对比性音频-文本嵌入；评估文本-音频对齐。 |
| EnCodec | 音乐编解码器 | Meta 的神经编解码器，MusicGen 使用；32 kHz，4 个码本。 |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开放自回归基准。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 声音设计的默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 开放 4B 完整歌曲生成器，2026 年 4 月。
- [Suno v5 platform docs](https://suno.com) — 商业质量领导者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 音乐 + 音效的潜在扩散。
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025 年 11 月先例。
