# 音频基础 — 波形、采样、傅里叶变换

> 波形是原始信号。频谱图是表示形式。梅尔特征是对机器学习友好的形式。每一个现代 ASR 和 TTS 流水线都沿着这个阶梯前进，第一个台阶就是理解采样和傅里叶。

**类型：** 学习
**语言：** Python
**前置条件：** 阶段 1 · 06（向量与矩阵），阶段 1 · 14（概率分布）
**时间：** ~45 分钟

## 问题

麦克风产生一个压力-时间信号。你的神经网络消费的是张量。两者之间有一系列约定，一旦被违反就会产生静默 bug：模型训练正常但 WER 翻倍，或 TTS 输出嘶嘶声，或语音克隆系统记住了麦克风而非说话人。

语音系统中的每个 bug 都可以追溯到以下三个问题之一：

1. 数据录制时的采样率是多少，模型期望的采样率是多少？
2. 信号是否发生了混叠？
3. 你是在原始样本上操作还是在频域表示上操作？

搞对了这些，阶段 6 的其余内容就都是可解的。搞错了，连 Whisper-Large-v4 都会产生垃圾输出。

## 概念

![波形、采样、DFT 和频率区间可视化](../assets/audio-fundamentals.svg)

**波形。** 一个一维浮点数数组，范围在 `[-1.0, 1.0]` 内。以样本序号为索引。转换成秒需要除以采样率：`t = n / sr`。一段 10 秒 16 kHz 的音频是一个有 160,000 个浮点数的数组。

**采样率 (sr)。** 每秒的样本数。2026 年的常见采样率：

| 采样率 | 用途 |
|--------|------|
| 8 kHz | 电话通信，传统 VOIP。奈奎斯特频率在 4 kHz，会丢失辅音。不推荐用于 ASR。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都使用 16 kHz。 |
| 22.05 kHz | 旧模型的 TTS 声码器训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频，音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农定理。** 采样率 `sr` 可以无歧义地表示最高 `sr/2` 的频率。`sr/2` 边界就是*奈奎斯特频率*。高于奈奎斯特的能量会被*混叠* — 折叠到较低频率中 — 从而损坏信号。在下采样之前务必使用低通滤波器。

**位深度。** 16 位 PCM（有符号 int16，范围 ±32,767）是通用的交换格式。24 位用于音乐，32 位浮点用于内部 DSP。像 `soundfile` 这样的库读取 int16 但返回 `[-1, 1]` 范围内的 float32 数组。

**傅里叶变换。** 任何有限信号都是不同频率正弦波的和。离散傅里叶变换 (DFT) 对 `N` 个样本计算 `N` 个复数系数 — 每个频率区间一个。`区间 k` 映射到频率 `k · sr / N` Hz。幅度是该频率的振幅，角度是相位。

**FFT。** 快速傅里叶变换：当 `N` 是 2 的幂时，一种 `O(N log N)` 的 DFT 算法。每个音频库底层都使用 FFT。16 kHz 下对 1024 个样本做 FFT 产生 512 个可用的频率区间，覆盖 0–8 kHz，分辨率 15.6 Hz。

**分帧 + 加窗。** 我们不会对整个音频做 FFT。而是将其切分为重叠的*帧*（通常 25 ms，步长 10 ms），每帧乘以窗函数（汉宁窗、汉明窗）以消除边缘不连续性，然后对每帧做 FFT。这就是短时傅里叶变换 (STFT)。课程 02 从这里继续。

```figure
mel-scale
```

## 动手构建

### 第 1 步：读取音频并绘制波形

`code/main.py` 仅使用标准库的 `wave` 模块以保持无需额外依赖。在生产中你会使用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 第 2 步：从第一原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

一个 16 kHz 下持续 1 秒的 440 Hz 正弦波（音乐会 A 音）有 16,000 个浮点数。使用 16 位 PCM 编码通过 `wave.open(..., "wb")` 写入。

### 第 3 步：手动计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — 对于 `N=256` 验证正确性尚可，对真实音频无用。实际代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 第 4 步：找出主频率

幅度峰值索引 `k_star` 映射到频率 `k_star * sr / N`。在 440 Hz 正弦波上运行应该返回在区间 `440 * N / sr` 处的峰值。

### 第 5 步：演示混叠

以 10 kHz 采样 7 kHz 正弦波（奈奎斯特频率 = 5 kHz）。7 kHz 音调高于奈奎斯特频率，折叠到 `10 − 7 = 3 kHz`。FFT 峰值出现在 3 kHz。这是经典的混叠演示，也是每个 DAC/ADC 都配备砖墙低通滤波器的原因。

## 使用它

你在 2026 年实际发货时会使用的技术栈：

| 任务 | 库 | 原因 |
|------|-----|------|
| 读/写 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 最快、稳定、返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠。 |
| STFT / 梅尔 | `torchaudio` 或 `librosa` | 支持 GPU；PyTorch 生态系统。 |
| 实时流式 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定。 |
| 检查文件 | `ffprobe` 或 `soxi` | 命令行、快速、报告采样率/声道数/编解码器。 |

决策法则：**先匹配采样率，再匹配其他任何东西**。Whisper 期望 16 kHz 单声道 float32。传给它 44.1 kHz 立体声，你会得到看似模型 bug 的垃圾输出。

## 输出

保存为 `outputs/skill-audio-loader.md`。该技能帮助你检查音频输入是否符合下游模型的期望，并在不符时正确重采样。

## 练习

1. **简单。** 在 16 kHz 下合成一段 1 秒的 220 Hz + 440 Hz + 880 Hz 混合音频。运行 DFT。确认在期望的区间处有三个峰值。
2. **中等。** 以 48 kHz 录制一段 3 秒的语音。使用 `torchaudio.transforms.Resample`（带抗混叠）下采样到 16 kHz，然后使用朴素抽取（每三个样本取一个）下采样到 16 kHz。对两者做 FFT。混叠出现在哪里？
3. **困难。** 仅使用 `math` 和第 3 步中的 DFT 从头构建 STFT。帧大小 400，步长 160，汉宁窗。使用 `matplotlib.pyplot.imshow` 绘制幅度图。这就是课程 02 中的频谱图。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 采样率 | 每秒的样本数 | ADC 测量信号的频率（Hz）。 |
| 奈奎斯特 | 你能表示的最高频率 | `sr/2`；高于它的能量会混叠回来。 |
| 位深度 | 每个样本的分辨率 | `int16` = 65,536 个层级；`float32` = `[-1, 1]` 中的 24 位精度。 |
| DFT | 序列的傅里叶变换 | `N` 个样本 → `N` 个复数频率系数。 |
| FFT | 快速 DFT | `O(N log N)` 算法，需要 `N` 为 2 的幂。 |
| 区间 | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 频谱图的底层实现 | 随时间分帧加窗的 FFT。 |
| 混叠 | 奇怪的频率幽灵 | 高于奈奎斯特的能量镜像到较低区间。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理背后的论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费、权威的 DSP 教科书。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 带代码的实用指南。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 为什么现实世界中的音频不是干净正弦波的参考。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10 分钟内理清频率区间直觉。
