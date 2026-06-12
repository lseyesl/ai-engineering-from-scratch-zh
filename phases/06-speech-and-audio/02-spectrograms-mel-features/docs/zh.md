# 频谱图、梅尔尺度与音频特征

> 神经网络不太擅长直接消费原始波形。它们消费频谱图。它们消费梅尔频谱图效果更好。2026 年的每一个 ASR、TTS 和音频分类器都取决于这一个预处理选择。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 6 · 01（音频基础）
**时间：** ~45 分钟

## 问题

取一段 10 秒的 16 kHz 音频。那就是 160,000 个浮点数，全部在 `[-1, 1]` 内，与"狗叫"或"猫这个词"的标签几乎完全不相关。原始波形包含信息，但形式让模型难以提取。两个相同的音素相隔 100 ms 说话，其原始样本完全不同。

频谱图解决了这个问题。它压缩了人类感知忽略的时间细节（微秒级抖动），同时保留了感知关注的结构（哪些频率在 ~10–25 ms 的时间窗口内具有能量）。

梅尔频谱图更进一步。人类对数感知音高：100 Hz vs 200 Hz 听起来"距离相同"于 1000 Hz vs 2000 Hz。梅尔尺度将频率轴扭曲以匹配这种感知。梅尔尺度的频谱图是 2010 年到 2026 年间语音机器学习中最重要的单一特征。

## 概念

![波形到 STFT 到梅尔频谱图到 MFCC 的阶梯](../assets/mel-features.svg)

**STFT（短时傅里叶变换）。** 将波形切片为重叠的帧（典型：25 ms 窗，10 ms 步长 = 16 kHz 下 400 个样本/160 个样本）。每帧乘以窗函数（汉宁窗是默认值；汉明窗有略微不同的权衡）。对每帧做 FFT。将幅度谱堆叠成一个形状为 `(n_frames, n_freq_bins)` 的矩阵。这就是你的频谱图。

**对数幅度。** 原始幅度跨越 5–6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围。每个生产流水线都使用对数幅度，而非原始幅度。

**梅尔尺度。** 频率 `f`（Hz）通过 `m = 2595 * log10(1 + f / 700)` 映射到梅尔 `m`。该映射在 1 kHz 以下大致呈线性，以上大致呈对数。80 个梅尔区间覆盖 0–8 kHz 是标准的 ASR 输入。

**梅尔滤波器组。** 一组在梅尔尺度上等间距的三角形滤波器。每个滤波器是相邻 FFT 区间的加权和。将 STFT 幅度乘以滤波器组矩阵即可通过一次矩阵乘法得到梅尔频谱图。

**对数梅尔频谱图。** `log(mel_spec + 1e-10)`。Whisper 的输入。Parakeet 的输入。SeamlessM4T 的输入。2026 年通用的音频前端。

**MFCC。** 取对数梅尔频谱图，应用 DCT（II 型），保留前 13 个系数。去相关特征并进一步压缩。在约 2015 年之前是主导特征，后来 CNN/Transformer 在原始对数梅尔频谱图上追赶上来。仍在说话人识别（x-vectors、ECAPA）中使用。

**分辨率权衡。** 更大的 FFT = 更好的频率分辨率，但更差的时间分辨率。25 ms / 10 ms 是音频-机器学习的默认值；50 ms / 12.5 ms 用于音乐；5 ms / 2 ms 用于瞬态检测（鼓点、爆破音）。

```figure
spectrogram-window
```

## 动手构建

### 第 1 步：对波形分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一个 10 秒 16 kHz 的音频，`frame_len=400, hop=160`，产生 998 帧。

### 第 2 步：汉宁窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在 FFT 之前逐元素相乘。消除因在非零点截断而产生的频谱泄漏。

### 第 3 步：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境使用 `torch.stft` 或 `librosa.stft`（基于 FFT，向量化）。这里的循环是教学性的；它在 `code/main.py` 中对短视频片段运行。

### 第 4 步：梅尔滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 个梅尔区间覆盖 0–8 kHz，`n_fft=400`，得到一个 `(80, 201)` 的矩阵。将 `(n_frames, 201)` 的 STFT 幅度乘以其转置得到 `(n_frames, 80)` 的梅尔频谱图。

### 第 5 步：对数梅尔

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见的替代方案：`librosa.power_to_db`（参考归一化的 dB）、`10 * log10(power + eps)`。Whisper 使用更复杂的剪切+归一化流程（参见 Whisper 的 `log_mel_spectrogram`）。

### 第 6 步：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个对数梅尔帧应用 DCT，保留前 13 个系数。这就是你的 MFCC 矩阵。第一个系数通常被丢弃（它编码了整体能量）。

## 使用它

2026 年的技术栈：

| 任务 | 特征 |
|------|------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 维对数梅尔，10 ms 步长，25 ms 窗 |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 维梅尔，5–12 ms 步长以获得精细时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 维对数梅尔，10 ms 步长 |
| 说话人嵌入（ECAPA-TDNN、WavLM） | 80 维对数梅尔或原始波形 SSL |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散 token（非梅尔） |
| 关键词检测 | 40 维 MFCC 用于小型设备 |

经验法则：**如果你处理的不是音乐，从 80 维对数梅尔开始**。任何偏离都需要举证。

## 2026 年仍在犯的陷阱

- **梅尔数量不匹配。** 训练用 80 个梅尔，推理用 128 个梅尔。静默失败。在两端记录特征形状。
- **上游采样率不匹配。** 在 22.05 kHz 下计算的梅尔与 16 kHz 下的不同。在特征化*之前*修复 SR。
- **dB 与 log。** Whisper 期望的是对数梅尔，而非 dB 梅尔。某些 HF 流水线会自动检测；你的自定义代码不会。
- **归一化偏移。** 训练时按话语归一化，推理时全局归一化。导致 WER 翻倍的生产 bug。
- **填充产生的泄漏。** 在音频末尾补零会在最后几帧产生平坦的频谱。对称填充或复制填充。

## 输出

保存为 `outputs/skill-feature-extractor.md`。该技能针对给定的模型目标选择特征类型、梅尔数量、帧/步长和归一化方式。

## 练习

1. **简单。** 运行 `code/main.py`。它合成一个啁啾（频率从 200 Hz 扫到 4000 Hz）并打印每帧的 argmax 梅尔区间。可选绘图并确认与扫频匹配。
2. **中等。** 使用 `n_mels` in `{40, 80, 128}` 和 `frame_len` in `{200, 400, 800}` 重新运行。测量整个时间轴上的尖锐峰值带宽。哪种组合最能解析啁啾？
3. **困难。** 实现 `power_to_db` 并比较 AudioMNIST 上一个小型 CNN 分类器在使用 (a) 原始对数梅尔、(b) `ref=max` 的 dB 梅尔、(c) MFCC-13 + delta + delta-delta 时的 ASR 准确率。报告 top-1 准确率。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 帧 | 一个切片 | 喂给一次 FFT 的 25 ms 波形块。 |
| 步长 | 跨度 | 连续帧之间的样本数；10 ms 是 ASR 默认值。 |
| 窗 | 汉宁/汉明函数 | 逐点乘数，将帧边缘逐渐减小到零。 |
| STFT | 频谱图生成器 | 分帧加窗的 FFT；产生时间×频率矩阵。 |
| 梅尔 | 扭曲的频率 | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| 滤波器组 | 矩阵 | 将 STFT 投影到梅尔区间的三角滤波器。 |
| 对数梅尔 | Whisper 的输入 | `log(mel_spec + eps)`；2026 年标准化。 |
| MFCC | 老派特征 | 对数梅尔的 DCT；13 个系数，去相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始的梅尔尺度。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 和 hop/window 的参考文档。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 模型的生产级流水线。
