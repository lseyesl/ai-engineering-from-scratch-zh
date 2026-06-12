# 音频分类 — 从 MFCC 上的 k-NN 到 AST 和 BEATs

> 从"狗叫 vs 警笛"到"这是什么语言"，一切都是音频分类。特征是梅尔。架构每十年换一次。评估指标始终是 AUC、F1 和各类别召回率。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 6 · 02（频谱图与梅尔），阶段 3 · 06（CNN），阶段 5 · 08（文本 CNN 与 RNN）
**时间：** ~75 分钟

## 问题

你得到一段 10 秒的音频。你想知道："这是什么？"城市声音（警笛、电钻、狗叫）、语音指令（是/否/停止）、语言识别（英语/西班牙语/阿拉伯语）、说话人情绪（愤怒/中性）或环境声音（室内/室外、嘈杂声）。所有这些都属于*音频分类*，到 2026 年基线架构已经成熟：对数梅尔 → CNN 或 Transformer → softmax。

核心困难不在于网络。在于数据。音频数据集存在严重的类别不平衡、强烈的领域偏移（干净 vs 嘈杂）和标签噪音（谁来决定"城市嘈杂声"vs"餐厅噪音"？）。80% 的问题是数据管理、数据增强和评估，而不是把 CNN 换成 Transformer。

## 概念

![音频分类阶梯：从 MFCC 上的 k-NN 到 AST 到 BEATs](../assets/audio-classification.svg)

**MFCC 上的 k-NN（1990 年代基线）。** 对每个音频展平 MFCC，与标注库计算余弦相似度，返回前 K 个的多数投票。在干净的小数据集（Speech Commands、ESC-50）上出奇地强大。无需 GPU 即可运行。

**对数梅尔上的 2D CNN（2015-2019）。** 将 `(T, n_mels)` 的对数梅尔视为图像。应用 ResNet-18 或 VGG 风格。对时间轴做全局均值池化。类别上的 softmax。仍然是大多数 2026 年 Kaggle 比赛的基线。

**音频频谱图 Transformer、AST（2021-2024）。** 将对数梅尔分块（例如 16×16 块），加上位置嵌入，送入 ViT。在 AudioSet 上达到监督学习的最高水平（mAP 0.485）。

**BEATs 和 WavLM-base（2024-2026）。** 在数百万小时数据上的自监督预训练。使用你原本需要的 1-10% 的有监督数据微调到你的任务上。2026 年，这是非语音音频的默认起点。BEATs-iter3 在 AudioSet 上超过 AST 1-2 mAP，同时只用四分之一的算力。

**Whisper 编码器作为冻结的骨干网络（2024）。** 使用 Whisper 的编码器，去掉解码器，附加一个线性分类器。在语言识别和简单事件分类上接近 SOTA，且无需音频增强。这是"免费午餐"基线。

### 类别不平衡是真正的挑战

ESC-50：50 个类别，每类 40 个样本 — 平衡，简单。UrbanSound8K：10 个类别，不平衡比例 10:1。AudioSet：632 个类别，长尾分布 100,000:1。有效的方法：

- 训练时使用平衡采样（评估时不用）。
- Mixup：线性插值两个片段（及其标签）作为增强。
- SpecAugment：掩盖随机的时间和频率区间。简单；至关重要。

### 评估

- 多类互斥（Speech Commands）：top-1 准确率、top-5 准确率。
- 多类多标签（AudioSet、UrbanSound 风格）：均值平均精度 (mAP)。
- 高度不平衡：各类别召回率 + 宏平均 F1。

你应该知道的 2026 年数据：

| 基准 | 基线 | 2026 SOTA | 来源 |
|------|------|-----------|------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs 论文 (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR 排行榜 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 结果 |

## 动手构建

### 第 1 步：特征化

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 第 2 步：固定长度摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但强大：对时间轴的均值+方差给出一个 26 维的固定嵌入（针对 13 系数的 MFCC）。即时运行。到 2017 年还在 ESC-50 上击败了最先进的 NN 基线。

### 第 3 步：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 第 4 步：升级到对数梅尔上的 CNN

在 PyTorch 中：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300 万个参数。在单张 RTX 4090 上，ESC-50 训练约 10 分钟。80%+ 准确率。

### 第 5 步：2026 年默认方案 — 微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于 BEATs，使用 `microsoft/BEATs-base` 通过 `beats` 库；transformers API 形状相同。

## 使用它

2026 年的技术栈：

| 情况 | 从什么开始 |
|------|-----------|
| 小数据集（<1000 片段） | 基于 MFCC 均值的 k-NN（你的基线）+ 音频增强 |
| 中等数据集（1K–100K） | BEATs 或 AST 微调 |
| 大数据集（>100K） | 从头训练或微调 Whisper 编码器 |
| 实时、边缘设备 | 40-MFCC CNN，量化为 int8（KWS 风格） |
| 多标签（AudioSet） | 带 BCE 损失的 BEATs-iter3 + mixup + SpecAugment |
| 语言识别 | MMS-LID、SpeechBrain VoxLingua107 基线 |

决策法则：**从冻结的骨干网络开始，而不是全新模型**。微调 BEATs 头部可以在数小时内获得 95% 的 SOTA 性能，而不是数周。

## 输出

保存为 `outputs/skill-classifier-designer.md`。针对给定的音频分类任务选择架构、增强策略、类别平衡策略和评估指标。

## 练习

1. **简单。** 运行 `code/main.py`。它在 4 类合成数据集（不同音高的纯音）上训练 k-NN MFCC 基线。报告混淆矩阵。
2. **中等。** 将 `summarize` 替换为 [均值、方差、偏度、峰度]。在相同的合成数据集上，4 矩池化是否优于均值+方差？
3. **困难。** 使用 `torchaudio`，在 ESC-50 的 fold 1 上训练 2D CNN。报告 5 折交叉验证准确率。添加 SpecAugment（时间掩码 = 20，频率掩码 = 10）并报告差值。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| AudioSet | 音频领域的 ImageNet | Google 的 200 万片段、632 类弱标注 YouTube 数据集。 |
| ESC-50 | 小型分类基准 | 50 类 × 40 段环境音片段。 |
| AST | 音频频谱图 Transformer | 对数梅尔 patch 上的 ViT；2021 年 SOTA。 |
| BEATs | 自监督音频 | 微软模型，iter3 截至 2026 年领跑 AudioSet。 |
| Mixup | 配对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于掩码的增强 | 将频谱图中随机的时间和频率区间归零。 |
| mAP | 主要多标签指标 | 跨类别和阈值的均值平均精度。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021–2024 年的经典架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024+ 的默认选择。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主导性的音频增强方法。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 50 类基准，生命力持久。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632 类 YouTube 分类体系；仍然是黄金标准。
