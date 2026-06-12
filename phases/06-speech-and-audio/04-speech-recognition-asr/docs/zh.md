# 语音识别 (ASR) — CTC、RNN-T、注意力

> 语音识别是在每个时间步上进行音频分类，由一个懂英语和静音的序列模型连接在一起。CTC、RNN-T 和注意力是三种实现方式。选一种并理解其原理。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 6 · 02（频谱图与梅尔），阶段 5 · 08（文本 CNN 与 RNN），阶段 5 · 10（注意力）
**时间：** ~45 分钟

## 问题

你有一段 10 秒的 16 kHz 音频。你想要一个字符串："turn on the kitchen lights"。挑战在于结构性的：音频帧与字符之间不是一对一对齐的。单词"okay"可能需要 200 ms 或 1200 ms。静音将话语分割成片。某些音素比其他音素长。输出 token 的数量事先是未知的。

有三种公式化方法可以解决这个问题：

1. **CTC（联结主义时序分类）。** 逐帧输出 token 概率，包括一个特殊的 *blank*。解码时折叠重复并去掉 blank。非自回归，速度快。用于 wav2vec 2.0、MMS。
2. **RNN-T（循环神经网络转录器）。** 联合网络根据编码器帧和之前的 token 预测下一个 token。可流式传输。用于 Google 的设备端 ASR、NVIDIA Parakeet。
3. **注意力编码器-解码器。** 编码器将音频压缩为隐藏状态，解码器通过交叉注意力自回归生成 token。用于 Whisper、SeamlessM4T。

2026 年，LibriSpeech test-clean 上的 SOTA WER 为 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。差异很小；但是部署差异巨大。

## 概念

![三种 ASR 公式化：CTC、RNN-T、注意力编码器-解码器](../assets/asr-formulations.svg)

**CTC 直觉。** 让编码器输出 `T` 个帧级别的分布，覆盖 `V+1` 个 token（V 个字符 + blank）。对于长度为 `U < T` 的目标字符串 `y`，任何能折叠为 `y` 的帧对齐方式都算在内。CTC 损失对所有这样的对齐方式求和。推理：逐帧 argmax，折叠重复，去掉 blank。

优点：非自回归、可流式、零前视。缺点：*条件独立假设* — 每一帧的预测独立于其他帧，因此没有内部语言模型。通过外部 LM 使用束搜索或浅融合来修复。

**RNN-T 直觉。** 增加一个*预测器*网络来嵌入 token 历史，和一个*连接器*将预测器状态与编码器帧组合成 `V+1` 上的联合分布（`+1` 表示空/不发射）。显式建模了 CTC 忽略的条件依赖。可流式，因为每一步只依赖于过去的帧和过去的 token。

优点：可流式 + 内部 LM。缺点：训练更复杂且更消耗内存（3D 损失格）；RNN-T 损失核本身就是一整类库。

**注意力编码器-解码器。** 编码器（6-32 个 Transformer 层）处理对数梅尔帧。解码器（6-32 个 Transformer 层）通过交叉注意力关注编码器输出，自回归生成 token。没有对齐约束 — 注意力可以看向音频的任何位置。除非限制注意力（分块 Whisper-Streaming，2024），否则不可流式。

优点：离线 ASR 质量最高，使用标准的 seq2seq 工具容易训练。缺点：自回归延迟与输出长度成正比；不经过工程处理无法流式。

### WER：唯一需要关注的数字

**词错误率** = `(S + D + I) / N`，其中 S=替换、D=删除、I=插入、N=参考词数。匹配词级别的 Levenshtein 编辑距离。越低越好。WER 超过 20% 通常不可用；低于 5% 对于朗读语音来说是人工水平的。2026 年标准基准数据：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 参数量 |
|------|------------------------|------------------------|--------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

所有这些都基于编码器-解码器或 RNN-T。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约为 1.8–2.1%。

## 动手构建

### 第 1 步：贪心 CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: 逐帧概率向量列表
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两个规则：折叠连续重复，去掉 blank。示例：`a a _ _ a b b _ c` → `a a b c`。

### 第 2 步：束搜索 CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境使用带 LM 融合的前缀树束搜索；这里是概念骨架。

### 第 3 步：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 第 4 步：使用 Whisper 推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强通用 ASR 的一行代码。在 24 GB GPU 上以约 20 倍实时速度运行。

### 第 5 步：使用 Parakeet 或 wav2vec 2.0 流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器注意力和结转状态；使用支持此功能的库（NeMo for Parakeet、带 `chunk_length_s` 的 `transformers` 流水线）。

## 使用它

2026 年的技术栈：

| 情况 | 选择 |
|------|------|
| 英语、离线、最高质量 | Whisper-large-v3-turbo |
| 多语言、鲁棒 | SeamlessM4T v2 |
| 流式、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘设备、移动端、<500 ms 延迟 | 量化后的 Whisper-Tiny 或 Moonshine (2024) |
| 长音频 | 带基于 VAD 的分块的 Whisper（WhisperX） |
| 特定领域（医疗、法律） | 微调 wav2vec 2.0 + 领域 LM 融合 |

## 2026 年仍在犯的陷阱

- **没有 VAD。** 在静音上运行 Whisper 会产生幻觉（"Thanks for watching!"）。始终用 VAD 做门控。
- **字符级 vs 词级 vs 子词 WER。** 报告归一化*后*的词级 WER（小写、去掉标点）。
- **语言识别偏移。** Whisper 的自动 LID 会将嘈杂的音频错误路由到日语或威尔士语；确定时强制指定 `language="en"`。
- **长音频未分块。** Whisper 有 30 秒的窗口。对任何更长的内容使用 `chunk_length_s=30, stride=5`。

## 输出

保存为 `outputs/skill-asr-picker.md`。针对给定的部署目标选择模型、解码策略、分块方式和 LM 融合。

## 练习

1. **简单。** 运行 `code/main.py`。它贪心地解码一个手工制作的 CTC 输出并计算与参考文本的 WER。
2. **中等。** 正确实现第 2 步中的前缀树束搜索（考虑 blank 合并规则）。在 10 个示例的合成数据集上与贪心解码比较。
3. **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`。计算前 100 段话语的 WER。与已发布的数据比较。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| CTC | blank token 损失 | 对所有帧到 token 对齐方式的边缘求和；非自回归。 |
| RNN-T | 流式损失 | CTC + 下一个 token 预测器；处理词序。 |
| 注意力 enc-dec | Whisper 风格 | 编码器 + 交叉注意力解码器；最佳离线质量。 |
| WER | 你报告的数字 | 词级别的 `(S+D+I)/N`。 |
| Blank | 空 | CTC 中表示"此帧不发射"的特殊 token。 |
| LM 融合 | 外部语言模型 | 在束搜索期间添加加权的 LM 对数概率。 |
| VAD | 静音门控 | 语音活动检测器；裁剪非语音部分。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年权威论文；2024 年 v3-turbo 扩展。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 年开放 ASR 排行榜领跑者。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 25+ 模型的实时基准测试。
