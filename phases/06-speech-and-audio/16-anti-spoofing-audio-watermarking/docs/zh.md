# 语音反欺骗与音频水印 — ASVspoof 5、AudioSeal、WaveVerify

> 语音克隆的发布速度超过了防御措施。2026 年的生产语音系统需要两样东西：一个检测器（AASIST、RawNet2）来分类真实 vs 伪造语音，和一个水印（AudioSeal）能抵抗压缩和编辑。两者都提供，否则不要发布语音克隆。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 6 · 06（说话人识别），阶段 6 · 08（语音克隆）
**时间：** ~75 分钟

## 问题

三种相关的防御措施：

1. **反欺骗 / 深度伪造检测。** 给定一段音频片段，它是合成的还是真实的？ASVspoof 基准（ASVspoof 2019 → 2021 → 5）是黄金标准。
2. **音频水印。** 在生成的音频中嵌入一个不可感知的信号，检测器可以后来提取。AudioSeal（Meta）和 WavMark 是开源选择。
3. **认证溯源。** 音频文件 + 元数据的加密签名。C2PA / Content Authenticity Initiative。

检测处理不合作的对手。水印处理合规性 — AI 生成的音频应能被识别为 AI 生成。2026 年两者都是必需的。

## 概念

![反欺骗 vs 水印 vs 溯源 — 三层防御](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025 年基准

与前几版的最大变化：

- **众包数据**（非录音室清洁）— 现实条件。
- **约 2000 个说话人**（之前约 100 个）。
- **32 种攻击算法。** TTS + 语音转换 + 对抗扰动。
- **两个赛道。** 反制措施（CM）独立检测；欺骗鲁棒 ASV（SASV）用于生物识别系统。

ASVspoof 5 上的 SOTA：约 7.23% EER。在较早的 ASVspoof 2019 LA 上：0.42% EER。现实世界部署：在野外片段上预期 5-10% EER。

### AASIST 和 RawNet2 — 检测模型家族

**AASIST**（2021，到 2026 年持续更新）。对频谱特征的图注意力。当前 ASVspoof 5 反制措施任务的 SOTA。

**RawNet2。** 原始波形上的卷积前端 + TDNN 骨干网络。更简单的基线；微调后仍有竞争力。

**NeXt-TDNN + SSL 特征。** 2025 变体：ECAPA 风格 + WavLM 特征 + 焦点损失。在 ASVspoof 2019 LA 上实现 0.42% EER。

### AudioSeal — 2024 年水印默认方案

Meta 的 **AudioSeal**（2024 年 1 月，v0.2 2024 年 12 月）。关键设计：

- **局部化。** 在 16 kHz 样本分辨率（1/16000 秒）下逐帧检测水印。
- **生成器 + 检测器联合训练。** 生成器学习嵌入不可听信号；检测器学习通过增强发现它。
- **鲁棒。** 能抵抗 MP3 / AAC 压缩、EQ、速度偏移 ±10%、噪声混合 +10 dB SNR。
- **快速。** 检测器以 485 倍实时速度运行；比 WavMark 快 1000 倍。
- **容量。** 16 位负载（可在每段话语中编码模型 ID、生成时间戳、用户 ID）。

### WavMark

AudioSeal 之前的开源基线。可逆神经网络，32 比特/秒。问题：

- 同步暴力破解慢。
- 可被高斯噪声或 MP3 压缩移除。
- 不实时友好。

### WaveVerify（2025 年 7 月）

解决 AudioSeal 的弱点 — 特别是时序操作（反转、变速）。使用基于 FiLM 的生成器 + 专家混合检测器。在标准攻击上与 AudioSeal 竞争；处理时序编辑。

### 对手利用的空白

来自 AudioMarkBench："在音高偏移下，所有水印的比特恢复准确率低于 0.6，表明几乎完全移除。"**音高偏移是通用攻击。** 没有 2026 年的水印对激进音高修改完全鲁棒。这就是为什么你需要检测（AASIST）与水印并存。

### C2PA / Content Authenticity Initiative

不是 ML 技术 — 是一个清单格式。音频文件携带关于创建工具、作者、日期的加密签名元数据。Audobox / Seamless 使用。对溯源有好处；但如果恶意行为者重新编码并剥离元数据则无效。

## 动手构建

### 第 1 步：简单的频谱特征检测器（玩具）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音通常具有异常平坦的高频能量。生产检测器使用 AASIST，而不是这个。但直觉成立。

### 第 2 步：AudioSeal 嵌入 + 检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: [0, 1] 中的浮点数 — 水印存在的概率
# decoded_payload: 16 位；与嵌入的负载匹配
```

### 第 3 步：评估 — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 第 4 步：生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每个生成内容包含：(1) 水印、(2) 签名清单、(3) 符合保留策略的审计日志。

## 使用它

| 用例 | 防御措施 |
|------|---------|
| 发布 TTS / 语音克隆 | 每个输出都嵌入 AudioSeal（不可协商） |
| 生物识别语音解锁 | AASIST + ECAPA 集成；活体挑战 |
| 呼叫中心欺诈检测 | 对 20% 的来电样本运行 AASIST |
| 播客真实性 | 上传时 C2PA 签名，如为 AI 生成则附加 AudioSeal |
| 研究 / 训练检测器 | ASVspoof 5 训练/开发/评估集 |

## 陷阱

- **有水印但从没运行过检测器。** 毫无意义。在 CI 中部署检测器。
- **无校准的检测。** 在 ASVspoof LA 上训练的 AASIST 会过拟合；现实世界准确率下降。在你的领域上校准。
- **音高偏移空白。** 激进的音高偏移会移除大多数水印。有检测回退方案。
- **元数据剥离和重新托管。** C2PA 可以通过重新编码轻易绕过。始终同时使用加密 + 感知（水印）防御。
- **将活体检测作为检测手段。** 让用户说一个随机短语。防止重放攻击但无法对付实时克隆。

## 输出

保存为 `outputs/skill-spoof-defender.md`。为语音生成部署选择检测模型、水印、溯源清单和操作手册。

## 练习

1. **简单。** 运行 `code/main.py`。在合成音频上运行玩具检测器 + 玩具水印嵌入/检测。
2. **中等。** 安装 `audioseal`，在 TTS 输出中嵌入 16 位负载，重新解码。用噪声破坏音频并测量比特恢复准确率。
3. **困难。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量 EER。在 F5-TTS 生成的保留测试集上测试 — 观察 OOD 检测如何退化。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| ASVspoof | 基准测试 | 两年一度挑战；2024 = ASVspoof 5。 |
| CM（反制措施） | 检测器 | 分类器：真实语音 vs 合成/转换。 |
| SASV | 说话人验证 + CM | 集成生物识别 + 欺骗检测。 |
| AudioSeal | Meta 水印 | 局部化，16 位负载，比 WavMark 快 485 倍。 |
| 比特恢复准确率 | 水印存活率 | 攻击后恢复的负载比特占比。 |
| C2PA | 溯源清单 | 关于创建/作者身份的加密元数据。 |
| AASIST | 检测器家族 | 基于图注意力的反欺骗 SOTA。 |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前基准。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 水印默认方案。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 用于时序攻击的 MoE 检测器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA 检测骨干网络。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 鲁棒性评估。
- [C2PA specification](https://c2pa.org/specifications/specifications/) — 溯源清单格式。
