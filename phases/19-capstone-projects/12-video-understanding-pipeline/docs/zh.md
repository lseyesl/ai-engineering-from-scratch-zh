# 顶点项目 12——视频理解管道（场景、问答、搜索）

> Twelve Labs 产品化了 Marengo + Pegasus。VideoDB 发布了视频 CRUD API。AI2 的 Molmo 2 发布了开源 VLM 检查点。Gemini 长上下文原生处理数小时的视频。TimeLens-100K 定义了大规模时间定位。2026 年的管道已经定型：场景分割、每场景标题 + 嵌入、转录对齐、多向量索引，以及用（开始、结束）时间戳加帧预览回答的查询。顶点项目是摄取 100 小时视频，在公共基准上达标，并测量计数和动作类问题上的幻觉。

**类型:** Capstone
**语言:** Python（管道）、TypeScript (UI)
**前置要求:** Phase 4（CV）、Phase 6（语音）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）
**涉及阶段:** P4 · P6 · P7 · P11 · P12 · P17
**时间:** 30 小时

## 问题

长视频问答是 2026 年规模上最消耗带宽的多模态问题。Gemini 2.5 Pro 可以原生读取 2 小时视频，但将 100 小时视频摄取到可查询的语料库中仍需要场景级索引。生产形态结合了场景分割（TransNetV2 或 PySceneDetect）、用 VLM 进行的每场景标题生成（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、转录对齐（带单词时间戳的 Whisper-v3-turbo）以及并排存储标题、帧嵌入和转录的多向量索引。查询管道用（开始、结束）时间戳加帧预览回答。

基准是公开的（ActivityNet-QA、NeXT-GQA）加上你自己的 100 个查询自定义集。计数和动作类型问题上的幻觉是已知的困难故障类别；顶点项目显式测量它。

## 概念

三个管道在摄取时并行运行。**场景分割**将视频切割成场景。**VLM 标题生成**为每个场景生成标题和关键帧的帧嵌入。**ASR 对齐**产生单词级时间戳。三个流按（scene_id、时间范围）连接。每个场景在多向量索引（Qdrant）中获得三种向量类型：标题嵌入、关键帧嵌入、转录嵌入。

在查询时，自然语言问题针对所有三种向量触发；结果用 RRF 合并；时间定位适配器（TimeLens 风格）在顶部场景内细化（开始、结束）窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收查询 + 顶部场景 + 裁剪帧，并用引用的时间戳和帧预览回答。

幻觉测量很重要。计数（"有多少人进入房间？"）和动作类型（"厨师在搅拌前先倒了吗？"）问题以不可靠著称。请与描述性问题分开报告准确性。

## 架构

```
视频文件 / URL
      |
      v
PySceneDetect / TransNetV2  (场景分割)
      |
      +--- 每场景关键帧 --- VLM 标题 + 帧嵌入
      |                      (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- 音频通道 --- Whisper-v3-turbo ASR + 单词时间戳
      |
      v
多向量 Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
查询:
  针对所有三个的稠密查询 -> RRF 合并 -> top-k 场景
      |
      v
TimeLens / VideoITG 时间定位 (在场景内细化开始/结束)
      |
      v
VLM 合成: 查询 + top 场景 + 帧预览
      |
      v
答案 + (开始, 结束) 时间戳 + 帧缩略图 + 引用
```

## 技术栈

- 场景分割：TransNetV2（2024-26 年最先进）或 PySceneDetect
- ASR：Whisper-v3-turbo，通过 faster-whisper，带单词时间戳
- VLM 标题生成器 + 回答器：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时间定位：TimeLens-100K 训练的适配器或 VideoITG
- 索引：Qdrant，多向量支持（标题/帧/转录）
- UI：Next.js 15，带 HTML5 视频播放器和场景缩略图
- 评估：ActivityNet-QA、NeXT-GQA、自定义 100 个查询的手工标记集
- 幻觉基准：计数和动作类型子集，带手工标签

## 构建它

1. **摄取遍历器。** 接受 YouTube URL 或本地 MP4。如果需要，降采样到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect 生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k-8k 场景。

3. **ASR 扫描。** 在音频上运行 Whisper-v3-turbo；导出单词级时间戳；分割成每场景转录切片。

4. **VLM 标题生成。** 每场景，用关键帧和简短标题模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。生成标题 + 帧嵌入。

5. **多向量索引。** Qdrant 集合，三种命名向量。负载：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题触发三个稠密查询；用倒数排名融合合并；top-k=5 场景。

7. **时间定位。** 在顶部场景上运行 TimeLens 风格适配器，以细化场景内的（开始、结束）窗口。

8. **VLM 合成。** 调用 Gemini 2.5 Pro，输入查询 + top-3 场景剪辑（作为图像或短视频剪辑）+ 转录。要求 `(video_id, start_ms, end_ms)` 引用。

9. **评估。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100 个查询的自定义集。报告总体准确性 + 按类别细分（计数、动作、描述性）。

## 使用它

```
$ video-qa ask --url=https://youtube.com/watch?v=X "第一分钟有多少辆车通过路口？"
[scene]    检测到 23 个场景
[asr]      转录完成, 4m12s
[index]    写入 69 个向量 (23 场景 x 3)
[query]    顶部场景: scene 3 [01:32-01:54], 置信度 0.84
[ground]   细化窗口: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 辆车在 00:12 至 00:58 间通过路口。
citations: [scene 3: 00:12-00:58]
          [帧预览 at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付物

`outputs/skill-video-qa.md` 是交付物。给定一个 YouTube URL 或上传的视频，管道索引场景并用带时间戳的引用回答问题。

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | 时间定位 IoU | 在留出定位集上的交并比 |
| 20 | QA 准确性 | NeXT-GQA 和自定义 100 查询 |
| 20 | 摄取吞吐量 | 每美元花费的小时视频数 |
| 20 | UI 和引用 UX | 时间戳链接、缩略图条、跳到帧 |
| 15 | 幻觉率 | 计数和动作类型准确性单独报告 |

## 练习

1. 在标题生成扫描上将 Gemini 2.5 Pro 替换为 Qwen3-VL-Max。在人工评级的 50 场景样本上报告标题质量差异。

2. 将每场景帧嵌入减少为一个池化向量而非多向量。测量检索回归。

3. 构建"严格计数"模式：合成器提取每个带时间戳的计数实例，用户可以点击验证。测量用户验证是否减少幻觉。

4. 基准测试摄取成本：三种 VLM 选择的每美元小时视频数。找到最佳点。

5. 添加说话人分离转录：在音频上运行 pyannote 说话人二值化并嵌入每说话人转录。演示"关于 X，Alice 说了什么？"查询。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Scene segmentation | "镜头检测" | 在镜头边界处将视频切割成场景 |
| Multi-vector index | "标题 + 帧 + 转录" | 每表示带有命名向量的 Qdrant 集合 |
| Temporal grounding | "它究竟何时发生" | 为查询答案细化（开始、结束）窗口 |
| Frame embedding | "视觉表示" | 关键帧的向量嵌入；用于场景视觉相似性 |
| RRF fusion | "倒数排名融合" | 跨多个排名列表的合并策略；经典的混合检索技巧 |
| Counting hallucination | "数错" | VLM 在"多少个 X"问题上的已知故障模式 |
| ActivityNet-QA | "视频 QA 基准" | 长视频 QA 准确性基准 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2)——开源 VLM 检查点
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens)——大规模时间定位
- [Gemini 视频长上下文](https://deepmind.google/technologies/gemini)——托管参考
- [VideoDB](https://videodb.io)——视频 CRUD API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io)——商业参考
- [TransNetV2](https://github.com/soCzech/TransNetV2)——场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)——经典开源替代
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467)——参考评估基准
