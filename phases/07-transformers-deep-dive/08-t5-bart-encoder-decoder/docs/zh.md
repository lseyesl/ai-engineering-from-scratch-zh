# T5、BART — 编码器-解码器模型

> 编码器理解。解码器生成。把它们放回一起，你得到一个为输入 → 输出任务构建的模型：翻译、摘要、重写、转录。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 7 · 05（完整 Transformer），阶段 7 · 06（BERT），阶段 7 · 07（GPT）
**时间：** ~45 分钟

## 问题

仅解码器的 GPT 和仅编码器的 BERT 各自为了不同目标对 2017 年的架构进行了精简。但许多任务天然是输入输出的：

- 翻译：英语 → 法语。
- 摘要：5,000 token 文章 → 200 token 摘要。
- 语音识别：音频 token → 文本 token。
- 结构化提取：散文 → JSON。

对于这些任务，编码器-解码器是最干净的选择。编码器产生源序列的稠密表示。解码器生成输出，在每一步交叉注意力到该表示。训练是在输出侧的偏移一位损失。与 GPT 相同的损失，只是以编码器输出为条件。

两篇论文定义了现代方案：

1. **T5**（Raffel 等人，2019）。"Text-to-Text Transfer Transformer。"每个 NLP 任务被重新定义为文本输入、文本输出。单一架构、单一词表、单一损失。在掩码跨度预测上预训练（在输入中损坏跨度，在输出中解码它们）。
2. **BART**（Lewis 等人，2019）。"Bidirectional and Auto-Regressive Transformer。"去噪自编码器：以多种方式损坏输入（打乱、掩码、删除、旋转），要求解码器重建原始内容。

在 2026 年，编码器-解码器格式在输入结构重要的场景中继续存在：

- Whisper（语音 → 文本）。
- 谷歌的翻译堆栈。
- 一些具有独特上下文和编辑结构的代码补全/修复模型。
- Flan-T5 及其变体用于结构化推理任务。

仅解码器赢得了聚光灯，但编码器-解码器从未消失。

## 概念

![带有交叉注意力的编码器-解码器](../assets/encoder-decoder.svg)

### 前向循环

```
源 token ─▶ 编码器 ─▶ (N_src, d_model)  ──┐
                                             │
目标 token ─▶ 解码器模块                     │
              ├─▶ 掩码自注意力                │
              ├─▶ 交叉注意力 ◀───────────────┘
              └─▶ FFN
             ↓
          下一个 token logits
```

关键的是，编码器对每个输入运行一次。解码器自回归运行，但在每一步都要交叉注意力到*相同*的编码器输出。缓存编码器输出对于长输入是一种免费的加速。

### T5 预训练 — 跨度损坏

选择输入的随机跨度（平均长度 3 个 token，总计 15%）。用唯一的哨兵 token 替换每个跨度：`<extra_id_0>`、`<extra_id_1>` 等。解码器仅输出带有哨兵前缀的被损坏跨度：

```
源：    The quick <extra_id_0> fox jumps <extra_id_1> dog
目标：  <extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列更廉价的信号。在 T5 论文的消融实验中，与 MLM（BERT）和前缀-LM（UniLM）相比具有竞争力。

### BART 预训练 — 多噪声去噪

BART 尝试了五种噪声函数：

1. Token 掩码。
2. Token 删除。
3. 文本填充（掩码一个跨度，解码器插入正确长度）。
4. 句子排列。
5. 文档旋转。

文本填充 + 句子排列的组合产生了最好的下游数据。解码器总是重建原始内容。BART 的输出是整个序列，而不仅仅是损坏的跨度——所以预训练计算量高于 T5。

### 推理

与 GPT 相同的自回归生成。贪心 / 束 / top-p 采样都适用。束搜索（宽度 4-5）是翻译和摘要的标准，因为输出分布比聊天更窄。

### 2026 年何时选择每种变体

| 任务 | 编码器-解码器？ | 为什么 |
|------|------------------|-----|
| 翻译 | 是，通常如此 | 清晰的源序列；固定的输出分布；束搜索有效 |
| 语音转文本 | 是（Whisper） | 输入模态与输出不同；编码器塑造音频特征 |
| 聊天 / 推理 | 否，仅解码器 | 没有持久的"输入"——对话本身就是序列 |
| 代码补全 | 通常否 | 仅解码器通过长上下文胜出；像 Qwen 2.5 Coder 这样的代码模型是仅解码器的 |
| 摘要 | 两者均可 | BART、PEGASUS 超过了早期的仅解码器基线；现代仅解码器 LLM 可以匹配它们 |
| 结构化提取 | 两者均可 | T5 很干净，因为"text → text"吸收了任何输出格式 |

自 2022 年以来的趋势：仅解码器接管了编码器-解码器曾经拥有的任务，因为 (a) 指令微调的仅解码器 LLM 通过提示泛化到任何任务，(b) 一种架构比两种更容易扩展，(c) RLHF 假设解码器。编码器-解码器在输入模态不同的情况下（语音、图像）或束搜索质量重要的情况下仍然存在。

## 动手实现

参见 `code/main.py`。我们为玩具语料库实现 T5 风格的跨度损坏——这是本课中最有用的单一部分，因为此后每个编码器-解码器预训练方案中都有它。

### 步骤 1：跨度损坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """选择总和约 mask_rate token 的跨度。返回 (corrupted_input, target)。"""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式是 T5 约定：`<sent0> span0 <sent1> span1 ...`。损坏的输入将未更改的 token 与跨度位置的哨兵 token 交替。

### 步骤 2：验证往返

给定损坏的输入和目标，重建原始句子。如果你的损坏是可逆的，前向传播就是良好定义的。这是一个健全性检查——真实训练从不这样做，但这个测试成本低廉，并能捕获跨度记账中的差一错误。

### 步骤 3：BART 加噪

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并显示结果。

## 使用

HuggingFace 参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的技巧：任务名称进入输入文本。同一个模型处理数十个任务，因为每个任务都是文本输入、文本输出。2026 年，这种模式已被指令微调的仅解码器模型泛化，但 T5 首先将其规范化。

## 产出

参见 `outputs/skill-seq2seq-picker.md`。该技能根据输入输出结构、延迟和质量目标，为新任务在编码器-解码器和仅解码器之间进行选择。

## 练习

1. **简单。** 运行 `code/main.py`，将跨度损坏应用于一个 30-token 句子，验证将非哨兵源 token 与解码后的目标跨度拼接起来是否能重现原始内容。
2. **中等。** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` token 替换随机跨度，解码器必须推断出正确的跨度长度和内容。展示一个示例。
3. **困难。** 在小型英语→猪拉丁语语料库（200 对）上微调 `flan-t5-small`。在保留的 50 对测试集上测量 BLEU。与在相同数据上使用相同计算量微调 `Llama-3.2-1B` 进行比较。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 编码器-解码器 | "Seq2seq transformer" | 两个堆栈：用于输入的双向编码器，用于输出的带交叉注意力的因果解码器。 |
| 交叉注意力（Cross-attention） | "源与目标对话之处" | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一位置。 |
| 跨度损坏（Span corruption） | "T5 的预训练技巧" | 用哨兵 token 替换随机跨度；解码器输出这些跨度。 |
| 去噪目标（Denoising objective） | "BART 的游戏" | 对输入应用噪声函数，训练解码器重建干净的序列。 |
| 哨兵 token（Sentinel token） | "`<extra_id_N>` 占位符" | 在源中标记损坏跨度并在目标中重新标记的特殊 token。 |
| Flan | "指令微调的 T5" | 在 1,800+ 任务上微调的 T5；使编码器-解码器在指令遵循方面具有竞争力。 |
| 束搜索（Beam search） | "解码策略" | 在每一步保留 top-k 部分序列；翻译/摘要的标准方法。 |
| 教师强制（Teacher forcing） | "训练时输入" | 在训练期间，向解码器输入真实的前一个输出 token，而不是采样的 token。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年编码器-解码器的典范。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。
