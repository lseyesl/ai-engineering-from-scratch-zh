# BERT — 掩码语言建模

> GPT 预测下一个词。BERT 预测缺失的词。一句话的差异——带来了长达五年以嵌入为核心的一切。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 7 · 05（完整 Transformer），阶段 5 · 02（文本表示）
**时间：** ~45 分钟

## 问题

在 2018 年，每个 NLP 任务——情感分析、NER、问答、蕴含——都在自己的标注数据上从头训练自己的模型。没有一个预训练好的"理解英语"的检查点可以微调。ELMo（2018）展示了你可以用双向 LSTM 预训练上下文嵌入；这有所帮助，但没有泛化。

BERT（Devlin 等人，2018）提出：如果我们拿一个 transformer 编码器，在互联网上的每个句子上训练它，并迫使它从两侧的上下文中预测缺失的词，会怎样？然后你在下游任务上微调一个头部。参数效率是一个启示。

结果：在 18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）统治了当时存在的每个 NLP 排行榜。到 2020 年，地球上每个搜索引擎、内容审核系统和语义搜索系统内部都有一个 BERT。

在 2026 年，仅编码器模型仍然是分类、检索和结构化提取的正确工具——它们每 token 比解码器快 5-10 倍，它们的嵌入是每个现代检索堆栈的骨干。ModernBERT（2024 年 12 月）将架构推进到 8K 上下文，使用 Flash Attention + RoPE + GeGLU。

## 概念

![掩码语言建模：选择 token，掩码它们，预测原始值](../assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机掩码 15% 的 token：

```
输入：  the [MASK] brown fox jumps [MASK] the lazy dog
目标：  the  quick brown fox jumps  over  the lazy dog
```

训练模型在掩码位置预测原始 token。因为编码器是双向的，在位置 1 预测 `[MASK]` 可以使用位置 2+ 的 `brown fox jumps`。这就是 GPT 做不到的事情。

### BERT 掩码规则

在被选中进行预测的 15% 的 token 中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为随机 token。
- 10% 保持不变。

为什么不总是 `[MASK]`？因为 `[MASK]` 在推理时从不出现。训练模型在 100% 的掩码位置期望 `[MASK]` 会在预训练和微调之间造成分布偏移。10% 随机 + 10% 保持不变让模型保持诚实。

### 下一句预测（NSP）——以及为什么被抛弃

原始 BERT 还在 NSP 上训练：给定两个句子 A 和 B，预测 B 是否跟在 A 后面。RoBERTa（2019）消融实验表明 NSP 有害无益。现代编码器跳过了它。

### 2026 年的变化：ModernBERT

2024 年的 ModernBERT 论文用 2026 年的原语重建了模块：

| 组件 | 原始 BERT（2018） | ModernBERT（2024） |
|-----------|----------------------|-------------------|
| 位置 | 学习的绝对位置 | RoPE |
| 激活 | GELU | GeGLU |
| 归一化 | LayerNorm | 预归一化 RMSNorm |
| 注意力 | 完整稠密 | 交替局部（128）+ 全局 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

与 2018 年的堆栈不同，它是原生支持 Flash Attention 的。在序列长度 8K 时，推理速度比 DeBERTa-v3 快 2-3 倍，且 GLUE 分数更高。

### 2026 年仍选择编码器的用例

| 任务 | 为什么编码器优于解码器 |
|------|---------------------------|
| 检索 / 语义搜索嵌入 | 双向上下文 = 每 token 更好的嵌入质量 |
| 分类（情感、意图、毒性） | 一次前向传播；无需生成开销 |
| NER / token 标注 | 逐位置输出，原生双向 |
| 零样本蕴含（NLI） | 编码器顶部的分类器头 |
| RAG 的重排序器 | 交叉编码器评分，比 LLM 重排序器快 10 倍 |

```figure
transformer-residual
```

## 动手实现

### 步骤 1：掩码逻辑

参见 `code/main.py`。函数 `create_mlm_batch` 接受 token ID 列表、词表大小和掩码概率。返回输入 ID（应用了掩码）和标签（仅在掩码位置，其他地方为 -100——PyTorch 的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # 否则：保持原样
    return input_ids, labels
```

### 步骤 2：在小型语料库上运行 MLM 预测

在包含 20 个词、200 个句子的词表上训练一个 2 层编码器 + MLM 头。没有梯度——我们做前向传播的健全性检查。完整训练需要 PyTorch。

### 步骤 3：比较掩码类型

展示三路规则如何让模型在没有 `[MASK]` 的情况下可用。在未掩码的句子和掩码的句子上进行预测。两者都应产生合理的 token 分布，因为模型在训练中看到了两种模式。

### 步骤 4：微调头部

在玩具情感数据集上用分类头替换 MLM 头。只有头部训练；编码器被冻结。这是每个 BERT 应用遵循的模式。

## 使用

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型是微调后的 BERT。** 像 `all-MiniLM-L6-v2` 这样的 `sentence-transformers` 模型是用对比损失训练的 BERT。编码器是相同的。改变的是损失。

**交叉编码器重排序器也是微调后的 BERT。** 对 `[CLS] query [SEP] doc [SEP]` 进行对分类。查询和文档之间的双向注意力正是交叉编码器在质量上优于双编码器的原因。

**2026 年何时不选 BERT。** 任何生成式任务。编码器没有合理的方式来自回归产生 token。还有：任何 1B 参数以下的场景，小型解码器可以用更大的灵活性匹配质量（Phi-3-Mini、Qwen2-1.5B）。

## 产出

参见 `outputs/skill-bert-finetuner.md`。该技能为新的分类或提取任务规划 BERT 微调（骨干选择、头部规范、数据、评估、停止条件）。

## 练习

1. **简单。** 运行 `code/main.py` 并打印 10,000 个 token 上的掩码分布。确认约 15% 被选中，其中约 80% 变为 `[MASK]`。
2. **中等。** 实现整词掩码：如果一个词被分词成子词，一起掩码所有子词或都不掩码。测量这是否提高了 500 句语料库上的 MLM 准确率。
3. **困难。** 在来自公共数据集的 10,000 个句子上训练一个小型（2 层、d=64）BERT。为 SST-2 情感微调 `[CLS]` token。与同等参数量的仅解码器基线进行比较——哪个更优？

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| MLM | "掩码语言建模" | 训练信号：随机将 15% 的 token 替换为 `[MASK]`，预测原始值。 |
| 双向（Bidirectional） | "双向查看" | 编码器注意力没有因果掩码——每个位置看到每个其他位置。 |
| `[CLS]` | "汇聚 token" | 预置到每个序列的特殊 token；其最终嵌入用作句子级表示。 |
| `[SEP]` | "段分隔符" | 分隔成对序列（例如查询/文档、句子 A/B）。 |
| NSP | "下一句预测" | BERT 的第二个预训练任务；在 RoBERTa 中被证明无用，2019 年后被抛弃。 |
| 微调（Fine-tuning） | "适应任务" | 保持编码器大部分冻结；在上面为下游任务训练一个小型头部。 |
| 交叉编码器（Cross-encoder） | "重排序器" | 将查询和文档都作为输入的 BERT，输出相关性分数。 |
| ModernBERT | "2024 年更新" | 使用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力、8K 上下文重建的编码器。 |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；废止了 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 在同等计算下，替换 token 检测优于 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 规范编码器参考。
