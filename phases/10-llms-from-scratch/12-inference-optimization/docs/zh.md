# 推理优化

> LLM 推理由两个阶段定义。Prefill 并行处理你的提示词——受计算限制。Decode 逐个生成 token——受内存限制。每一项优化都针对其中之一或两者。

**类型：** Build
**语言：** Python
**前置要求：** 阶段 10，课程 01-08（Transformer 架构、注意力机制）
**预计时间：** ~120 分钟

## 学习目标

- 实现 KV-cache 以消除自回归 token 生成过程中的冗余计算
- 解释 LLM 推理的 prefill 与 decode 阶段，以及为何它们有不同的瓶颈（计算受限 vs 内存受限）
- 实现连续批处理和 PagedAttention 概念，以在并发请求下最大化 GPU 利用率
- 比较推理优化技术（KV-cache、投机解码、flash attention）及其吞吐量/延迟权衡

## 问题

你将 Llama 3 70B 部署在 4xA100 GPU 上。单个用户获得约 50 tokens/秒。感觉很快。然后 100 个用户同时访问端点。吞吐量下降到 3 tokens/秒/用户。你每月 25,000 美元的 GPU 账单在服务的响应速度比人类打字还慢。

模型本身在 1 个用户和 100 个用户之间没有变化。相同的权重、相同的架构、相同的数学计算。变化的是你如何调度工作。朴素推理浪费了 90% 以上的可用 GPU 算力。一个等待第 47 个 token 的用户让整个批处理槽保持开放，而 GPU 内存总线在矩阵乘之间空闲。与此同时，一个新用户的 2,000 token 提示词可以用有用的计算填满那段空闲时间。

这不是一个扩展问题。这是一个调度问题。本课程中的技术——KV 缓存、连续批处理、PagedAttention、投机解码、前缀缓存——是将 25,000 美元/月的推理账单变成 5,000 美元/月（服务相同流量）的关键。

vLLM 在 4xA100-80GB 上服务 Llama 3 70B，在低并发时达到约 50 tokens/秒/用户，并通过连续批处理和 PagedAttention 在 100 个并发请求时保持 15-25 TPS/用户。没有这些优化，相同硬件在该并发量下只能服务 5 TPS/用户。相同的 GPU、相同的模型，4 倍的吞吐量。

## 概念

### Prefill vs Decode

每个 LLM 推理请求都有两个不同的阶段。

**Prefill** 处理整个输入提示词。所有 token 都是已知的，因此注意力可以在整个序列上并行计算。这是一个大型矩阵乘法——GPU 核心保持忙碌。瓶颈是计算：你的硬件每秒能提供多少 FLOPS。A100 可提供 312 TFLOPS（BF16）。在单个 A100 上，70B 模型的 4,096 token 提示词的 Prefill 大约需要 400ms。

**Decode** 逐个生成输出 token。每个新 token 关注所有之前的 token，但每次前向传递只产生一个 token。权重矩阵与 prefill 时大小相同，但你是将它们乘以单个向量而不是矩阵。GPU 核心在微秒内完成计算，然后等待下一批权重从内存到达。瓶颈是内存带宽：你能多快将模型权重从 HBM 流式传输到计算单元。A100 拥有 2 TB/s 的带宽。FP16 的 70B 模型为 140 GB。完整读取模型一次需要 70ms——这是单次 decode 步骤的下限。

```mermaid
graph LR
    subgraph "Prefill（计算受限）"
        P1["所有提示词 token"] --> P2["并行注意力"]
        P2 --> P3["完全矩阵乘利用率"]
    end

    subgraph "Decode（内存受限）"
        D1["一次一个 token"] --> D2["顺序生成"]
        D2 --> D3["等待内存读取"]
    end

    P3 --> D1
```

**ops:byte 比率**（也称为算术强度）捕捉了这种权衡。它衡量每次从内存加载一个字节时执行了多少次操作。

```
ops:byte 比率 = 每 token 的 FLOPs / 从内存读取的字节数
```

在批大小为 4,096 的 prefill 期间，每加载一个权重执行约 4,096 次乘加运算。比率很高——你受计算限制。在批大小为 1 的 decode 期间，每加载一个权重执行约 1 次操作。比率很低——你受内存限制。

基本洞见：*decode 受内存限制，因为你读取整个模型来产生一个 token*。下面的每项优化要么减少读取量，要么增加每次读取处理的 token 批大小，要么完全避免读取。

### KV 缓存

在注意力机制中，每个 token 的 query 关注每个之前 token 的 key 和 value 向量。没有缓存，生成 token N 需要重新计算所有 N-1 个先前 token 的 key 和 value 投影。Token 1 在生成 token 2 时被投影，然后在 token 3 时再次投影，然后在 token 4 时再次投影。到 token 1,000 时，token 1 总共被投影了 999 次。

KV 缓存存储所有先前 token 的 key 和 value 投影。生成 token N 时，你只计算 token N 的 key 和 value，然后将它们与缓存的 token 1 到 N-1 的 K/V 连接起来。

```mermaid
graph TD
    subgraph "没有 KV 缓存"
        A1["Token 5：重新计算 token 1-4 的 K,V"]
        A2["Token 6：重新计算 token 1-5 的 K,V"]
        A3["Token 7：重新计算 token 1-6 的 K,V"]
    end

    subgraph "有 KV 缓存"
        B1["Token 5：计算 K5,V5，从缓存读取 K1-4,V1-4"]
        B2["Token 6：计算 K6,V6，从缓存读取 K1-5,V1-5"]
        B3["Token 7：计算 K7,V7，从缓存读取 K1-6,V1-6"]
    end
```

**KV 缓存的内存公式：**

```
KV 缓存大小 = 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_param
```

对于 Llama 3 70B（80 层，使用 GQA 的 8 个 KV head，head_dim=128，BF16）：

```
每 token：2 * 80 * 8 * 128 * 2 字节 = 327,680 字节 = 320 KB
4,096 tokens 时：320 KB * 4,096 = 1.28 GB
128K tokens 时：320 KB * 131,072 = 40 GB
```

Llama 3 70B 的一个 128K 上下文对话消耗 40 GB 的 KV 缓存——一半的 A100 内存。100 个并发用户各使用 4K tokens，仅 KV 缓存就需要 128 GB。这就是为什么 KV 缓存管理是推理优化的核心挑战。

### 连续批处理

静态批处理等待一批 N 个请求到达，一起处理它们，然后等待*所有*请求完成后再接受新请求。如果一个请求需要 500 个 token 而另一个需要 10 个，短请求在完成后还要空闲等待 490 个 decode 步骤。

连续批处理（也称为迭代级批处理）在任何请求完成时立即将新请求插入批处理。批处理在每个 decode 步骤时重新评估。一个在 10 个 token 后完成的请求会立即被一个等待中的请求替换。

```mermaid
sequenceDiagram
    participant GPU
    participant R1 as 请求 1（50 tokens）
    participant R2 as 请求 2（10 tokens）
    participant R3 as 请求 3（30 tokens）
    participant R4 as 请求 4（等待中）

    Note over GPU: 静态批处理
    GPU->>R1: 处理批 [R1, R2, R3]
    Note over R2: R2 在第 10 步完成
    Note over R2: 浪费 40 步...
    Note over R3: R3 在第 30 步完成
    Note over R3: 浪费 20 步...
    GPU->>R4: 第 50 步终于开始 R4

    Note over GPU: 连续批处理
    GPU->>R1: 处理批 [R1, R2, R3]
    Note over R2: R2 在第 10 步完成
    GPU->>R4: 在第 11 步插入 R4
    Note over R3: R3 在第 30 步完成
```

吞吐量的提升取决于输出长度变化多少。长度均匀时，连续批处理与静态批处理相当。长度变化时（常见情况），连续批处理可以提供 2-5 倍的更高吞吐量，因为 GPU 槽永远不会闲置。

### PagedAttention

每个请求的 KV 缓存是一个连续的内存块。随着请求的到达和离开，内存碎片化——就像操作系统中的 RAM 碎片化。一个 4K token 的请求需要 1.28 GB 连续内存。即使你总共有 2 GB 空闲，你可能没有 1.28 GB *连续*的空闲空间。你要么浪费内存，要么拒绝请求。

PagedAttention（来自 vLLM）将操作系统风格的虚拟内存应用于 KV 缓存。它不是为每个请求分配一个连续块，而是分配固定大小的"页"（通常每页 16 个 token）。页可以位于物理 GPU 内存中的任何位置。页表将每个请求的逻辑序列位置映射到物理页位置。

```mermaid
graph TD
    subgraph "连续分配"
        C1["请求 A：2GB 块"]
        C2["[空闲：0.5GB]"]
        C3["请求 B：1GB 块"]
        C4["[空闲：1.5GB -- 但已碎片化]"]
    end

    subgraph "PagedAttention"
        P1["页池：256 页，每页 16 个 token"]
        P2["请求 A：页 3,7,12,45,88..."]
        P3["请求 B：页 1,4,9,22,67..."]
        P4["无碎片化，无浪费"]
    end
```

PagedAttention 还支持共享前缀的**写时复制**。如果 50 个请求共享相同的系统提示词，该系统提示词的 KV 缓存页只存储一次，由所有 50 个请求引用。只有当某个请求产生分歧（不同的用户消息）时，它才会获得自己的页。这对于具有共享系统提示词的应用程序来说大幅减少了内存使用。

vLLM 报告通过 PagedAttention 实现了近乎为零的内存浪费（约 4%，而朴素分配约为 60-80%）。

### 投机解码

Decode 很慢，因为它是顺序的——你生成一个 token，将其反馈回去，生成下一个。但如果你能廉价地猜测接下来的 5 个 token，然后一次性验证它们呢？

投机解码使用一个小型、快速的**草稿模型**生成 K 个候选 token。大型**目标模型**然后在一个单次前向传递中处理所有 K 个候选（这看起来像 prefill——并行的、计算受限的、高效的）。如果目标模型同意草稿模型的预测，你在一次目标前向传递的时间内接受所有 K 个 token。如果在位置 j 上出现分歧，你接受 token 1 到 j-1 并丢弃其余部分。

```mermaid
graph LR
    D["草稿模型（1B）"] -->|"生成 5 个 token<br/>~5ms"| C["候选：the cat sat on the"]
    C --> T["目标模型（70B）"]
    T -->|"一次传递验证全部 5 个<br/>~70ms"| V{"匹配？"}
    V -->|"5 个中 4 个匹配"| A["75ms 内接受 4 个 token<br/>对比 280ms 顺序生成"]
    V -->|"位置 5 不匹配"| R["拒绝 token 5<br/>从目标重新采样"]
```

加速取决于**接受率**——草稿模型的预测与目标匹配的频率。对于 Llama 3 8B 为 Llama 3 70B 起草，自然语言上的接受率通常为 70-85%。这转化为 2-3 倍的 decode 加速。

投机解码的三种方法：

| 方法 | 草稿来源 | 接受率 | 开销 |
|--------|-------------|-----------------|----------|
| 草稿-目标（Leviathan 等） | 独立小模型 | 70-85% | 草稿模型内存 |
| EAGLE（Li 等） | 目标上的轻量级 head | 75-90% | ~1% 额外参数 |
| N-gram 查找 | Token n-gram 表 | 40-60% | 可忽略 |

**EAGLE** 在目标模型的隐藏状态之上训练一个小的自回归 head。它使用目标模型倒数第二层的特征来预测下一个 token 的嵌入。因为它在目标模型自身的表示（而不是独立模型）上操作，所以能以最小的额外内存实现更高的接受率。EAGLE-2 添加了一个动态草稿树，根据上下文调整候选数量。

**N-gram 投机解码**维护一个来自当前上下文或预构建语料库的 n-gram 延续表。如果草稿匹配同一对话中之前出现的内容（重复模式、代码、结构化输出），它就能零神经网络开销地触发。平均接受率较低，但每次推测的成本基本为零。

投机解码在*数学上是精确的*——输出分布与目标模型的分布完全相同。这不是近似。验证步骤确保每个被接受的 token 具有目标模型会分配的确切概率。

### 前缀缓存

许多请求共享相同的前缀。一个聊天机器人的系统提示词。一个 RAG 上下文块。一组 few-shot 示例。没有前缀缓存，每个请求都从头开始重新计算这些共享 token 的 KV 缓存。

前缀缓存存储常见前缀的 KV 缓存，并在请求之间重用。当带有已知前缀的新请求到达时，系统复制（或引用）缓存的 KV 条目，只计算唯一后缀部分的 KV。

对于一个跨所有请求共享的 2,000 token 系统提示词，前缀缓存消除了每个请求约 400ms 的 prefill 时间。在 100 请求/秒时，这节省了每秒 40 秒的 GPU 计算——超过一个 GPU 的工作量。

SGLang 的 RadixAttention 使用基数树（trie）实现前缀缓存，该树按 token 内容索引前缀。任何匹配存储前缀的请求都能免费获得其 KV 缓存。该树支持部分前缀匹配——如果你与缓存条目共享 2,000 个前缀 token 中的 1,500 个，你重用那 1,500 个，只重新计算 500 个。

### 推理引擎

三个引擎主导生产级 LLM 服务：

| 引擎 | 关键创新 | 最适合 |
|--------|---------------|----------|
| vLLM | PagedAttention、连续批处理 | 通用服务，最高兼容性 |
| SGLang | RadixAttention（前缀缓存）、结构化生成 | 多轮聊天机器人、受约束解码 |
| TensorRT-LLM | NVIDIA 内核融合、FP8 量化 | NVIDIA 硬件上的最大单 GPU 吞吐量 |

**vLLM** 是默认的起点。它支持最广泛的模型，可在任何 GPU 供应商（NVIDIA、AMD、Intel）上运行，并通过 PagedAttention + 连续批处理实现强大的吞吐量。兼容 OpenAI 的 API 意味着你可以将其直接替换为任何 OpenAI API 调用。

**SGLang** 建立在与 vLLM 相同的基础上，但增加了用于前缀缓存的 RadixAttention 和用于结构化 LLM 程序的领域特定语言。如果你的工作负载涉及多轮对话、工具使用或受约束解码（JSON 输出、正则表达式引导的生成），SGLang 通常通过前缀重用以 2-5 倍的性能超越 vLLM。

**TensorRT-LLM** 将模型编译为优化的 NVIDIA GPU 内核。它融合操作（注意力 + 线性 + 激活在一个内核中），在 H100 GPU 上使用 FP8，并与 NVIDIA Triton 推理服务器集成以实现生产部署。它在 NVIDIA 硬件上实现了最高的单 GPU 吞吐量，但需要更多设置，且仅适用于 NVIDIA GPU。

Llama 3 70B（4xA100-80GB，BF16）的真实数据：

| 指标 | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|---------------|
| 吞吐量（1 用户） | ~50 TPS | ~55 TPS | ~65 TPS |
| 吞吐量（100 用户） | ~2,500 总 TPS | ~3,200 总 TPS | ~3,000 总 TPS |
| 首 token 时间 | ~400ms | ~300ms（前缀命中） | ~350ms |
| 最大上下文 | 128K | 128K | 128K |

### Ops:Byte 框架

你无法优化你无法测量的东西。ops:byte 比率告诉你你受计算限制还是内存限制，这决定了哪些优化重要。

```
计算上限：GPU 的峰值 FLOPS
内存上限：峰值带宽 * ops:byte 比率
```

当 ops:byte 较低时（decode、小批大小），你达到了内存带宽上限。增加更多计算能力（更高时钟频率、更多核心）没有帮助。你需要减少内存读取（量化、KV 缓存压缩）或增加批大小以将读取分摊到更多有用工作上。

当 ops:byte 较高时（prefill、大批大小），你达到了计算上限。内存带宽优化没有帮助。你需要更快的 GPU、内核融合或降低精度以挤出更多 FLOPS。

| 场景 | ops:byte | 受限 | 优化方式 |
|----------|----------|-------|---------------|
| Prefill，批=1 | ~4,096 | 计算 | 内核融合，FP8 |
| Decode，批=1 | ~1 | 内存 | 量化，KV 压缩 |
| Decode，批=32 | ~32 | 内存 | 更大批大小，连续批处理 |
| Decode，批=256 | ~256 | 过渡中 | 两者都重要 |
| Decode，批=1024 | ~1,024 | 计算 | 内核融合，张量并行 |

A100 上的交叉点大约在 ops:byte = 156（312 TFLOPS / 2 TB/s）。低于 156，你受内存限制。高于 156，你受计算限制。连续批处理通过每次迭代打包更多 token 将 decode 推向这个交叉点。

```figure
context-window-slide
```

## 构建它

### 步骤 1：从零开始构建 KV 缓存

我们构建一个多头 KV 缓存，按层、按头存储 key 和 value 投影，并展示内存增长模式。

```python
import numpy as np

class KVCache:
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len, dtype=np.float16):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        self.k_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.v_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.seq_len = 0

    def update(self, layer_idx, new_keys, new_values):
        num_new = new_keys.shape[1]
        end = self.seq_len + num_new
        self.k_cache[layer_idx, :, self.seq_len:end, :] = new_keys
        self.v_cache[layer_idx, :, self.seq_len:end, :] = new_values
        return (
            self.k_cache[layer_idx, :, :end, :],
            self.v_cache[layer_idx, :, :end, :]
        )

    def advance(self, num_tokens):
        self.seq_len += num_tokens

    def memory_bytes(self):
        return self.k_cache.nbytes + self.v_cache.nbytes

    def used_bytes(self):
        per_token = 2 * self.num_layers * self.num_heads * self.head_dim * np.dtype(self.dtype).itemsize
        return per_token * self.seq_len
```

### 步骤 2：使用 KV 缓存的注意力

一个简化的多头注意力，在 decode 步骤中使用 KV 缓存。

```python
def scaled_dot_product_attention(query, keys, values):
    head_dim = query.shape[-1]
    scores = np.matmul(query, keys.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    seq_len_q = scores.shape[-2]
    seq_len_k = scores.shape[-1]
    if seq_len_q > 1:
        mask = np.triu(np.ones((seq_len_q, seq_len_k), dtype=np.float32), k=seq_len_k - seq_len_q + 1)
        scores = scores + mask * (-1e9)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, values)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale

    def forward(self, x, kv_cache=None, layer_idx=0):
        batch, seq_len, d_model = x.shape
        Q = np.matmul(x, self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            K_full, V_full = kv_cache.update(layer_idx, K[0], V[0])
            K = K_full[np.newaxis, :, :, :]
            V = V_full[np.newaxis, :, :, :]
            if seq_len == 1:
                kv_cache.advance(1)

        attn_out = scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, -1, d_model)
        return np.matmul(attn_out, self.W_o)
```

### 步骤 3：连续批处理模拟器

这个模拟器展示了静态批处理和连续批处理之间的调度差异。

```python
import heapq

class Request:
    def __init__(self, request_id, prompt_tokens, output_tokens, arrival_step):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.arrival_step = arrival_step
        self.tokens_generated = 0
        self.start_step = None
        self.end_step = None

    def is_done(self):
        return self.tokens_generated >= self.output_tokens


def simulate_static_batching(requests, batch_size):
    step = 0
    completed = []
    queue = list(requests)
    queue.sort(key=lambda r: r.arrival_step)

    while queue:
        batch = []
        while queue and len(batch) < batch_size:
            r = queue.pop(0)
            r.start_step = max(step, r.arrival_step)
            batch.append(r)

        if batch:
            step = max(step, max(r.start_step for r in batch))
            max_output = max(r.output_tokens for r in batch)
            for r in batch:
                r.tokens_generated = r.output_tokens
                r.end_step = step + max_output
            step += max_output
            completed.extend(batch)

    return completed


def simulate_continuous_batching(requests, batch_size):
    step = 0
    completed = []
    queue = sorted(requests, key=lambda r: r.arrival_step)
    queue_idx = 0
    active = []
    waiting = []

    while queue_idx < len(queue) or active or waiting:
        while queue_idx < len(queue) and queue[queue_idx].arrival_step <= step:
            waiting.append(queue[queue_idx])
            queue_idx += 1

        while waiting and len(active) < batch_size:
            r = waiting.pop(0)
            r.start_step = step
            active.append(r)

        if not active:
            if waiting:
                step += 1
                continue
            elif queue_idx < len(queue):
                step = queue[queue_idx].arrival_step
                continue
            else:
                break

        for r in active:
            r.tokens_generated += 1

        done = [r for r in active if r.is_done()]
        for r in done:
            r.end_step = step + 1
            completed.append(r)
        active = [r for r in active if not r.is_done()]

        step += 1

    return completed


def batching_stats(completed):
    latencies = [r.end_step - r.arrival_step for r in completed]
    total_time = max(r.end_step for r in completed) - min(r.arrival_step for r in completed)
    total_tokens = sum(r.output_tokens for r in completed)
    return {
        "avg_latency": np.mean(latencies),
        "p50_latency": np.median(latencies),
        "p99_latency": np.percentile(latencies, 99),
        "total_time": total_time,
        "throughput": total_tokens / total_time if total_time > 0 else 0,
    }
```

### 步骤 4：前缀缓存

一个基于 trie 的前缀缓存，为共享前缀存储 KV 条目。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.kv_data = None
        self.hit_count = 0


class PrefixCache:
    def __init__(self, max_entries=1000):
        self.root = TrieNode()
        self.max_entries = max_entries
        self.total_entries = 0
        self.hits = 0
        self.misses = 0

    def _walk(self, token_ids):
        node = self.root
        depth = 0
        for tid in token_ids:
            if tid not in node.children:
                break
            node = node.children[tid]
            depth += 1
        return node, depth

    def lookup(self, token_ids):
        node, depth = self._walk(token_ids)
        if depth > 0:
            self.hits += 1
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                current.hit_count += 1
            kv_entries = []
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                if current.kv_data is not None:
                    kv_entries.append(current.kv_data)
            return depth, kv_entries
        self.misses += 1
        return 0, []

    def insert(self, token_ids, kv_per_token):
        node = self.root
        for i, tid in enumerate(token_ids):
            if tid not in node.children:
                if self.total_entries >= self.max_entries:
                    return i
                node.children[tid] = TrieNode()
                self.total_entries += 1
            node = node.children[tid]
            if i < len(kv_per_token):
                node.kv_data = kv_per_token[i]
        return len(token_ids)

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

### 步骤 5：投机解码模拟器

我们模拟草稿-目标投机解码，具有可配置的接受率。

```python
class DraftModel:
    def __init__(self, vocab_size, acceptance_rate=0.8):
        self.vocab_size = vocab_size
        self.acceptance_rate = acceptance_rate

    def generate(self, context, num_tokens):
        tokens = np.random.randint(0, self.vocab_size, size=num_tokens)
        return tokens

    def get_probs(self, context, token):
        probs = np.random.dirichlet(np.ones(self.vocab_size))
        return probs


class TargetModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def get_probs(self, context, tokens=None):
        if tokens is not None:
            return [np.random.dirichlet(np.ones(self.vocab_size)) for _ in tokens]
        return np.random.dirichlet(np.ones(self.vocab_size))


def speculative_decode(draft_model, target_model, context, num_speculative=5,
                       draft_cost=1.0, target_cost=10.0, verify_cost=12.0):
    total_tokens = 0
    total_cost = 0.0
    accepted_counts = []
    context = list(context)

    max_tokens = 100

    while total_tokens < max_tokens:
        draft_tokens = draft_model.generate(context, num_speculative)
        total_cost += draft_cost * num_speculative

        target_probs = target_model.get_probs(context, draft_tokens)
        total_cost += verify_cost

        accepted = 0
        for i, token in enumerate(draft_tokens):
            draft_p = draft_model.get_probs(context + list(draft_tokens[:i]), token)
            target_p = target_probs[i]

            r = np.random.random()
            acceptance_prob = min(1.0, target_p[token] / (draft_p[token] + 1e-10))

            if r < draft_model.acceptance_rate:
                accepted += 1
                context.append(token)
                total_tokens += 1
            else:
                new_token = np.random.choice(draft_model.vocab_size, p=target_p)
                context.append(new_token)
                total_tokens += 1
                break

        accepted_counts.append(accepted)

        if accepted == num_speculative:
            bonus_probs = target_model.get_probs(context)
            bonus_token = np.random.choice(draft_model.vocab_size, p=bonus_probs)
            context.append(bonus_token)
            total_tokens += 1

    sequential_cost = total_tokens * target_cost
    return {
        "total_tokens": total_tokens,
        "speculative_cost": total_cost,
        "sequential_cost": sequential_cost,
        "speedup": sequential_cost / total_cost if total_cost > 0 else 1.0,
        "avg_accepted": np.mean(accepted_counts),
        "acceptance_rate": np.mean(accepted_counts) / num_speculative,
    }


def compare_speculation_strategies(vocab_size=1000, num_trials=20):
    results = {}

    for name, acceptance_rate, spec_tokens in [
        ("草稿-目标（8B->70B）", 0.78, 5),
        ("EAGLE", 0.85, 6),
        ("N-gram", 0.50, 4),
        ("无推测", 0.0, 0),
    ]:
        if spec_tokens == 0:
            results[name] = {
                "speedup": 1.0,
                "acceptance_rate": 0.0,
                "avg_accepted": 0.0,
            }
            continue

        trial_results = []
        for _ in range(num_trials):
            draft = DraftModel(vocab_size, acceptance_rate=acceptance_rate)
            target = TargetModel(vocab_size)
            context = list(np.random.randint(0, vocab_size, size=10))
            result = speculative_decode(draft, target, context, num_speculative=spec_tokens)
            trial_results.append(result)

        results[name] = {
            "speedup": np.mean([r["speedup"] for r in trial_results]),
            "acceptance_rate": np.mean([r["acceptance_rate"] for r in trial_results]),
            "avg_accepted": np.mean([r["avg_accepted"] for r in trial_results]),
        }

    return results
```

### 步骤 6：KV 缓存内存剖析器

计算真实模型配置的 KV 缓存内存需求。

```python
MODEL_CONFIGS = {
    "Llama-3-8B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 8, "gqa": True,
    },
    "Llama-3-70B": {
        "num_layers": 80, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 70, "gqa": True,
    },
    "Llama-3-405B": {
        "num_layers": 126, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 405, "gqa": True,
    },
    "Mistral-7B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 7, "gqa": True,
    },
    "GPT-4-est": {
        "num_layers": 120, "num_kv_heads": 96, "head_dim": 128,
        "model_params_b": 1800, "gqa": False,
    },
}


def kv_cache_memory(config, seq_len, dtype_bytes=2):
    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * dtype_bytes
    total = per_token * seq_len
    return {
        "per_token_bytes": per_token,
        "per_token_kb": per_token / 1024,
        "total_bytes": total,
        "total_mb": total / (1024 ** 2),
        "total_gb": total / (1024 ** 3),
    }


def memory_budget(config, gpu_memory_gb, model_dtype_bytes=2, kv_dtype_bytes=2):
    model_memory_gb = config["model_params_b"] * 1e9 * model_dtype_bytes / (1024 ** 3)
    overhead_gb = gpu_memory_gb * 0.1
    available_for_kv = gpu_memory_gb - model_memory_gb - overhead_gb

    if available_for_kv <= 0:
        return {"error": "模型不适合 GPU 内存", "model_memory_gb": model_memory_gb}

    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * kv_dtype_bytes
    max_tokens = int(available_for_kv * (1024 ** 3) / per_token)

    return {
        "gpu_memory_gb": gpu_memory_gb,
        "model_memory_gb": round(model_memory_gb, 1),
        "overhead_gb": round(overhead_gb, 1),
        "available_for_kv_gb": round(available_for_kv, 1),
        "max_total_tokens": max_tokens,
        "max_users_at_2k": max_tokens // 2048,
        "max_users_at_4k": max_tokens // 4096,
        "max_users_at_32k": max_tokens // 32768,
    }
```

## 使用它

使用 vLLM：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=4,
    enable_prefix_caching=True,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["用一段话解释推理优化。"], params)
```

使用 SGLang 实现前缀缓存 + 结构化输出：

```python
import sglang as sgl

@sgl.function
def classify(s, text):
    s += sgl.system("你是一个分类器。只输出 JSON。")
    s += sgl.user(f"将这段文本分类：{text}")
    s += sgl.assistant(sgl.gen("result", regex=r'\{"label": "(positive|negative|neutral)"\}'))

runtime = sgl.Runtime(model_path="meta-llama/Llama-3-70B-Instruct", tp_size=4)
sgl.set_default_backend(runtime)

results = classify.run_batch([
    {"text": "这个产品太棒了！"},
    {"text": "糟糕的体验。"},
    {"text": "还可以吧，我想。"},
])
```

使用 TensorRT-LLM：

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("./llama-70b-trt-engine/", rank=0)

outputs = runner.generate(
    batch_input_ids=[tokenizer.encode("解释 KV 缓存。")],
    max_new_tokens=256,
    temperature=0.7,
)
```

## 交付物

本课程产出：
- `outputs/skill-inference-optimization.md` —— 用于诊断和优化 LLM 推理服务的技能

## 练习

1. 修改 KV 缓存分析器，比较 FP16 vs FP8 vs INT4 KV 缓存量化。对于 Llama 3 70B 在 4K 上下文中，计算每种量化方式在 4xA100-80GB 上的最大并发用户数。INT4 的 KV 量化应该大约将用户容量提高 4 倍。

2. 扩展连续批处理模拟器以跟踪 GPU 利用率（每步填充的批处理槽比例）。绘制 50 个请求（其输出长度遵循 Pareto 分布，shape=1.5，scale=20）的静态和连续批处理的利用率随时间变化图。连续批处理应保持 >80% 的利用率。

3. 实现 KV 缓存的分组查询注意力（GQA）版本，其中 `num_kv_heads < num_query_heads`。Llama 3 70B 使用 64 个 query head 但只有 8 个 KV head。计算相对于完整多头注意力的内存节省（KV 缓存大小减少 8 倍）。

4. 构建一个使用 LRU 淘汰机制的前缀缓存。将 max_entries 设为 500，生成 1,000 个请求，其中 60% 共享 5 个常见前缀之一。测量命中率并与无限制缓存进行比较。良好的淘汰策略应使命中率保持在 55% 以上。

5. 扩展投机解码模拟器以实现基于树的推测（EAGLE-2 风格）。不是使用单个 K 个草稿 token 的链，而是生成一个候选树（例如，3 个级别各 2 个分支 = 8 个叶子候选）。比较每轮验证中接受的总 token 数与线性推测的对比。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| Prefill | "处理提示词" | 在所有输入 token 上并行计算注意力——受计算限制，因为完整的矩阵乘法使 GPU 核心保持忙碌 |
| Decode | "生成 token" | 每次前向传递产生一个 token，每次读取完整的模型权重——受内存限制，因为计算在下一批权重到达之前就已完成 |
| KV 缓存 | "缓存注意力状态" | 存储所有先前 token 的 key 和 value 投影，以避免在每个 decode 步骤重新计算——用内存换取计算 |
| 连续批处理 | "动态批处理" | 在任何请求完成时立即将新请求插入正在运行的批处理，在每个 decode 迭代时重新评估，而不是等待整个批处理完成 |
| PagedAttention | "KV 缓存的虚拟内存" | 以固定大小的页而不是连续块分配 KV 缓存，消除内存碎片，并为共享前缀启用写时复制 |
| 投机解码 | "起草和验证" | 使用快速草稿模型提出多个 token，然后在一次目标模型前向传递中全部验证——数学上精确，2-3 倍加速 |
| EAGLE | "自投机解码" | 一种投机解码变体，在目标模型自身的隐藏状态上训练轻量级 head，比单独的草稿模型实现更高的接受率 |
| 前缀缓存 | "重用系统提示词 KV" | 存储常见前缀（系统提示词、few-shot 示例）的计算过的 KV 缓存条目，并在请求间重用，以跳过冗余的 prefill |
| Ops:byte 比率 | "算术强度" | 计算操作与内存读取字节的比率——决定工作负载是受计算限制（高比率）还是受内存限制（低比率） |
| 首 token 时间 | "TTFT" | 从接收到请求到产生第一个输出 token 的延迟——对于长提示词，主要由 prefill 时间决定 |

## 延伸阅读

- Kwon 等，"Efficient Memory Management for Large Language Model Serving with PagedAttention"（2023）——提出分页 KV 缓存管理的 vLLM 论文，现为推理服务的行业标准
- Leviathan 等，"Fast Inference from Transformers via Speculative Decoding"（2023）——证明草稿-验证推测产生精确目标模型分布的同时实现 2-3 倍加速的基础论文
- Li 等，"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"（2024）——通过在目标模型自身特征上训练 head 而非使用单独的草稿模型，实现更高的接受率
- Zheng 等，"SGLang: Efficient Execution of Structured Language Model Programs"（2024）——引入用于前缀缓存的 RadixAttention 和多调用 LLM 程序的编程模型
- Williams 等，"Roofline: An Insightful Visual Performance Model for Multicore Architectures"（2009）——形式化了 ops:byte 框架用于推理计算与内存瓶颈的原始 roofline 论文
