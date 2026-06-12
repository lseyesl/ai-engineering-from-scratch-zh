# KV 缓存、Flash Attention 与推理优化

> 训练是并行且受 FLOP 限制的。推理是串行且受内存限制的。不同的瓶颈，不同的技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 7 · 02（自注意力），阶段 7 · 05（完整 Transformer），阶段 7 · 07（GPT）
**时间：** ~75 分钟

## 问题

一个朴素的自回归解码器需要 `O(N²)` 的工作量来生成 `N` 个 token：在每一步它都要重新计算整个前缀上的注意力。对于 4K token 的响应，这是 1600 万次注意力操作，其中大多数是冗余的。前缀 token 的每个隐藏状态一旦计算就是确定性的——你只需要将新 token 的查询与之前所有内容的缓存键和值进行匹配。

除此之外，注意力本身移动了大量数据。标准注意力会实例化一个 N×N 的分数矩阵、N×d 的 softmax 输出、N×d 的最终输出——对 HBM 的读写太多了。对于 N≥2K，注意力在受 FLOP 限制之前就已经受内存限制了。经典的注意力核使现代 GPU 的利用率低 4-10 倍。

两个优化——均来自 Dao 等人——将前沿推理从"慢"推进到"快"：

1. **KV 缓存。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力是对缓存键的一次查询。推理从每步 `O(N²)` 降至 `O(N)`。
2. **Flash Attention。** 分块注意力计算，使得完整的 N×N 矩阵永远不会触及 HBM。所有 softmax + 矩阵乘法都在 SRAM 中完成。在 A100 上获得 2-4 倍的挂钟加速；在 H100 上使用 FP8 获得 5-10 倍加速。

到 2026 年，两者都已普及。每个生产推理堆栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都假设它们存在。每个前沿模型都启用了 Flash Attention。

## 概念

![KV 缓存增长和 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV 缓存数学

每解码器层、每 token、每头：

```
每层每 token 字节数 = 2 * d_head * dtype_size
                      ^
                      K 和 V
```

对于 7B 模型，32 层、32 头、d_head=128、fp16：

```
每层每 token = 2 * 128 * 2 = 512 字节
每 token（32 层）= 16 KB
每 32K 上下文 = 512 MB
```

对于 Llama 3 70B（80 层、d_head=128、GQA 使用 8 个 KV 头）：

```
每层每 token = 2 * 8 * 128 * 2 = 4096 字节（4 KB）
每 32K 上下文 = 10.4 GB
```

这 10 GB 就是为什么在 batch size 为 1 时，Llama 3 70B 在 128K 上下文下需要大部分 40 GB A100 仅用于 KV 缓存。

**GQA 是 KV 缓存的胜利。** 具有 64 头的 MHA 将是 32 GB。MLA 进一步压缩。

拖动维度滑块，观察缓存大小的变化。增加序列长度或 batch size，看看它多快就能超过单个 GPU 的容量：

```figure
kv-cache-sizer
```

### Flash Attention — 分块技巧

标准注意力：

```
S = Q @ K^T          （HBM 读取，N×N，HBM 写入）
P = softmax(S)       （HBM 读取，HBM 写入）
O = P @ V            （HBM 读取，HBM 写入）
```

三次 HBM 往返。在 H100 上，HBM 带宽为 3 TB/s；SRAM 为 30 TB/s。每次 HBM 传输相对于将一切保持在片上都是 10 倍的减速。

Flash Attention：

```
对每个 Q 块（分块大小约 128 × 128）：
    将 Q_tile 加载到 SRAM
    对每个 K、V 块：
        将 K_tile、V_tile 加载到 SRAM
        计算 S_tile = Q_tile @ K_tile^T     （SRAM）
        运行中 softmax 聚合                 （SRAM）
        累积到 O_tile                       （SRAM）
    将 O_tile 写入 HBM
```

每次分块一次 HBM 传输。总内存占用从 `O(N²)` 降至 `O(N)`。反向传播重新计算前向传播中的一些值而不是存储它们——另一个内存优势。

**数值技巧。** 运行中 softmax 在分块之间维护 `(max, sum)`，以便最终归一化是精确的。不是近似——Flash Attention 计算与标准注意力逐位相同的输出（在 fp16 非结合性误差范围内）。

**版本演变：**

| 版本 | 年份 | 关键变化 | 参考硬件上的加速 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 分块 SRAM 核 | A100 上 2 倍 |
| Flash 2 | 2023 | 更好的并行性、因果优先排序 | A100 上 3 倍 |
| Flash 3 | 2024 | Hopper 异步、FP8 | H100 上 1.5-2 倍（约 740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5 级流水线、软件 exp2 | 推理优先（最初仅前向） |

Flash 4 在发布时仅支持前向传播。训练仍使用 Flash 3。Flash 4 的 GQA 和 varlen 支持待定（2026 年中）。

### 推测解码——另一个延迟优势

廉价模型提出 N 个 token。大模型并行验证所有 N 个。如果验证接受 k 个 token，你为 k 次生成支付了 1 次大模型前向传播。在代码和散文上典型的 k=3-5。

2026 年默认选择：
- **EAGLE 2 / Medusa。** 集成草稿头，共享验证器的隐藏状态。2-3 倍加速，无质量损失。
- **带草稿模型的推测解码。** 消费硬件上 2-4 倍加速。
- **Lookahead 解码。** Jacobi 迭代；不需要草稿模型。小众但免费。

### 连续批处理

经典批处理推理：等待最慢的序列完成，然后开始新的批次。当短响应提前完成时浪费 GPU。

连续批处理（首次在 Orca 中实现，现在在 vLLM、TensorRT-LLM、SGLang 中）：只要旧请求完成，就在批次中交换新请求。典型聊天工作负载的吞吐量提升 5-10 倍。

### PagedAttention — KV 缓存即虚拟内存

vLLM 的头牌功能。KV 缓存以 16-token 块分配；页表将逻辑位置映射到物理块。允许你在并行样本（束搜索、并行采样）之间共享 KV，为提示缓存热交换前缀，并整理内存碎片。相比朴素连续分配，吞吐量提升 4 倍。

```figure
flash-attention-memory
```

## 动手实现

参见 `code/main.py`。我们实现：

1. 一个朴素的 `O(N²)` 增量解码器。
2. 一个 `O(N)` 带 KV 缓存的解码器。
3. 一个模拟 Flash Attention 运行中最大算法风格的分块 softmax。

### 步骤 1：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：为每个解码器层、每个头持续增长每 token 的 K、V 向量。

### 步骤 2：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention 风格的 softmax(qK^T)V，带运行中最大/求和。"""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

一次计算即产生与 `softmax(qK) V` 逐位相同的输出，但任何时候工作集只是一个 `tile × d_head` 块，而不是完整的 `N × d_head`。

### 步骤 3：比较 100-token 生成中的朴素 vs 缓存解码

计数注意力操作。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码打印两者。

## 使用

```python
# HuggingFace transformers 在仅解码器 generate() 上自动启用 KV 缓存。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # 如果是 Hopper 则使用 FA3
    torch_dtype="bfloat16",
)
# generate() 自动使用 KV 缓存
```

vLLM 生产环境：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是 2026 年的一个重大优势——相同的系统提示、少样本示例或长上下文文档可以在调用之间重用 KV。对于具有重复工具提示的代理工作负载，前缀缓存通常能带来 5 倍的吞吐量提升。

## 产出

参见 `outputs/skill-inference-optimizer.md`。该技能为新的推理部署选择注意力实现、KV 缓存策略、量化和推测解码。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素解码器和缓存解码器产生相同的输出；注意操作计数差异。
2. **中等。** 实现前缀缓存：给定一个提示 P 和多个补全，对 P 进行一次前向传播以填充 KV 缓存，然后为每个补全分支。测量与每次重新编码 P 相比的加速。
3. **困难。** 实现一个玩具 PagedAttention：KV 缓存在固定 16-token 块中，带空闲列表。当序列完成时，将其块归还到池中。模拟 1,000 次不同长度的聊天补全。比较与连续分配的内存碎片情况。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| KV 缓存 | "使解码快速的技巧" | 存储每个前缀 token 的 K 和 V；新查询与之进行注意力计算，而非重新计算。 |
| HBM | "GPU 主内存" | 高带宽内存；H100 上 80 GB，B200 上 192 GB。约 3 TB/s 带宽。 |
| SRAM | "片上内存" | 每 SM 快速内存，H100 上每 SM 约 256 KB。约 30 TB/s 带宽。 |
| Flash Attention | "分块注意力核" | 无需在 HBM 中实例化 N×N 矩阵即可计算注意力。 |
| 连续批处理（Continuous batching） | "无等待批处理" | 将完成的序列换出、新序列换入，无需排空批次。 |
| PagedAttention | "vLLM 的头牌" | KV 缓存在固定块中分配，带页表；消除碎片。 |
| 前缀缓存（Prefix caching） | "重用长提示" | 在请求之间缓存共享前缀的 KV；大幅降低代理成本。 |
| 推测解码（Speculative decoding） | "草稿 + 验证" | 廉价草稿模型提出 token；大模型一次验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 发布说明 (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5 级流水线和软件 exp2 技巧；阅读仓库 README，了解本课提到的仅前向发布的注意事项。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 推测解码。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 论文，本课引用的集成草稿方法。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 与 EAGLE 并列引用的 Medusa 方法。
- [vLLM 文档 — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 关于 16-token 块和页表设计的规范深度解析。
