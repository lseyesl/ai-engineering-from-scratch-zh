# vLLM 服务内部机制：PagedAttention、Continuous Batching、Chunked Prefill

> vLLM 在 2026 年的主导地位并非依赖单一技巧，而是三个相互叠加的默认特性。PagedAttention 始终开启。Continuous batching 在解码迭代之间将新请求注入活动批次。Chunked prefill 将长提示切分，使解码 token 永远不会被饿死。三者全部开启时，Llama 3.3 70B FP8 在一个 H100 SXM5 上可在 128 并发下达到 2,200-2,400 tok/s — 比 vLLM 自身默认值高出约 25%，是朴素 PyTorch 循环的 3-4 倍。本课以你可以画图的方式解读调度器和注意力内核，并以 `code/main.py` 中的一个玩具级 continuous batcher 收尾，它按照 vLLM 的方式调度预填充和解码。

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 minutes

## Learning Objectives

- 将 PagedAttention 解释为 KV 缓存分配器：块、块表，以及为什么在生产负载下碎片率保持在 4% 以下。
- 在迭代级别绘制 continuous batching 的示意图：完成的序列如何离开批次，新序列如何加入而不需要排空。
- 用一句话描述 chunked prefill，并说出它保护的是哪个延迟指标（提示：是 TTFT 尾部，而非平均吞吐量）。
- 说出 2026 年 vLLM v0.18.0 的陷阱，即团队同时启用所有优化时会踩到的坑。

## 问题

一个朴素的 PyTorch 服务循环一次运行一个请求：分词、预填充、解码直到 EOS、返回。一个用户时没问题。一百个用户时，就是一个耐心排队的问题。明显的修复 — 静态批处理 — 将每个请求填充到窗口中最长提示的长度，将每个解码填充到最长预期输出的长度，并且整个批次被最慢的序列拖住。你为从未使用的填充付费，快速请求等待慢速请求。

vLLM 同时解决了三个问题。PagedAttention 阻止了 KV 缓存碎片像经典连续分配那样吞噬 60-80% 的 GPU 内存。Continuous batching 允许请求在每个解码迭代之间加入和离开批次，因此批次始终充满真正的工作。Chunked prefill 将 32k token 的提示分解为约 512 token 的切片，与解码交错执行，因此长提示不会冻结 GPU 上的每个解码 token。

2026 年的生产默认值是三者全部开启。你需要理解每个特性的作用，因为失败模式都在调度器上，而非模型上。

## 概念

### PagedAttention 作为虚拟内存系统

KV 缓存每个序列的大小为 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对于 Llama 3.3 70B 在 8192 token 下，BF16 下每个序列约 1.25 GB。如果你为每个请求预保留 8192 个槽位，但平均请求只使用 1500 个 token，你浪费了约 82% 的预留 HBM。经典批处理承担了这种浪费。

PagedAttention 借鉴了操作系统虚拟内存的思想。KV 缓存不是每个序列连续的。它以固定大小的块（默认 16 个 token）分配。每个序列有一个块表，将其逻辑 token 位置映射到物理块 ID。当一个序列增长超过其分配的块时，添加一个块。当它完成时，其块返回池中。

碎片率从 60-80%（经典）下降到 4% 以下（PagedAttention）。你不需要通过标志启用 PagedAttention — 它是 vLLM 唯一提供的分配器。可调参数是 `--gpu-memory-utilization`（默认 0.9），它告诉 vLLM 在加载权重和激活后为 KV 块保留多少 HBM。

### 迭代级别的 Continuous Batching

旧的"动态批处理"等待一个窗口（比如 10 ms）来填充批次，然后运行预填充 + 解码 + 解码 + 解码直到每个序列完成。快速序列提前离开，在 GPU 完成慢速序列时空闲等待。

Continuous batching 在每个解码步骤之间操作。将运行中的序列集合称为 `RUNNING` 列表。在每次迭代中：

1. 任何在 `RUNNING` 中刚达到 EOS 或 max_tokens 的序列被移除。
2. 调度器查看等待队列。如果有空闲的 KV 块，它接纳新序列（预填充或恢复）。
3. 前向传播在现在 `RUNNING` 中的任何序列上运行，每个序列产生一个新 token。

批次大小从不填充到固定数量。处于输出不同位置的序列共享一个融合前向传播。在 2026 年的 vLLM 中，这被称为 `V1 scheduler`。关键不变性：调度器每个解码迭代运行一次，而不是每个请求一次。

### Chunked Prefill 保护 TTFT 尾部

预填充是计算密集型的。在 Llama 3.3 70B 上，一个 32k token 的提示在一个 H100 上需要约 800 ms 的纯预填充。当预填充运行时，批次中每个其他序列的解码 token 都在等待。在一个服务循环中，一个长提示的首 token 延迟 (TTFT) 变成了数十个其他用户的 token 间延迟 (ITL) 尖峰。

Chunked prefill 将预填充拆分为固定大小的块（默认 512 token），并将每个块作为一个单元调度。在块之间，调度器可以推进解码序列一个 token。你以很小的绝对预填充延迟代价（每块几毫秒）换取更低的解码时间抖动。在已发表的基准测试中，混合负载下的 P99 ITL 从约 50 ms 下降到约 15 ms。

### 三个默认特性相互依赖

这三个特性相互假设。PagedAttention 为调度器提供了细粒度的 KV 资源进行交易。Continuous batching 需要这种细粒度资源，以便接纳新序列不会强制全局重排。Chunked prefill 是调度器在同一个 `RUNNING` 列表上做出的决策 — 它只是一个额外的调度策略，而不是一个独立的系统。

你不需要知道每个标志。你需要知道调度器优化什么：在 KV 块预算约束下，受 chunked prefill 切片影响的 goodput。

### 2026 年 v0.18.0 的陷阱

在 vLLM v0.18.0 中，你不能同时使用 `--enable-chunked-prefill` 和 draft-model 推测解码 (`--speculative-model`)。文档记录的例外是 V1 调度器中的 N-gram GPU 推测解码。团队如果没读发行说明就开启所有标志，会在启动时遇到运行时错误，而不是软回归。如果你的推测增益值得启用 chunked prefill，重新审视这个选择 — 2026 年正确的答案通常是 EAGLE-3 不加 chunked prefill，而不是 draft model 加无法编译的 chunked prefill。

### 你应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三者全开：2,200-2,400 tok/s。
- 相同模型，默认 vLLM（无 chunked prefill）：约 1,800 tok/s。
- 相同模型，朴素 PyTorch 前向循环：约 600 tok/s。
- PagedAttention 在生产负载下的 KV 碎片浪费：<4%。
- 混合负载下的 P99 ITL：有 chunked prefill 约 15 ms，无 chunked prefill 约 50 ms。

### 调度器的样子

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # 在一个批次中调度预填充块 + 解码
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # 例如 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # 一次融合 GPU 调用
```

`code/main.py` 正是这个循环，使用 stdlib Python 和模拟的 token 计数及前向延迟。运行它展示了 chunked prefill 如何在长预填充期间保持解码序列存活。

```figure
tensor-parallel
```

## Use It

`code/main.py` 模拟一个 vLLM 风格的调度器，具有可切换的特性。运行它可以看到：

- `NAIVE` 模式：一次一个请求，无批处理。
- `STATIC` 模式：填充并等待，经典批处理。
- `CONTINUOUS` 模式：迭代级别的接纳和释放。
- `CONTINUOUS + CHUNKED` 模式：预填充切片与解码交错。

输出显示总吞吐量（每虚拟秒 token 数）、TTFT 均值和 P99 ITL。`CONTINUOUS + CHUNKED` 行应在混合流量上占优。

## Ship It

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定服务配置（批次大小、KV 内存利用率、chunked prefill 大小、推测配置），它生成一个调度器诊断，指出三个默认特性中哪个是瓶颈以及如何调优。

## Exercises

1. 运行 `code/main.py`。在混合短请求和长请求的工作负载上比较 `STATIC` 和 `CONTINUOUS`。吞吐量差距来自哪里 — 预填充效率、解码效率还是尾部延迟？
2. 修改玩具调度器，添加 `--max-num-batched-tokens`。在运行 Llama 3.3 70B FP8 的 H100 上，合适的值是多少？（提示：它是 KV 块大小和空闲块数量的函数，而非原始 HBM。）
3. 重新阅读 vLLM v0.18.0 发行说明。哪些标志组合是互斥的？列出它们。
4. 计算 1,000 个请求的 KV 缓存碎片浪费，平均输出 1,500 token，标准差 600 token，在 (a) 每个请求连续分配，最大 8192，(b) PagedAttention 使用 16 token 块。
5. 用一段话解释为什么 chunked prefill 有助于 P99 ITL 但单独来看不影响吞吐量。在实践中，吞吐量提升来自哪里？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| PagedAttention | "KV 技巧" | KV 缓存的固定大小块分配器；碎片率 <4% |
| Block table | "页表" | 每个序列从逻辑 token 位置到物理 KV 块的映射 |
| Continuous batching | "动态批处理，但做对了" | 每个解码迭代做出接纳/释放决策 |
| Chunked prefill | "预填充拆分" | 将长预填充分解为 512 token 切片，与解码交错 |
| TTFT | "首 token 时间" | 预填充 + 队列 + 网络；长提示时由预填充主导 |
| ITL | "token 间延迟" | 连续解码 token 之间的时间；由批次大小主导 |
| Goodput | "满足 SLO 的吞吐量" | 每个请求仍达到 TTFT 和 ITL 目标的 token/秒 |
| V1 scheduler | "新调度器" | vLLM 的 2026 年调度器；N-gram 推测解码是与 chunked prefill 兼容的路径 |
| `--gpu-memory-utilization` | "内存旋钮" | 在权重和激活之后为 KV 块保留的 HBM 比例 |

## Further Reading

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于 chunked prefill 和推测解码兼容性的官方来源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 年发布节奏和版本特定行为。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 仍然定义如何思考分配器的原始文章。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — 碎片分析和调度器设计。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — 带有火焰图的详细 V1 调度器讲解。