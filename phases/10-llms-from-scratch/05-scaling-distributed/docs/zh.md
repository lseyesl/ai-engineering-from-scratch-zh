# 扩展：分布式训练、FSDP、DeepSpeed

> 你的 124M 模型在一张 GPU 上训练好了。现在试试 70 亿参数。模型放不进内存。单台机器上数据需要数周时间。分布式训练在大规模下不是可选项，而是唯一的路径。

**类型:** Build
**语言:** Python
**前置知识:** Phase 10, 课程 04 (Pre-Training a Mini GPT)
**时间:** ~120 分钟

## 学习目标

- 解释三种并行方式（数据并行、张量并行、流水线并行），以及每种方式在何种模型规模和集群规模下是必要的
- 使用 PyTorch DDP 实现数据并行训练，在多个 GPU 之间同步梯度
- 计算给定模型大小的内存预算（权重 + 优化器状态 + 梯度 + 激活值），以确定最低硬件要求
- 配置 FSDP 或 DeepSpeed ZeRO 阶段，跨 GPU 分片模型状态，使超出单 GPU 内存的模型能够训练

## 问题

一个 FP16 的 7B 参数模型仅权重就需要 14GB。Adam 优化器存储每个参数的额外两份副本（一阶和二阶矩估计）。这又是 28GB。反向传播期间的梯度再增加 14GB。在存储任何激活值之前，你已经用掉了 56GB。

NVIDIA A100 拥有 80GB 内存。

80GB 中已消耗 56GB。剩下 24GB 给激活值——前向传播期间计算的中间值，这些值必须保留以备反向传播。对于 2048 token 序列和 4096 维模型，单层的激活值约使用 64MB。32 层需要每样本 2GB。batch size 为 8 时需要 16GB。你有 24GB。batch size 为 12 时就爆了。

现在试试 70B 参数。仅权重：FP16 下 140GB。一张 GPU 放不下。你至少需要 2 张 A100（2 x 80GB = 160GB）才能容纳权重。加上优化器状态和梯度，你需要更多：最少 3 张 GPU，实际取决于分片策略，需要 8-16 张。

Llama 3 405B 在 16,384 张 NVIDIA H100 GPU 上训练。训练运行估计花费了 1 亿美元的计算成本。DeepSeek V3 通过在架构（Mixture of Experts 意味着每个 token 只激活一小部分参数）和训练效率上的巧妙设计，以大约 560 万美元训练了可比模型。

本课程涵盖使大规模训练成为可能的四种策略：数据并行、张量并行、流水线并行和全分片数据并行。你将在纯 Python 中模拟每一种策略，在接触任何分布式训练框架之前理解其机制。

## 概念

### 为什么需要分布式

以下是实际模型的内存计算。每个数字都是计算出来的，不是估算的。

| 模型 | 参数量 | 权重 (FP16) | Adam 状态 | 梯度 (FP16) | 总计（不含激活值） |
|-------|--------|----------------|-------------|------------------|----------------------|
| GPT-2 Small | 124M | 248 MB | 992 MB | 248 MB | 1.5 GB |
| Llama 3 8B | 8B | 16 GB | 64 GB | 16 GB | 96 GB |
| Llama 3 70B | 70B | 140 GB | 560 GB | 140 GB | 840 GB |
| Llama 3 405B | 405B | 810 GB | 3,240 GB | 810 GB | 4,860 GB |

"Adam 状态"一列是杀手。Adam 为每个参数存储运行均值 (m) 和运行方差 (v)，两者都是 FP32。对于 70B 模型，即 70B x 4 字节 x 2 = 560GB。仅优化器就需要七张 A100。

单张 H100 有 80GB。Llama 3 405B 至少需要 61 张 H100 来容纳权重、优化器和梯度。加上激活值，数量进一步增长。Meta 使用 16,384 张 GPU 不是因为他们想，而是因为他们不得不这样做。

### 数据并行

最简单的分布式策略。将完整模型复制到 N 张 GPU 上。将每个训练 batch 分成 N 等份。每张 GPU 在其数据分片上运行前向和反向传播。反向传播后，在所有 GPU 上平均梯度。每张 GPU 使用相同的平均梯度更新其权重副本，保持所有副本同步。

**优点：** 线性吞吐量扩展。N 张 GPU 每步处理 N 倍数据。通信仅限于梯度平均，可与计算重叠。

**缺点：** 每张 GPU 持有模型的完整副本，包括优化器状态和梯度。对于 70B 模型，每张 GPU 需要 840GB。数据并行不减少每 GPU 内存，只减少训练时间。

**数学计算：** 有效 batch size = per_gpu_batch_size x N。对于 N=64 张 GPU，每 GPU batch 为 16，有效 batch 为 1,024。Llama 3 每步使用 1600 万 token 的有效 batch size。

```mermaid
graph TD
    subgraph DataParallel["Data Parallelism (N=4 GPUs)"]
        B["Full Batch\n(1024 samples)"] --> S["Split"]
        S --> G1["GPU 1\nFull Model Copy\n256 samples"]
        S --> G2["GPU 2\nFull Model Copy\n256 samples"]
        S --> G3["GPU 3\nFull Model Copy\n256 samples"]
        S --> G4["GPU 4\nFull Model Copy\n256 samples"]
        G1 --> AR["AllReduce\nAverage Gradients"]
        G2 --> AR
        G3 --> AR
        G4 --> AR
        AR --> U["Update\n(identical on all GPUs)"]
    end

    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style G1 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G2 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G3 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G4 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style AR fill:#1a1a2e,stroke:#51cf66,color:#fff
    style U fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### 张量并行

将单个层拆分到多张 GPU。一个矩阵乘法被划分到多张 GPU 上，每张计算部分结果。

考虑前馈层中形状为 (8192, 8192) 的权重矩阵。使用 4 路张量并行，每张 GPU 持有 (8192, 2048) 的分片。每张 GPU 将输入乘以其分片，产生部分结果。部分结果通过 all-reduce 或 all-gather 组合产生完整输出。

**优点：** 减少模型权重的每 GPU 内存。70B 模型拆分到 8 张 GPU 意味着每张 GPU 持有约 8.75B 参数的权重。

**缺点：** 每层之后需要快速的 GPU 间通信。每个 matmul 后的 all-reduce 增加延迟。这在 NVLink（同一节点内 GPU 间 900 GB/s）下效果良好，但在跨节点 InfiniBand（400 Gb/s，约 50 GB/s）连接时效果不佳。张量并行几乎总是限于单个节点内（8 张 GPU）。

**实际使用：** Megatron-LM 开创了张量并行。Llama 3 405B 在每个节点内使用 8 路张量并行。

### 流水线并行

按层拆分模型。GPU 1 运行第 1-8 层。GPU 2 运行第 9-16 层。GPU 3 运行第 17-24 层。GPU 4 运行第 25-32 层。数据流经流水线：GPU 1 计算其层并将激活值发送给 GPU 2，GPU 2 计算其层并发送给 GPU 3，依此类推。

**优点：** GPU 间通信最小——仅在层边界传输激活值，与梯度或权重相比很小。由于带宽要求低，可以在节点间工作。

**缺点：** 流水线气泡。当 GPU 4 正在计算微 batch 1 的前向传播时，GPU 1、2 和 3 处于空闲（它们已经完成了自己的部分）。反向传播期间，模式反转。使用朴素流水线，N 个流水线阶段的 GPU 利用率仅为 1/N。

**GPipe 和 PipeDream** 通过将 batch 拆分为微 batch 来解决气泡问题。GPU 1 在完成微 batch 1 的前向传播后立即开始微 batch 2。这使计算在流水线阶段之间重叠。对于 M 个微 batch 和 N 个阶段，气泡比例降至 (N-1)/M。使用 M=16 个微 batch 和 N=4 个阶段，气泡为 3/16 = 18.75% 的空闲时间。

### FSDP：全分片数据并行

FSDP 结合了数据并行的可扩展性和分片的内存效率。不是每张 GPU 持有模型的完整副本，而是每张 GPU 只持有 1/N 的参数、梯度和优化器状态。

在某一层的前向传播之前，FSDP 执行 **all-gather** 从所有 GPU 收集完整参数到每张 GPU 内存中。前向传播之后，每张 GPU 丢弃非本地参数。反向传播期间，all-gather 再次运行以重建参数用于梯度计算。反向传播之后，**reduce-scatter** 分发梯度分片，使每张 GPU 只存储 1/N 的梯度。

**70B 模型在 8 张 GPU 上的数学计算：**

| 组件 | 无 FSDP | 有 FSDP |
|-----------|-------------|-----------|
| 权重 (FP16) | 每 GPU 140 GB | 每 GPU 17.5 GB |
| Adam 状态 (FP32) | 每 GPU 560 GB | 每 GPU 70 GB |
| 梯度 (FP16) | 每 GPU 140 GB | 每 GPU 17.5 GB |
| **总计** | **每 GPU 840 GB** | **每 GPU 105 GB** |

没有 FSDP，你无法将 70B 模型放入单个 80GB GPU。使用 FSDP 在 8 张 GPU 上，每张 GPU 使用 105GB——等等，这仍然放不下。你至少需要 16 张 GPU 才能降到每 GPU 80GB 以下，或者将 FSDP 与激活检查点结合（在反向传播期间重新计算激活值，而不是存储它们）。

通信成本高于普通数据并行，因为每层之前需要 all-gather。但内存节省使以前不可能的训练运行成为可能。

```mermaid
graph TD
    subgraph FSDP["FSDP: Fully Sharded Data Parallel (4 GPUs)"]
        direction TB
        S["Model: 4 layers, sharded"]

        subgraph GPU1["GPU 1"]
            G1S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end
        subgraph GPU2["GPU 2"]
            G2S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end
        subgraph GPU3["GPU 3"]
            G3S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end
        subgraph GPU4["GPU 4"]
            G4S["Shard: 1/4 params\n1/4 optimizer\n1/4 gradients"]
        end

        AG["All-Gather\n(reconstruct full params\nbefore each layer)"]
        FW["Forward Pass\n(full params temporarily)"]
        RS["Reduce-Scatter\n(distribute gradient shards\nafter backward)"]

        S --> GPU1
        S --> GPU2
        S --> GPU3
        S --> GPU4
        GPU1 --> AG
        GPU2 --> AG
        GPU3 --> AG
        GPU4 --> AG
        AG --> FW
        FW --> RS
    end

    style G1S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G2S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G3S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G4S fill:#1a1a2e,stroke:#0f3460,color:#fff
    style AG fill:#1a1a2e,stroke:#e94560,color:#fff
    style FW fill:#1a1a2e,stroke:#51cf66,color:#fff
    style RS fill:#1a1a2e,stroke:#e94560,color:#fff
```

### DeepSpeed ZeRO

DeepSpeed 的 ZeRO（零冗余优化器）在概念上与 FSDP 相同，但由微软独立开发。它定义了三个阶段，每个阶段更积极地分片：

| 阶段 | 分片内容 | 内存节省 | 通信量 |
|-------|--------|---------------|---------------|
| ZeRO-1 | 仅优化器状态 | ~4x 减少 | 与数据并行相同 |
| ZeRO-2 | + 梯度 | ~8x 减少 | 略多 |
| ZeRO-3 | + 参数 | ~Nx 减少（N 张 GPU） | 每层 all-gather |

ZeRO-3 等同于 FSDP。命名不同，机制相同。PyTorch 在 DeepSpeed 验证了概念之后，将 FSDP 作为原生实现添加。

DeepSpeed 还引入了 ZeRO-Offload（将优化器状态卸载到 CPU RAM，更便宜且更大）和 ZeRO-Infinity（卸载到 NVMe SSD）。这些以计算速度换取内存容量——卸载的操作更慢，但释放了 GPU 内存。

### 混合精度训练

现代训练同时使用多种浮点格式：

- **前向传播**：FP16 或 BF16（16 位）。内存为 FP32 的一半。矩阵乘法在 tensor core 上运行速度提升 2 倍。
- **主权重**：FP32（32 位）。由优化器维护，保证权重更新时的数值精度。
- **损失缩放**：在反向传播前将损失乘以一个大常数，防止 FP16 梯度下溢到零。在优化器步骤前除以相同的常数。

BF16（Brain Float 16）与 FP32 具有相同的指数范围（8 位指数），但精度降低（7 位尾数 vs FP32 的 23 位）。它很少需要损失缩放，因为它能表示相同的数值范围。FP16 有 5 位指数和 10 位尾数——它能表示精细粒度的值，但在极端数值时会溢出/下溢。

Google 的 TPU 原生使用 BF16。NVIDIA 的 A100 和 H100 都支持 FP16 和 BF16。业界已基本转向 BF16，因为它消除了损失缩放的麻烦。

**7B 模型的内存对比：**

| 精度 | 权重 | 优化器 | 梯度 | 总计 |
|-----------|---------|-----------|-----------|-------|
| 全部 FP32 | 28 GB | 56 GB | 28 GB | 112 GB |
| 混合 (BF16 + FP32 master) | 14 GB | 56 GB | 14 GB | 84 GB |

此模型的混合精度节省 28GB。优化器状态无论哪种精度都保持在 FP32——这是大部分内存的去向。

### Megatron-LM 与 3D 并行

真正的大规模训练结合了所有三种并行方式：

- **数据并行**跨节点组（扩展 batch size）
- **张量并行**在节点内（将层拆分到 8 张 GPU）
- **流水线并行**跨节点（将层组拆分到多台机器）

Llama 3 405B 在 16,384 张 H100 上：
- 每个节点内 8 路张量并行（每节点 8 张 GPU）
- 跨节点 16 路流水线并行（16 个流水线阶段）
- 剩余维度上 128 路数据并行（16,384 / 8 / 16 = 128）

这种 3D 分解（8 x 16 x 128 = 16,384）就是你扩展到数千张 GPU 的方式。每张 GPU 看到不同的数据分片（数据并行），持有每层的一个切片（张量并行），并计算不同的层组（流水线并行）。

DeepSeek V3 采取了不同的方法。其 Mixture of Experts 架构每个 token 仅激活 671B 参数中的 37B。这意味着每张 GPU 只需要计算（并存储激活值）活跃参数。他们在 2,048 张 H800 GPU 上训练——不到 Meta GPU 数量的 1/8——花费 560 万美元，而 Meta 估计花费 1 亿美元。

```mermaid
graph TD
    subgraph ThreeD["3D Parallelism (Llama 3 405B)"]
        direction TB
        subgraph DP["Data Parallel (128-way)\nSplit batch across 128 groups"]
            subgraph PP["Pipeline Parallel (16-way)\nSplit layers across 16 stages"]
                subgraph TP["Tensor Parallel (8-way)\nSplit each layer across 8 GPUs"]
                    G1["GPU 1\nSlice of layers 1-N"]
                    G2["GPU 2\nSlice of layers 1-N"]
                    G8["GPU 8\nSlice of layers 1-N"]
                end
            end
        end
    end

    N1["Total: 8 x 16 x 128 = 16,384 GPUs"]

    style G1 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G2 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style G8 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style N1 fill:#1a1a2e,stroke:#e94560,color:#fff
```

```figure
paged-kv-cache
```

## 构建它

### 第 1 步：模拟数据并行

将 batch 拆分到模拟的 GPU 上。每张 GPU 在其数据分片上计算前向传播。平均"梯度"（我们模拟为损失值）。

```python
import numpy as np

def simulate_data_parallelism(data, num_gpus, model_fn):
    batch_size = len(data)
    shard_size = batch_size // num_gpus
    remainder = batch_size % num_gpus

    gpu_losses = []
    gpu_gradients = []

    offset = 0
    for gpu_id in range(num_gpus):
        extra = 1 if gpu_id < remainder else 0
        shard = data[offset:offset + shard_size + extra]
        offset += shard_size + extra

        loss, grad = model_fn(shard)
        gpu_losses.append(loss)
        gpu_gradients.append(grad)

    avg_loss = np.mean(gpu_losses)
    avg_gradient = np.mean(gpu_gradients, axis=0)

    return avg_loss, avg_gradient
```

All-reduce 操作（平均梯度）是数据并行中唯一的通信。在实践中，这使用 NVIDIA GPU 上的 NCCL 库，它实现了 ring all-reduce：每张 GPU 将其 1/N 的梯度发送给邻居，从另一个邻居接收 1/N，经过 N-1 步后每张 GPU 都拥有完整的平均值。总通信量：2 x gradient_size x (N-1)/N，对于大 N 趋近于 2 倍梯度大小。

### 第 2 步：模拟张量并行

将权重矩阵拆分到多张 GPU。每张 GPU 计算部分矩阵乘法。组合结果。

```python
def simulate_tensor_parallelism(input_data, weight_matrix, num_gpus):
    d_in, d_out = weight_matrix.shape
    assert d_out % num_gpus == 0, f"d_out {d_out} not divisible by num_gpus {num_gpus}"
    shard_size = d_out // num_gpus

    partial_results = []
    for gpu_id in range(num_gpus):
        start = gpu_id * shard_size
        end = start + shard_size
        weight_shard = weight_matrix[:, start:end]

        partial = input_data @ weight_shard
        partial_results.append(partial)

    full_output = np.concatenate(partial_results, axis=-1)

    direct_output = input_data @ weight_matrix
    error = np.abs(full_output - direct_output).max()

    return full_output, error
```

误差应该正好为零（或机器精度）。张量并行在数学上是精确的——它产生与在一张 GPU 上计算完整 matmul 相同的结果。拆分是沿着输出维度进行的，因此每张 GPU 产生不同的列块，拼接重建完整结果。

对于列并行线性层（拆分输出维度），你进行拼接。对于行并行（拆分输入维度），你进行求和。在 transformer FFN 中，第一个线性层（扩展）使用列并行，第二个线性层（收缩）使用行并行。这避免了两层之间的 all-reduce。

### 第 3 步：模拟流水线并行

将模型的层拆分到虚拟 GPU 上。展示早期阶段在等待后期阶段计算时处于空闲的气泡问题。

```python
def simulate_pipeline_parallelism(num_layers, num_stages, num_microbatches):
    layers_per_stage = num_layers // num_stages

    timeline = {}
    clock = 0

    for mb in range(num_microbatches):
        for stage in range(num_stages):
            start_time = max(
                timeline.get((stage, mb - 1, "fwd"), (0, 0))[1] if mb > 0 else 0,
                timeline.get((stage - 1, mb, "fwd"), (0, 0))[1] if stage > 0 else 0,
            )
            end_time = start_time + layers_per_stage
            timeline[(stage, mb, "fwd")] = (start_time, end_time)

    last_fwd_end = max(v[1] for v in timeline.values())

    for mb in range(num_microbatches - 1, -1, -1):
        for stage in range(num_stages - 1, -1, -1):
            deps = [last_fwd_end]
            if mb < num_microbatches - 1 and (stage, mb + 1, "bwd") in timeline:
                deps.append(timeline[(stage, mb + 1, "bwd")][1])
            if stage < num_stages - 1 and (stage + 1, mb, "bwd") in timeline:
                deps.append(timeline[(stage + 1, mb, "bwd")][1])
            start_time = max(deps)
            end_time = start_time + layers_per_stage
            timeline[(stage, mb, "bwd")] = (start_time, end_time)

    total_time = max(v[1] for v in timeline.values())
    compute_time = num_microbatches * num_stages * layers_per_stage * 2
    bubble_fraction = 1.0 - compute_time / (total_time * num_stages)

    return timeline, total_time, bubble_fraction
```

对于 4 个阶段和 1 个微 batch，气泡分数为 75%——任何时候四张 GPU 中有三张空闲。对于 16 个微 batch，降至约 19%。消除气泡的代价是内存：你必须同时存储所有正在进行的微 batch 的激活值。

### 第 4 步：内存计算器

计算训练任何模型大小的精确内存需求。

```python
def memory_calculator(
    params_billions,
    precision_bytes=2,
    optimizer="adam",
    num_gpus=1,
    sharding="none",
    sequence_length=2048,
    batch_size_per_gpu=1,
    hidden_dim=None,
    num_layers=None,
):
    params = params_billions * 1e9

    weight_memory = params * precision_bytes

    if optimizer == "adam":
        optimizer_memory = params * 4 * 2
    elif optimizer == "sgd":
        optimizer_memory = params * 4
    else:
        optimizer_memory = 0

    gradient_memory = params * precision_bytes

    total_no_activation = weight_memory + optimizer_memory + gradient_memory

    if hidden_dim and num_layers:
        activation_per_layer = (
            sequence_length * batch_size_per_gpu * hidden_dim * precision_bytes * 4
        )
        activation_memory = activation_per_layer * num_layers
    else:
        activation_memory = params * precision_bytes * 0.5

    if sharding == "fsdp" or sharding == "zero3":
        weight_memory /= num_gpus
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
    elif sharding == "zero2":
        optimizer_memory /= num_gpus
        gradient_memory /= num_gpus
    elif sharding == "zero1":
        optimizer_memory /= num_gpus

    per_gpu_total = weight_memory + optimizer_memory + gradient_memory + activation_memory

    return {
        "params_billions": params_billions,
        "weights_gb": weight_memory / 1e9,
        "optimizer_gb": optimizer_memory / 1e9,
        "gradients_gb": gradient_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "per_gpu_total_gb": per_gpu_total / 1e9,
        "total_across_gpus_gb": per_gpu_total * num_gpus / 1e9,
        "fits_on_80gb": per_gpu_total / 1e9 <= 80,
        "num_gpus": num_gpus,
        "sharding": sharding,
    }
```

这个计算器回答每个 ML 工程师都会问的问题："我需要多少张 GPU？"输入模型大小，看看是否放得下。调整分片策略，直到每 GPU 总量降到 80GB 以下。

### 第 5 步：混合精度模拟

比较 FP32、FP16 和混合精度训练之间的内存使用。

```python
def mixed_precision_comparison(params_billions):
    params = params_billions * 1e9

    fp32_weights = params * 4
    fp32_optimizer = params * 4 * 2
    fp32_gradients = params * 4
    fp32_total = fp32_weights + fp32_optimizer + fp32_gradients

    fp16_weights = params * 2
    fp16_master = params * 4
    fp16_optimizer = params * 4 * 2
    fp16_gradients = params * 2
    fp16_total = fp16_weights + fp16_master + fp16_optimizer + fp16_gradients

    mixed_weights = params * 2
    mixed_optimizer = params * 4 * 2
    mixed_gradients = params * 2
    mixed_total = mixed_weights + mixed_optimizer + mixed_gradients

    return {
        "fp32_total_gb": fp32_total / 1e9,
        "fp16_with_master_gb": fp16_total / 1e9,
        "mixed_bf16_gb": mixed_total / 1e9,
        "savings_vs_fp32": 1 - mixed_total / fp32_total,
    }
```

对大多数人来说最大的意外：混合精度并没有将内存减半。优化器状态（Adam 的 m 和 v）无论精度如何都保持在 FP32。对于 7B 模型，FP32 训练使用 112GB。混合精度使用 84GB。这是 25% 的减少，不是 50%。优化器占主导地位。

## 使用它

### 运行所有模拟

```python
def run_all_demos():
    print("=" * 70)
    print("DATA PARALLELISM SIMULATION")
    print("=" * 70)

    np.random.seed(42)
    data = np.random.randn(64, 32)
    weight = np.random.randn(32, 16)

    def model_fn(batch):
        output = batch @ weight
        loss = np.mean(output ** 2)
        grad = 2 * batch.T @ (batch @ weight) / len(batch)
        return loss, grad

    for n_gpus in [1, 2, 4, 8]:
        loss, grad = simulate_data_parallelism(data, n_gpus, model_fn)
        print(f"  {n_gpus} GPUs: loss={loss:.4f}, grad_norm={np.linalg.norm(grad):.4f}")

    print()
    print("=" * 70)
    print("TENSOR PARALLELISM SIMULATION")
    print("=" * 70)

    x = np.random.randn(4, 8192)
    W = np.random.randn(8192, 8192)

    for n_gpus in [1, 2, 4, 8]:
        output, error = simulate_tensor_parallelism(x, W, n_gpus)
        print(f"  {n_gpus} GPUs: output_shape={output.shape}, max_error={error:.2e}")

    print()
    print("=" * 70)
    print("PIPELINE PARALLELISM SIMULATION")
    print("=" * 70)

    for n_mb in [1, 4, 8, 16, 32]:
        _, total_t, bubble = simulate_pipeline_parallelism(32, 4, n_mb)
        print(f"  {n_mb:2d} micro-batches: total_time={total_t:4d}, bubble={bubble:.1%}")

    print()
    print("=" * 70)
    print("MEMORY CALCULATOR")
    print("=" * 70)

    configs = [
        (7, "none", 1),
        (7, "fsdp", 8),
        (70, "none", 1),
        (70, "fsdp", 8),
        (70, "fsdp", 16),
        (405, "fsdp", 64),
        (405, "fsdp", 128),
    ]

    print(f"  {'Model':>8} {'Sharding':>8} {'GPUs':>5} {'Per-GPU':>10} {'Fits 80GB':>10}")
    print("  " + "-" * 50)
    for params, shard, gpus in configs:
        result = memory_calculator(params, num_gpus=gpus, sharding=shard)
        fits = "Yes" if result["fits_on_80gb"] else "No"
        print(f"  {params:>6}B {shard:>8} {gpus:>5} {result['per_gpu_total_gb']:>8.1f}GB {fits:>10}")

    print()
    print("=" * 70)
    print("MIXED PRECISION COMPARISON")
    print("=" * 70)

    for params_b in [7, 13, 70, 405]:
        result = mixed_precision_comparison(params_b)
        print(f"  {params_b}B: FP32={result['fp32_total_gb']:.0f}GB, "
              f"Mixed BF16={result['mixed_bf16_gb']:.0f}GB, "
              f"Savings={result['savings_vs_fp32']:.0%}")
```

## 交付它

本课程产出 `outputs/prompt-distributed-training-planner.md`——一个提示词，接收模型大小和可用硬件，然后生成完整的分布式训练计划：并行策略、内存预算、通信开销和预期吞吐量。

## 练习

1. 修改内存计算器以包含激活检查点。使用检查点时，只每 K 层存储激活值（典型 K=1，即全部重新计算）。展示内存-计算权衡：检查点节省多少内存，又使训练减慢多少（完全检查点时大约多 33% 的计算量）？

2. 扩展流水线并行模拟，实现 PipeDream 使用的 1F1B（一次前向、一次反向）调度。比较 4 个阶段和 8 个微 batch 下与朴素调度的气泡分数。1F1B 调度应该有更小的峰值内存，因为它更早开始反向传播。

3. 实现梯度累积模拟。不是在每个微 batch 后做 all-reduce，而是本地累积 K 步的梯度，然后进行 all-reduce。展示这如何将通信减少 K 倍，同时产生相同的最终梯度（因此训练也相同）。

4. 构建成本估算器。给定模型大小、目标 token 数量、GPU 类型（A100 每小时 2 美元，H100 每小时 3.50 美元）和并行策略，估算总训练成本（美元）。对照已知成本验证：Llama 3 405B 据报道花费约 1 亿美元，DeepSeek V3 花费约 560 万美元。

5. 将 ZeRO-Offload 添加到内存计算器。假设每节点 CPU RAM 为 512GB，NVMe 为 2TB。展示将优化器状态卸载到 CPU 如何使 70B 模型在 4 张 GPU 上训练（而不是 16 张），代价是优化器步骤减慢 30-50%。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 数据并行 | "将模型复制到每张 GPU" | 每张 GPU 处理不同的数据分片；每步后通过 all-reduce 平均梯度 |
| 张量并行 | "将层拆分到多张 GPU" | 划分权重矩阵，使每张 GPU 计算部分 matmul；需要快速 NVLink 互连 |
| 流水线并行 | "将层拆分到多张 GPU" | 每张 GPU 运行不同的层组；数据流经流水线，使用微 batch 减少气泡 |
| FSDP | "分片一切" | 全分片数据并行——每张 GPU 持有 1/N 的权重、梯度和优化器状态；计算前执行 all-gather |
| ZeRO | "DeepSpeed 版的 FSDP" | 零冗余优化器，分 3 个阶段：分片优化器（阶段 1），+ 梯度（阶段 2），+ 参数（阶段 3） |
| All-reduce | "跨 GPU 平均" | 集体操作，每张 GPU 最终得到所有 GPU 输入的总和（或平均）——通常实现为 ring all-reduce |
| All-gather | "从所有 GPU 收集" | 集体操作，每张 GPU 最终得到所有 GPU 数据的拼接——FSDP 中用于重建完整参数 |
| Reduce-scatter | "求和并分发" | 集体操作，减少（求和）数据并将不同块分发给不同 GPU——FSDP 中用于梯度分片 |
| 混合精度 | "以半精度训练" | 前向/反向使用 FP16/BF16，优化器状态使用 FP32——节省约 25% 内存，不是 50%，因为优化器占主导 |
| 流水线气泡 | "流水线中的空闲时间" | GPU 等待上一阶段数据时空闲的时间比例——通过使用更多微 batch 减少 |

## 延伸阅读

- [Rajbhandari et al., 2020 — "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"](https://arxiv.org/abs/1910.02054) —— DeepSpeed ZeRO 论文，定义了三个分片阶段
- [Shoeybi et al., 2020 — "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"](https://arxiv.org/abs/1909.08053) —— NVIDIA 用于 transformer 的张量并行
- [Narayanan et al., 2021 — "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"](https://arxiv.org/abs/2104.04473) —— 结合数据、张量和流水线的 3D 并行
- [Zhao et al., 2023 — "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"](https://arxiv.org/abs/2304.11277) —— PyTorch 的原生 FSDP 实现
- [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) —— 16,384 GPU 训练的 3D 并行细节
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) —— MoE 架构如何将训练成本降低一个数量级
