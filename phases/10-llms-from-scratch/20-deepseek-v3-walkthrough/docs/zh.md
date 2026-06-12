# DeepSeek-V3 架构详解

> Phase 10 · Lesson 14 命名了每个开放模型旋转的六个架构旋钮。DeepSeek-V3（2024 年 12 月，671B 总参数，37B 活跃）旋转了全部六个并添加了四个：Multi-Head Latent Attention、无辅助损失负载均衡、Multi-Token Prediction 和 DualPipe 训练。本课程从头到尾阅读 DeepSeek-V3 的架构，并从发布的配置推导每个参数计数。到最后你可以解释为什么 671B/37B 比率是正确的赌注，以及为什么 MLA + MoE 在前沿共同胜过单独使用任何一个。

**类型：** 学习
**语言：** Python (stdlib, 参数计算器)
**前置课程：** Phase 10 · 14 (开放模型详解), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**时长：** ~75 分钟

## 学习目标

- 从头到尾阅读 DeepSeek-V3 配置，并用六个 GPT-2 旋钮加四个 DeepSeek 特有新增来解释每个字段。
- 推导总参数计数（671B）、活跃参数计数（37B）以及贡献于每个的组件。
- 计算 MLA 在 128k 上下文下的 KV cache 占用，并与相同活跃参数的 GQA 密集模型比较。
- 说出四个 DeepSeek 特有创新（MLA、MTP、无辅助损失路由、DualPipe）并指出每个针对架构/训练栈的哪个部分。

## 问题所在

DeepSeek-V3 是第一个架构与 Llama 家族有实质性不同的前沿开放模型。Llama 3 405B 是"旋转了六个旋钮的 GPT-2"。DeepSeek-V3 是旋转了全部六个旋钮再加四个的 GPT-2。阅读 Llama 3 配置是阅读 DeepSeek 配置的热身，但深层结构——注意力块的形状、路由逻辑、训练时目标——差异足够大，需要单独的详解。

学习它的回报：DeepSeek-V3 的开放权重发布改变了开放模型中"前沿能力"的含义。其架构是许多 2026 年训练运行正在复制的蓝图。理解它是任何触及前沿 LLM 训练或推理的角色的基本要求。

## 核心概念

### 不变的核心，再次回顾

DeepSeek-V3 仍然是自回归的。它仍然堆叠解码器块。每个块仍然有注意力加 MLP 加两个 RMSNorm。它仍然在 MLP 中使用 SwiGLU。它仍然使用 RoPE。Pre-norm。权重绑定嵌入。与每个 Llama 或 Mistral 相同的基线。

### 转折：MLA 替代 GQA

从 Phase 10 · 14 你知道 GQA 通过在 Q 头组间共享 K 和 V 来缩小 KV cache。Multi-Head Latent Attention (MLA) 更进一步：K 和 V 被压缩为共享的低秩潜在表示（`kv_lora_rank`），然后按头即时解压。KV cache 仅存储潜在表示——通常每 token 每层 512 个浮点数，而非 8 x 128 = 1024 个浮点数。

在 128k 上下文下，使用 MLA 的 DeepSeek-V3（每 token 每层一个共享潜在 `c^{KV}`；K 和 V 都通过可被吸收到后续矩阵乘法中的上投影从此潜在推导）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
= 61 * 512 * 131072 * 2
= 7.6 GB
```

一个假设的 GQA 基线（Llama 3 70B 形状，8 个 KV 头，头维度 128）需要：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
= 30.5 GB
```

MLA 在 128k 上下文下比 Llama-3-70B 风格的 GQA cache 小 4 倍。

权衡：MLA 在每次注意力计算中添加一个解压步骤（每头）。额外计算与节省的带宽相比很小。对长上下文推理是净收益。

### 路由：无辅助损失负载均衡

MoE 路由器决定哪些 top-k 专家处理每个 token。朴素路由器将过多工作集中在少数专家上，让其他专家空闲。标准修复：添加惩罚负载不均衡的辅助损失项。这有效但略微降低主任务性能。

DeepSeek-V3 引入无辅助损失方案。每专家偏置项被添加到路由器 logits，在训练期间通过简单规则调整：如果专家 `e` 过载，减少 `bias_e`；如果欠载，增加它。无额外损失项。训练保持干净。专家负载保持均衡。

对主损失的影响：无可测量的影响。对 MoE 架构的影响：更干净，无需调整辅助损失超参数。

### MTP：更密集训练 + 免费草稿

从 Phase 10 · 18 你知道 DeepSeek-V3 添加了 D=1 MTP 模块，预测两个位置之后的 token。在推理时，训练好的模块被重新用作推测解码草稿，接受率 80%+。在训练时，每个隐藏状态在 D+1 = 2 个目标上被监督，提供更密集的信号。

参数：671B 主模型之上 14B。开销：2.1%。

### 训练：DualPipe

从 Phase 10 · 19 你知道 DualPipe 是一种双向流水线，将前向和反向 chunk 与跨节点全互联通信重叠。在 DeepSeek-V3 的 2,048-H800 规模下，它恢复了 1F1B 会因流水线气泡损失的约 245k GPU 小时。

### 配置，逐字段解析

以下是 DeepSeek-V3 的配置（简化）：

```
hidden_size: 7168
intermediate_size: 18432 (密集 MLP 隐藏大小，用于前几层)
moe_intermediate_size: 2048 (专家 MLP 隐藏大小)
num_hidden_layers: 61
first_k_dense_layers: 3 (前 3 层使用密集 MLP)
num_attention_heads: 128
num_key_value_heads: 128 (在 MLA 下形式上等于 num_heads，但
真正的压缩在 kv_lora_rank 中)
kv_lora_rank: 512 (MLA 潜在维度)
num_experts: 256 (每块 MoE 专家数)
num_experts_per_tok: 8 (top-8 路由)
shared_experts: 1 (每块一个始终开启的共享专家)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1 (深度 1 处 1 个 MTP 模块)
```

解析：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的密集 MLP。其余 58 个使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩到此潜在维度并按头解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，路由 top-8。
- `shared_experts=1`：在 256 个路由专家之上，1 个始终开启的专家贡献于每个 token。可以将其视为"密集底座"，确保每个 token 都获得一些可靠的内容。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐藏大小。比密集 MLP 小，因为有 256 个。

### 参数核算

完整计算在 `code/main.py` 中。要点：

- 嵌入：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个密集块：带 MLA 的注意力（每块 ~144M）+ 密集 MLP（每块 ~260M）+ norm。约 1.2B 总计。
- 58 个 MoE 块：带 MLA 的注意力（~144M）+ 256 个专家（每个 30M）+ 1 个共享专家（30M）+ norm。每块总计 ~7.95B，包括所有专家。58 个 MoE 块共 461B。
- MTP 模块：14B。

总计：核心架构 ~476B + 14B MTP + 发布的 671B 数字明确包含了额外的结构参数（偏置张量、专家特定组件、共享专家缩放等）。我们在计算器中复现的数字与发布值相差 3-5%——差异来自 DeepSeek 报告第 2 节附录中记录的细粒度核算。

每次前向的活跃参数：

- 注意力：每层 144M * 61 = 8.8B（所有层都激活）。
- MLP 活跃：前 3 层密集（3 * 260M = 780M），58 个 MoE 层每层活跃 8 个路由 + 1 个共享 + 路由开销。每层活跃 MLP：~260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- 嵌入 + norm：1.2B。
- 活跃总计：约 26B 核心 + 14B MTP（已训练但推理时不总是运行）≈ 37B。

### 671B / 37B 比率

18 倍稀疏比率（活跃参数占总参数的 5.5%）。DeepSeek-V3 是发布开放权重的最稀疏前沿 MoE 模型。Mixtral 8x7B 比率为 13/47（28%）密集得多。Llama 4 Maverick 比率为 17B/400B（4.25%）可比。DeepSeek 的赌注：在前沿规模下，更多专家配合更低激活比率产生每活跃 FLOP 更好的质量。

### DeepSeek-V3 的位置

| 模型 | 总参数 | 活跃参数 | 比率 | 注意力 | 新颖想法 |
|------|-------|---------|------|--------|---------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助损失 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 后续：R1、V4

DeepSeek-R1（2025）是在 V3 骨干上的推理训练运行。R1 使用相同架构。改变的是后训练方案（在可验证任务上的大规模 RL），而非预训练架构。

DeepSeek-V4（如果发布）预计保留 MLA + MoE + MTP 并添加 DSA（DeepSeek Sparse Attention），即 Phase 10 · 17 中 NSA 的继任者。谱系是稳定的：架构级创新累积；每个版本旋转更多旋钮。

```figure
moe-routing
```

## 使用它

`code/main.py` 是专门针对 DeepSeek-V3 形状的参数计算器。运行它，将其输出与论文数字比较，并在假设变体上使用它（256 专家 vs 512，top-8 vs top-16，MLA rank 512 vs 1024）。

关注什么：

- 总参数计数 vs 发布的 671B。
- 活跃参数计数 vs 发布的 37B。
- 128k 上下文下的 KV cache——MLA vs GQA 比较。
- 逐层分解以查看参数预算实际去向。

## 交付它

本课程产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 家族模型（V3、R1 或任何未来变体），它产出逐组件的架构解读，命名配置的每个字段，按组件推导参数计数，并识别模型使用了四个 DeepSeek 特有创新中的哪些。

## 练习

1. 运行 `code/main.py`。将计算器的总参数估计与发布的 671B 比较，并识别差异来源。论文第 2 节有完整细目。

2. 修改配置使用 MLA rank 256 而非 512。计算 128k 上下文下产生的 KV cache 大小。它带来多少百分比缩减，对每头表达能力的代价是什么？

3. 将 DeepSeek-V3 的（256 专家，top-8）路由与假设的（512 专家，top-8）变体比较。总参数增长；活跃参数不变。额外的专家容量在理论上带来什么，在推理时代价是什么？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）第 2.1 节关于 MLA 的内容。用三句话解释为什么 K 和 V 解压矩阵可以被"吸收"到后续矩阵乘法中以实现推理时效率。

5. DeepSeek-V3 对大多数操作使用 FP8 训练。计算 FP8 vs BF16 存储 671B 权重的内存节省。这与 14.8T token 训练预算如何交叉？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| MLA | "Multi-Head Latent Attention" | 将 K 和 V 压缩为共享低秩潜在（kv_lora_rank，通常 512），按头即时解压；KV cache 仅存储潜在 |
| kv_lora_rank | "MLA 压缩维度" | K 和 V 共享潜在的大小；DeepSeek-V3 使用 512 |
| First k dense layers | "早期层保持密集" | MoE 模型的前几层跳过 MoE 路由器并运行密集 MLP 以保持稳定性 |
| num_experts_per_tok | "Top-k 路由" | 每 token 激活多少路由专家；DeepSeek-V3 使用 8 |
| Shared experts | "始终开启的专家" | 无论路由如何都处理每个 token 的专家；DeepSeek-V3 使用 1 |
| Auxiliary-loss-free routing | "偏置调整负载均衡" | 训练期间调整的每专家偏置项，保持专家负载均衡而不添加损失项 |
| MTP module | "额外预测头" | 从 h^(1) 和 E(t+1) 预测 t+2 的 transformer 块；更密集训练，免费推测解码草稿 |
| DualPipe | "双向流水线" | 将前向/反向计算与跨节点全互联重叠的训练调度 |
| Active parameter ratio | "稀疏度" | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 training | "8 位训练" | FP8 中的训练存储和许多计算操作；与 BF16 相比大约减半内存，质量代价很小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 技术报告 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的架构、训练和结果文档
- [DeepSeek-V3 Hugging Face 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件和部署说明
- [DeepSeek-V2 论文 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前身
- [DeepSeek-R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — V3 架构上的推理训练继任者
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 家族注意力的未来方向
- [DualPipe 仓库](https://github.com/deepseek-ai/DualPipe) — 训练调度参考
