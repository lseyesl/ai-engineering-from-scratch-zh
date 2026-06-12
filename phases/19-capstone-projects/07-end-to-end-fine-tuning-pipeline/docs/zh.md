# 顶点项目 07——端到端微调管道（数据到 SFT 到 DPO 到服务）

> 一个在你自己的数据上训练、在你自己的偏好上 DPO 对齐、量化、推测解码并以可衡量的 $/1M tokens 服务的 8B 模型。2026 年开源栈是 Axolotl v0.8、TRL 0.15、用于迭代的 Unsloth、用于量化的 GPTQ/AWQ/GGUF、以及带 EAGLE-3 的 vLLM 0.7。顶点项目是可重现地运行整个管道——YAML 输入，服务端点输出——并在 2026 年模型开放框架下发布模型卡。

**类型:** Capstone
**语言:** Python（管道）、YAML（配置）、Bash（脚本）
**前置要求:** Phase 2（ML）、Phase 3（DL）、Phase 7（Transformer）、Phase 10（LLM 从零开始）、Phase 11（LLM 工程）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时间:** 35 小时

## 问题

2026 年，每个正经的 AI 团队都保持一个随时可用的微调管道。不是因为他们发布前沿基础模型，而是因为下游适配——领域 SFT、针对标记偏好的 DPO、用于推测解码的蒸馏草稿、使用 EAGLE-3 服务——是可衡量的胜利所在。Axolotl v0.8 处理多 GPU SFT 配置。TRL 0.15 处理 DPO 和 GRPO。Unsloth 提供快速的单 GPU 迭代。带 EAGLE-3 的 vLLM 0.7 在不损失质量的情况下将解码吞吐量提升 2-3 倍。工具是现成的；技艺在于 YAML、数据卫生和评估纪律。

你将通过 SFT 然后 DPO 在任务特定数据上运行 8B 基础模型（Llama 3.3、Qwen3 或 Gemma 3），量化以用于服务，并针对 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 衡量增益。你将在 2026 年模型开放框架下生成模型卡。关键是可重现性——一个命令端到端地重新运行整个管道。

## 概念

管道有五个阶段。**Data**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII 擦洗、针对公共基准污染的分割卫生检查。**SFT**：Axolotl YAML、8xH100 上的 ZeRO-3、余弦调度、打包序列、2-3 轮。**DPO 或 GRPO**：TRL 配置、1 轮、偏好对（人工标记或模型评判）、beta 调优。**Quantize**：GPTQ + AWQ + GGUF 以实现部署灵活性。**Serve**：带 EAGLE-3 推测头的 vLLM 0.7（或带 SpecForge 的 SGLang）、K8s 部署、基于队列等待的 HPA。

消融实验是交付物：SFT-only vs SFT+DPO vs SFT+GRPO，在三个任务特定基准上。服务指标：bs=1/8/32 下 tokens/s、EAGLE-3 接受率、$/1M tokens。安全评估：Llama Guard 4 通过率。模型卡：偏见评估、可重现性种子、数据许可。

## 架构

```
原始数据 (HF datasets + 内部)
    |
    v
Datatrove 去重 + Nemotron-CC 质量过滤 + PII 擦洗
    |
    v
分割卫生 (MMLU-Pro 污染检查)
    |
    v
Axolotl SFT 配置 (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO 配置       ---> 4xH100, 1 轮
    |
    v
GPTQ + AWQ + GGUF 量化
    |
    v
vLLM 0.7 + EAGLE-3 推测解码
    |
    v
K8s 部署, HPA 基于队列等待
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
模型卡 (2026 MOF) + 安全评估 (Llama Guard 4)
```

## 技术栈

- 数据：Datatrove 用于去重、Nemotron-CC 分类器用于质量、Presidio 用于 PII
- 基础模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8 配合 ZeRO-3、Flash Attention 3、打包序列
- 偏好调优：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 迭代
- 量化：GPTQ（Marlin）、AWQ、通过 llama.cpp 的 GGUF
- 服务：带 EAGLE-3 推测解码的 vLLM 0.7（或 SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA device plugin、基于队列等待指标的 HPA
- 可观测性：W&B 用于训练、Langfuse 用于推理

## 构建它

1. **数据管道。** 在原始语料库上运行 Datatrove 去重。应用 Nemotron-CC 风格质量分类器。Presidio 擦洗 PII。使用显式种子编写训练/验证分割。

2. **污染检查。** 对每个验证分割，计算针对 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集的 MinHash。拒绝任何重叠。

3. **Axolotl SFT。** YAML 配置 ZeRO-3、FA3、序列打包。在 8xH100 上训练 2-3 轮。记录到 W&B。

4. **TRL DPO / GRPO。** 取 SFT 检查点，对偏好对运行 1 轮 DPO（或在数学/代码上使用可验证奖励的 GRPO）。扫描 beta。

5. **量化。** 生成三种量化：GPTQ-INT4-Marlin、AWQ-INT4、用于 llama.cpp 的 GGUF-Q4_K_M。记录大小和标称吞吐量。

6. **带推测解码的服务。** vLLM 0.7 配置 EAGLE-3 草稿头，通过 Red Hat Speculators 训练。测量 bs=1/8/32 下的接受率和尾延迟。报告与 Anthropic/OpenAI 在相同评估上的 $/1M tokens。

7. **评估矩阵。** 在基础模型、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成表格。

8. **安全评估。** 开发集上的 Llama Guard 4 通过率。ShieldGemma-2 输出过滤器。

9. **模型卡。** MOF 2026 模板：数据、训练、评估、安全、许可、可重现性部分，附带 YAML 和 commit SHA。

## 使用它

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k 去重, 12k 过滤, 280k 接受 (seed=7)
[SFT]     3 轮, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 轮, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md 在 2026 MOF 下生成
```

## 交付物

`outputs/skill-finetuning-pipeline.md` 描述了交付物。一个命令将数据通过 SFT 经过 DPO 经过量化经过服务经过评估，并输出模型卡 + 服务端点。

| 权重 | 标准 | 测量方式 |
|:---:|---|---|
| 25 | 评估 vs 基础模型 | 在目标任务上的可衡量增益（MMLU-Pro、MT-Bench-v2、任务特定）|
| 20 | 管道可重现性 | 一个命令使用相同种子端到端重新运行 |
| 20 | 数据卫生 | 去重率、PII 擦洗覆盖率、污染检查通过 |
| 20 | 服务效率 | bs=1/8/32 下 tokens/s、EAGLE-3 接受率、$/1M tokens |
| 15 | 模型卡 + 安全评估 | 2026 MOF 完整性 + Llama Guard 4 通过率 |

## 练习

1. 在相同任务特定基准上运行 SFT-only vs SFT+DPO vs SFT+GRPO。报告哪种偏好方法胜出及其程度。

2. 将 Llama 3.3 8B 替换为 Qwen3 14B。在匹配质量下测量 $/1M tokens。

3. 测量领域数据 vs 通用 ShareGPT 上的 EAGLE-3 接受率。报告差异及其对延迟预算的意义。

4. 注入 1% 的污染（将 MMLU-Pro 答案泄露到训练数据中）并重新运行评估。观察 MMLU-Pro 准确性不现实地飙升。构建一个捕获此问题的污染检查 CI 门。

5. 添加 LoRA SFT 作为全量微调的替代方案。在 10 倍低内存下测量质量差距。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Axolotl | "SFT 训练器" | 用于 SFT、DPO 和蒸馏的统一 YAML 驱动训练器 |
| TRL | "偏好调优器" | Hugging Face 库，用于 LLM 的 DPO、GRPO、PPO |
| GRPO | "组相对策略优化" | DeepSeek R1 的 RL 配方，使用可验证奖励 |
| EAGLE-3 | "推测解码草稿" | 预测 N 个 token 的草稿头；vLLM 用目标模型验证 |
| MOF | "模型开放框架" | 2026 年标准，对模型发布在数据、代码、许可方面进行分级 |
| Contamination check | "分割卫生" | 基于 MinHash 的测试集泄露到训练中的检测 |
| Acceptance rate | "EAGLE / MTP 指标" | 目标模型接受的草稿 token 的比例 |

## 延伸阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/)——参考 SFT / DPO 训练器
- [TRL 文档](https://huggingface.co/docs/trl)——DPO 和 GRPO 参考实现
- [Unsloth](https://github.com/unslothai/unsloth)——单 GPU 迭代参考
- [DeepSeek R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)——GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai)——参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge)——备选推测解码训练器
- [Model Openness Framework 2026](https://isocpp.org/)——开放发布分级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)——权威评估运行器
