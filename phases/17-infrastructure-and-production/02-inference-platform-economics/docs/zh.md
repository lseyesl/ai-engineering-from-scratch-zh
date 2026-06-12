# 推理平台经济学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理市场不再是 GPU 时间租赁。它分化为定制芯片 (Groq、Cerebras、SambaNova)、GPU 平台 (Baseten、Together、Fireworks、Modal) 和 API 优先市场 (Replicate、DeepInfra)。Fireworks 于 2026 年 5 月 1 日将每 GPU 价格提高 $1/小时，$4B 估值和每天 10T+ token 的处理量告诉你，以量驱动的模式是有效的。Baseten 于 2026 年 1 月以 $5B 估值完成 $300M E 轮融资。竞争定位规则很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业级体验，Modal 优化 Python 原生开发者体验，Replicate 优化多模态覆盖，Anyscale 优化分布式 Python。本课给你一个可以直接交给创始人的对比矩阵。

**Type:** Learn
**Languages:** Python (stdlib, toy per-call economics comparator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## Learning Objectives

- 说出三个市场细分（定制芯片、GPU 平台、API 优先）并将每个供应商映射到对应细分。
- 解释为什么"按 token"的 API 定价模式趋向于服务引擎的成本曲线，而非硬件的成本曲线。
- 计算至少三个供应商的每次请求有效成本，并解释何时按分钟计费 (Baseten、Modal) 优于按 token 计费。
- 针对给定工作负载（无服务器突发型、稳定高吞吐、微调变体、多模态）确定哪个平台是合适的默认选择。

## 问题

你评估了托管云厂商平台。你决定需要一个更窄、更快的供应商 — Fireworks 追求延迟，Together 追求广度，Baseten 追求微调定制模型。现在你有六个真实选择，而定价页面并不对齐。Fireworks 显示 $/M tokens；Baseten 显示 $/分钟；Modal 显示 $/秒；Replicate 显示 $/预测。如果不建模工作负载，你无法直接比较它们。

更糟的是，每个定价页面背后的商业模式都不同。Fireworks 在共享 GPU 上运行自己的定制引擎 (FireAttention)；按 token 费率反映了它们的利用率曲线。Baseten 提供 Truss 加专用 GPU；按分钟计费反映了独占性。Modal 是真正的 Python 无服务器 — 按秒计费，亚秒级冷启动。相同的输出（LLM 响应），三种不同的成本函数。

本课对六个平台建模，告诉你每个平台何时胜出。

## 概念

### 三个细分市场

**定制芯片** — Groq (LPU)、Cerebras (WSE)、SambaNova (RDU)。在相同模型上，解码速度通常比基于 GPU 的集群快 5-10 倍。每 token 价格更高（Groq 在 2025 年末 Llama-70B 上约 $0.99/M），但对于延迟敏感型用例无可匹敌。Groq 是语音代理和实时翻译的生产选择。

**GPU 平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA (H100、H200、B200，2026 年) 或有时 AMD 上。位于"裸 GPU 租赁" (RunPod、Lambda) 和"云厂商托管服务" (Bedrock) 之间的经济层。

**API 优先市场** — Replicate、DeepInfra、OpenRouter、Fal。目录广泛，按预测或按秒付费，强调首次调用时间。

### Fireworks — 延迟优化的 GPU 平台

- FireAttention 引擎（定制）；号称在等效配置下延迟比 vLLM 低 4 倍。
- 批处理层约按无服务器费率的 50%，适用于非交互式工作负载。
- 微调模型以与基础模型相同的费率提供服务 — 与那些为你的 LoRA 收取溢价的供应商相比，这是一个真正的差异化优势。
- 2026 年中：2026 年 5 月 1 日起按需 GPU 租赁有效提价 $1/小时。大规模时可协商批量定价。
- 财务信号：$4B 估值，每天处理 10T+ token。

### Together — 广度优化

- 200+ 模型，包括上游发布后几天内的开源版本。
- 在等效 LLM 模型上比 Replicate 便宜 50-70% — "AI 原生云"的定位是量和目录。
- 推理 + 微调 + 训练，统一 API。

### Baseten — 企业级体验优化

- Truss 框架：模型打包，依赖、密钥、服务配置在一个清单中。
- GPU 范围从 T4 到 B200。按分钟计费，合理的冷启动缓解。
- SOC 2 Type II，HIPAA 就绪。常见的金融科技和医疗选择。
- $5B 估值，2026 年 1 月 E 轮融资 ($300M，来自 CapitalG、IVP、NVIDIA)。

### Modal — Python 原生优化

- 纯 Python 的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰一个函数，一个命令部署。
- 按秒计费。冷启动 2-4 秒（预热后）；小模型 <1 秒。
- $87M B 轮融资，$1.1B 估值 (2025)。在独立调查中开发者体验评分最高。

### Replicate — 多模态广度

- 按预测付费。图像、视频和音频模型的默认平台。
- 集成生态 (Zapier、Vercel、CMS 插件)。
- 在 LLM 每 token 费率上竞争力较弱，但在多模态多样性上胜出。

### Anyscale — Ray 原生

- 基于 Ray 构建；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）。
- 最适合分布式 Python 工作负载，其中推理步骤是更大图中的一个节点。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 紧密集成。

### 按 token 与按分钟 — 何时胜出

按 token 在延迟不敏感且突发的工作负载下有意义 — 你只为使用付费。按分钟在利用率高且可预测时有意义 — 一旦你饱和了 GPU，按分钟就优于按 token。

粗略规则：对于专用 GPU 持续利用率约 30% 以上的工作负载，按分钟 (Baseten、Modal) 开始优于按 token (Fireworks、Together)。低于此，按 token 胜出，因为你避免了为空闲付费。

### 定制引擎是真正的护城河

每个平台在 vLLM 和 SGLang 之上都声称有定制引擎。FireAttention、RayTurbo、Baseten 的推理栈。定制引擎的声称中营销成分不少 — 诚实的说法是 vLLM 加 SGLang 代表了约 80% 的生产开源推理，平台层的差异化在于开发者体验、归因和 SLA。

### 你应该记住的数字

- Fireworks GPU 租赁：2026 年 5 月 1 日起提价 $1/小时。
- Fireworks 声称：等效配置下延迟比 vLLM 低 4 倍。
- Together：在 LLM 上比 Replicate 便宜 50-70%。
- Baseten 估值：$5B (E 轮，2026 年 1 月，$300M 轮次)。
- Modal 估值：$1.1B (B 轮，2025 年)。
- 按分钟在约 30% 持续利用率以上优于按 token。

```figure
cost-per-token
```

## Use It

`code/main.py` 在合成工作负载上比较六个供应商的定价模型。报告 $/天 和有效 $/M tokens。运行它，找到按 token 和按分钟之间的盈亏平衡点。

## Ship It

本课产出 `outputs/skill-inference-platform-picker.md`。给定工作负载配置、SLA 和预算，选择主推理平台并命名亚军。

## Exercises

1. 运行 `code/main.py`。对于 70B 模型在一个 H100 上，Baseten（按分钟）在什么持续利用率下优于 Fireworks（按 token）？自己推导交叉点并与经验法则比较。
2. 你的产品同时提供图像生成、聊天和语音转文本。为每种模态选择平台，并命名统一它们的网关模式。
3. Fireworks 在你主要模型上提价 $1/小时。如果 40% 的流量转移到批处理层（半价），建模混合成本影响。
4. 一个受监管客户需要 SOC 2 Type II 加 HIPAA 加专用 GPU。哪三个平台可行，哪个在 FinOps 上胜出？
5. 比较 Llama 3.1 70B 在 Fireworks 无服务器、Together 按需、Baseten 专用和 Replicate API 上每 1000 次预测的成本。每天 10 次预测时哪个最便宜？每天 10,000 次呢？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Custom silicon | "非 GPU 芯片" | Groq LPU、Cerebras WSE、SambaNova RDU — 针对解码优化 |
| FireAttention | "Fireworks 引擎" | 定制注意力内核；号称延迟比 vLLM 低 4 倍 |
| Truss | "Baseten 的格式" | 模型打包清单；依赖 + 密钥 + 服务配置 |
| Per-token | "API 定价" | 按消耗的 token 收费；不为空闲付费 |
| Per-minute | "专用定价" | 按挂钟 GPU 时间收费；高利用率时胜出 |
| Per-prediction | "Replicate 定价" | 按模型调用收费；常见于图像/视频 |
| RayTurbo | "Anyscale 引擎" | Ray 上的专有推理；在 Ray 集群上与 vLLM 竞争 |
| Batch tier | "半价" | 非交互式队列，费率降低；常见于 Fireworks、OpenAI |
| Fine-tuned at base rate | "Fireworks LoRA" | 以基础模型费率对 LoRA 服务请求收费（差异化优势） |

## Further Reading

- [Fireworks Pricing](https://fireworks.ai/pricing) — 按 token 费率、批处理层、GPU 租赁。
- [Baseten Pricing](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业层级。
- [Modal Pricing](https://modal.com/pricing) — 按秒 GPU 费率和免费层。
- [Together AI Pricing](https://www.together.ai/pricing) — 模型目录和按 token 费率。
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo 和托管 Ray 定价。
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 对比评估。
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商格局。