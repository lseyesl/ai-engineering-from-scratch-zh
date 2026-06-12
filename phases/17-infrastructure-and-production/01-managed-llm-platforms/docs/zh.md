# 托管 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大云厂商，三种截然不同的策略。AWS Bedrock 是一个模型市场 — Claude、Llama、Titan、Stability、Cohere 统一在一个 API 后面。Azure OpenAI 是 OpenAI 独家合作加上 Provisioned Throughput Units (PTU) 提供专用容量。Vertex AI 以 Gemini 为先，拥有最佳的长上下文和多模态能力。2026 年 Artificial Analysis 测得 Azure OpenAI 中位数延迟约 50 ms，Bedrock 在 Llama 3.1 405B 等效模型上约 75 ms — PTU 解释了这一差距，因为专用容量优于共享按需。决策规则不是"哪个最快"，而是"哪个模型目录和 FinOps 界面匹配我的产品"。本课教你基于写下来的权衡做选择，而不是凭感觉。

**Type:** Learn
**Languages:** Python (stdlib, toy cost-and-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60 minutes

## Learning Objectives

- 说出三种平台策略（市场 vs 独家 vs Gemini 优先）并将每种策略匹配到产品用例。
- 解释 Provisioned Throughput Units (PTU) 在 Azure OpenAI 中能买到什么，以及为什么按需 Bedrock 在 405B 规模下通常慢约 25 ms。
- 绘制每个平台的 FinOps 归因面（Bedrock Application Inference Profiles vs Vertex 按团队项目 vs Azure 作用域加 PTU 预留）。
- 制定一个"至少两个供应商"策略，并解释为什么单一供应商锁定是 2026 年代价最高的错误。

## 问题

你为产品选择了 Claude 3.7 Sonnet。现在需要提供服务。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者通过网关。直接 API 最简单；Bedrock 增加了 BAA、VPC 端点、IAM 和 CloudWatch 归因。网关增加了故障转移、统一计费和跨供应商速率限制。

更深层的问题是目录。如果你需要在同一个产品中使用 Claude、Llama 和 Gemini，你无法从一个地方全部买到，除非同时使用 Bedrock、Vertex 和 Azure OpenAI。三大云厂商不可互换 — 它们各自对谁拥有模型层做出了不同的赌注。

本课映射了这三种赌注、延迟差距、FinOps 差距和锁定风险。

## 概念

### 三种策略

**AWS Bedrock** — 市场模式。Claude (Anthropic)、Llama (Meta)、Titan (AWS 自研)、Stability (图像)、Cohere (嵌入)、Mistral，加上图像和嵌入子目录。一个 API，一个 IAM 面，一个 CloudWatch 导出。Bedrock 的赌注是客户更想要可选性，而不是单一模型。

**Azure OpenAI** — 独家合作。你可以在 Azure 数据中心获得 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper 以及 OpenAI 模型的微调。Azure OpenAI Service 目录中没有非 OpenAI 模型 — 那些归 Azure AI Foundry（独立产品）。Azure 的赌注是 OpenAI 保持前沿地位，客户希望在这个特定关系上获得企业级控制。

**Vertex AI** — Gemini 优先，其他其次。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，加上 Model Garden（第三方）。Vertex 的赌注是多模态长上下文 — 100 万 token 的 Gemini 上下文是其差异化优势。

### 规模下的延迟差距

Artificial Analysis 持续运行基准测试。在等效的 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI 中位数首 token 延迟约 50 ms；Bedrock 约 75 ms。这一差距不是 AWS 的失败 — 而是容量模型的差异。Azure 销售 PTU (Provisioned Throughput Units)，为你的租户预留 GPU 容量。Bedrock 的等效产品 (Provisioned Throughput) 也存在，但每个单元起价约 $21/小时，大多数客户仍使用共享按需。

按需共享容量与所有其他客户的流量竞争。专用容量则不会。如果你的产品 SLA 要求 P99 TTFT < 100 ms，你要么在 Azure 上购买 PTU，要么购买 Bedrock Provisioned Throughput，要么接受默认的方差。

### Provisioned Throughput 经济学

Azure PTU：预留的推理计算块。对于可预测的工作负载，相比按需可节省高达约 70%。无论流量如何，按固定小时费率付费 — 即使空闲也要支付预留费用。盈亏平衡点通常在约 40-60% 的持续利用率。

Bedrock Provisioned Throughput：$21-$50/小时，取决于模型和区域。类似的数学 — 盈亏平衡点约为峰值利用率的一半。需要月度承诺。

Vertex 的预置容量按 Gemini SKU 销售；定价因模型和区域而异，公开宣传较少。

### FinOps 面 — 真正的差异化因素

**Bedrock Application Inference Profiles** 是市场上最清晰的归因方式。用 `team`、`product`、`feature` 标记一个 profile；将所有模型调用路由通过它；CloudWatch 无需后处理即可按 profile 分解成本。2025 年新增，仍然是粒度最细的云厂商原生方案。

**Vertex** 的归因是按团队项目加处处标签。你将每个团队建模为一个 GCP 项目，在每个资源上打标签，使用 BigQuery Billing Export 加 DataStudio 进行汇总。工作量更大，但 BigQuery 让你可以在成本数据上运行任意 SQL。

**Azure** 依赖订阅/资源组作用域加标签，PTU 预留作为一等成本对象。标签从资源组继承，而非请求，因此按请求归因需要 Application Insights 自定义指标或一个能打标头的网关。

模式：Bedrock 原生最清晰，Vertex 通过 BigQuery 最灵活，Azure 最不透明，除非你自行埋点。

### 锁定是 2026 年的风险

当单一模型主导时，单一云厂商承诺是可以的。2026 年，前沿每月都在移动 — 这个季度是 Claude 3.7，下个季度是 Gemini 2.5，再下个季度是 GPT-5。锁定在一个平台上，你就被锁在了三分之二的前沿之外。

成功团队采用的模式：对任何产品关键型 LLM 调用，至少两个供应商。Bedrock 加 Azure OpenAI 是常见的组合 — 一个用 Claude，另一个用 GPT，两者之间故障转移，同一个网关。成本增加可以忽略不计，因为网关路由最优；在故障期间（如 2025 年 1 月 Azure OpenAI 事件、AWS us-east-1 宕机）的可用性提升是决定性的。

### 数据驻留、BAA 和受监管行业

Bedrock：大多数区域提供 BAA；VPC 端点；护栏。常见的金融科技默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；企业受监管默认选择。
Vertex：HIPAA、GDPR、按区域数据驻留；Google Cloud 的合规栈。

三者都满足基本要求。差异在于数据保留策略、日志处理方式，以及滥用监控是否读取你的流量（大多数默认 opt-in；企业版可 opt-out）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等效模型上的中位数 TTFT：约 50 ms（使用 PTU）。
- Bedrock 按需中位数 TTFT：约 75 ms。
- Bedrock Provisioned Throughput：$21-$50/小时/单元。
- Azure PTU 盈亏平衡点：约 40-60% 持续利用率。
- 高利用率下 PTU 相比按需节省：高达 70%。

## Use It

`code/main.py` 在合成工作负载上比较三个平台 — 它模拟按需 vs PTU 经济学、TTFT 方差和成本归因保真度。运行它，看看 PTU 在什么情况下划算，以及市场的模型广度何时胜过 TTFT 差距。

## Ship It

本课产出 `outputs/skill-managed-platform-picker.md`。给定工作负载配置（所需模型、TTFT SLA、日处理量、合规要求），它推荐一个主平台、一个备用平台和一个 FinOps 埋点方案。

## Exercises

1. 运行 `code/main.py`。对于 70B 类模型，Azure PTU 在什么持续利用率下优于按需？计算盈亏平衡点并与宣传的 40-60% 区间比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个双供应商部署 — 哪个模型放在哪个云厂商，前面放什么网关，故障转移策略是什么？
3. 一个受监管的医疗客户需要 BAA、美东数据驻留和低于 100ms 的 P99 TTFT。选择一个平台并用三个具体特性证明。
4. 你发现 Bedrock 账单本月上涨了 4 倍，但流量没有变化。没有 Application Inference Profiles，你如何找到原因？有了 profiles，需要多长时间？
5. 阅读 Azure OpenAI 和 Bedrock 的定价页面。对于每月 1 亿 token 的 Claude 工作负载，哪个更便宜 — 直接 Anthropic API、Bedrock 按需还是 Bedrock Provisioned Throughput？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Bedrock | "AWS LLM 服务" | 跨 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | "Azure 的 ChatGPT" | Azure 数据中心中的独家 OpenAI 模型，带企业级控制 |
| Vertex AI | "Google 的 LLM" | Gemini 优先平台，Model Garden 提供第三方模型 |
| PTU | "专用容量" | Provisioned Throughput Unit — 预留推理 GPU，按小时定价 |
| Application Inference Profile | "Bedrock 标签" | 带标签的按产品成本/用量 profile，CloudWatch 原生 |
| Model Garden | "Vertex 目录" | Vertex AI 的第三方模型区，独立于 Gemini |
| Two-provider minimum | "LLM 冗余" | 每条关键 LLM 路径跨 ≥2 个云厂商运行的策略 |
| BAA | "HIPAA 文书" | Business Associate Agreement；PHI 必需；三者均提供 |
| Abuse monitoring | "日志监视器" | 供应商侧对提示/输出的安全检查；企业版可 opt-out |

## Further Reading

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — 官方费率卡和 Provisioned Throughput 定价。
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU 经济学和费率卡。
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 层级和 Model Garden 附加费。
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — 跨供应商的持续延迟和吞吐量基准。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 归因机制对比。