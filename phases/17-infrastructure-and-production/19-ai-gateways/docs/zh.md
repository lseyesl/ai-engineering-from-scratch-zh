# AI 网关 — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关位于你的应用和模型提供商之间。核心功能包括提供商路由、故障转移、重试、速率限制、密钥引用、可观测性、护栏（guardrails）。2026 年市场格局：**LiteLLM** 是 MIT 开源，支持 100+ 提供商，兼容 OpenAI API，但在约 2000 RPS 时崩溃（8 GB 内存，已发布基准中显示级联故障）；最适合 Python、<500 RPS、开发/原型设计。**Portkey** 定位于控制平面（护栏、PII 脱敏、越狱检测、审计追踪），2026 年 3 月转为 Apache 2.0 开源，20-40 ms 延迟开销，$49/月生产版。**Kong AI Gateway** 基于 Kong Gateway 构建——Kong 在相同 12 CPU 上的基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%；定价 $100/模型/月（Plus 版最多 5 个模型）；如果你已在使用 Kong，则是企业级选择。**Bifrost**（Maxim AI）——带可配置退避的自动重试，OpenAI 429 时回退到 Anthropic。**Cloudflare / Vercel AI Gateways**——托管式、零运维、基本重试。数据驻留需求驱动自托管决策；Portkey 和 Kong 位于中间地带，提供开源 + 可选的托管服务。

**Type:** Learn
**Languages:** Python（stdlib，简易网关路由模拟器）
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms)、Phase 17 · 16 (Model Routing)
**Time:** ~60 分钟

## 学习目标

- 列举六大核心网关功能（路由、故障转移、重试、速率限制、密钥、可观测性、护栏）。
- 将四个 2026 年网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到规模上限和用例。
- 引用 Kong 基准测试（比 Portkey 快 228%，比 LiteLLM 快 859%）并解释为什么它对 >500 RPS 场景很重要。
- 根据数据驻留和运维预算选择自托管与托管方案。

## 问题

你的产品调用 OpenAI、Anthropic 和自托管的 Llama。每个提供商都有不同的 SDK、错误模型、速率限制和认证方案。你需要故障转移（如果 OpenAI 返回 429，尝试 Anthropic）、单一凭据存储、统一的可观测性和每租户速率限制。

在应用层重新实现这些功能会将每个服务耦合到每个提供商。网关层将其整合到一个进程中，使用一个 API（通常兼容 OpenAI）来分发到各个提供商。

## 概念

### 六大核心功能

1. **提供商路由**——OpenAI、Anthropic、Gemini、自托管等在同一个 API 后面。
2. **故障转移**——在 429、5xx 或质量失败时，在其他地方重试。
3. **重试**——指数退避，有限次数。
4. **速率限制**——按租户、按密钥、按模型。
5. **密钥引用**——运行时从 vault 拉取凭据（绝不在应用中）。
6. **可观测性**——OTel + GenAI 属性（Phase 17 · 13）+ 成本归因。
7. **护栏**——PII 脱敏、越狱检测、允许主题过滤器。

### LiteLLM——MIT 开源，Python

- 100+ 提供商，兼容 OpenAI API，路由器配置，故障转移，基本可观测性。
- 在 Kong 的基准测试中约 2000 RPS 时崩溃；8 GB 内存占用，持续负载下级联故障。
- 最适合：Python 应用，<500 RPS，开发/预发布网关，实验性路由。
- 成本：开源 $0；存在云免费层。

### Portkey——控制平面定位

- 2026 年 3 月起 Apache 2.0 开源。护栏、PII 脱敏、越狱检测、审计追踪。
- 每请求 20-40 ms 延迟开销。
- $49/月生产版，包含数据保留 + SLA。
- 最适合：需要捆绑护栏 + 可观测性的受监管行业。

### Kong AI Gateway——规模方案

- 基于 Kong Gateway（成熟的 API 网关产品，lua+OpenResty）构建。
- Kong 在 12-CPU 等效环境中的基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 版最多 5 个模型。
- 最适合：已在使用 Kong；>1000 RPS；愿意付费使用许可证。

### Bifrost（Maxim AI）

- 带可配置退避的自动重试。
- OpenAI 429 时回退到 Anthropic 是一个经典方案。
- 较新的进入者；商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管式，零运维。基本重试和可观测性。
- 最适合：在 Cloudflare/Vercel 上的边缘服务 JavaScript 应用。
- 在护栏和速率限制方面比 Kong/Portkey 功能有限。

### 自托管 vs 托管

数据驻留是驱动因素。医疗和金融默认选择自托管（LiteLLM 或 Portkey OSS 或 Kong）。消费产品默认选择托管（Cloudflare AI Gateway）或中端方案（Portkey 托管）。混合方案：受监管租户用自托管，其他用托管。

### 延迟预算

- LiteLLM：典型 5-15 ms 开销。
- Portkey：20-40 ms 开销。
- Kong：3-8 ms 开销。
- Cloudflare/Vercel：1-3 ms 开销（边缘优势）。

网关延迟直接增加 TTFT。对于 TTFT P99 < 100 ms 的 SLA，选择 Kong 或 Cloudflare。对于 P99 < 500 ms，任何都可接受。

### 速率限制语义很重要

简单的令牌桶（token-bucket）在中低规模下可行。多租户需要滑动窗口（sliding-window）+ 突发配额 + 每租户分层。LiteLLM 提供令牌桶；Kong 提供滑动窗口；Portkey 提供分层方案。

### 网关 + 可观测性 + 路由的组合

Phase 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产中是同一层。选择一个覆盖三者的工具，或仔细地连接它们：大多数 2026 年部署将 Helicone（可观测性）或 Portkey（护栏）与 Kong（规模）组合以实现角色分离。

### 你应该记住的数字

- LiteLLM：约 2000 RPS 时崩溃，8 GB 内存。
- Portkey：20-40 ms 开销；2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 版最多 5 个。
- Cloudflare/Vercel：边缘 1-3 ms 开销。

## 使用它

`code/main.py` 模拟跨 3 个提供商的网关路由，在注入 429/5xx 错误下进行故障转移。报告延迟、重试率和故障转移命中率。

## 交付物

本课程产出 `outputs/skill-gateway-picker.md`。根据规模、运维姿态、合规要求、延迟预算选择网关。

## 练习

1. 运行 `code/main.py`。配置从 OpenAI→Anthropic→自托管的故障转移。在 5% 提供商错误率下，预期命中率是多少？
2. 你的 SLA 是在 300 ms 基线上 TTFT P99 < 200 ms。哪些网关保持在预算内？
3. 医疗客户要求自托管 + PII 脱敏 + 审计。选择 Portkey OSS 或 Kong。
4. 比较 LiteLLM vs Kong：团队应该在什么 RPS 上限时迁移？
5. 为多租户 SaaS 设计速率限制策略：免费层、试用层、付费层。令牌桶还是滑动窗口？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "API 代理" | 位于应用和提供商之间的进程 |
| LiteLLM | "那个 MIT 的" | Python 开源，100+ 提供商，2K RPS 时崩溃 |
| Portkey | "护栏网关" | 控制平面 + 可观测性，Apache 2.0 |
| Kong AI Gateway | "那个规模的" | 基于 Kong Gateway 构建，基准测试领先 |
| Bifrost | "Maxim 的网关" | 重试 + Anthropic 故障转移方案 |
| Cloudflare AI Gateway | "边缘托管" | 边缘部署的托管网关，零运维 |
| PII redaction | "数据清洗" | 发送到模型前通过正则 + NER 脱敏 |
| Jailbreak detection | "提示注入防护" | 对用户输入的分类器 |
| Audit trail | "受监管日志" | 每个 LLM 调用的不可变记录 |
| Token-bucket | "简单速率限制" | 基于令牌补充的速率限制器 |
| Sliding-window | "精确速率限制" | 时间窗口速率限制器；更好的公平性 |

## 延伸阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
