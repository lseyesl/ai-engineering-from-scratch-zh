# vLLM 生产栈与 LMCache KV 卸载

> vLLM 的生产栈是参考性的 Kubernetes 部署——路由器、引擎和可观测性组件相互连接。LMCache 是 KV 卸载层，将 KV 缓存从 GPU 内存中提取出来，并在查询和引擎之间复用（CPU DRAM，然后是磁盘/Ceph）。vLLM 0.11.0 的 KV Offloading Connector（2026 年 1 月）使其成为异步且可通过 Connector API（v0.9.0+）插拔的。卸载延迟不会影响用户。即使没有共享前缀，LMCache 也很有价值——当 GPU 的 KV 槽位耗尽时，被抢占的请求可以从 CPU 恢复，而无需重新计算预填充。已发表的基准测试在 16x H100（80GB HBM）跨 4 个 a3-highgpu-4g 上：当 KV 缓存超过 HBM 时，原生 CPU 卸载和 LMCache 都能显著提高吞吐量；在低 KV 占用时，所有配置与基线匹配，仅有少量开销。

**类型：** 学习
**语言：** Python（stdlib，玩具级 KV 溢出模拟器）
**前置要求：** Phase 17 · 04（vLLM 服务内部原理），Phase 17 · 06（SGLang/RadixAttention）
**时间：** ~60 分钟

## 学习目标

- 绘制 vLLM 生产栈的各层：路由器、引擎、KV 卸载、可观测性。
- 解释 KV Offloading Connector API（v0.9.0+）以及 0.11.0 异步路径如何隐藏卸载延迟。
- 量化 LMCache CPU-DRAM 何时有帮助（KV > HBM）vs 何时增加开销（KV 小到足以容纳在 HBM 中）。
- 根据部署约束，在原生 vLLM CPU 卸载和 LMCache 连接器之间做出选择。

## 问题

你的 vLLM 服务显示 GPU HBM 使用率达到 100%，每当并发度升高时就会出现抢占事件。请求被驱逐、重新排队，你在短短一分钟内四次重新预填充同一个 2K token 的提示。GPU 算力浪费在冗余的预填充上；有效吞吐量远低于原始吞吐量。

增加更多 GPU 成本线性增长。增加更多 HBM 是不可能的。但 CPU DRAM 很便宜——一个插槽有 512 GB 以上，延迟比 HBM 差几个数量级，但对于"临时温热"的 KV 缓存来说足够了。

LMCache 将 KV 缓存提取到 CPU DRAM，使被抢占的请求快速恢复，并且跨引擎的重复前缀可以共享缓存，而无需每个引擎重新预填充。

## 概念

### vLLM 生产栈

`github.com/vllm-project/production-stack` 是参考性的 Kubernetes 部署：

- **路由器**——缓存感知（Phase 17 · 11）。消费 KV 事件。
- **引擎**——vLLM 工作节点。每个 GPU 或每个 TP/PP 组一个。
- **KV 缓存卸载**——LMCache 部署或原生连接器。
- **可观测性**——Prometheus 采集、Grafana 仪表板、OTel 追踪。
- **控制平面**——服务发现、配置、滚动更新。

以 Helm chart + operator 形式提供。

### KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入了用于可插拔 KV 缓存后端的 Connector API。你的引擎将块卸载到连接器；连接器存储它们（RAM、磁盘、对象存储、LMCache）。当请求需要某个块时，连接器将其加载回来。

vLLM 0.11.0（2026 年 1 月）增加了异步卸载路径——卸载可以在后台进行，因此在常见情况下引擎不会阻塞。端到端延迟和吞吐量仍然取决于工作负载形状、KV 缓存命中率和系统压力；vLLM 自己的说明指出，自定义内核卸载在低命中率下可能降低吞吐量，并且异步调度与推测解码存在已知的交互问题。

### 原生 CPU 卸载 vs LMCache

**原生 vLLM CPU 卸载**：引擎本地。将 KV 块存储在主机 RAM 中。实现快速，零网络跳转。不能跨引擎。

**LMCache 连接器**：集群级别。将块存储在共享的 LMCache 服务器中（CPU DRAM + Ceph/S3 层级）。任何引擎都可以访问块。已发表 16x H100 基准测试。

当单个引擎有 HBM 压力时选择原生。当多个引擎共享前缀时选择 LMCache（具有公共系统提示的 RAG、具有共享模板的多租户）。

### 基准测试行为

16x H100（80 GB HBM）分布在 4 个 a3-highgpu-4g 上的测试：

- 低 KV 占用（短提示、低并发）：所有配置与基线匹配，LMCache 增加约 3-5% 开销。
- 中等占用：LMCache 开始在跨引擎前缀复用方面发挥作用。
- KV 超过 HBM：原生 CPU 卸载和 LMCache 都显著提高吞吐量；LMCache 收益更大，因为可以跨引擎共享。

### LMCache 何时起决定性作用

- 多租户服务，其中系统提示在租户之间共享。
- RAG，其中文档块在查询之间重复。
- 同一基座上的微调变体（LoRA），其中基座模型的 KV 复用减少了冗余工作。
- 抢占密集型工作负载：从 CPU 恢复比重新预填充更便宜。

### 何时不应启用

- HBM 压力小——你为没有收益的开销买单。
- 短上下文（<1K token）——传输时间 > 重新预填充。
- 单租户单提示工作负载——没有可捕获的复用。

### 与分离式服务的集成

Phase 17 · 17 的分离式服务 + LMCache 叠加：从预填充池到解码池的 KV 传输如果未被使用则落入 LMCache；后续查询从 LMCache 拉取。Phase 17 · 11 的缓存感知路由器可以将请求路由到其本地或 LMCache 共享缓存匹配的引擎。

### 你应该记住的数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响取决于工作负载、KV 命中率和系统压力（非绝对保证）。
- 16x H100 基准测试：当 KV 占用超过 HBM 时 LMCache 有帮助。
- HBM 压力小时：3-5% 开销，无收益。

```figure
zero-sharding
```

## 使用它

`code/main.py` 模拟有和没有 LMCache 的抢占密集型工作负载。报告避免的重新预填充次数、吞吐量增益以及盈亏平衡的 HBM 利用率。

## 交付物

本课程产出 `outputs/skill-vllm-stack-decider.md`。根据工作负载形状和 vLLM 部署，决定使用原生、LMCache 还是两者都不用。

## 练习

1. 运行 `code/main.py`。在什么 HBM 利用率下 LMCache 开始产生收益？
2. 一个租户在 200 次查询/小时中共享一个 6K token 的系统提示。计算每个租户预期的 LMCache 节省。
3. LMCache 服务器是单点故障。设计高可用策略（副本、回退到原生）。
4. LMCache 将数据存储到旋转磁盘上的 Ceph。对于 70B FP8 上 4K token 的 KV（500 MB），读取时间与重新预填充相比如何？
5. 论证 vLLM 0.11.0 异步路径是否"免费"——开销隐藏在哪里？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| Production-stack | "参考部署" | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | "KV 后端接口" | vLLM 0.9.0+ 可插拔 KV 存储接口 |
| Native CPU offload | "引擎本地溢出" | 将 KV 存储在同一引擎的主机 RAM 中 |
| LMCache | "集群 KV 缓存" | CPU DRAM + 磁盘上的跨引擎 KV 缓存服务器 |
| 0.11.0 async | "非阻塞卸载" | 隐藏在引擎流背后的卸载 |
| Preemption | "驱逐以腾出空间" | HBM 满时的 KV 缓存重排 |
| Prefix reuse | "相同系统提示" | 多个查询共享开头；缓存命中 |
| Ceph tier | "磁盘层级" | 缓存层次结构中 DRAM 之下的持久存储 |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector 实现。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — 异步路径详情。