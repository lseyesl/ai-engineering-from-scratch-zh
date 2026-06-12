# 边缘推理 — Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 边缘推理的核心约束是内存带宽，而非算力。移动 DRAM 约为 50-90 GB/s；数据中心 HBM3 超过 2-3 TB/s——相差 30-50 倍。解码受内存带宽限制，因此这一差距是决定性的。到 2026 年，边缘推理格局分为四个方向。Apple M4/A18 Neural Engine 峰值达 38 TOPS，采用统一内存（CPU↔NPU 无需拷贝）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达 45 TOPS。WebGPU + WebLLM 在 M3 Max 上运行 Llama 3.1 8B（Q4）约 41 tok/s（约为原生的 70-80%）；GitHub 17.6k 星标，兼容 OpenAI API，移动端覆盖约 70-75%。NVIDIA Jetson Orin Nano Super（8GB）可运行 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 运行 gpt-oss-20b 约 40 tok/s；Jetson T4000（JetPack 7.1）性能为 AGX Orin 的 2 倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、分块预填充——Bosch、ThunderSoft、MediaTek 在 CES 2026 上已展示。

**Type:** Learn
**Languages:** Python（stdlib，简易带宽受限解码模拟器）
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)、Phase 17 · 09 (Production Quantization)
**Time:** ~60 分钟

## 学习目标

- 解释为什么移动端 LLM 推理受内存带宽限制，而算力是次要因素。
- 列举四大边缘目标平台（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）并将每个匹配到用例。
- 说出 2026 年 WebGPU 的覆盖缺口（Firefox Android 仍在追赶）和 Safari iOS 26 的正式支持。
- 为每个目标选择合适的量化格式（ANE 用 Core ML INT4 + FP16，Hexagon 用 QNN INT8/INT4，浏览器用 WebGPU Q4，Jetson Thor 用 NVFP4）。

## 问题

客户需要一个设备端聊天机器人：语音优先、默认隐私、支持离线。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 运行约 55 tok/s——没问题。在 iPhone 16 Pro 上，同一模型运行仅 3 tok/s——不行。在搭载 Snapdragon 8 Gen 3 的中端 Android 上，7 tok/s。在浏览器中通过 WebGPU（Chrome Android v121+）上，4-8 tok/s，取决于设备。

吞吐量方差不是移植问题。它是带宽差距乘以量化格式再乘以 NPU 是否可从用户空间访问的综合结果。2026 年的边缘推理是四个不同的问题，有四种不同的解决方案。

## 概念

### 带宽是真正的天花板

解码为每个 token 读取全部权重。一个 7B 模型的 Q4 版本为 3.5 GB。以 50 GB/s 读取 3.5 GB 需要 70 ms——理论上限约 14 tok/s。在 90 GB/s（高端移动 DRAM）下，上限升至约 25 tok/s。在此数字以下，任何算力都无济于事。

数据中心 HBM3 以 3 TB/s 读取同样的 3.5 GB 仅需 1.2 ms——上限为 830 tok/s。相同的模型，相同的权重。不同的内存子系统。

### Apple Neural Engine（M4 / A18）

- 高达 38 TOPS。统一内存（CPU 和 ANE 共享同一内存池）——无拷贝开销。
- 通过 Core ML + `.mlmodel` 编译模型访问，或通过 PyTorch 的 Metal Performance Shaders（MPS）访问。
- Llama.cpp Metal 后端使用 MPS，而非直接使用 ANE；原生 ANE 需要 Core ML 转换。
- 2026 年 iOS 应用的最佳实践路径：Core ML 搭配 INT4 权重 + FP16 激活。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 高达 45 TOPS。集成在 SoC 中但与 CPU 和 GPU 分离的内存域。
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换。
- Chat 模板、Llama 3.2、Phi-3 均作为一等公民在 AI Hub 上提供。

### Intel / AMD NPU（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件生态落后于 Apple/Qualcomm；OpenVINO 正在改进但仍是小众。
- 最适合 Windows ARM copilot 应用；在 AMD/Intel 桌面端上用于本地优先场景。

### WebGPU + WebLLM

- 通过 WebGPU 计算着色器在浏览器中运行模型；无需安装。
- M3 Max 上 Llama 3.1 8B Q4 约 41 tok/s——约为通过相同后端的原生性能的 70-80%。
- WebLLM GitHub 17.6k 星标；兼容 OpenAI 的 JS API；Apache 2.0。
- 2026 年覆盖范围：Chrome Android v121+、Safari iOS 26 GA、Firefox Android 仍在追赶。移动端整体覆盖约 70-75%。

### NVIDIA Jetson 系列

- Orin Nano Super（8GB）：可运行 Llama 3.2 3B、Phi-3，tok/s 表现良好。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 约 40 tok/s。
- Thor / T4000（JetPack 7.1）：性能为 AGX Orin 的 2 倍，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 投机解码、NVFP4 权重、分块预填充——数据中心优化技术已移植到边缘。

### 各目标平台的量化选择

| 目标平台 | 格式 | 说明 |
|--------|--------|-------|
| Apple ANE | INT4 权重 + FP16 激活 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC（q4f16_1）| 使用 `mlc_llm convert_weight` + 编译的 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存受限 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘端的长上下文陷阱

Llama 3.1 的 128K 上下文是数据中心特性。在 8 GB RAM 的手机上，4 GB 模型 + 2 GB KV 缓存（32K token）+ 操作系统开销 = OOM。边缘部署将上下文保持在 4K-8K，除非接受激进的 KV 量化（Q4 KV）。

### 语音是杀手级应用

语音代理对延迟敏感（首 token < 500 ms）。本地推理完全消除了网络延迟。结合语音转文本（Whisper Turbo 变体可在边缘运行），边缘推理成为生产级语音循环。

### 你应该记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 约 41 tok/s。
- AGX Orin：通过 vLLM 在 gpt-oss-20b 上约 40 tok/s。
- 数据中心与边缘带宽差距：30-50 倍。
- WebGPU 移动端覆盖：约 70-75%（Firefox Android 滞后）。

## 使用它

`code/main.py` 根据带宽受限数学计算各边缘目标的理论解码吞吐量上限。与实测基准比较，突出显示带宽（而非算力）是瓶颈的情况。

## 交付物

本课程产出 `outputs/skill-edge-target-picker.md`。根据平台（iOS/Android/浏览器/Jetson）、模型、延迟/内存预算选择合适的量化格式和转换流水线。

## 练习

1. 运行 `code/main.py`。对于 Snapdragon 8 Gen 3（约 77 GB/s 带宽）上 Q4 的 7B 模型，计算解码上限。与实测的 6-8 tok/s 比较——运行时的效率如何？
2. 安卓上的 WebGPU 需要 Chrome v121+。为旧版浏览器设计一个后备方案——通过同一兼容 OpenAI 的 API 进行服务端处理。
3. 你的 iOS 应用需要 4K 上下文流式输出。哪种模型/格式组合能让你在 iPhone 16 上保持 4 GB 以内的活跃内存？
4. Jetson AGX Orin 以 40 tok/s 运行 gpt-oss-20b。Jetson Nano 仅能容纳 3B 模型。如果你的产品同时面向两者，如何统一推理栈？
5. 论证"WebLLM 在 2026 年是否已达到生产就绪"。引用覆盖率、性能和 Firefox Android 的差距。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| ANE | "Apple 神经引擎" | M 系列和 A 系列中的设备端 NPU；统一内存 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | "浏览器 GPU" | W3C 标准化的浏览器 GPU API；2026 年支持 Chrome/Safari |
| WebLLM | "浏览器 LLM 运行时" | MLC-LLM 项目；Apache 2.0；兼容 OpenAI 的 JS |
| Jetson | "NVIDIA 边缘" | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | "边缘 TensorRT" | TensorRT-LLM 的 2026 边缘移植版；支持 EAGLE-3 + NVFP4 |
| Unified memory | "共享池" | CPU 和 NPU 共享同一内存；无拷贝开销 |
| Bandwidth-bound | "内存受限" | 解码吞吐量由读取权重的字节/秒决定 |
| Core ML | "Apple 转换框架" | 用于 ANE 原生模型的 Apple 框架 |
| QNN | "Qualcomm 栈" | Qualcomm Neural Network SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/)——格局和基准测试。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/)——Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/)——2026 年边缘移植版公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2)——设计和基准测试。
- [Apple Core ML](https://developer.apple.com/documentation/coreml)——ANE 原生转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)——为 Hexagon 预转换的模型。
