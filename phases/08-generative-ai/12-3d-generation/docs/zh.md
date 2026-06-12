# 3D 生成

> 3D 是从 2D 到 3D 迁移学习最强的模态。2023 年的突破是 3D 高斯泼溅。2024-2026 年的生成式推进在其上叠加了多视角扩散 + 3D 重建，以从单个提示或照片生成物体和场景。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 4（视觉），阶段 8 · 07（潜在扩散）
**时间：** ~45 分钟

## 问题

3D 内容制作是痛苦的：

- **表示。** 网格、点云、体素网格、符号距离场（SDF）、神经辐射场（NeRF）、3D 高斯。每个都有权衡。
- **数据稀缺。** ImageNet 有 1400 万张图像。最大的干净 3D 数据集（Objaverse-XL，2023）约有 1000 万个物体，大多数质量较低。
- **内存。** 一个 512³ 的体素网格是 1.28 亿个体素；一个有用的场景 NeRF 每射线需要 100 万个样本。生成比重建更难。
- **监督。** 对于 2D 图像，你有像素。对于 3D，你通常只有少数几个 2D 视图，需要将其提升到 3D。

2026 年的技术栈将两个问题分开处理。首先，使用扩散模型生成*2D 多视角图像*。其次，将*3D 表示*（通常是高斯泼溅）拟合到这些图像上。

## 概念

![3D 生成：多视角扩散 + 3D 重建](../assets/3d-generation.svg)

### 表示：3D 高斯泼溅（Kerbl 等人，2023）

将场景表示为约 100 万个 3D 高斯的点云。每个有 59 个参数：位置（3）、协方差（6，或四元数 4 + 缩放 3）、不透明度（1）、球谐颜色（3 阶 48，0 阶 3）。

渲染 = 投影 + alpha 合成。快速（4090 上 1080p 约 100 fps）。可微。通过梯度下降对着真实照片拟合。一个场景在消费级 GPU 上 5-30 分钟即可拟合。

在其上的两项 2023-2024 年创新：
- **生成式高斯泼溅。** 像 LGM、LRM、InstantMesh 这样的模型直接从一张或几张图像预测高斯点云。
- **4D 高斯泼溅。** 为动态场景提供每帧偏移的高斯。

### 多视角扩散

微调一个预训练的图像扩散模型，从文本提示或单张图像生成同一物体的多个一致视图。Zero123（Liu 等人，2023）、MVDream（Shi 等人，2023）、SV3D（Stability，2024）、CAT3D（Google，2024）。通常输出物体周围的 4-16 个视图，通过高斯泼溅或 NeRF 提升到 3D。

### 文本到 3D 管线

| 模型 | 输入 | 输出 | 时间 |
|-------|-------|--------|------|
| DreamFusion（2022） | 文本 | 通过 SDS 的 NeRF | 每个资产约 1 小时 |
| Magic3D | 文本 | 网格 + 纹理 | 约 40 分钟 |
| Shap-E（OpenAI，2023） | 文本 | 隐式 3D | 约 1 分钟 |
| SJC / ProlificDreamer | 文本 | NeRF / 网格 | 约 30 分钟 |
| LRM（Meta，2023） | 图像 | 三平面 | 约 5 s |
| InstantMesh（2024） | 图像 | 网格 | 约 10 s |
| SV3D（Stability，2024） | 图像 | 新视角 | 约 2 分钟 |
| CAT3D（Google，2024） | 1-64 图像 | 3D NeRF | 约 1 分钟 |
| TripoSR（2024） | 图像 | 网格 | 约 1 s |
| Meshy 4（2025） | 文本 + 图像 | PBR 网格 | 约 30 s |
| Rodin Gen-1.5（2025） | 文本 + 图像 | PBR 网格 | 约 60 s |
| 腾讯 Hunyuan3D 2.0（2025） | 图像 | 网格 | 约 30 s |

2025-2026 年方向：具有适用于游戏引擎的 PBR 材质的直接文本到网格模型。多视角扩散中间步骤仍然是通用物体的最佳性能配方。

### NeRF（作为背景）

神经辐射场（Mildenhall 等人，2020）。一个微小的 MLP 接收 `(x, y, z, view direction)` 并输出 `(color, density)`。通过沿光线积分来渲染。在质量上战胜基于网格的新视角合成，但渲染速度慢 100-1000 倍。在大多数实时用途中已被高斯泼溅取代，但在研究中仍然占主导地位。

## 动手实现

`code/main.py` 实现了一个玩具 2D "高斯泼溅"拟合：将合成的目标图像（一个平滑渐变）表示为 2D 高斯泼溅的总和。通过梯度下降优化位置、颜色和协方差以匹配目标。你将看到两个核心操作：前向渲染（泼溅 + alpha 合成）和通过梯度下降拟合。

### 步骤 1：2D 高斯泼溅

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 步骤 2：通过求和泼溅渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的 3D 高斯泼溅按深度对高斯排序并按顺序进行 alpha 合成。我们的 2D 玩具只是求和。

### 步骤 3：通过梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 陷阱

- **视角不一致。** 如果你独立生成 4 个视图但它们对物体结构有分歧，3D 拟合结果会模糊。修复：使用共享注意力的多视角扩散。
- **背面幻觉。** 单张图像到 3D 必须发明看不见的一面。质量变化很大。
- **高斯泼溅爆炸。** 不受约束的训练会增长到 1000 万个高斯并过度拟合。密集化 + 剪枝启发式（来自原始 3D-GS 论文）是必要的。
- **拓扑问题。** 来自隐式场（SDF）的网格通常有孔洞或自相交。在发布前运行重新网格化（例如 Blender 的体素重新网格化）。
- **训练数据的许可。** Objaverse 的许可混合；商业使用因模型而异。

## 使用

| 任务 | 2026 年选择 |
|------|-----------|
| 从照片进行场景重建 | 高斯泼溅（3DGS、Gsplat、Scaniverse） |
| 用于游戏的文本到 3D | Meshy 4 或 Rodin Gen-1.5（PBR 输出） |
| 图像到 3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 从少量图像进行新视角合成 | CAT3D、SV3D |
| 动态场景重建 | 4D 高斯泼溅 |
| 虚拟形象 / 穿衣人体 | Gaussian Avatar、HUGS |
| 研究 / SOTA | 上周发布的最新成果 |

对于在游戏或电商管线中部署生产级 3D：Meshy 4 或 Rodin Gen-1.5 输出可直接导入 Unity / Unreal 的 PBR 网格。

## 产出

保存 `outputs/skill-3d-pipeline.md`。技能接受 3D 简报（输入：文本 / 单张图像 / 少量图像；输出：网格 / 泼溅 / NeRF；用途：渲染 / 游戏 / VR）并输出：管线（多视角扩散 + 拟合，或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需的材质通道。

## 练习

1. **简单。** 使用 4、16、64 个高斯运行 `code/main.py`。报告最终的 MSE 与目标对比。
2. **中等。** 扩展到彩色高斯（RGB）。确认重建匹配目标颜色模式。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片的拍摄中重建一个真实物体。报告拟合时间和保留视角上的最终 SSIM。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 3D 高斯泼溅 | "3DGS" | 场景表示为 3D 高斯点云；可微的 alpha 合成渲染。 |
| NeRF | "神经辐射场" | 在 3D 点输出颜色 + 密度的 MLP；通过光线积分渲染。 |
| 三平面 | "三个 2-D 平面" | 将 3D 分解为三个 2-D 轴对齐的特征网格；比体素更便宜。 |
| SDS | "分数蒸馏采样" | 通过使用 2D 扩散分数作为伪梯度来训练 3D 模型。 |
| 多视角扩散 | "同时生成多个视角" | 输出一批一致相机视图的扩散模型。 |
| PBR | "基于物理的渲染" | 带有反照率、粗糙度、金属度、法线通道的材质。 |
| 密集化 | "增长泼溅" | 3DGS 训练启发式：在高梯度区域分裂/克隆泼溅。 |

## 生产说明：3D 还没有共享基础

与图像（潜在扩散 + DiT）和视频（时空 DiT）不同，3D 在 2026 年没有一个单一的主导运行时。生产决策树根据表示分叉：

- **NeRF / 三平面。** 推理是光线行进 + 每个样本的 MLP 前向传播。512² 渲染需要数百万次 MLP 前向传播。积极地对射线样本进行批处理；SDPA/xformers 适用。
- **多视角扩散 + LRM 重建。** 两阶段管线。阶段 1（多视角 DiT）是一个扩散服务器，就像第 7 课所述。阶段 2（LRM Transformer）是对视图的一次性前向传播。整体延迟分布是"扩散 + 一次性"——相应地选择每阶段的服务原语。
- **SDS / DreamFusion。** 每个资产的优化，而非推理。构建作业，而非请求处理器。

对于大多数 2026 年的产品，正确的答案是"按请求运行多视角扩散模型，异步重建为 3DGS，将 3DGS 用于实时查看"。这可以干净地将工作负载分为 GPU 推理服务器（快速）和离线优化器（慢速）。

## 延伸阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — 多视角扩散。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。
