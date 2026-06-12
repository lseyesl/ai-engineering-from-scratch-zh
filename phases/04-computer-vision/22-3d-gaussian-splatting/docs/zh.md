# 3D 高斯泼溅——从头构建

> 一个场景是数百万个 3D 高斯的点云。每个高斯有一个位置、方向、尺度、不透明度和一个依赖于视角的颜色。将它们光栅化，通过光栅化反向传播，完成。

**类型：** 构建
**语言：** Python
**前置知识：** 第四阶段第13课（3D 视觉与 NeRF），第一阶段第12课（张量运算），第四阶段第10课（可选的扩散基础）
**时间：** ~90分钟

## 学习目标

- 解释为什么 3D 高斯泼溅在 2026 年取代了 NeRF 作为逼真 3D 重建的生产默认选择
- 陈述每个高斯的六个参数（位置、旋转四元数、尺度、不透明度、球谐颜色、可选特征）以及每个贡献多少浮点数
- 从头实现一个使用 alpha 合成的 2D 高斯泼溅光栅化器，然后展示 3D 情况如何投影到同一循环
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 从 20-50 张照片重建场景，并导出为 `KHR_gaussian_splatting` glTF 扩展或 OpenUSD 26.03 的 `UsdVolParticleField3DGaussianSplat` 模式

## 问题

NeRF 将一个场景存储为 MLP 的权重。每个渲染像素是沿光线的数百次 MLP 查询。训练需要数小时，渲染需要数秒，并且权重无法编辑——如果你想在场景中移动一把椅子，你必须重新训练。

3D 高斯泼溅（Kerbl、Kopanas、Leimkühler、Drettakis，SIGGRAPH 2023）取代了这一切。一个场景是一组显式的 3D 高斯。渲染是 100+ fps 的 GPU 光栅化。训练只需几分钟。编辑是直接的：平移一部分高斯，你就移动了椅子。到 2026 年，Khronos 集团已经批准了高斯泼溅的 glTF 扩展，OpenUSD 26.03 提供了高斯泼溅模式，Zillow 和 Apartments.com 使用它们渲染房地产，大多数关于 3D 重建的新研究论文都是核心 3DGS 思想的变体。

心智模型很简单，但数学有足够多的活动部件，以至于大多数介绍从光栅化开始，跳过了投影和球谐函数。本课构建了完整的系统——先做 2D 版本，然后扩展到 3D。

## 概念

### 一个高斯携带的内容

一个 3D 高斯是空间中的一个参数化斑点，具有以下属性：

```
position         mu         (3,)    世界坐标中的中心
rotation         q          (4,)    编码方向的单位四元数
scale            s          (3,)    每轴对数尺度（渲染时取指数）
opacity          alpha      (1,)    Sigmoid 后的不透明度 [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   视角相关的颜色
```

旋转 + 尺度构建了一个 3x3 协方差矩阵：`Sigma = R S S^T R^T`。这就是高斯在 3D 中的形状。球谐函数让颜色随视角方向变化——镜面高光、细微光泽、视角相关的辉光——而无需存储每视角纹理。使用 3 阶 SH，每个颜色通道有 16 个系数，每个高斯用于颜色就需要 48 个浮点数。

一个场景通常有 100-500 万个高斯。每个存储大约 60 个浮点数（3 + 4 + 3 + 1 + 48 + misc）。对于一个 500 万高斯的场景，这大约是 240 MB——远小于带逐点纹理的等效点云，也小于以高分辨率重新渲染的 NeRF 的 MLP 权重。

### 光栅化，而非光线步进

```mermaid
flowchart LR
    SCENE["数百万个 3D 高斯<br/>（位置、旋转、尺度、<br/>不透明度、SH 颜色）"] --> PROJ["投影到 2D<br/>（相机外参 + 内参）"]
    PROJ --> TILES["分配到 Tile<br/>（16x16 屏幕空间）"]
    TILES --> SORT["逐 Tile<br/>深度排序"]
    SORT --> ALPHA["从前往后<br/>Alpha 合成"]
    ALPHA --> PIX["像素颜色"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五个步骤，全部 GPU 友好。无需每像素 MLP 查询。一张 RTX 3080 Ti 以 147 fps 渲染 600 万个泼溅。

### 投影步骤

在世界位置 `mu` 处具有 3D 协方差 `Sigma` 的 3D 高斯投影到屏幕位置 `mu'` 处具有 2D 协方差 `Sigma'` 的 2D 高斯：

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = 视角变换（相机的旋转 + 平移）
J = 在 mu' 处透视投影的雅可比矩阵
```

2D 高斯的足迹是一个椭圆，其轴是 `Sigma'` 的特征向量。该椭圆内的每个像素都接收高斯的贡献，权重为 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`。

### Alpha 合成规则

对于一个像素，覆盖它的高斯从后到前排序（或等价地从前到后，使用反转公式）。颜色使用与自 1980 年代以来每个半透明光栅化器相同的方程合成：

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       透射率到 i 为止
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   局部贡献
c_i = eval_SH(SH_i, view_direction)    视角相关颜色
```

**这与 NeRF 的体渲染方程相同**，只是在一个显式的稀疏高斯集上而不是沿光线的密集样本上。这个恒等式就是渲染质量匹配 NeRF 的原因——两者都在积分相同的辐射场方程。

### 为什么这是可微的

每一步——投影、Tile 分配、Alpha 合成、SH 评估——相对于高斯参数都是可微的。给定真值图像，计算渲染像素损失，通过光栅化器反向传播，通过梯度下降更新所有 `(mu, q, s, alpha, c_lm)`。经过约 30,000 次迭代，高斯找到它们正确的位置、尺度和颜色。

### 密集化和剪枝

一组固定的高斯无法覆盖复杂场景。训练包括两种自适应机制：

- **克隆** — 当梯度幅度高但尺度小时，在当前位置克隆一个高斯——重建需要更多细节。
- **分裂** — 当梯度高时，将一个大尺度的高斯分裂为两个更小的高斯——一个大高斯太平滑，无法拟合该区域。
- **剪枝** — 移除不透明度低于阈值的高斯——它们没有贡献。

密集化每 N 次迭代运行一次。一个场景通常从约 10 万个初始高斯（从 SfM 点播种）增长到训练结束时的 100-500 万个。

### 球谐函数（一段话）

视角相关颜色是单位球面上的函数 `c(direction)`。球谐函数是球面上的傅里叶基。截断到 `L` 阶，每个通道有 `(L+1)^2` 个基函数。为新视角评估颜色是学习到的 SH 系数与在视角方向评估的基函数之间的点积。0 阶 = 1 个系数 = 恒定颜色。3 阶 = 16 个系数 = 足以捕捉朗伯着色、镜面反射和轻微反射。SD 高斯泼溅论文默认使用 3 阶。

### 2026 年的生产堆栈

```
1. 采集         智能手机 / DJI 无人机 / 手持扫描仪
2. SfM / MVS    COLMAP 或 GLOMAP 推导相机位姿 + 稀疏点
3. 训练 3DGS    nerfstudio / gsplat / inria official / PostShot（RTX 4090 上约 10-30 分钟）
4. 编辑          SuperSplat / SplatForge（清除漂浮物，分割）
5. 导出          .ply -> glTF KHR_gaussian_splatting 或 .usd（OpenUSD 26.03）
6. 查看          Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 和生成变体

- **4D 高斯泼溅** — 高斯是时间的函数；用于体视频。
- **生成泼溅** — 文本到泼溅模型（World Labs 的 Marble），可以生成整个场景。
- **3D 高斯无迹变换** — NVIDIA NuRec 的变体，用于自动驾驶仿真。

## 构建

### 第一步：2D 高斯

我们先构建一个 2D 光栅化器。3D 情况在投影后简化为它。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      中心
    covs:   (G, 2, 2)   协方差矩阵
    points: (H, W, 2)   像素坐标
    返回： (G, H, W)   每个高斯在每个像素的密度
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` 对每个（高斯，像素）对执行二次型 `diff^T Sigma^-1 diff`。

### 第二步：2D 泼溅光栅化器

从前到后的 Alpha 合成。2D 中深度无意义，所以我们使用一个学习的每高斯标量来确定顺序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     [0, 1]
    depths:    (G,)     用于排序的每高斯标量
    image_size: (H, W)
    返回：   (H, W, 3) 渲染图像
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

不快——真正的实现使用基于 tile 的 CUDA 内核——但数学完全正确且完全可微。

### 第三步：可训练的 2D 泼溅场景

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # 2D 中单个角度
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是无约束参数，在渲染时通过正确的激活函数映射。这是每个 3DGS 实现的标准模式。

### 第四步：将 2D 高斯拟合到目标图像

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # 红色圆形
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # 蓝色方形
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"步 {step:3d}  mse {loss.item():.4f}")
```

200 步后，64 个高斯稳定到两个形状中。这就是整个想法——在显式几何基元上的梯度下降。

### 第五步：从 2D 到 3D

3D 扩展保持相同的循环。新增：

1. 每高斯旋转是四元数而不是单个角度。
2. 协方差是 `R S S^T R^T`，其中 `R` 从四元数构建，`S = diag(exp(log_scale))`。
3. 投影 `(mu, Sigma) -> (mu', Sigma')` 使用相机外参和在 `mu` 处的透视投影的雅可比矩阵。
4. 颜色变成球谐展开；在视角方向评估它。
5. 深度排序使用实际的相机空间 z 而不是学习的标量。

每个生产实现（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）都使用基于 tile 的 CUDA 内核在 GPU 上精确地执行此操作。

### 第六步：球谐函数评估

最高 3 阶的 SH 基有每个通道 16 个项。评估：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   最后一个维度是 RGB 通道
    dirs:      (..., 3)       单位向量
    返回：   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # 为简洁起见省略了 3 阶项；完整的 16 系数版本在代码文件中
    return result
```

学习到的 `sh_coeffs` 存储了该高斯的"每个方向的颜色"。在渲染时，你针对当前视角方向评估它，得到一个 3 向量 RGB。

## 使用

对于真正的 3DGS 工作，使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的 3DGS 训练器。对于典型场景，在 RTX 4090 上运行需要 10-30 分钟。

2026 年重要的导出选项：

- `.ply` — 原始高斯云（可移植，文件最大）。
- `.splat` — PlayCanvas / SuperSplat 量化格式。
- glTF `KHR_gaussian_splatting` — Khronos 标准，跨查看器可移植（2026 年 2 月 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat` — USD 原生，用于 NVIDIA Omniverse 和 Vision Pro 流水线。

对于 4D / 动态场景，`4DGS` 和 `Deformable-3DGS` 使用随时间变化的位置和不透明度扩展了相同的机制。

## 交付

本课产出：

- `outputs/prompt-3dgs-capture-planner.md` — 一个提示词，为给定场景类型规划采集会话（照片数量、相机路径、光照）。
- `outputs/skill-3dgs-export-router.md` — 一个技能，根据下游查看器或引擎选择正确的导出格式（`.ply` / `.splat` / glTF / USD）。

## 练习

1. **（简单）** 在不同的合成图像上运行上面的 2D 泼溅训练器。改变 `num_splats` 为 `[16, 64, 256]`，绘制每个的 MSE 与步长曲线。识别收益递减的点。
2. **（中等）** 扩展 2D 光栅化器，支持通过 2 阶球谐函数依赖于标量"视角角度"的每高斯 RGB 颜色。在一对目标图像上训练并验证模型重建了二者。
3. **（困难）** 克隆 `nerfstudio` 并在你拥有的任何场景（书桌、植物、人脸、房间）的 20 张照片采集上训练 `splatfacto`。导出为 glTF `KHR_gaussian_splatting` 并在查看器（Three.js `GaussianSplats3D`、SuperSplat、Babylon.js V9）中打开。报告训练时间、高斯数量和渲染 fps。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| 3DGS | "高斯泼溅" | 显式场景表示为数百万个 3D 高斯，每高斯有位置、旋转、尺度、不透明度、SH 颜色 |
| 协方差 | "高斯的形状" | `Sigma = R S S^T R^T`；一个高斯的方向和各向异性尺度 |
| Alpha 合成 | "从后到前混合" | 与 NeRF 的体渲染相同的方程，现在在显式稀疏集上 |
| 密集化 | "克隆和分裂" | 在重建欠拟合处自适应地添加新高斯 |
| 剪枝 | "删除低不透明度" | 移除在训练中坍缩到接近零不透明度的高斯 |
| 球谐函数 | "视角相关颜色" | 球面上的傅里叶基；将颜色存储为视角方向的函数 |
| Splatfacto | "nerfstudio 的 3DGS" | 2026 年训练 3DGS 的最简单路径 |
| `KHR_gaussian_splatting` | "glTF 标准" | Khronos 2026 扩展，使 3DGS 跨查看器和引擎可移植 |

## 延伸阅读

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — 原始论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — 生产级 CUDA 光栅化器
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — 参考训练方法
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026 年可移植格式
- [OpenUSD 26.03 release notes](https://openusd.org/release/) — `UsdVolParticleField3DGaussianSplat` 模式
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — 行业概览
