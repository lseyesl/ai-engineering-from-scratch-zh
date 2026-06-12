# SAM 3 与开放词汇分割

> 给模型一个文本提示和一张图像，获得每个匹配物体的遮罩。SAM 3 将其变为一次前向传播。

**类型：** 使用 + 构建
**语言：** Python
**前置知识：** 第四阶段第07课（U-Net），第四阶段第08课（Mask R-CNN），第四阶段第18课（CLIP）
**时间：** ~60分钟

## 学习目标

- 区分 SAM（仅视觉提示）、Grounded SAM / SAM 2（检测器 + SAM）和 SAM 3（通过可提示概念分割实现原生文本提示）
- 解释 SAM 3 架构：共享骨干 + 图像检测器 + 基于记忆的跟踪器 + 存在性头 + 解耦检测器-跟踪器设计
- 使用 Hugging Face `transformers` 的 SAM 3 集成进行文本提示的检测、分割和视频跟踪
- 根据延迟、概念复杂度和部署目标在 SAM 3、Grounded SAM 2、YOLO-World 和 SAM-MI 之间做出选择

## 问题

2023 年的 SAM 是一个仅视觉提示的模型：你点击一个点或画一个框，它返回一个遮罩。对于"给我这张照片中所有橙子"，你需要一个检测器（Grounding DINO）来产生框，然后 SAM 来分割每个框。Grounded SAM 将其变成了一个流水线，但它是两个冻结模型的级联，不可避免地存在错误累积。

SAM 3（Meta，2025 年 11 月，ICLR 2026）将级联合并了。它接受一个短名词短语或一个图像样本作为提示，并在一次前向传播中返回所有匹配的遮罩和实例 ID。这就是**可提示概念分割（PCS）**。结合 2026 年 3 月的 Object Multiplex 更新（SAM 3.1），它高效地跟踪视频中同一概念的多个实例。

本课讲的是这所代表的结构性转变。2D 分割、检测和文本-图像基础已经合并到一个模型中。生产问题不再是"我应该将哪些流水线串联起来"，而是"哪个可提示模型可以端到端地处理我的用例"。

## 概念

### 三代模型

```mermaid
flowchart LR
    subgraph SAM1["SAM (2023)"]
        A1["图像 + 点/框提示"] --> A2["ViT 编码器"] --> A3["遮罩解码器"]
        A3 --> A4["该提示的遮罩"]
    end
    subgraph GSAM2["Grounded SAM 2 (2024)"]
        B1["文本"] --> B2["Grounding DINO"] --> B3["框"] --> B4["SAM 2"] --> B5["遮罩 + 跟踪"]
        B6["图像"] --> B2
        B6 --> B4
    end
    subgraph SAM3["SAM 3 (2025)"]
        C1["文本或图像样本"] --> C2["共享骨干"]
        C3["图像"] --> C2
        C2 --> C4["图像检测器 + 记忆跟踪器<br/>+ 存在性头"]
        C4 --> C5["所有匹配遮罩<br/>+ 实例 ID"]
    end

    style SAM1 fill:#e5e7eb,stroke:#6b7280
    style GSAM2 fill:#fef3c7,stroke:#d97706
    style SAM3 fill:#dcfce7,stroke:#16a34a
```

### 可提示概念分割

一个"概念提示"是一个短名词短语（"yellow school bus"、"striped red umbrella"、"hand holding a mug"）或一个图像样本。模型返回图像中匹配该概念的每个实例的分割遮罩，加上每个匹配的唯一实例 ID。

这与经典的视觉提示 SAM 在三个方面不同：

1. 无需逐实例提示——一个文本提示返回所有匹配。
2. 开放词汇——概念可以是自然语言可描述的任何事物。
3. 一次返回多个实例，而不是每个提示一个遮罩。

### 关键架构组件

- **共享骨干** — 一个 ViT 处理图像。检测器头和基于记忆的跟踪器都从中读取。
- **存在性头** — 预测概念是否存在于图像中。将"这个在吗？"与"它在哪？"解耦。减少不存在概念上的假阳性。
- **解耦检测器-跟踪器** — 图像级检测和视频级跟踪有独立的头，互不干扰。
- **记忆库** — 跨帧存储每个实例的特征用于视频跟踪（与 SAM 2 相同的机制）。

### 大规模训练

SAM 3 在**400 万个独特概念**上训练，由数据引擎通过 AI + 人工审查迭代标注和纠正生成。新的 **SA-CO 基准**包含 27 万个独特概念，比之前的基准大 50 倍。SAM 3 在 SA-CO 上达到人类水平的 75-80%，并在图像 + 视频 PCS 上将现有系统翻倍。

### SAM 3.1 Object Multiplex

2026 年 3 月更新：**Object Multiplex** 引入了一个共享记忆机制，用于同时跟踪同一概念的多个实例。以前，跟踪 N 个实例需要 N 个独立的记忆库。Multiplex 将其压缩为一个共享记忆加每个实例的查询。结果：多目标跟踪显著加快而不牺牲准确率。

### Grounded SAM 在 2026 年仍然适用的场景

- 当你需要更换特定的开放词汇检测器（DINO-X、Florence-2）时。
- 当 SAM 3 的许可（HF 门控）成为障碍时。
- 当你需要对检测器阈值有比 SAM 3 暴露的更多控制时。
- 用于检测器组件的研究 / 消融工作时。

模块化流水线仍有其一席之地。对于大多数生产工作，SAM 3 是更简单的答案。

### YOLO-World vs SAM 3

- **YOLO-World** — 仅开放词汇检测器（无遮罩）。实时。当你需要高速 fps 的框时最佳。
- **SAM 3** — 完整分割 + 跟踪。较慢但输出更丰富。

生产分工：YOLO-World 用于快速检测流水线（机器人导航、快速仪表板），SAM 3 用于任何需要遮罩或跟踪的场景。

### SAM-MI 效率

SAM-MI（2025-2026）解决了 SAM 解码器的瓶颈。关键思想：

- **稀疏点提示** — 使用少量精心选择的点而不是密集提示；将解码器调用减少 96%。
- **浅层遮罩聚合** — 将粗略的遮罩预测合并为一个更锐利的遮罩。
- **解耦遮罩注入** — 解码器接收预计算的遮罩特征而不是重新运行。

结果：在开放词汇基准上比 Grounded-SAM 快约 1.6 倍。

### 三个模型的输出格式

都返回相同的总体结构（框 + 标签 + 分数 + 遮罩 + ID），这很有帮助——你的下游流水线不必根据运行的模型分支。

## 构建

### 第一步：提示构建

构建一个辅助函数，将用户句子转换为 SAM 3 概念提示列表。这是"用户输入的内容"与"模型消费的内容"之间的边界。

```python
def split_concepts(sentence):
    """
    多概念提示的启发式分词器。
    返回短名词短语列表。
    """
    for sep in [",", ";", "and", "or", "&"]:
        if sep in sentence:
            parts = [p.strip() for p in sentence.replace("and ", ",").split(",")]
            return [p for p in parts if p]
    return [sentence.strip()]

print(split_concepts("cats, dogs and balloons"))
```

SAM 3 每次前向传播接受一个概念；对于多概念查询，循环或批处理它们。

### 第二步：后处理辅助函数

将 SAM 3 的原始输出转换为与我们第四阶段第 16 课流水线契约匹配的干净检测列表。

```python
from dataclasses import dataclass
from typing import List

@dataclass
class ConceptDetection:
    concept: str
    instance_id: int
    box: tuple          # (x1, y1, x2, y2)
    score: float
    mask_rle: str       # 游程编码


def rle_encode(binary_mask):
    flat = binary_mask.flatten().astype("uint8")
    runs = []
    prev, count = flat[0], 0
    for v in flat:
        if v == prev:
            count += 1
        else:
            runs.append((int(prev), count))
            prev, count = v, 1
    runs.append((int(prev), count))
    return ";".join(f"{v}x{c}" for v, c in runs)
```

RLE 使得即使对于许多高分辨率遮罩，响应负载也很小。相同格式适用于 SAM 2、SAM 3、Grounded SAM 2。

### 第三步：统一的开放词汇分割接口

将你拥有的任何后端（SAM 3、Grounded SAM 2、YOLO-World + SAM 2）包装在一个单一方法后面。你的下游代码在后端更改时不需要变化。

```python
from abc import ABC, abstractmethod
import numpy as np

class OpenVocabSeg(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray, concept: str) -> List[ConceptDetection]:
        ...


class StubOpenVocabSeg(OpenVocabSeg):
    """
    用于流水线测试（未加载真实模型时）的确定性桩。
    """
    def detect(self, image, concept):
        h, w = image.shape[:2]
        return [
            ConceptDetection(
                concept=concept,
                instance_id=0,
                box=(w * 0.2, h * 0.3, w * 0.5, h * 0.8),
                score=0.89,
                mask_rle="0x100;1x50;0x200",
            ),
            ConceptDetection(
                concept=concept,
                instance_id=1,
                box=(w * 0.55, h * 0.25, w * 0.85, h * 0.75),
                score=0.74,
                mask_rle="0x80;1x40;0x220",
            ),
        ]
```

真正的 `SAM3OpenVocabSeg` 子类将包装 `transformers.Sam3Model` 和 `Sam3Processor`。

### 第四步：Hugging Face SAM 3 用法（参考）

对于实际模型，`transformers` 集成：

```python
from transformers import Sam3Processor, Sam3Model
import torch

processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").eval()

inputs = processor(images=pil_image, return_tensors="pt")
inputs = processor.set_text_prompt(inputs, "yellow school bus")

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(
    outputs.masks, inputs.original_sizes, inputs.reshaped_input_sizes
)
boxes = outputs.boxes
scores = outputs.scores
```

一个提示，所有匹配在一次调用中返回。

### 第五步：衡量 Grounded SAM 2 免费给了你什么

一个诚实的基准：在用 SAM 3 替换 Grounded SAM 2 时，实际流水线中会发生什么？

- **延迟**：SAM 3 节省了一次前向传播（没有独立的检测器），但模型本身更重；通常是中性或略有加速。
- **准确率**：SAM 3 在罕见或组合概念（"striped red umbrella"）上明显更好。在常见的单词语义概念上相似。
- **灵活性**：Grounded SAM 2 允许你更换检测器（DINO-X、Florence-2、Grounding DINO 1.5）；SAM 3 是整体式的。

结论：SAM 3 是 2026 年开放词汇分割的默认选择。Grounded SAM 2 在你需要检测器灵活性或不同的许可条款时仍然是正确的答案。

## 使用

生产部署模式：

- **实时标注** — SAM 3 + CVAT 的标签即文本提示功能。标注员选择一个标签名；SAM 3 预标注每个匹配的实例。审查并纠正。
- **视频分析** — SAM 3.1 Object Multiplex 用于多目标跟踪；将帧馈送到基于记忆的跟踪器。
- **机器人** — SAM 3 用于开放词汇操作（"pick up the red cup"）；作为规划基元运行。
- **医学影像** — 在医学概念上微调 SAM 3；需要在 HF 上提交访问请求。

Ultralytics 在其 Python 包中包装了 SAM 3：

```python
from ultralytics import SAM

model = SAM("sam3.pt")
results = model(image_path, prompts="yellow school bus")
```

与 YOLO 和 SAM 2 接口相同。

## 交付

本课产出：

- `outputs/prompt-open-vocab-stack-picker.md` — 根据延迟、概念复杂度和许可选择 SAM 3 / Grounded SAM 2 / YOLO-World / SAM-MI 的提示词。
- `outputs/skill-concept-prompt-designer.md` — 将用户话语转换为格式良好的 SAM 3 概念提示（分割、消歧、回退）的技能。

## 练习

1. **（简单）** 在你选择的 10 张图像上使用概念提示运行 SAM 3。与相同图像上的 SAM 2 + Grounding DINO 1.5 进行比较。报告每个模型遗漏了哪些概念。
2. **（中等）** 在 SAM 3 之上构建一个"点击包含 / 点击排除"的 UI：文本提示返回候选实例；用户点击保留哪些作为正例。将最终概念集输出为 JSON。
3. **（困难）** 在自定义概念集（例如 5 种电子元件）上微调 SAM 3，每种有 20 张标注图像。与相同测试集上的零样本 SAM 3 比较；测量遮罩 IoU 提升。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| 开放词汇分割 | "按文本分割" | 为自然语言描述的物体生成遮罩，而非固定标签集 |
| PCS | "可提示概念分割" | SAM 3 的核心任务——给定名词短语或图像样本，分割所有匹配实例 |
| 概念提示 | "文本输入" | 短名词短语或图像样本；不是完整句子 |
| 存在性头 | "它在这里吗？" | SAM 3 模块，在定位之前决定概念是否存在于图像中 |
| SA-CO | "SAM 3 基准" | 27 万概念开放词汇分割基准；比之前的开放词汇基准大 50 倍 |
| Object Multiplex | "SAM 3.1 更新" | 共享记忆多目标跟踪；快速联合跟踪多个实例 |
| Grounded SAM 2 | "模块化流水线" | 检测器 + SAM 2 级联；在检测器交换重要时仍然相关 |
| SAM-MI | "高效 SAM 变体" | Mask Injection，比 Grounded-SAM 快 1.6 倍 |

## 延伸阅读

- [SAM 3: Segment Anything with Concepts (arXiv 2511.16719)](https://arxiv.org/abs/2511.16719)
- [SAM 3.1 Object Multiplex (Meta AI, March 2026)](https://ai.meta.com/blog/segment-anything-model-3/)
- [SAM 3 model page on Hugging Face](https://huggingface.co/facebook/sam3)
- [Grounded SAM 2 tutorial (PyImageSearch)](https://pyimagesearch.com/2026/01/19/grounded-sam-2-from-open-set-detection-to-segmentation-and-tracking/)
- [Ultralytics SAM 3 docs](https://docs.ultralytics.com/models/sam-3/)
- [SAM3-I: Instruction-aware SAM (arXiv 2512.04585)](https://arxiv.org/abs/2512.04585)
