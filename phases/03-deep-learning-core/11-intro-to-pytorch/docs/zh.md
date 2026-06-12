# PyTorch 入门

> 你从数学上构建了它。现在你是认真的。PyTorch 是你将用于实际生产的框架——理解其设计意味着你可以驾驭它，而不是与它对抗。

**类型：** 参考
**语言：** Python
**前置知识：** 课程 03.10（微框架）
**时间：** ~60 分钟

## 学习目标

- 将 tinyframe 构建块映射到它们的 PyTorch 对应物，并在两者之间逐字翻译训练循环
- 使用 `nn.Module`、`nn.Linear`、`nn.Sequential` 和 `nn.Parameter` 构建架构
- 加载、检查并可视化真实的 MNIST 数据，实现自定义 `DataLoader`
- 根据您对 tinyframe 的了解诊断形状不匹配和梯度问题

## 问题

您从头构建了一个小型的、可用的框架。现在使用真正的框架。从零开始重新学习的设计是细微且精确的：PyTorch 的 `nn.Module` 处理参数注册、设备移动、训练/评估模式以及子模块递归。对于大型项目来说，其抽象级别更高。但如果你理解设计，所有内容都可以逻辑地映射。

真正的挑战：当你的 PyTorch 代码出现问题时——形状不匹配、梯度溢出、不收敛——你将有一个思维模型来说明哪里出了问题。因为你从裸机层构建了同样的东西，所以你可以进行推理，而不是猜测。

## 概念

### 从 tinyframe 到 PyTorch

| tinyframe | PyTorch |
|-----------|---------|
| `Tensor`（标量） | `torch.Tensor`（N 维） |
| `Module` | `nn.Module` |
| `Linear` | `nn.Linear` |
| `Sequential` | `nn.Sequential` |
| `SGD`、`Adam` | `torch.optim.SGD`、`torch.optim.Adam` |
| `backward()` | `tensor.backward()` |

### 关键差异

- **张量维度**：PyTorch 默认处理 N 维批处理
- **设备**：张量位于 CPU 或 GPU 上——每个张量调用 `.to(device)` 完成移动
- **梯度累积**：默认情况下 PyTorch 累积梯度；您必须在每个步骤调用 `optimizer.zero_grad()`
- **训练模式**：`model.train()` 和 `model.eval()` 控制 Dropout 和 BatchNorm 的行为

### 批量张量 vs. 单个张量

在您的 tinyframe 中，一次处理一个样本。PyTorch 默认处理批次。形状约定：
- 单个图像：[1, 28, 28]
- 图像批次：[N, 1, 28, 28]
- 单句：[seq_len, vocab_size]
- 句子批次：[N, seq_len, vocab_size]

第一个维度始终是批次维度。所有包含批次维度的操作（矩阵乘法、损失函数、前向传播）都需要形状匹配。

```python
# 批次中的单个图像
x = torch.randn(1, 28, 28)

# 8 张图像的批次
x = torch.randn(8, 1, 28, 28)

# 展平（例如，用于 MLP 的线性层）：
x = x.view(8, -1)  # 形状：(8, 784)
```

### 设备管理

一个常见的错误是混合设备（有些张量在 CPU 上，有些在 GPU 上）：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
data = data.to(device)
targets = targets.to(device)
```

## 构建它

### 第 1 步：导入并检查数据

```python
from torchvision import datasets, transforms
import torch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist = datasets.MNIST('data', train=True, download=True, transform=transform)
```

### 第 2 步：定义模型（映射自 tinyframe）

```python
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
```

### 第 3 步：训练循环（逐行映射）

```python
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
```

### 第 4 步：评估循环

```python
def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)')
```

### 第 5 步：完整训练脚本

将所有内容组合成一个脚本：

```python
model = MNISTModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.NLLLoss()

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader)
```

```figure
dropout-mask
```

## 使用它

将模型迁移到 GPU 并启动训练：

```bash
python3 mnist_pytorch.py
```

输出应显示 MNIST 上在 5 个 epoch 内测试准确率超过 95%。

## 交付物

本课程产出：
- `mnist_pytorch.py`——用 PyTorch 编写的完整 MNIST 训练脚本
- `outputs/skill-pytorch-to-tinyframe.md`——将 tinyframe 概念映射到 PyTorch 的技能

## 练习

1. 扩展模型以包含 BatchNorm1d 层。打印层前后的均值和标准差。看到规范化的效果了吗？
2. 将损失函数切换为 CrossEntropyLoss（移除 LogSoftmax）并重新训练。验证输出相同。
3. 添加 TensorBoard 日志记录。
4. 实现梯度裁剪以在 backward() 和 optimizer.step() 之间应用。
5. 找到并修复此着色器中的类型不匹配：将目标设为 `targets = torch.randint(0, 10, (8, 1))` 并且不使用 `.view_as`。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| `nn.Module` | "模型类" | 拥有参数、处理训练/评估模式并管理子模块的基类 |
| `nn.Parameter` | "可训练张量" | 包装张量并由 `nn.Module.parameters()` 自动注册的张量子类 |
| `torch.no_grad()` | "不跟踪梯度" | 禁用梯度跟踪的上下文管理器——用于推理以节省内存 |
| `model.eval()` | "评估模式" | 禁用训练专用行为（Dropout、BatchNorm）并冻结模型以进行推理 |
| `tensor.view()` | "重塑" | 在不复制数据的情况下更改张量形状 |
| `nn.Sequential` | "层列表" | 一个按顺序执行层的容器 Module |
| 设备 | "CPU 或 GPU" | 张量或模块所在的硬件上下文（cpu、cuda、mps） |

## 延伸阅读

- PyTorch 官方教程 (https://pytorch.org/tutorials/)
- PyTorch 文档（nn 模块）(https://pytorch.org/docs/stable/nn.html)
- "Deep Learning with PyTorch" by Stevens, Antiga & Viehmann (Manning, 2020)
