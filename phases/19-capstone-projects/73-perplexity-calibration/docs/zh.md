# 困惑度与校准

> 如果你的模型对一千个答案说 90% 确信，但只答对了六百个，那它校准得不好。校准是可信评估的一半。另一半是困惑度，它告诉你模型是否认为留出文本是合理的。

**类型：** Build
**语言：** Python
**前置条件：** Phase 19 Track B 基础，课程 70 和 71
**时间：** ~90 分钟

## 学习目标

- 从模型适配器提供的 token 负对数概率计算留出语料上的 token 级困惑度。
- 从分箱的预测概率计算分类器或多选评估的期望校准误差（ECE）。
- 计算 Brier 分数（对正确性指标的均方误差）并解释它在什么情况下做了 ECE 做不到的事。
- 构建绘制置信度-准确率曲线所需的可靠性图数据。
- 将三者接入评估框架，使运行器可以将 `perplexity`、`ece` 和 `brier` 数值附加到模型报告。

## 困惑度告诉你什么

困惑度是每个 token 的指数化平均负对数似然。越低越好。困惑度为一意味着模型对每个实际 token 分配概率一。困惑度等于词汇表大小意味着模型是均匀的，什么都没学到。真实数字落在中间：一个 2026 年的强基础模型在 WikiText-103 上大约在八到十二之间。一个差的模型在同一文本上在五十以上。

框架本身不计算对数概率。那些来自模型适配器。框架做聚合：它接收一个每 token 负对数概率列表、一个每序列 token 计数列表，返回语料困惑度。

```python
def perplexity(neg_log_probs, token_counts):
    total_nll = sum(neg_log_probs)
    total_tokens = sum(token_counts)
    return math.exp(total_nll / total_tokens)
```

实现处理零 token 边界情况，并断言负对数概率为非负。一个常见错误是忘记取负：返回 `log p` 而非 `-log p` 的适配器会产生低于一的困惑度，这是不可能的。函数将此捕获为契约违反。

## ECE 度量什么

期望校准误差按置信度将预测分组到固定数量的箱中，然后测量每个箱中置信度与准确率之间的平均差距，按箱大小加权。

```mermaid
flowchart TD
    A[N predictions with confidence p and correctness y] --> B[bin by p into M bins]
    B --> C[for each bin compute avg confidence and avg accuracy]
    C --> D[gap = abs avg conf - avg acc]
    D --> E[weighted by bin size / N]
    E --> F[ECE = sum of weighted gaps]
```

标准公式在 `[0, 1]` 上使用十个等宽箱。实现支持任何正整数计数。我们暴露 `bins` 参数，使运行器可以在发布惯例（10）和比较惯例（15）之间选择。

ECE 受箱数和样本量偏差影响。十个箱和一百个预测，你无法区分 0.02 的 ECE 和随机噪声。实现返回已填充箱的数量以及 ECE，使运行器可以在样本太少时拒绝报告单个数字。

## Brier 分数做了 ECE 做不到的事

ECE 只关心平均差距。一个模型在一半箱上过度自信、在另一半上不够自信，可以有低 ECE 但局部校准很差。Brier 分数对每个预测测量与真实结果的平方误差，因此直接惩罚离散度。

对于二元结果，Brier 是 `mean((p_i - y_i)^2)`。它分解为可靠性、分辨率和不确定性。我们计算分数和分解。运行器报告标量但记录分解供仪表板使用。

```python
def brier(p, y):
    return float(np.mean((p - y) ** 2))
```

## 可靠性图数据

可靠性图绘制每个箱中的预测置信度与经验准确率。对角线是完美校准。函数返回三个数组：每箱平均置信度、每箱平均准确率和每箱计数。绘图代码在下游；本课停在数据结构。

```mermaid
flowchart LR
    A[predictions, confidences] --> B[bin edges 0 to 1]
    B --> C[per-bin mean confidence]
    B --> D[per-bin mean accuracy]
    B --> E[per-bin count]
    C --> R[reliability data triple]
    D --> R
    E --> R
```

返回的元组是调用层绘制图表或计算自定义 ECE 变体（自适应 ECE、扫描 ECE 等）所需的数据。我们返回 numpy 数组，使下游代码无需转换。

## 置信度来源

框架不假设置信度来自 softmax。它接受每个预测 `[0, 1]` 中的任何数字。对于多选题任务，自然置信度是 `选项对数似然上的 softmax`。对于自由文本，自然置信度是模型自报告的概率或平均对数似然的指数。评估只消费这个数字。它来自哪里是适配器的工作。

## 边界情况

- 所有预测错误：ECE 是平均置信度，Brier 很高，困惑度是模型对文本的看法。
- 所有预测正确且高置信度：ECE 接近零，Brier 接近零。
- 完全不确定的预测器在 p=0.5：ECE 是 0.5 减去准确率，Brier 是 0.25 减去一个修正项。
- 空输入：ECE、Brier 和可靠性返回 `0.0`（或零填充数组）。困惑度对零 token 情况返回 `NaN`。这些路径都不发出警告；运行器检查值并决定是报告还是跳过。

这些情况都写入了测试。真实模型在真实基准上不会命中它们，但有缺陷的适配器或微小样本会，运行器不应崩溃。

## 分发

校准不是像 F1 那样的每任务度量。它是每模型报告。运行器在整个评估中累积 `(confidence, correct)` 对，一次性计算 ECE、Brier 和可靠性数据。困惑度在留出文本语料上计算，与逐任务评分分开。

接口是：

```python
report = CalibrationReport.from_predictions(confidences, correct)
report.ece          # float
report.brier        # float
report.reliability  # tuple of three numpy arrays
report.populated_bins  # int
```

`PerplexityResult.from_token_nll(neg_log_probs, token_counts)` 返回困惑度和每 token 的平均负对数似然。

## 本课不做的事

本课不调用模型。本课不实现 softmax。本课不从输出 token 估计置信度；那是适配器的工作。本课不做温度缩放或 Platt 缩放；那些是事后修复，属于不同的课程。本课的要点是让三个数字（困惑度、ECE、Brier）可信且可复现。

## 如何阅读代码

`main.py` 定义了 `perplexity`、`expected_calibration_error`、`brier_score`、`reliability_diagram` 以及 `CalibrationReport` / `PerplexityResult` 数据类。演示在已知真值的合成预测上运行：一个校准良好的模型、一个过度自信的模型和一个不够自信的模型。`code/tests/test_calibration.py` 中的测试固定了每个边界情况以及合成预测器的参考值。

从头到尾阅读 `main.py`。函数排序从标量到向量到报告。每个函数都有简短的文档字符串，包含数学和契约。

## 延伸阅读

校准是已发表评估中最被忽视的轴。大多数排行榜报告一个准确率数字就完事了。一个在准确率上获胜但在 Brier 上失败的模型，比一个在准确率上低几分但可靠报告不确定性的模型，是更差的生产部署。一旦你有了校准管道，在留出验证切片上添加温度缩放，重新计算 ECE，观察差距缩小。那是单独的课程，但基础在这里。
