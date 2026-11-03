# 混合精度与数值范围

> 状态：提纲

> 前置内容：[LogSoftmax 与数值稳定性](../01_probability/03_logsoftmax_numerical_stability.md)、[梯度优化、AdamW 与梯度裁剪](./02_optimization.md)。

本篇要回答：哪些计算可以降低精度，哪些数值问题需要单独处理？

## 1. 浮点数的精度与范围

- 比较不同格式的指数范围与有效精度，区分溢出、下溢和舍入误差。
- 用一个大数和一个小梯度说明仅增加小数精度不能解决全部问题。

## 2. 混合精度的计算路径

- 记录参数、前向激活、矩阵累加、梯度与优化器状态实际使用的 dtype。
- 解释 autocast 的职责，并核对哪些运算由框架保留为较高精度。

## 3. Loss Scaling

- 推导缩放 loss、反向传播与反缩放梯度的关系。
- 说明溢出检测、跳过更新与梯度裁剪的顺序，区分 FP16 / BF16 的使用条件。

## 4. 最小核对与边界

- 对同一小批次比较 FP32 与混合精度的 loss、梯度和内存占用。
- 记录设备、框架版本与算子选择；不要将低精度训练简化成转换所有参数 dtype。

## 阅读入口

- [PyTorch：Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)
