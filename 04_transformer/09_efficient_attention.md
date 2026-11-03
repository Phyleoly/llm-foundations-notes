# Attention 的计算与存储优化

> 状态：提纲

> 前置内容：[Attention 的基本计算](./01_attention.md)、[参数量、计算量与显存](./08_parameter_count_and_compute.md)。
>
> 阅读位置：建议完成基础主线后再展开。

本篇要回答：如何在保留 attention 定义的同时减少中间结果的存储与搬运？

## 1. 标准实现的中间结果

- 从 QK、Softmax、加权求和逐步标出需要读写哪些张量。
- 分别讨论计算量、显存占用与内存访问，避免用一个复杂度概括全部成本。

## 2. 分块与在线 Softmax

- 为分块计算列出需要维护的归一化统计量，并给出小矩阵分块核对目标。
- 说明重新计算与保存中间结果之间的权衡。

## 3. FlashAttention 的边界

- 解释精确 attention 的计算重排与稀疏 / 近似 attention 的区别。
- 区分减少 IO、减少显存和改变运算复杂度；误差核对需注明 dtype。

## 4. 测量框架

- 固定形状、设备、dtype、mask 和 warm-up，比较时间、峰值显存与输出误差。
- 把模型结构的 MQA / GQA 与计算实现的优化分开记录。

## 阅读入口

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
