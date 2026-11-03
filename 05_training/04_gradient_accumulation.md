# 梯度累积与有效 Batch

> 状态：提纲

> 前置内容：[Batching、Padding、Packing 与 Loss Mask](./01_batching_padding_and_masking.md)、[梯度优化、AdamW 与梯度裁剪](./02_optimization.md)。

本篇要回答：显存只能容纳小 batch 时，怎样累积梯度而不改变损失权重？

## 1. 有效 Batch 的定义

- 定义 micro-batch、累积次数、数据并行规模与有效样本数。
- 同时记录有效 token 数，区分样本数相同与损失分母相同。

## 2. Loss 缩放与聚合

- 推导有效 token 数相同时的缩放方式，再考虑变长 micro-batch。
- 用一个短 batch 和一个长 batch 检查平均梯度是否符合预期的全局目标。

## 3. 更新边界

- 标出 zero_grad、backward、clipping、optimizer step 与 scheduler step 的调用边界。
- 讨论最后一个不足完整累积周期的 batch，以及分布式梯度同步时机。

## 4. 等价性的条件

- 列出与大 batch 一次更新比较时要固定的条件，包括随机性与损失归约。
- 设计一次小模型梯度核对，不把近似相同的训练曲线当作逐步数值等价。
