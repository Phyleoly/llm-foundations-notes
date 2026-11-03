# 训练循环、监控与 Checkpoint

> 状态：提纲

> 前置内容：[Decoder-only Transformer 的完整前向过程](../04_transformer/07_decoder_only_transformer.md)、[Batching、Padding、Packing 与 Loss Mask](./01_batching_padding_and_masking.md)、[梯度优化、AdamW 与梯度裁剪](./02_optimization.md)、[学习率调度与训练预算](./03_learning_rate_schedule.md)、[梯度累积与有效 Batch](./04_gradient_accumulation.md)、[混合精度与数值范围](./05_mixed_precision.md)。

本篇要回答：怎样描述一条能够检查、恢复并复现实验的最小训练流程？

## 1. 最小训练循环

- 按数据读取、前向、loss、反向、更新、验证和保存的顺序写伪代码。
- 明确 model.train / eval、梯度开关、label shifting 与有效 token 计数。

## 2. 训练检查与监控

- 安排小数据过拟合、初始 loss 合理性和单步参数变化检查。
- 分别记录训练与验证 loss、学习率、梯度范数、token 吞吐和显存。

## 3. Checkpoint 与恢复

- 区分仅用于推理的权重文件与用于续训的完整状态。
- 列出模型、优化器、scheduler、scaler、随机数及数据读取进度的保存与恢复需求。

## 4. 最小实验记录

- 固定一套小模型配置，列出数据、版本、种子、训练预算、命令与结果记录表。
- 说明 smoke test、从头训练、精确续训与论文复现分别验证了什么。
