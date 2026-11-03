# 第五章 Training

本章讨论在概率目标和 Transformer 结构确定之后，如何实际组织和优化模型训练。

## 精选阅读路线

1. [Batching、Padding、Packing 与 Loss Mask](./01_batching_padding_and_masking.md)：先确认输入、目标和有效 token 的组织方式。
2. [梯度优化、AdamW 与梯度裁剪](./02_optimization.md) → [学习率调度与训练预算](./03_learning_rate_schedule.md)：理解一次参数更新与多步训练的关系。
3. [训练循环、监控与 Checkpoint](./08_training_loop_and_checkpoints.md)：连接数据读取、更新、验证与恢复。

训练设置需要扩展时，再阅读 [梯度累积](./04_gradient_accumulation.md)、[混合精度](./05_mixed_precision.md) 与 [分布式训练](./06_distributed_training.md)。数据侧的问题集中在 [预训练语料与数据组织](./07_pretraining_data.md)。

## 必要基础与相关阅读

先理解 [概率训练目标](../01_probability/01_token_probability_model.md) 和 [Decoder-only Transformer 的完整前向过程](../04_transformer/07_decoder_only_transformer.md)。各篇开头进一步列出该问题需要的最小前置内容，不要求一次读完所有训练主题。

## 全部笔记

以下索引由文章标题与状态生成，阅读顺序以上方路线为准。

<!-- notes:index:start -->
| 笔记 | 状态 |
| --- | --- |
| [Batching、Padding、Packing 与 Loss Mask](./01_batching_padding_and_masking.md) | 提纲 |
| [梯度优化、AdamW 与梯度裁剪](./02_optimization.md) | 提纲 |
| [学习率调度与训练预算](./03_learning_rate_schedule.md) | 提纲 |
| [梯度累积与有效 Batch](./04_gradient_accumulation.md) | 提纲 |
| [混合精度与数值范围](./05_mixed_precision.md) | 提纲 |
| [分布式训练与状态分片](./06_distributed_training.md) | 提纲 |
| [预训练语料与数据组织](./07_pretraining_data.md) | 提纲 |
| [训练循环、监控与 Checkpoint](./08_training_loop_and_checkpoints.md) | 提纲 |

<!-- notes:index:end -->
