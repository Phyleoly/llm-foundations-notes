# Batching、Padding、Packing 与 Loss Mask

> 状态：提纲

> 前置内容：[Teacher Forcing 与标签移位](../01_probability/02_teacher_forcing_and_label_shifting.md)、[Causal Attention Mask](../04_transformer/04_causal_attention_mask.md)。

本篇要回答：如何批量训练，同时保证可见范围与损失统计都符合样本语义？

## 1. 样本与 Batch

- 区分原始文档、训练样本、截断片段与 batch，写清输入和 target 的形状。
- 讨论左右 padding、截断长度与位置编号的关系。

## 2. 三种 Mask 的职责

- 区分 causal mask、padding attention mask 与 loss mask，分别列出它们作用于哪个计算。
- 用带 PAD 的样本说明屏蔽 key 和忽略 target 为什么需要分别处理。

## 3. Sequence Packing

- 比较保留文档边界的 packing 与允许跨文档上下文的拼接目标。
- 明确文档边界、EOS、position IDs 和标签移位，避免把不同训练约定混为一谈。

## 4. Loss 的聚合与例子

- 对两个有效长度不同的样本，比较按 token 平均与按序列平均。
- 给出包含 padding 和边界 target 的最小 batch，标出每个损失项是否计入。
