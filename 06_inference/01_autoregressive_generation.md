# 自回归生成的完整流程

> 状态：提纲

> 前置内容：[Decoder-only Transformer 的完整前向过程](../04_transformer/07_decoder_only_transformer.md)、[Teacher Forcing 与标签移位](../01_probability/02_teacher_forcing_and_label_shifting.md)。

本篇要回答：模型怎样把一次前向计算变成连续生成的文本？

## 1. Prompt 与下一 Token

- 跟踪 prompt 编码、前向计算和用于选择下一 token 的 logits。
- 说明 batch 中序列长度不同时，最后一个有效位置与 padding 的关系。

## 2. 生成循环

- 写出选择 token、追加上下文、更新位置与继续前向的伪代码。
- 区分训练中的真实前缀与推理中的模型前缀，连接误差累积问题。

## 3. 停止与输出

- 分别记录 EOS、最大新 token 数、上下文长度和字符串停止条件。
- 说明解码文本、特殊 token 与多序列结束状态如何对应。

## 4. 最小例子与边界

- 手工跟踪三次生成，记录每一步输入、预测位置和选中 token。
- 先建立不使用 cache 的参考流程，再链接 KV cache 与采样策略。
