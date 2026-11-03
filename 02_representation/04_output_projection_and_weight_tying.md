# 输出投影与 Weight Tying

> 状态：提纲

> 前置内容：[离散 Token 表示与 Token Embedding](./01_token_embedding.md)、[Token 到概率模型](../01_probability/01_token_probability_model.md)。

本篇要回答：模型如何把每个位置的 hidden state 转换成整个词表上的打分？

## 1. 输出投影的形状

- 定义批次、序列、模型宽度和词表大小，写出 hidden states 到 logits 的映射。
- 区分输出投影、Softmax 与损失；说明哪些张量带 batch / sequence 维。

## 2. 输入与输出权重共享

- 在 embedding matrix 按行存 token 的约定下，写出 weight tying 需要的转置与形状。
- 说明输入与输出两条路径如何向同一参数贡献梯度。

## 3. 维度与参数量

- 比较共享和不共享参数时的计数，避免重复计算同一矩阵。
- 讨论 embedding dimension 与 model dimension 不同时需要哪些额外映射。

## 4. 最小例子与边界

- 用小词表和一个 hidden state 手算 logits、概率与目标 token 的损失。
- 区分输入 embedding、上下文 hidden state 与输出词表打分的含义。
