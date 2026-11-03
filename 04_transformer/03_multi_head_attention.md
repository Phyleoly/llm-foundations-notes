# Multi-Head Attention 与 MQA / GQA

> 状态：提纲

> 前置内容：[Self-Attention](./02_self_attention.md)。

本篇要回答：多个 attention head 如何共同产生一个模型宽度的输出？

## 1. 多头投影与形状

- 定义 model dimension、head dimension 与 head 数量，标注拆分和转置前后的张量形状。
- 区分独立投影矩阵与一次线性投影后 reshape 的等价表示。

## 2. 并行计算与输出投影

- 写出每个 head 的 attention、拼接和输出投影，检查残差支路的宽度。
- 解释多头结构提供的建模方式，不把每个 head 固定解释为某种语言学功能。

## 3. MQA 与 GQA

- 分别定义 query 头数和 key/value 头数，画出共享关系。
- 比较 MHA、MQA、GQA 的参数与缓存形状；缓存如何复用留到 Inference。

## 4. 最小例子与边界

- 跟踪一个小 batch 的拆头、拼接与输出过程。
- 说明哪些整除关系来自所选实现约定，计算参数量时包含输出投影。

## 阅读入口

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
