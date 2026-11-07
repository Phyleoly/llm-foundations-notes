# Multi-Head Attention 与 MQA / GQA

> 状态：草稿

> 前置内容：[Self-Attention](./02_self_attention.md)。

本篇要回答：多个 attention head 如何共同产生一个模型宽度的输出？

## 1. 多头投影与形状

省略 batch 维度，设输入 $X\in\mathbb R^{T\times d}$，其中 $T$ 是序列长度， $d$ 是模型宽度。在串行 Pre-Norm block 中， $X$ 是 Attention 前归一化后的残差流。各 head 从同一 $X$ 计算自己的 Q、K、V。若每个 head 的 value 维度为 $d_h$，则它汇集 value 后得到 $T\times d_h$ 的输出。

## 2. 并行计算与输出投影

设 $H$ 是 head 数。拼接各 head 后乘总输出矩阵，等价于把该矩阵按 head 分为 $H$ 个行块，各自投影后相加。对标准多头 Attention，省略偏置和 dropout，有

```math
\Delta R^{(h)}=A^{(h)}(X)XW_V^{(h)}W_O^{(h)},
\qquad
\mathrm{Attn}_\ell(X)=\sum_{h=1}^{H}\Delta R^{(h)}.
```

其中 $A^{(h)}(X)\in\mathbb R^{T\times T}$ 是第 $h$ 个 head 的注意力权重，每个 query 的权重在所有可见 key 上归一化。 $W_V^{(h)}\in\mathbb R^{d\times d_h}$ 提取 value， $W_O^{(h)}\in\mathbb R^{d_h\times d}$ 将结果投影回残差空间。单个 head 的写入落在 $W_O^{(h)}$ 的行空间中，维数至多为 $d_h$。各 head 的写入量相加得到一个 $T\times d$ 的更新。

用 QK 与 OV 的语言说，QK 决定从哪些位置读取，OV 决定将读到的信息转换成什么。某个位置获得的注意力权重，只给出了这次写入中的加权系数。整个 block 如何累积这些更新，见[残差流的计算图](../08_interpretability/residual_stream.md#2-计算图中的残差流)。[Elhage 等，2021，Attention Heads](https://transformer-circuits.pub/2021/framework/index.html)

## 3. MQA 与 GQA

- 分别定义 query 头数和 key/value 头数，画出共享关系。
- 比较 MHA、MQA、GQA 的参数与缓存形状；缓存如何复用留到 Inference。

## 4. 最小例子与边界

- 跟踪一个小 batch 的拆头、拼接与输出过程。
- 说明哪些整除关系来自所选实现约定，计算参数量时包含输出投影。

## 阅读入口

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
