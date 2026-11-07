# 残差连接、LayerNorm 与 RMSNorm

> 状态：提纲

> 前置内容：[Multi-Head Attention 与 MQA / GQA](./03_multi_head_attention.md)、[逐位置 MLP 与门控前馈层](./05_mlp.md)。

本篇要回答：Transformer block 为什么同时需要残差连接与归一化？

## 1. 残差连接

残差连接将子层输入与同形状的更新量相加。ResNet 的设计动机是让深层网络学习相对于输入的更新，缓解优化困难。Transformer 中各子层连续更新同一状态的计算过程见[残差流](../08_interpretability/residual_stream.md)。[He 等，2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

- 写出残差相加与对应的形状约束，画出主支路和残差支路。
- 从梯度传播与表示更新解释其作用，避免把它描述为保证训练成功的充分条件。

## 2. LayerNorm 与 RMSNorm

- 分别定义统计量、归一化轴、可学习参数与 epsilon。
- 比较是否减均值、缩放行为与参数形式；区分 token 维和特征维。

## 3. Pre-Norm 与 Post-Norm

- 画出 attention / MLP 子层前后放置归一化的两种结构。
- 讨论最终归一化、残差路径与训练稳定性的关系，注明所分析的架构。

## 4. 最小例子与边界

- 对一个短向量手算两种归一化，并分别改变平移量与整体尺度。
- 检查低精度、近零方差和 epsilon 的影响；实现细节以实际框架为准。
