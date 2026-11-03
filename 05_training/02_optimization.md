# 梯度优化、AdamW 与梯度裁剪

> 状态：提纲

> 前置内容：[Token 到概率模型](../01_probability/01_token_probability_model.md)、[Decoder-only Transformer 的完整前向过程](../04_transformer/07_decoder_only_transformer.md)。

本篇要回答：训练损失如何通过一次优化步骤改变模型参数？

## 1. 梯度与参数更新

- 定义参数、梯度、学习率与优化器状态，先写出 SGD 更新。
- 解释梯度清零、反向传播与参数更新的顺序，以及梯度如何跨 batch 累积。

## 2. Adam 与 AdamW

- 列出一阶矩、二阶矩、偏差修正和 epsilon 的作用。
- 区分 weight decay 与向目标加入 L2 正则，明确所用更新公式。

## 3. 梯度裁剪与参数组

- 定义 global norm clipping，并标注裁剪发生在更新前的哪个位置。
- 记录不同参数组的学习率和 weight decay 约定，不把某一实现的默认值当作普遍规则。

## 4. 最小例子与检查

- 用一个标量参数手算前两步更新，核对优化器状态。
- 说明 loss、梯度范数、参数变化量各能暴露什么问题。

## 阅读入口

- [PyTorch：AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
