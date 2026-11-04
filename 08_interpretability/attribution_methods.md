# 输入归因、梯度与局部解释

> 状态：提纲

> 前置内容：[可解释性的问题与证据](./overview.md)、[Token Embedding](../02_representation/01_token_embedding.md)；梯度方法还需要链式法则。

本篇要回答：哪些输入部分影响了一个指定输出分数，不同归因方法中的“影响”分别指什么？

## 1. 固定解释对象

- 指定预测位置与标量目标：某个 logit、两个候选的 logit difference、log-probability 或序列得分。
- 区分输入 token、词语、子词组和 embedding 维度；先定义分组，再讨论归因如何聚合。
- 比较完整上下文与参考输入，说明要解释的是相对哪种基线的变化。

## 2. 扰动方法与梯度方法

| 方法 | 待展开的问题 | 需要检查的选择 |
| --- | --- | --- |
| 遮挡、删除或替换输入 | 改变某个输入部分后，分数如何变化 | 替换内容、位置变化及是否仍在合理输入分布中 |
| Gradient / Gradient × Input | 当前位置附近的微小变化如何影响目标 | 梯度饱和、符号、取范数和聚合方式 |
| Integrated Gradients | 沿参考输入到实际输入的路径累计梯度 | 参考点、积分步数、路径和完备性误差 |
| LIME | 局部扰动样本能否被简单代理模型近似 | 扰动分布、局部权重和代理模型拟合质量 |
| SHAP | 相对所定义的特征联盟价值，如何分配贡献 | 缺失特征的处理、背景分布、相关特征和计算近似 |

完整展开时，对同一标量目标比较这些方法。通用方法用于建立概念，不要求把所有算法用于每一个 LLM 实验。

## 3. 文本输入的特殊约束

- token ID 是离散索引，梯度通常对查表后的连续 embedding 计算；不能把 token ID 的数值大小当作连续语义。
- embedding 插值点未必对应真实文本，删除词语也会改变语法、长度和位置；需解释这些干预对应的问题。
- 多 token 答案要定义完整序列得分和长度处理，不能只取首个 token 却声称解释了整段答案。
- 大的归因值只在该方法、目标和参考选择下有意义，不直接给出一个内部计算回路。

## 4. 最小练习与完成标准

先用可手算的线性函数比较梯度、Gradient × Input 和相对零基线的贡献。再为一个固定的 next-token 目标设计两种合理替换基线，记录哪些 token 的排名会改变，并解释原因。

扩展到代码时，加入模型参数随机化等 sanity check。结果要区分视觉上突出、数值上敏感和机制上必要，当前练习尚未执行。

## 来源

- [Sundararajan 等，2017：Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html)：Integrated Gradients 的定义、公理与基线。
- [Ribeiro 等，2016：LIME](https://arxiv.org/abs/1602.04938)：局部代理解释。
- [Lundberg 与 Lee，2017：SHAP](https://arxiv.org/abs/1705.07874)：加性特征归因框架。
- [Adebayo 等，2018：Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)：解释图的检验；向文本迁移时需重新定义合适的对照。
