# 残差流、Logit Lens 与输出贡献

> 状态：提纲

> 前置内容：[残差连接与归一化](../04_transformer/06_residual_and_layernorm.md)、[输出投影](../02_representation/04_output_projection_and_weight_tying.md)、[Multi-Head Attention](../04_transformer/03_multi_head_attention.md)。

本篇要回答：如何从 residual stream 观察逐层信息变化，输出投影又能告诉我们哪些内容？

## 1. 把残差流放回计算图

以串行 Pre-Norm decoder block 为例，省略 batch 维度，设序列长度为 $T$，表示维度为 $d$，层输入为 $R_\ell\in\mathbb R^{T\times d}$：

```math
\begin{aligned}
R'_\ell &= R_\ell + \mathrm{Attn}_\ell(\mathrm{Norm}_{\ell,a}(R_\ell)),\\
R_{\ell+1} &= R'_\ell + \mathrm{MLP}_\ell(\mathrm{Norm}_{\ell,m}(R'_\ell)).
\end{aligned}
```

这里 attention 项包含其输出投影，两项输出都与残差流同形状。该式用于定位观察点；Post-Norm、并行 attention / MLP 等架构需要按实际前向改写。归一化的完整定义由第四章维护。

- 标出子层输入、归一化后输入、各 head 输出、残差相加前后的位置。
- 区分“读了某个方向”和“向残差流写入了某个方向”；残差更新并不意味着每个方向都有独立语义。
- 比较不同实现的缓存时，核对是否折叠归一化、中心化权重或改变输出投影约定。

## 2. 用词表方向读取中间表示

设 $V$ 为词表大小， $W_U\in\mathbb R^{d\times V}$ 为输出投影， $b_U\in\mathbb R^V$ 为可选偏置。将其应用到中间残差流可构造一种 Logit Lens 读取方式：

```math
Z^{\mathrm{lens}}_\ell=\mathrm{Norm}_f(R_\ell)W_U+b_U.
```

结果形状为 $T\times V$，偏置沿位置广播。这里使用实际模型的最终归一化 $\mathrm{Norm}_f$；没有该操作的模型需相应调整。中间层的读取分数不等于模型在该层已经形成最终答案。

- 比较原始 Logit Lens 和需要训练逐层映射的 Tuned Lens，说明各自增加的假设。
- 检查不同层的坐标与尺度差异；训练 lens 时单独记录数据、目标和留出评估。

## 3. Direct Logit Attribution 的边界

- 先在无最终归一化的线性输出设置中，推导残差各加性项对目标 logit 的分解。
- 有 LayerNorm 或 RMSNorm 时，说明是否固定某次前向的归一化统计量，以及仿射项如何处理；不能直接把归一化当作跨干预不变的线性函数。
- 区分某次前向中沿词表方向的贡献，与移除该组件后重新运行模型得到的总效应；后者会改变后续计算。

## 4. 最小练习与完成标准

画出一层 Pre-Norm block，在每个观察点写出张量形状。选定一个预测位置和两个候选 token，说明如何读取逐层分数，以及哪些分数只是观察量、哪些需要干预后重新测量。

## 来源

- [Elhage 等，2021：A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)：residual stream、QK / OV 分解与简化模型；阅读时保留其架构假设。
- [Belrose 等：Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112)：逐层读取器及原始 Logit Lens 的局限，初稿发布于 2023 年。
