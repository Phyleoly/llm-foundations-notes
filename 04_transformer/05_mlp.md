# 逐位置 MLP 与门控前馈层

> 状态：草稿

> 前置内容：[离散 Token 表示与 Token Embedding](../02_representation/01_token_embedding.md)。

本篇要回答：attention 之后，MLP 如何变换每个位置的特征？

## 1. 逐位置前馈计算

在串行 Pre-Norm block 中，记 $R'_\ell$ 为第 $\ell$ 层 Attention 更新后的残差状态。MLP 对每个位置使用同一组参数。取其中一行 $r=R'_\ell[t]\in\mathbb R^d$，其归一化结果 $x=\mathrm{Norm}_{\ell,m}(r)$ 是 MLP 的输入。设隐藏宽度为 $d_{\mathrm{ff}}$。下式按列展开输入矩阵，按行展开输出矩阵。

```math
\begin{aligned}
a &= \sigma\!\left(
  x\,\underbrace{\begin{bmatrix}k_1 & \cdots & k_{d_{\mathrm{ff}}}\end{bmatrix}}_{W_{\mathrm{in}}}
  + b_{\mathrm{in}}\right),\\[6pt]
\Delta r &= \underbrace{\begin{bmatrix}a_1 & \cdots & a_{d_{\mathrm{ff}}}\end{bmatrix}}_{a}
  \underbrace{\begin{bmatrix}v_1\\ \vdots\\ v_{d_{\mathrm{ff}}}\end{bmatrix}}_{W_{\mathrm{out}}}
  + b_{\mathrm{out}},\\[4pt]
R_{\ell+1}[t] &= r+\Delta r.
\end{aligned}
```

第 $j$ 列 $k_j$ 读取当前输入，经激活函数 $\sigma$ 得到标量 $a_j$。这个标量乘以第 $j$ 行 $v_j$，就是该单元写入残差流的向量。Geva 等将这对输入列和输出行分别称为 key 和 value，并通过词表投影分析 value 与哪些词相关。[Geva 等，2021](https://aclanthology.org/2021.emnlp-main.446/)、[Geva 等，2022](https://aclanthology.org/2022.emnlp-main.3/)

这里的列 $k_j$ 和行 $v_j$ 都是训练学到的参数。当前 token 的输入决定激活值 $a_j$，也就决定该位置沿 $v_j$ 写入多少。[残差流的计算图](../08_interpretability/residual_stream.md#2-计算图中的残差流)

## 2. 激活与门控

- 比较常规 FFN 与 GLU / SwiGLU 的计算路径，说明逐元素乘法所在位置。
- 列出 gate、up、down projection 的形状，区分激活名称与整个网络结构。

## 3. 参数量与宽度选择

- 推导常规 FFN 和门控 FFN 的参数量，明确 bias 是否计入。
- 在相同参数预算下比较中间宽度，避免只按矩阵个数判断成本。

## 4. 最小例子与边界

- 逐步追踪一个位置的向量经过投影、激活或门控再返回模型宽度。
- 说明非线性与残差的职责，残差和归一化的完整分析留到下一篇。
