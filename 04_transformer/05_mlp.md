# 逐位置 MLP 与门控前馈层

> 状态：提纲

> 前置内容：[离散 Token 表示与 Token Embedding](../02_representation/01_token_embedding.md)。

本篇要回答：attention 之后，MLP 如何变换每个位置的特征？

## 1. 逐位置前馈计算

- 写出两层线性映射与激活函数，标注 model dimension 和 intermediate dimension。
- 解释参数如何跨位置共享，以及 MLP 自身是否混合不同 token 的信息。

## 2. 激活与门控

- 比较常规 FFN 与 GLU / SwiGLU 的计算路径，说明逐元素乘法所在位置。
- 列出 gate、up、down projection 的形状，区分激活名称与整个网络结构。

## 3. 参数量与宽度选择

- 推导常规 FFN 和门控 FFN 的参数量，明确 bias 是否计入。
- 在相同参数预算下比较中间宽度，避免只按矩阵个数判断成本。

## 4. 最小例子与边界

- 逐步追踪一个位置的向量经过投影、激活或门控再返回模型宽度。
- 说明非线性与残差的职责，残差和归一化的完整分析留到下一篇。
