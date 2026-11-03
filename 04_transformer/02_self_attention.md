# Self-Attention

> 状态：草稿

> 前置内容：[离散 Token 表示与 Token Embedding](../02_representation/01_token_embedding.md) 与 [Attention 的基本计算](./01_attention.md)。

本篇要回答：单头 self-attention 如何计算，在无位置机制、无 mask 时为何具有置换等变性？

## 1. 单头 Self-Attention

以下考虑不含位置机制、且不带 attention mask 的单头 self-attention。设序列长度为 $T$，query、key 和 value 按行组成 $Q,K\in\mathbb R^{T\times d_k}$ 与 $V\in\mathbb R^{T\times d_v}$，其中 $d_k$、 $d_v$ 分别为 query / key 和 value 向量的维度。Attention 的计算为

```math
\mathrm{Attention}(Q,K,V)
=
\mathrm{Softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
```

其中， $QK^\top\in\mathbb R^{T\times T}$ 记录 token 两两之间的内积，Softmax 沿每一行计算。

### 待补充：从同一输入构造 Q/K/V

- 定义输入矩阵与三个投影矩阵，写出 Q、K、V 的构造过程和形状。
- 说明 self-attention 中 query 与 key/value 来自同一序列；用 cross-attention 作简短对照。
- 用一组小矩阵跟踪输入投影、attention 和输出，核对每个位置的表示如何依赖其他位置。

## 2. 无 Mask 时的置换等变性

用置换矩阵 $P\in\mathbb R^{T\times T}$ 表示输入的重排，且 $P^\top P=I$。由于各位置共享生成 query、key 和 value 的线性映射，输入重排后， $Q$、 $K$、 $V$ 也会按相同的顺序变为 $PQ$、 $PK$、 $PV$。

记原来的分数矩阵为 $S=QK^\top/\sqrt{d_k}$。重排输入相当于同时重排 $S$ 的行与列，使其变为 $PSP^\top$。逐行 Softmax 会保留这种重排关系：

```math
\mathrm{Softmax}(PSP^\top)
=
P\,\mathrm{Softmax}(S)P^\top.
```

这是因为，调整行的顺序，只会使各行的 Softmax 结果随之换位；调整列的顺序，不会改变每一行的归一化分母，因此该行的概率也只会随之换位。

因此，重排后的 attention 输出满足

```math
\begin{aligned}
\mathrm{Attention}(PQ,PK,PV)
&=
\mathrm{Softmax}\left(
\frac{(PQ)(PK)^\top}{\sqrt{d_k}}
\right)PV
\\
&=
\mathrm{Softmax}(PSP^\top)PV
\\
&=
P\,\mathrm{Softmax}(S)P^\top PV
\\
&=
P\,\mathrm{Attention}(Q,K,V).
\end{aligned}
```

也就是说，**输入中的 token 如何换位，它们的输出表示就随之进行相同的换位**。对于其中的某个 token，将它移到另一个位置后，其对应的输出向量仍与原来相同。这一性质称为**置换等变性（permutation equivariance）**。

这说明，在上述条件下，self-attention 可以根据 token 的内容建立关联，但并未利用“某个 token 位于哪里”或“两个 token 相隔多远”这些信息。位置表示的作用，就是将这些位置关系引入模型，使它们能够参与表示与 attention 的计算。

参考：[Attention Is All You Need，§3.2.1](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1)。

加入 causal mask 后的可见范围与置换关系见 [Causal Attention Mask](./04_causal_attention_mask.md)；显式位置机制见 [位置表示](../02_representation/02_position_representation.md)。
