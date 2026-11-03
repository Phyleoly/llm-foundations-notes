# Self-Attention

> 状态：草稿

> 前置内容：[离散 Token 表示与 Token Embedding](../02_representation/01_token_embedding.md) 与 [Attention 的基本计算](./01_attention.md)。

本篇要回答：同一序列如何生成 Q/K/V 并得到单头 self-attention 的输出，在无位置机制、无 mask 时为何具有置换等变性？下面用一组三 token 输入逐步计算。

## 1. 单头 Self-Attention

先考虑一个序列、一个 attention head，不含位置机制、attention mask 或 dropout。为突出矩阵计算，线性投影省略偏置。

### 1.1 从同一输入构造 Q/K/V

设序列长度为 $T$，每个 token 的输入表示维度为 $d$。将这些表示按行排列，得到输入矩阵 $X\in\mathbb R^{T\times d}$，第 $i$ 行为 $\mathbf x_i$。这里的输入是 embedding 或前面网络层产生的连续表示，不是 token ID。

对同一个 $X$ 使用三个可学习的投影矩阵：

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
```

其中， $d_k$ 是 query 和 key 的维度， $d_v$ 是 value 的维度。各矩阵的形状为：

| 矩阵 | 形状 | 含义 |
| --- | --- | --- |
| $X$ | $T\times d$ | 输入表示，每行对应一个 token |
| $W_Q$、 $W_K$ | $d\times d_k$ | query 与 key 的投影参数 |
| $W_V$ | $d\times d_v$ | value 的投影参数 |
| $Q$、 $K$ | $T\times d_k$ | 每个位置的 query 与 key |
| $V$ | $T\times d_v$ | 每个位置的 value |

**在行向量约定下，投影矩阵乘在输入的右侧。** 例如， $Q$ 的形状由 $(T\times d)(d\times d_k)=T\times d_k$ 得到。每个位置都使用同一组参数，但 $W_Q$、 $W_K$、 $W_V$ 是分别学习的，因此“来自同一输入”并不表示 $Q=K=V$。

这也是 self-attention 与 cross-attention 的区别：前者的 Q/K/V 都来自同一序列；后者的 Query 来自一组表示，Key/Value 来自另一组表示，query 数量与 key/value 数量可以不同。

### 1.2 从分数到输出

先计算缩放点积分数，再沿每一行做 Softmax，最后汇总 Value：

```math
\begin{aligned}
S&=\frac{QK^\top}{\sqrt{d_k}}\in\mathbb R^{T\times T},\\
A&=\mathrm{Softmax}(S)\in\mathbb R^{T\times T},\\
O&=AV\in\mathbb R^{T\times d_v}.
\end{aligned}
```

分数矩阵的第 $i$ 行对应 query 位置，第 $j$ 列对应 key/value 位置。具体来说，

```math
S_{ij}=\frac{\mathbf q_i\mathbf k_j^\top}{\sqrt{d_k}},
\qquad
A_{ij}=\frac{\exp(S_{ij})}{\sum_{\ell=1}^{T}\exp(S_{i\ell})},
\qquad
\mathbf o_i=\sum_{j=1}^{T}A_{ij}\mathbf v_j.
```

因此，每一行权重之和为 $1$，输出的每一行仍对应原来的 query 位置，但它已经汇总了其他位置的 Value。这里的 $O$ 是单个 head 的加权汇总结果；多头拼接与输出投影见 [Multi-Head Attention](./03_multi_head_attention.md)。

## 2. 一组三 Token 的完整计算

用 $(a,b,c)$ 标记三个 token，取 $T=3$、 $d=d_k=2$、 $d_v=1$。下面的输入与投影参数是为手算选取的数值，不赋予向量分量具体语义。

### 2.1 输入与投影

输入矩阵的三行依次对应 $a$、 $b$、 $c$：

```math
X=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},
\qquad
W_Q=\begin{bmatrix}1&0\\0&1\end{bmatrix},
\quad
W_K=\begin{bmatrix}0&1\\1&0\end{bmatrix},
\quad
W_V=\begin{bmatrix}1\\2\end{bmatrix}.
```

右乘相应投影矩阵，得到：

```math
Q=XW_Q=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},
\qquad
K=XW_K=
\begin{bmatrix}
0&1\\
1&0\\
1&1
\end{bmatrix},
\qquad
V=XW_V=
\begin{bmatrix}1\\2\\3\end{bmatrix}.
```

例如， $a$ 的输入 $(1,0)$ 分别得到 $\mathbf q_1=(1,0)$、 $\mathbf k_1=(0,1)$ 和 $\mathbf v_1=(1)$。投影过程逐行独立进行，跨 token 的信息汇总发生在后面的 attention 计算中。

### 2.2 分数与权重

先将每个 query 与全部 key 做点积，再除以 $\sqrt{2}$：

```math
S=\frac{QK^\top}{\sqrt{2}}
=\frac{1}{\sqrt{2}}
\begin{bmatrix}
0&1&1\\
1&0&1\\
1&1&2
\end{bmatrix}.
```

例如，第一行的 query 为 $(1,0)$，与三个 key 的点积依次为 $0,1,1$。记 $s=1/\sqrt{2}$，这一行的 Softmax 为：

```math
A_{1,:}
=\mathrm{Softmax}(0,s,s)
=\frac{(1,e^s,e^s)}{1+2e^s}
\approx(0.1978,0.4011,0.4011).
```

对其余两行做同样的归一化，得到完整权重矩阵：

```math
A\approx
\begin{bmatrix}
0.1978&0.4011&0.4011\\
0.4011&0.1978&0.4011\\
0.2483&0.2483&0.5035
\end{bmatrix}.
```

这里没有 mask，三个 query 都能访问 $a,b,c$。每行在这三个候选位置之间分配权重。本文小数保留四位，输出使用未舍入的权重计算，因此展示值的行和可能略偏离 $1$。

### 2.3 汇总 Value

第一行输出为：

```math
\mathbf o_1
=A_{11}\mathbf v_1+A_{12}\mathbf v_2+A_{13}\mathbf v_3
=\frac{1+2e^s+3e^s}{1+2e^s}
\approx 2.2033.
```

三个位置的输出合在一起为：

```math
O=AV\approx
\begin{bmatrix}
2.2033\\
2.0000\\
2.2552
\end{bmatrix}
\in\mathbb R^{3\times1}.
```

输出维度由 $d_v=1$ 决定，不必等于输入维度 $d=2$。例如， $a$ 的输入只在第一行，但它的输出已经使用了 $b$、 $c$ 的 Key 和 Value；这些输入变化时，第一行的权重或汇总结果也可能变化。

## 3. 无 Mask 时的置换等变性

仍沿用前面的条件：没有位置机制、mask 或 dropout，且各位置共享投影参数。用置换矩阵 $P\in\mathbb R^{T\times T}$ 表示输入的重排，且 $P^\top P=I$。由于 $(PX)W_Q=P(XW_Q)=PQ$，输入重排后， $Q$、 $K$、 $V$ 也会按相同的顺序变为 $PQ$、 $PK$、 $PV$。

重排输入相当于同时重排分数矩阵 $S$ 的行与列，使其变为 $PSP^\top$。逐行 Softmax 会保留这种重排关系：

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

例如，将上面的 $(a,b,c)$ 重排为 $(b,a,c)$，并保持投影参数不变，得到：

```math
P=\begin{bmatrix}0&1&0\\1&0&0\\0&0&1\end{bmatrix},
\qquad
O'=PO\approx
\begin{bmatrix}2.0000\\2.2033\\2.2552\end{bmatrix}.
```

输出跟着换位，而不是整个输出矩阵保持原样；这也是“等变”与“不变”的区别。

这说明，在上述条件下，self-attention 可以根据 token 的内容建立关联，但并未利用“某个 token 位于哪里”或“两个 token 相隔多远”这些信息。位置表示的作用，就是将这些位置关系引入模型，使它们能够参与表示与 attention 的计算。

加入 causal mask 后的可见范围与置换关系见 [Causal Attention Mask](./04_causal_attention_mask.md)；显式位置机制见 [位置表示](../02_representation/02_position_representation.md)。

参考：缩放点积与投影形式见 [Attention Is All You Need，§3.2.1–§3.2.2](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1)，self-attention 与 cross-attention 见 [§3.2.3](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS3)。本文的小矩阵例子与置换等变性推导按上述定义展开。
