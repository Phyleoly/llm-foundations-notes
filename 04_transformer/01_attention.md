# Attention 的基本计算

> 前置内容：[离散 Token 表示与 Token Embedding](../02_representation/01_token_embedding.md)、[LogSoftmax 与数值稳定性](../01_probability/03_logsoftmax_numerical_stability.md)。

## 1. Attention 机制

### 1.1 从普通平均到按需加权平均

**Attention 的意义，是让每个位置根据当前需要，从可访问的上下文中汇总信息。** 在基于 Softmax 的注意力计算中，这个汇总过程在数学上就是对 Value 向量做加权平均。

以刚经过 embedding lookup 的 token 为例，它的向量由自身的 token ID 决定，还没有结合当前上下文。要利用其他位置的信息，就需要把那些位置的内容汇总过来。

假设有 $n$ 个候选内容向量 $\mathbf v_1,\dots,\mathbf v_n$。一种直接的汇总方式是取普通平均：

```math
\bar{\mathbf v}
=
\frac{1}{n}\sum_{j=1}^{n}\mathbf v_j.
```

这种方式让每个候选项使用相同的系数。对于同一组候选信息，无论当前查询是什么，都会得到同一份汇总。但不同位置可能需要不同的信息，因此汇总的比例也应该能够随查询变化。

Attention 把等权平均改成了**由当前查询决定的加权平均**。记第 $j$ 个 Value 的权重为 $\alpha_j$。Softmax 使这些权重非负且总和为 $1$，因此

```math
\mathbf o
=
\frac{\sum_{j=1}^{n}\alpha_j\mathbf v_j}
     {\sum_{j=1}^{n}\alpha_j}
=
\sum_{j=1}^{n}\alpha_j\mathbf v_j.
```

所以，常见公式中的“加权求和”已经是一个加权平均，不需要再除以候选数量 $n$。它的特点在于：**平均时使用的系数随查询变化，而不是始终固定为 $1/n$。**

### 1.2 Query 与 Key 决定如何平均 Value

这些系数由 Query 与 Key 的匹配关系产生：

- **Query** 表达当前查询需要匹配什么信息。
- **Key** 提供每条候选信息用于匹配的特征。
- **Value** 提供这条候选信息实际参与汇总的内容。

每个 $\mathbf k_j$ 都与一个 $\mathbf v_j$ 配对。Query 与各个 Key 先算出一组匹配分数，再经过 Softmax 得到权重。**第 $j$ 个分数归一化后得到的 $\alpha_j$，就是乘在第 $j$ 个 Value 上的系数。** Query 和 Key 决定权重，Value 提供被汇总的内容。

例如，取两个示意用的 Value： $\mathbf v_1=(10,0)$、 $\mathbf v_2=(0,5)$。下面人为设定两组 attention 权重，与普通平均对照：

| 汇总方式 | $\mathbf v_1$ 的系数 | $\mathbf v_2$ 的系数 | 输出向量 |
| --- | ---: | ---: | --- |
| 普通平均 | 0.5 | 0.5 | $(5,2.5)$ |
| 针对查询 A 的加权平均 | 0.8 | 0.2 | $(8,1)$ |
| 针对查询 B 的加权平均 | 0.2 | 0.8 | $(2,4)$ |

三种结果都来自相同的两个 Value，但保留了不同的内容比例。实际模型中的 attention 权重由 Query 和 Key 计算得到，而不是人为指定。

在 Transformer 中，汇总得到的向量会参与当前位置表示的更新，让后续计算能够利用其他位置提供的内容。

## 2. Query、Key 与 Value

在模型中，**Query、Key、Value 通常由当前 attention 层的输入表示经过不同的可学习投影得到。**

按行向量约定，设 $\mathbf x$ 是产生当前查询的输入表示， $\mathbf y_j$ 是第 $j$ 条候选信息的输入表示，则

```math
\mathbf q=\mathbf xW_Q,
\qquad
\mathbf k_j=\mathbf y_jW_K,
\qquad
\mathbf v_j=\mathbf y_jW_V.
```

其中， $W_Q$、 $W_K$、 $W_V$ 是训练时学习的参数矩阵； $\mathbf q$、 $\mathbf k_j$、 $\mathbf v_j$ 则是每次前向计算根据当前输入得到的向量。输入表示既可以来自 embedding，也可以来自前面网络层的输出。将多条 Query、Key、Value 分别按行排列，就得到矩阵 $Q$、 $K$、 $V$。

同一条候选信息分别生成 Key 和 Value，使模型可以用一组特征判断“是否匹配当前查询”，再用另一组特征提供“应该传递的内容”。两组特征由训练共同学习，其各个分量通常没有预先指定的语义。

在 self-attention 中，查询和候选信息来自同一序列；在 cross-attention 中，Query 来自一组表示，Key 和 Value 来自另一组表示。后面的点积匹配要求 Query 与 Key 的维度相同，而 Value 可以使用不同的维度；输出向量的维度与 Value 一致。

## 3. Scaled Dot-Product Attention

Scaled Dot-Product Attention 用 Query 与 Key 的点积计算匹配分数，再除以 $\sqrt{d_k}$，经过 Softmax 得到 Value 的权重。其中， $d_k$ 是 Query 和 Key 的特征维度。下面讨论单头注意力的基本计算，以及缩放与分数尺度、Softmax 梯度的关系。

### 3.1 放大 logits 对 Softmax 的影响

Softmax 的输入分数称为 **logits**。对于包含 $n$ 个元素的分数向量 $\mathbf z=[z_1,z_2,\dots,z_n]$，第 $i$ 个位置的输出为：

```math
\mathrm{softmax}(\mathbf z)_i
=\frac{\exp(z_i)}{\sum_{j=1}^{n}\exp(z_j)}.
```

将所有 logits 同时乘以 $t\gt 0$，任意两个位置的概率之比为：

```math
\frac{\mathrm{softmax}(t\mathbf z)_i}
{\mathrm{softmax}(t\mathbf z)_j}
=\exp\bigl(t(z_i-z_j)\bigr).
```

因此，影响相对权重的是 logits 之间的差值。对于固定的 $z_i\gt z_j$，增大 $t$ 会使第 $i$ 个位置相对第 $j$ 个位置的权重更大。

假设分数向量存在唯一最大值 $z_m$：

```math
z_m\gt z_j,
\qquad \forall j\neq m.
```

最大值所在位置的概率可以写成：

```math
\begin{aligned}
\mathrm{softmax}(t\mathbf z)_m
&=\frac{\exp(tz_m)}{\sum_{j=1}^{n}\exp(tz_j)}\\
&=\frac{1}{\sum_{j=1}^{n}\exp\bigl(t(z_j-z_m)\bigr)}\\
&=\frac{1}{1+\sum_{j\neq m}\exp\bigl(t(z_j-z_m)\bigr)}.
\end{aligned}
```

由于 $j\neq m$ 时有 $z_j-z_m\lt 0$，所以当 $t\to\infty$ 时，上式求和中的每一项都趋于 $0$，从而：

```math
\lim_{t\to\infty}\mathrm{softmax}(t\mathbf z)_m=1.
```

对于其余位置，同样将分子、分母除以 $\exp(tz_m)$，得到：

```math
\mathrm{softmax}(t\mathbf z)_j
=\frac{\exp\bigl(t(z_j-z_m)\bigr)}
{1+\sum_{k\neq m}\exp\bigl(t(z_k-z_m)\bigr)}
\longrightarrow 0,
\qquad \forall j\neq m.
```

因此，**在存在唯一最大值时，按同一个正系数不断放大 logits，会使 Softmax 输出趋近于 one-hot：最大项的概率趋于 $1$，其余项趋于 $0$。** 如果有 $r$ 个并列最大值，分母中会保留 $r$ 个值为 $1$ 的项，因此每个最大项的概率趋于 $1/r$，其余项趋于 $0$。

### 3.2 点积的尺度与缩放

设 Query 和 Key 的特征维度为 $d_k$：

```math
\begin{aligned}
\mathbf q&=[q_1,q_2,\dots,q_{d_k}],\\
\mathbf k&=[k_1,k_2,\dots,k_{d_k}].
\end{aligned}
```

为分析点积的尺度，沿用[Attention Is All You Need §3.2.1 脚注 1](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1)中的简化假设： $q_1,\dots,q_{d_k},k_1,\dots,k_{d_k}$ 这 $2d_k$ 个随机变量相互独立，并且：

```math
\begin{aligned}
\mathbb E[q_i]=\mathbb E[k_i]&=0,\\
\mathrm{Var}(q_i)=\mathrm{Var}(k_i)&=1.
\end{aligned}
```

先看单个乘积项。由 $q_i$ 与 $k_i$ 的独立性，有

```math
\mathbb E[q_i k_i]
=\mathbb E[q_i]\mathbb E[k_i]
=0.
```

又因为 $\mathbb E[q_i^2]=\mathrm{Var}(q_i)+(\mathbb E[q_i])^2=1$，同理 $\mathbb E[k_i^2]=1$，所以

```math
\begin{aligned}
\mathrm{Var}(q_i k_i)
&=\mathbb E[q_i^2 k_i^2]-(\mathbb E[q_i k_i])^2\\
&=\mathbb E[q_i^2]\mathbb E[k_i^2]-0\\
&=1.
\end{aligned}
```

当 $i\neq j$ 时，两个乘积项 $q_i k_i$ 与 $q_j k_j$ 也相互独立，因此协方差为

```math
\begin{aligned}
\mathrm{Cov}(q_i k_i,q_j k_j)
&=\mathbb E[q_i k_i q_j k_j]
-\mathbb E[q_i k_i]\mathbb E[q_j k_j]\\
&=0.
\end{aligned}
```

将各维度的乘积相加，就得到点积的均值与方差：

```math
\begin{aligned}
\mathbb E[\mathbf q\cdot\mathbf k]
&=\sum_{i=1}^{d_k}\mathbb E[q_i k_i]
=0,\\
\mathrm{Var}(\mathbf q\cdot\mathbf k)
&=\sum_{i=1}^{d_k}\mathrm{Var}(q_i k_i)
+2\sum_{1\leq i\lt j\leq d_k}\mathrm{Cov}(q_i k_i,q_j k_j)\\
&=d_k.
\end{aligned}
```

因此，**点积的均值仍为 $0$，但标准差为 $\sqrt{d_k}$。** 维度增大时，点积的典型波动幅度也会增大。

为了控制点积的尺度，可以对它进行缩放。根据方差的缩放性质，除以 $\sqrt{d_k}$ 和除以 $d_k$ 分别得到：

```math
\begin{aligned}
\mathrm{Var}\left(\frac{\mathbf q\cdot\mathbf k}{\sqrt{d_k}}\right)
&=\frac{1}{d_k}\mathrm{Var}(\mathbf q\cdot\mathbf k)
=1,\\
\mathrm{Var}\left(\frac{\mathbf q\cdot\mathbf k}{d_k}\right)
&=\frac{1}{d_k^2}\mathrm{Var}(\mathbf q\cdot\mathbf k)
=\frac{1}{d_k}.
\end{aligned}
```

因此，**除以 $\sqrt{d_k}$ 可以使点积的方差保持为 $1$，不再随维度变化。** 而除以 $d_k$ 后，方差会随维度增大而趋于 $0$，点积的波动幅度也会不断缩小。

3.1 表明，Softmax 取决于分数之间的差值。把同样的假设扩展到一个 Query $\mathbf q$ 和两个 Key $\mathbf k_a,\mathbf k_b$：三者的所有分量相互独立，均值为 $0$、方差为 $1$。此时， $k_{a,i}-k_{b,i}$ 的均值为 $0$、方差为 $2$，且与 $q_i$ 独立，因此

```math
\begin{aligned}
\mathrm{Var}(\mathbf q\cdot\mathbf k_a-\mathbf q\cdot\mathbf k_b)
&=\sum_{i=1}^{d_k}\mathrm{Var}\bigl(q_i(k_{a,i}-k_{b,i})\bigr)\\
&=2d_k,\\
\mathrm{Var}\left(
\frac{\mathbf q\cdot\mathbf k_a-\mathbf q\cdot\mathbf k_b}{\sqrt{d_k}}
\right)
&=2.
\end{aligned}
```

因此，除以 $\sqrt{d_k}$ 也使同一 Query 下分数差的方差不再随维度增长。实际训练得到的 Query 和 Key 不保证满足上述简化假设，缩放也不保证 Softmax 始终远离饱和。

### 3.3 Softmax 的梯度与饱和

Softmax 输出趋近于 one-hot 时，对 logits 的小幅变化会越来越不敏感。这可以从它的偏导数看出来。记

```math
p_i=\mathrm{softmax}(\mathbf z)_i
=\frac{\exp(z_i)}{\sum_{k=1}^{n}\exp(z_k)}.
```

先考察 $p_i$ 对自身对应的 logit $z_i$ 的偏导。此时分子和分母都随 $z_i$ 变化，由商法则得到：

```math
\begin{aligned}
\frac{\partial p_i}{\partial z_i}
&=\frac{\exp(z_i)\sum_{k=1}^{n}\exp(z_k)-\exp(2z_i)}
{\left(\sum_{k=1}^{n}\exp(z_k)\right)^2}\\
&=\frac{\exp(z_i)}{\sum_{k=1}^{n}\exp(z_k)}
\left(1-\frac{\exp(z_i)}{\sum_{k=1}^{n}\exp(z_k)}\right)\\
&=p_i(1-p_i).
\end{aligned}
```

当 $j\neq i$ 时，分子与 $z_j$ 无关，只有分母随 $z_j$ 变化，因此：

```math
\begin{aligned}
\frac{\partial p_i}{\partial z_j}
&=-\frac{\exp(z_i)\exp(z_j)}
{\left(\sum_{k=1}^{n}\exp(z_k)\right)^2}\\
&=-p_i p_j,
\qquad j\neq i.
\end{aligned}
```

将这些偏导排列成雅可比矩阵，其第 $i$ 行、第 $j$ 列表示第 $j$ 个 logit 对第 $i$ 个概率的影响。把概率写成列向量 $\mathbf p=[p_1,\dots,p_n]^{\mathsf T}$，则

```math
J=\left[\frac{\partial p_i}{\partial z_j}\right]_{i,j=1}^{n}
=\mathrm{diag}(\mathbf p)-\mathbf p\mathbf p^{\mathsf T}.
```

其中， $\mathrm{diag}(\mathbf p)$ 的对角线元素为 $p_1,\dots,p_n$，其余元素为 $0$。

当某个 $p_m\to 1$、其余概率趋于 $0$ 时，所有对角项和非对角项都趋于 $0$：

```math
\begin{aligned}
p_i(1-p_i)&\to 0,\\
-p_i p_j&\to 0,\qquad i\neq j.
\end{aligned}
```

即 $J\to\mathbf 0$。此时小幅改变 logits，概率也几乎不变，这就是这里所说的 **Softmax 饱和**。若有多个并列最大值，最大项之间仍有非零的偏导，整个 $J$ 不会趋于零。

以两个 logits 为例，下面给出概率及其对第一个 logit 的偏导，数值均为近似值：

| logits $(z_1,z_2)$ | $p_1$ | $p_2$ | $\partial p_1/\partial z_1=p_1(1-p_1)$ |
| --- | ---: | ---: | ---: |
| $(0,0)$ | 0.50000 | 0.50000 | 0.25000 |
| $(5,0)$ | 0.99331 | 0.00669 | 0.00665 |
| $(10,0)$ | 0.99995 | 0.00005 | 0.00005 |

设最终损失为 $\mathcal L$。沿 $\mathbf z\to\mathbf p\to\mathcal L$ 这条计算路径，链式法则为：

```math
\nabla_{\mathbf z}\mathcal L
=J^{\mathsf T}\nabla_{\mathbf p}\mathcal L.
```

其中， $\nabla_{\mathbf p}\mathcal L$ 是后续计算传回给概率向量的上游梯度。**若上游梯度保持有界**，就可以由 $J\to\mathbf 0$ 推出沿这条路径传回 logits 的梯度也趋于 $0$。在 attention 中，这意味着经由 Softmax 传回匹配分数的学习信号会变弱。

最终损失的形式会影响这一结论的适用性。例如，若直接对 Softmax 概率使用交叉熵 $\mathcal L=-\log p_c$，其中 $c$ 是目标类别，则

```math
\frac{\partial\mathcal L}{\partial z_i}
=-\frac{1}{p_c}\frac{\partial p_c}{\partial z_i}
=\begin{cases}
p_c-1, & i=c,\\
p_i, & i\neq c.
\end{cases}
```

当模型几乎把全部概率分给错误类别时， $p_c\to 0$，上游梯度 $-1/p_c$ 无界，而目标类别的 logit 梯度趋于 $-1$。因此，**Softmax 的局部导数趋于零，并不保证最终损失对 logits 的梯度也趋于零。** 交叉熵的定义见[负对数似然与交叉熵](../01_probability/01_token_probability_model.md#10-负对数似然与交叉熵)。

### 3.4 完整的 Attention 计算

给定一个 Query $\mathbf q$ 和 $n$ 组 Key/Value $(\mathbf k_j,\mathbf v_j)$，其中 Query 和 Key 的维度为 $d_k$，Value 的维度为 $d_v$。

首先，计算 Query 与每个 Key 的点积，并进行缩放：

```math
z_j=\frac{\mathbf q\cdot\mathbf k_j}{\sqrt{d_k}},
\qquad j=1,\dots,n.
```

接着，对这 $n$ 个分数做 Softmax：

```math
\alpha_j=\frac{\exp(z_j)}{\sum_{\ell=1}^{n}\exp(z_\ell)}.
```

每个 $\alpha_j$ 对应一组 Key/Value，所有权重之和为 $1$。最后，按这些权重汇总 Value：

```math
\mathbf o=\sum_{j=1}^{n}\alpha_j\mathbf v_j,
\qquad
\mathbf o\in\mathbb R^{d_v}.
```

输出 $\mathbf o$ 是一个与 Value 维度相同的向量。

当有 $m$ 个 Query 时，可以将它们按行堆叠成矩阵 $Q$，并将 $n$ 组 Key/Value 按相同的候选顺序分别堆叠成 $K$ 和 $V$：

```math
Q\in\mathbb R^{m\times d_k},
\qquad
K\in\mathbb R^{n\times d_k},
\qquad
V\in\mathbb R^{n\times d_v}.
```

所有 Query 与 Key 的缩放点积分数可以一次计算为：

```math
Z=\frac{QK^{\mathsf T}}{\sqrt{d_k}},
\qquad
Z\in\mathbb R^{m\times n}.
```

其中， $Z_{ij}$ 表示第 $i$ 个 Query 与第 $j$ 个 Key 的缩放点积分数。每一行对应一个 Query，每一列对应一组 Key/Value。

**对 $Z$ 的每一行分别做 Softmax**，得到权重矩阵 $A$：

```math
A=\mathrm{softmax}(Z),
\qquad
A\in\mathbb R^{m\times n}.
```

具体来说，第 $i$ 个 Query 在全部 $n$ 个候选项之间分配权重：

```math
A_{ij}=\frac{\exp(Z_{ij})}{\sum_{\ell=1}^{n}\exp(Z_{i\ell})},
\qquad
\sum_{j=1}^{n}A_{ij}=1.
```

最后，将权重矩阵与 Value 矩阵相乘，得到完整的矩阵表达：

```math
O=AV
=\mathrm{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V,
\qquad
O\in\mathbb R^{m\times d_v}.
```

输出矩阵 $O$ 的第 $i$ 行，就是第 $i$ 个 Query 对全部 Value 加权汇总得到的 $d_v$ 维向量。

实际计算 Softmax 时，可以先减去每一行的最大分数，再计算指数和归一化。这个平移不改变注意力权重，用于避免指数溢出；除以 $\sqrt{d_k}$ 的作用是调整分数差的尺度。数值稳定计算的推导见 [Stable Softmax](../01_probability/03_logsoftmax_numerical_stability.md#4-stable-softmax)。
