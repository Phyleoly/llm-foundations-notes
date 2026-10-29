# Attention 的基本计算

> 前置内容：[离散 Token 表示与 Token Embedding](../02_representation/01_token_embedding.md)、[LogSoftmax 与数值稳定性](../01_probability/03_logsoftmax_numerical_stability.md)。

## 1. Attention 机制

### 1.1 从普通平均到按需加权平均

**Attention 的意义，是让每个位置根据当前需要，从可访问的上下文中汇总信息。** 在常见的 Softmax attention 中，这个汇总过程在数学上就是对 Value 向量做加权平均。

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

每个 $\mathbf k_j$ 都与一个 $\mathbf v_j$ 配对。Query 与各个 Key 先算出一组匹配分数，再经过 Softmax 得到权重。**第 $j$ 个分数归一化后得到的 $\alpha_j$，就是乘在第 $j$ 个 Value 上的系数。** Q、K 决定怎样平均，V 是被平均的向量。

例如，取两个示意用的 Value：$\mathbf v_1=(10,0)$、$\mathbf v_2=(0,5)$。下面人为设定两组 attention 权重，与普通平均对照：

| 汇总方式 | $\mathbf v_1$ 的系数 | $\mathbf v_2$ 的系数 | 输出向量 |
| --- | ---: | ---: | --- |
| 普通平均 | 0.5 | 0.5 | $(5,2.5)$ |
| 针对查询 A 的加权平均 | 0.8 | 0.2 | $(8,1)$ |
| 针对查询 B 的加权平均 | 0.2 | 0.8 | $(2,4)$ |

三种结果都来自相同的两个 Value，但保留了不同的内容比例。实际模型中的 attention 权重由 Query 和 Key 计算得到，而不是人为指定。

这使得每个查询都能从同一组候选信息中得到针对自身的汇总。输出仍然是一个特征向量；在 Transformer 中，这类上下文信息会参与当前位置表示的更新，让后续计算能够利用其他位置提供的内容。

## 2. Query、Key 与 Value

在模型中，**$Q$、$K$、$V$ 都是数值特征向量，通常由输入到当前 attention 层的表示经过不同的可学习投影得到。**

按行向量约定，设 $\mathbf x \in \mathbb{R}^N$ 是产生当前查询的输入表示， $\mathbf{y_j}$ 是第 $j$ 条候选信息的候选表示，

则可以如下构造：

```math
\mathbf q=\mathbf xW_Q,
\qquad
\mathbf k_j=\mathbf y_jW_K,
\qquad
\mathbf v_j=\mathbf y_jW_V.
```

其中，$W_Q$、$W_K$、$W_V$ 是训练时学习的参数矩阵；Q、K、V 则是每次前向计算根据当前输入得到的结果。输入表示既可以来自 embedding，也可以来自前面网络层的输出。

同一条候选信息分别生成 Key 和 Value，使模型可以用一组特征判断“是否匹配当前查询”，再用另一组特征提供“应该传递的内容”。两组特征由训练共同学习，其各个分量通常没有预先指定的语义。

在 self-attention 中，查询和候选信息来自同一序列；在 cross-attention 中，两者可以来自不同序列。具体来源与构造见 [Self-Attention](./02_self_attention.md)。后面的点积匹配要求 Query 与 Key 的维度相同，而 Value 可以使用不同的维度；输出向量的维度与 Value 一致。

## 3. Scaled Dot-Product Attention

- 推导点积、缩放、逐行 Softmax 与对 value 加权求和的完整表达。
- 在明确分量分布假设后解释缩放因子的动机，区分向量维度与序列长度。

## 4. 矩阵形状与最小例子

- 用两条 query 和三组 key/value 手算 score、权重与输出。
- 检查 Softmax 的归一化轴，以及输出长度和输出宽度由什么决定。

## 5. 边界与后续

- 讨论极端 logits、数值稳定性，以及 attention 权重的解释边界。
- 同源 Q/K/V、置换等变性、multi-head 与 mask 分别进入后续笔记。

## 阅读入口

- [Attention Is All You Need，§3.2](https://arxiv.org/html/1706.03762v7)
