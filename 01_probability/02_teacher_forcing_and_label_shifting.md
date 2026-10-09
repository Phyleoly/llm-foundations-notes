# Teacher Forcing 与标签移位

> 前置内容：[Token 到概率模型：参数化语言模型](./01_token_probability_model.md#6-参数化语言模型)

## 1. Teacher Forcing

前文将自回归语言模型写为

```math
p_\theta(x_{1:T})
=
\prod_{i=1}^{T}
p_\theta(x_i\mid x_{<i}).
```

这里自然会产生一个训练层面的问题：

> 当模型预测 $x_i$ 时，条件中的真实历史 $x_{<i}$ 从哪里得到？

对于训练样本，我们已经知道完整的真实序列

$$
x=(x_1,x_2,\dots,x_T).
$$

因此，在位置 $i$ 的训练过程中，可以直接使用真实的历史 token

$$
(x_1,x_2,\dots,x_{i-1})
$$

作为模型上下文。这种训练方式称为 **Teacher Forcing**。

假设真实序列为

$$
(x_1,x_2,x_3,x_4),
$$

则训练过程中对应的预测任务为

$$
\begin{aligned}
(\mathrm{BOS}) &\rightarrow x_1,\\
(\mathrm{BOS},x_1) &\rightarrow x_2,\\
(\mathrm{BOS},x_1,x_2) &\rightarrow x_3,\\
(\mathrm{BOS},x_1,x_2,x_3) &\rightarrow x_4.
\end{aligned}
$$

例如，即使模型在第 $3$ 个位置预测错误，

$$
\hat{x}_3\neq x_3,
$$

训练第 $4$ 个位置时仍然使用真实 token $x_3$ 作为上下文，而不会使用模型预测得到的 $\hat{x}_3$。

## 2. Causal Mask 与并行训练

Teacher Forcing 提供了训练时使用的真实历史序列，但 decoder-only Transformer 还需要保证位置 $i$ 不能访问未来的 token。

对于位置 $i$，模型可以访问当前及之前的位置，但不能访问

$$
x_{i+1},x_{i+2},\dots.
$$

实际实现中，这一约束通常通过 **causal mask** 实现。

由于完整训练序列已经给定，各位置的 hidden state 可以在一次前向传播中并行计算，而不需要像推理阶段那样逐 token 生成。

因此需要区分：

```text
Teacher Forcing
→ 决定训练时历史上下文使用真实 token

Causal Mask
→ 决定每个位置允许看到哪些 token
```

## 3. Label Shifting

在 decoder-only Transformer 中，可以将真实序列一次性输入模型，并由不同位置分别预测下一个 token。

| 位置 | 当前输入 token | 预测目标 |
|---|---|---|
| 1 | BOS | $x_1$ |
| 2 | $x_1$ | $x_2$ |
| 3 | $x_2$ | $x_3$ |
| 4 | $x_3$ | $x_4$ |
| 5 | $x_4$ | EOS |

等价地，可以写成

$$
\begin{aligned}
\text{Input}
&=
(\mathrm{BOS},x_1,x_2,x_3,x_4),\\
\text{Target}
&=
(x_1,x_2,x_3,x_4,\mathrm{EOS}).
\end{aligned}
$$

因此，input 和 target 在位置上相差一个 token。计算损失时，需要将每个输入位置产生的预测与对应的下一个真实 token 对齐，这种位置错位通常称为 **label shifting**。

完成 label shifting 后，每个有效位置都会得到一组 logits 以及对应的真实 target token。单个位置的训练目标仍然是

$$
-\log p_\theta(x_i\mid x_{<i}).
$$

其与 Cross-Entropy 的关系见 [Token 到概率模型：负对数似然与交叉熵](./01_token_probability_model.md#10-负对数似然与交叉熵)。实际计算中的数值稳定性问题见 [LogSoftmax 与数值稳定性](./03_logsoftmax_numerical_stability.md)。
