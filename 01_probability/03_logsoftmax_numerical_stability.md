# LogSoftmax 与数值稳定性

> 前置内容：[Token 到概率模型：负对数似然与交叉熵](./01_token_probability_model.md#10-负对数似然与交叉熵)

前文已经得到单个 token 的负对数似然

$$
\ell_i=-\log p_{i,c},
$$

其中

$$
p_i=\operatorname{softmax}(z_i).
$$

从数学表达式看，可以先计算 Softmax，再对真实类别对应的概率取对数。但在有限精度浮点计算中，这种直接实现可能产生数值溢出或下溢。

## 1. Softmax 的数值溢出问题

假设 logits 为

$$
z=(z_1,z_2,\cdots,z_n),
$$

则 Softmax 在第 $i$ 个位置的输出为

$$
\operatorname{softmax}(z)_i
=
\frac{\exp z_i}
{\sum_{j=1}^{n}\exp z_j}.
$$

如果某个 logit 很大，例如 $z_i=1000$，直接计算时将出现 $e^{1000}$。该数值可能超出浮点数的可表示范围，从而导致数值溢出，使 Softmax 的计算结果失效。

## 2. LogSoftmax

为避免先显式计算 Softmax 再取对数，可以将二者合并为 LogSoftmax：

$$
\begin{aligned}
\operatorname{logsoftmax}(z)_i
&=
\log\frac{\exp z_i}{\sum_{j=1}^{n}\exp z_j}\\
&=
z_i-\log\sum_{j=1}^{n}\exp z_j.
\end{aligned}
$$

其中第二项

$$
\operatorname{LSE}(z)
=
\log\sum_{j=1}^{n}\exp z_j
$$

称为 **LogSumExp**。

## 3. LogSumExp 的稳定计算

LogSumExp 中仍然包含指数运算，因此直接计算同样存在溢出风险。

令

$$
m=\max_j z_j.
$$

则

$$
\begin{aligned}
\operatorname{LSE}(z)
&=
\log\sum_{j=1}^{n}\exp z_j\\
&=
\log\sum_{j=1}^{n}\exp(z_j-m+m)\\
&=
\log\left(\exp m\sum_{j=1}^{n}\exp(z_j-m)\right)\\
&=
m+\log\sum_{j=1}^{n}\exp(z_j-m).
\end{aligned}
$$

由于

$$
z_j-m\leq 0,
$$

因此

$$
0<\exp(z_j-m)\leq 1.
$$

同时，最大 logit 对应的指数项恰好为

$$
\exp 0=1.
$$

因此，将所有 logits 减去最大值后，可以避免产生过大的指数值；同时，由于至少有一项等于 $1$，所有指数项也不会同时下溢。

## 4. Stable Softmax

Stable Softmax 使用相同的平移技巧。Softmax 对所有 logits 同时减去同一个常数保持不变：

$$
\operatorname{softmax}(z)
=
\operatorname{softmax}(z-c),
\qquad
c\in\mathbb R.
$$

取

$$
c=\max_j z_j,
$$

则有

$$
\begin{aligned}
\operatorname{softmax}(z)_i
&=
\frac{\exp z_i}{\sum_{j=1}^{n}\exp z_j}\\
&=
\frac{\exp(z_i-\max_j z_j)}
{\sum_{k=1}^{n}\exp(z_k-\max_j z_j)}.
\end{aligned}
$$

这就是 Stable Softmax 的基本形式。

实际框架中的交叉熵损失通常直接接收 logits，而不是先显式计算 Softmax 概率再取对数。例如，PyTorch 的 `CrossEntropyLoss` 会将相关计算合并，从而利用 LogSoftmax / LogSumExp 的稳定形式完成损失计算。

训练阶段的 logits 与 target 如何通过 label shifting 对齐，见 [Teacher Forcing 与标签移位](./02_teacher_forcing_and_label_shifting.md)。
