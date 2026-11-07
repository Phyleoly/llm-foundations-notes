# Logit Lens：读取中间层的词表预测

> 状态：草稿

> 前置内容：[残差流](./residual_stream.md)、[输出投影与 Weight Tying](../02_representation/04_output_projection_and_weight_tying.md)。

本篇要回答：如何把中间层的表示读成词表分数，这些分数又能说明什么？

## 1. 把输出端接到中间层

Logit Lens 复用模型已有的输出端，将中间残差流转换为词表上的分数，用来观察逐层读取结果的变化。它不需要额外训练。

以具有最终归一化的 Pre-Norm 模型为例，省略 batch 维度。设模型有 $n$ 个 block，序列长度为 $T$，表示维度为 $d$，词表大小为 $V$。沿用残差流笔记的编号，第 $\ell$ 个 block 的输出为 $R_{\ell+1}\in\mathbb R^{T\times d}$。模型最终通过输出投影 $W_U\in\mathbb R^{d\times V}$ 和可选偏置 $b_U\in\mathbb R^V$ 得到 logits：

```math
Z=\mathrm{Norm}_f(R_{n+1})W_U+b_U.
```

这里 $\mathrm{Norm}_f$ 是模型的最终归一化，偏置沿 token 位置广播。Logit Lens 将相同的归一化和输出投影应用到第 $\ell$ 个 block 的输出：

```math
Z_\ell^{\mathrm{lens}}
=\mathrm{Norm}_f(R_{\ell+1})W_U+b_U,
\qquad 1\leq\ell\leq n.
```

得到的分数形状为 $T\times V$。对每个位置沿词表维做 Softmax，就得到该位置的读取分布：

```math
P_\ell^{\mathrm{lens}}
=\mathrm{Softmax}_{\mathrm{vocab}}\!\left(Z_\ell^{\mathrm{lens}}\right).
```

在 next-token prediction 中，第 $t$ 个位置对应对下一个 token 的预测。比较同一位置在不同层的高分 token 或候选 token 的分数，就能观察读取结果如何变化。这里相当于跳过后续 block 的更新，直接连接输出端；当 $\ell=n$ 时，读取结果与模型的最终输出一致。没有最终归一化的模型需相应调整公式。

## 2. 如何理解读取结果

Logit Lens 回答的是：如果用最终输出端读取当前表示，哪些 token 会得到较高分数？这个结果不等于模型在该层已经确定了最终答案，也不保证每经过一层都更接近最终预测。

不同层的表示可能与最终输出端不完全匹配。因此，某个 token 在中间层得分较低，不能单独证明该层没有相关信息；得分较高，也不能单独证明这部分信息决定了最终输出。它提供的是一种观察方式，若要判断某个组件是否对预测起作用，还需要结合干预实验。

Tuned Lens 针对中间表示与输出端不匹配的问题，为不同层训练仿射映射后再读取。它增加了可学习参数，属于对 Logit Lens 的扩展。

## 来源

- [nostalgebraist，2020：interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)：Logit Lens 的最初介绍。
- [Belrose 等，2023：Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/html/2303.08112v4#S2)：第 2 节式（1）—（3）给出残差递推与 Logit Lens 读取公式，并讨论其局限；第 3 节介绍 Tuned Lens。
