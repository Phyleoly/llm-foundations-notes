# Causal Attention Mask

> 状态：草稿

> 前置内容：[Self-Attention](./02_self_attention.md)。

本篇要回答：causal mask 如何限制可见范围，固定或同步重排 mask 会怎样改变置换关系？部分推导仍待展开。

## 1. 可见范围与加性 Mask

自回归语言模型使用 causal mask，使每个位置在计算 attention 时只能访问自身及之前的位置。

对于长度为 $T$ 的序列，以 $i$ 表示 query 位置、 $j$ 表示 key 位置，定义一个 $T\times T$ 的加性 mask $M$：

```math
M_{i,j}
=
\begin{cases}
0, & j\leq i,\\
-\infty, & j\gt i.
\end{cases}
```

待补充：

- 在不使用显式位置机制的条件下，写出加入 $M$ 后的 attention 公式。
- 用 $(a,b,c)$ 与重排后的 $(b,a,c)$ 画出相同的下三角可见性矩阵，再列出同一个 token 在换位前后能访问哪些 token。
- 说明 mask 绑定的是序列中的位置关系，输入换位后，可见的 token 内容会随之改变。

## 2. 固定 Mask 时的置换

- 沿用 [Self-Attention](./02_self_attention.md) 的分数矩阵 $S=QK^\top/\sqrt{d_k}$ 与置换矩阵 $P$，分别展开“先重排输入再计算”和“先计算再重排输出”。
- 对照两种分数矩阵： $PSP^\top+M$ 与 $P(S+M)P^\top$。
- 找出关键差异：固定 mask 时，一般有 $M\neq PMP^\top$，因此不能继续沿用无 mask 时的置换推导。这里的 $PMP^\top$ 表示对 $M$ 同步重排行与列。
- 回到前面的例子解释结论：输出一般不再只是输入的对应换位；这不要求每个 token 的输出都发生变化。

## 3. 如果 Mask 也随输入一起置换

- 将 $M$ 同步重排为 $PMP^\top$，补出此时输入、mask 与输出之间的等变关系。
- 解释这种操作保留了原有的 token 可见关系，相当于对整套关系重新编号。
- 区分两种操作：在固定的因果顺序下重排 token，以及连同可见关系一起重排。后者得到的 mask 在新的排列下通常不再是标准下三角形式。

## 4. 与 Padding / Loss Mask 的边界（待补充）

- 区分限制历史可见范围、屏蔽 padding key，以及不计入某些 target 的损失。
- 用一个变长 batch 标出三者作用的位置；批处理、组合 mask 和有效 token 归约在 [Batching 与 Masking](../05_training/01_batching_padding_and_masking.md) 中展开。

位置机制的分类见 [位置表示](../02_representation/02_position_representation.md)；顺序约束与隐式位置学习的区别，留在 [NoPos / NoPE 论文阅读与复现](../02_representation/03_nope_reading_list.md) 中结合论文展开。

参考：[Attention Is All You Need，§3.1 与 §3.2.3](https://arxiv.org/html/1706.03762v7)。
