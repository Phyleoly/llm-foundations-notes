# 位置表示

> 前置内容：[离散 Token 表示与 Token Embedding](./01_token_embedding.md)。

## 1. 位置信息的必要性

**位置表示是让模型在处理 token 内容时，同时利用它在序列中的位置，以及它与其他 token 的先后和间隔。**

token embedding 本身只由 token ID 决定：同一个 token 无论出现在序列中的哪个位置，经过 embedding lookup 后都会得到相同的向量。它提供了 token 的内容表示，却没有说明这个 token 当前出现在什么位置。

但顺序会影响语言的含义。为方便说明，将“我 / 喜欢 / 你”看作三个 token：“我喜欢你”与“你喜欢我”包含相同的 token，却表达了不同的关系。

这些 token 的向量虽然会按顺序组成输入矩阵，**矩阵中的行号却不会自动成为向量的一部分**。例如，对每一行使用相同的线性变换时，计算只读取该行向量，并不会额外读取它的行号。没有位置机制、没有 mask 的标准 self-attention 也只会让输出随输入一起换位；同一个 token 的对应输出不会仅因换了位置而改变。相关推导见 [Self-Attention](../04_transformer/02_self_attention.md)。

因此，要让模型利用顺序，就需要让位置或位置关系真正参与表示与计算。这样，模型在关联两个 token 时，才能结合“谁在前、谁在后、相隔多远”等线索，学习与顺序有关的模式。位置表示提供这些线索，具体如何使用则由模型在训练中学习。

> 位置信息也可以来自模型结构。causal mask 限定每个位置只能访问自身及之前的位置，已经引入了顺序约束，也可能使模型学到隐式位置信息。因此，“没有显式位置编码”不等于“完全没有位置线索”。这部分见 [Causal Attention Mask](../04_transformer/04_causal_attention_mask.md) 与 [NoPos / NoPE](./03_nope_reading_list.md)。

## 2. 位置表示的基本概念

理解位置表示，可以先区分两类信息：单个 token 的**绝对位置**，以及两个 token 之间的**相对位置**。

### 2.1 绝对位置

绝对位置表示 “这个 token 位于当前序列的第几个位置” 。

对于长度为 $T$ 的序列，本文从 $0$ 开始编号，位置索引为 $0,1,\dots,T-1$。

例如，在“我 / 喜欢 / 你”中，三个 token 的位置分别是 $0$、 $1$、 $2$。“喜欢”的 token ID 标识词表中的条目，位置索引 $1$ 则标识它这次出现的位置。同一个 token 可以在序列中出现多次，具有相同的 token ID 和不同的位置索引。

一种常见做法是根据位置索引 $i$ 构造位置向量：

```math
\mathbf p_i=f(i).
```

其中， $f$ 可以是可学习的位置查找表，也可以是固定的数学函数。这样，相同的 token 出现在不同位置时，就可以配上不同的位置表示。后面的 learned absolute positional embedding 与 sinusoidal positional encoding 都从这一思路展开。

### 2.2 相对位置

相对位置以另一个位置为参照。以位置 $i$ 为参照、考察位置 $j$ 时，可定义为

```math
r_{i,j}=j-i.
```

其绝对值 $|r_{i,j}|$ 表示两个位置相隔多少步，正负号表示方向：负数表示 $j$ 在 $i$ 之前，正数表示 $j$ 在 $i$ 之后，零表示同一位置。

仍以“我 / 喜欢 / 你”为例，固定“喜欢”所在的位置 $i=1$：

| 考察的 token | 绝对位置 $j$ | 相对位移 $j-i$ | 相对于“喜欢”的位置 |
| --- | ---: | ---: | --- |
| 我 | 0 | -1 | 前一个 token |
| 喜欢 | 1 | 0 | 自身 |
| 你 | 2 | +1 | 后一个 token |

如果把两个位置的编号同时增加相同的非零偏移量 $c$，它们的绝对位置会改变，相对位移则保持不变：

```math
(j+c)-(i+c)=j-i.
```

因此，“前一个 token”这样的关系可以出现在序列中的不同地方，而不必绑定某个固定位置。这里的间隔以 token 为单位，不一定等于字符数或词数。不同资料也可能采用 $i-j$ 的符号约定，阅读公式时需要先确认参照方向。

### 2.3 位置关系如何进入计算

位置索引与相对位移描述了位置关系，位置机制则把这些关系转化为模型可以使用的数值。常见的作用方式包括：

| 作用位置 | 基本做法 | 后续方法 |
| --- | --- | --- |
| 输入表示 | 将位置向量与 token 表示相加 | Learned absolute、sinusoidal（第 3 节） |
| Attention 分数 | 根据相对位置给匹配分数加入偏置 | Relative position bias、ALiBi（第 4 节） |
| Query 与 Key | 根据各自的位置对向量施加旋转 | RoPE（第 5 节） |

这些方式都能让位置参与计算，因此位置表示并不总是表现为“给 token embedding 加一个位置向量”。例如，[原始 Transformer](https://arxiv.org/html/1706.03762v7#S3.SS5) 在输入端相加，而 [ALiBi](https://arxiv.org/abs/2108.12409) 直接调整 attention 分数。

绝对位置与相对位置也不是对所有方法都互斥的分类。[RoPE](https://arxiv.org/abs/2104.09864) 根据绝对位置构造旋转，但旋转后的 query 与 key 在内积中体现相对位移。分析一种方法时，需要同时看它用什么位置量构造表示，以及最终计算中体现了什么位置关系。

## 3. 绝对位置表示

### 3.1 Learned Absolute Positional Embedding

**可学习的绝对位置嵌入为每个可用位置设置一个可训练向量，使用时按位置索引查表。**

沿用每个 token 以行向量表示的约定，并假设 token embedding 的维度与模型维度相同，即 $d_e=d_{\mathrm{model}}=d$。长度为 $T$ 的序列经过 embedding lookup 后，得到 token 表示矩阵：

```math
X_{\mathrm{emb}}\in\mathbb R^{T\times d}.
```

若位置 $i$ 的 token ID 为 $k_i$，则该位置的 token 向量为 $E_{k_i,:}$。

类似地，用一张包含 $L$ 行的可学习位置表存放位置向量：

```math
P\in\mathbb R^{L\times d}.
```

表的每一行对应一个绝对位置，可用的位置索引为 $0,1,\dots,L-1$。按索引 $i$ 取出对应行，就得到位置向量：

```math
\mathbf p_i=f(i)=P_{i,:},
\qquad
0\leq i\lt L.
```

将 token 向量与对应的位置向量逐元素相加，得到 Transformer 层的初始输入：

```math
\mathbf h_i^{(0)}
=
E_{k_i,:}+P_{i,:}.
```

同一个 token 出现在不同位置时，使用相同的 token 向量，并加上各自的位置向量。不同样本中，位置索引相同的 token 共享同一个位置向量。

当位置从 $0$ 开始连续编号，且序列长度满足 $T\leq L$ 时，可以将整个序列的初始表示写为

```math
H^{(0)}
=
X_{\mathrm{emb}}+P_{0:T,:}
\in\mathbb R^{T\times d}.
```

其中， $P_{0:T,:}$ 表示位置表的前 $T$ 行，对应位置 $0$ 到 $T-1$。逐元素相加不会改变序列长度或向量维度。若 $d_e\neq d_{\mathrm{model}}$，可先将 token embedding 投影到模型维度，再与相同维度的位置向量相加，具体见 [Token Embedding](./01_token_embedding.md)。

位置表 $P$ 通常随机初始化，并在语言模型训练中与其他参数一起更新。这里的“learned”指位置向量通过训练学得；位置索引 $i$ 只用于查表，本身不是可训练参数。整张位置表共有 $Ld$ 个参数。

位置表的行数 $L$ 决定了可直接查找的位置范围。当 $i\geq L$ 时，查表会越界，因此需要限制输入范围、扩展位置表，或调整位置索引。**索引裁剪是其中一种可选策略，并非该方法的必需步骤。**

例如，可以将非负位置索引裁剪为

```math
\tilde i=\min(i,L-1),
\qquad
\mathbf p_i=P_{\tilde i,:},
```

这样，所有 $i\geq L-1$ 的位置都会读取最后一行。例如， $L=4$ 时，位置 $3$、 $4$、 $5$ 都读取 $P_{3,:}$。裁剪可以避免索引越界，但也会让这些位置共享同一个位置向量，因而无法仅凭该向量区分它们。避免查表越界，并不意味着模型能有效处理更长序列。

实现参考：[GPT-2 的 token / position embedding 查表与相加](https://github.com/openai/gpt-2/blob/master/src/model.py)。

### 3.2 Sinusoidal Positional Encoding

**正弦位置编码使用固定的正弦和余弦函数表示绝对位置，无需学习位置参数。**

[Attention Is All You Need](https://arxiv.org/html/1706.03762v7#S3.SS5) 中的定义为

```math
\begin{aligned}
PE_{(pos,2i)}
&=\sin\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right),\\
PE_{(pos,2i+1)}
&=\cos\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right).
\end{aligned}
```

其中， $pos$ 表示 token 的位置， $d_{\mathrm{model}}$ 表示编码维度， $i$ 是维度对的索引。

各个位置共享同一组编码函数，仅输入的位置索引不同。训练过程中，模型学习如何使用这些位置特征，编码函数本身保持固定。

#### 3.2.1 正余弦配对与位置关系

同一频率的正弦与余弦组成一对分量。将其写成行向量，有

```math
\mathbf u_i(pos)=[\sin(pos\omega_i),\cos(pos\omega_i)],
\qquad
\|\mathbf u_i(pos)\|_2^2=1.
```

这对分量可以区分一圈内的不同相位。例如，相位 $0$ 与 $\pi$ 的正弦值相同，配对后却分别为 $[0,1]$ 与 $[0,-1]$。

位置增加 $k$ 时，三角函数加法公式给出

```math
\mathbf u_i(pos+k)
=\mathbf u_i(pos)
\begin{bmatrix}
\cos(k\omega_i)&-\sin(k\omega_i)\\
\sin(k\omega_i)&\cos(k\omega_i)
\end{bmatrix}.
```

右侧矩阵只依赖偏移 $k$ 与频率 $\omega_i$，与起点无关。这为学习相对位置关系提供了线索，但编码仍由绝对位置生成。

将所有维度对拼成完整位置向量 $\mathbf p_{pos}$，其模长固定，两个位置的内积只依赖位置差：

```math
\begin{aligned}
\|\mathbf p_{pos}\|_2^2&=\frac d2,\\
\mathbf p_m\mathbf p_n^{\top}
&=\sum_{i=0}^{d/2-1}\cos\bigl((m-n)\omega_i\bigr).
\end{aligned}
```

这些性质只针对原始位置向量。与 token 表示相加并经过可学习投影后，attention 分数还取决于内容和模型参数。

#### 3.2.2 长度泛化

固定公式可以继续生成训练范围之外的位置编码。例如，训练仅覆盖位置 $0,\dots,127$ 时，仍能计算 $128,\dots,255$ 的编码；使用预计算表时，需相应扩展表。

但**编码可计算，不等于模型能有效利用更长的上下文**。

## 4. 相对位置表示

### 4.1 相对位置向量

- 从位置对 $(i,j)$ 转向相对位移 $j-i$。
- 以 Shaw 等人的方法为例，说明相对位置向量如何进入 attention 计算。
- 说明距离截断的动机，以及截断后保留和丢失的信息。

参考：[Self-Attention with Relative Position Representations](https://aclanthology.org/N18-2074/)。

### 4.2 Relative Position Bias 与 ALiBi

- 先写出向 attention logits 加入标量位置偏置的一般形式。
- 以 ALiBi 为例，解释相对距离、不同 attention head 的斜率与线性惩罚。
- 区分位置偏置与 attention mask 的作用。

参考：[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)。

## 5. Rotary Position Embedding（RoPE）

### 5.1 从二维旋转开始

- 定义二维旋转矩阵，说明旋转角与位置索引的关系。
- 从一对分量推广到多对分量，解释不同维度对使用不同频率的方式。

### 5.2 对 Query 和 Key 施加旋转

- 写出旋转前后的 query / key，并说明所用维度属于单个 attention head。
- 标出 RoPE 在 attention 中的作用位置，与输入端相加的位置向量对照。

### 5.3 内积中的相对位置关系

- 推导旋转矩阵的组合关系，再推导旋转后 query / key 的内积。
- 解释“由绝对位置构造旋转，内积中的显式位置关系由相对位移决定”。
- 说明这一性质针对位置变换；完整 attention score 仍取决于 query / key 的内容。

参考：[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)。

## 6. 方法比较与使用边界

### 6.1 方法比较

- 汇总 learned absolute、sinusoidal、相对位置向量、ALiBi 与 RoPE。
- 对照维度：注入位置、额外可学习参数、相对位置的体现方式、长度限制。

### 6.2 长度泛化

- 区分训练长度、推理长度与长度外推。
- 比较位置表范围、编码可计算性与实际模型表现。
- 将 RoPE scaling 等上下文扩展方法列为后续阅读。

### 6.3 Position IDs、Padding 与 Mask

- 区分 token ID、position ID 与 padding 后的张量下标。
- 说明位置编号与 causal mask / padding mask 各自负责什么。
- padding 的训练处理与增量生成的位置衔接分别链接到 [Training](../05_training/README.md) 和 [Inference](../06_inference/README.md)。

## 小结

- 回答：token 表示缺少什么信息，位置机制如何补充这些信息？
- 回答：为什么不能把所有模型的输入都写成 token embedding 加 position embedding？
- 用一句话概括绝对位置、相对位置与 RoPE 的关系，并衔接 [Transformer](../04_transformer/README.md)。
