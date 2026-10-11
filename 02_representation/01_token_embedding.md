# 离散 Token 表示与 Token Embedding

> 前置内容：tokenizer 如何将文本转换为 token 序列见 [Tokenization](../03_tokenization/README.md)。

语言模型首先将文本离散化为 token，并进一步将每个 token 转换为整数 token ID。然而，token ID 本身只是离散类别的索引，不能直接表达 token 之间的关系。

为了让神经网络处理这些离散符号，需要进一步将 token ID 映射到连续向量空间。
## 1. Token ID：离散类别的索引

设模型词表为

```math
\mathcal V
=
\{v_1,v_2,\dots,v_K\},
\qquad
K=|\mathcal V|.
```

Tokenizer 为词表中的每个 token 分配一个唯一的整数索引。可以定义索引映射

```math
\mathrm{id}:
\mathcal V
\rightarrow
\{1,2,\dots,K\}.
```

若

```math
x=v_k,
```

则

```math
\mathrm{id}(x)=k.
```

因此，token ID 的作用只是标识某个 token 对应词表中的哪一个离散类别。

需要注意，token ID 的整数值本身通常没有语义上的数值意义。例如，若

```math
k_1\lt k_2,
```

并不能推出对应 token 在语义、句法或统计意义上存在某种大小关系。

类似地，

```math
|k_1-k_2|
```

也不能用于衡量两个 token 的相似程度。

因此，token ID 应当理解为 **categorical index**，而不是普通的数值特征。

实际 tokenizer 通常输出一个整数序列，例如

```math
(k_1,k_2,\dots,k_T).
```

模型随后使用这些整数作为索引，从 embedding matrix 中读取对应的连续向量。

> 数学推导中可以将 token ID 编号为 $1,\dots,K$；实际程序通常使用 $0,\dots,K-1$。这只是索引约定不同，不影响后续原理。

## 2. One-hot Representation

一种直接表示离散类别的方法是 one-hot representation。

对于 token $v_k$，定义其 one-hot 列向量

```math
\mathbf e_k
=
\begin{bmatrix}
e_1^{(k)}\\
e_2^{(k)}\\
\vdots\\
e_K^{(k)}
\end{bmatrix}
\in
\{0,1\}^K,
```

其中

```math
e_i^{(k)}
=
\begin{cases}
1, & i=k,\\
0, & i\neq k.
\end{cases}
```

因此，$\mathbf e_k$ 只有第 $k$ 个分量为 $1$，其余分量均为 $0$。

对于任意两个 token $v_m$ 和 $v_n$，有

```math
\mathbf e_m^\top\mathbf e_n
=
\begin{cases}
1, & m=n,\\
0, & m\neq n.
\end{cases}
```

因此，不同 token 对应的 one-hot 向量彼此正交。

当

```math
m\neq n
```

时，

```math
\begin{aligned}
\|\mathbf e_m-\mathbf e_n\|_2^2
&=
\|\mathbf e_m\|_2^2
+
\|\mathbf e_n\|_2^2
-
2\mathbf e_m^\top\mathbf e_n
\\
&=
1+1-0
\\
&=
2.
\end{aligned}
```

从而

```math
\boxed{
\|\mathbf e_m-\mathbf e_n\|_2
=
\sqrt 2
}
\qquad
(m\neq n).
```

也就是说，在 one-hot 空间中，任意两个不同 token 之间的欧氏距离完全相同。

对于不同 token，其 cosine similarity 同样满足

```math
\frac{
\mathbf e_m^\top\mathbf e_n
}{
\|\mathbf e_m\|_2
\|\mathbf e_n\|_2
}
=
0,
\qquad
m\neq n.
```

因此，one-hot representation 能够区分不同离散类别，却没有预先编码类别之间的相似关系。

即使两个 token 在语言中高度相关，它们与两个完全无关 token 一样，在 one-hot 空间中仍然具有相同的距离和内积关系。

此外，one-hot 向量的维度等于词表大小

```math
K,
```

但每个向量只有一个非零元素，因此是一种高维稀疏表示。

实际神经网络通常不会显式构造 one-hot 向量，而是直接使用 token ID 完成 embedding lookup。

## 3. Embedding Matrix

为了将离散 token 映射到连续向量空间，引入可学习的 embedding matrix。

设 embedding dimension 为

```math
d_e,
```

定义

```math
E
\in
\mathbb R^{K\times d_e}.
```

将其按行写为

```math
E
=
\begin{bmatrix}
E_{1,:}\\
E_{2,:}\\
\vdots\\
E_{K,:}
\end{bmatrix}.
```

其中第 $k$ 行

```math
E_{k,:}
\in
\mathbb R^{d_e}
```

对应词表中第 $k$ 个 token 的 embedding。

因此，对于

```math
x=v_k,
```

定义

```math
\boxed{
\mathrm{Emb}(x)
=
E_{k,:}
}.
```

于是，embedding matrix 可以理解为一个可学习的查找表：

```math
\boxed{
\text{token ID }k
\longrightarrow
E_{k,:}\in\mathbb R^{d_e}
}.
```

Embedding matrix 是模型参数的一部分，会在训练过程中通过梯度下降不断更新。

需要注意，embedding 在初始化时通常并不天然具有“语义”。其中出现的词法、句法、共现模式以及其他统计结构，是在模型训练过程中逐渐形成的。

## 4. Embedding Lookup

给定 token ID

```math
k,
```

embedding lookup 所做的操作就是从 embedding matrix 中读取第 $k$ 行：

```math
k
\xrightarrow{\text{Embedding Lookup}}
E_{k,:}.
```

因此，

```math
\boxed{
\mathrm{Emb}(v_k)
=
E_{k,:}
}.
```

从线性代数的角度看，这一操作与 one-hot representation 和 embedding matrix 的矩阵乘法完全等价。

由于 $\mathbf e_k$ 是 $K$ 维列向量，

```math
E^\top\mathbf e_k
=
E_{k,:}^\top.
```

等价地，将 one-hot 写成行向量后，

```math
\boxed{
\mathbf e_k^\top E
=
E_{k,:}
}.
```

因为

```math
\begin{aligned}
\mathbf e_k^\top E
&=
\sum_{i=1}^{K}
e_i^{(k)}E_{i,:}
\\
&=
E_{k,:}.
\end{aligned}
```

One-hot 中只有第 $k$ 个位置的系数为 $1$，因此矩阵乘法实际上只保留了 $E$ 的第 $k$ 行。

所以从数学上看，

```math
\boxed{
\text{Embedding Lookup}
\equiv
\text{One-hot}\times E
}.
```

二者得到完全相同的结果。

但在实际实现中，没有必要显式构造维度为 $K$ 的 one-hot 向量再执行矩阵乘法。

深度学习框架可以直接根据整数 token ID 读取 embedding matrix 中对应的行，从而避免构造高维稀疏向量和执行大量结果必然为零的乘法。

因此，embedding layer 可以概括为

```math
\boxed{
\text{token ID}
\longrightarrow
\text{row lookup in }E
\longrightarrow
\text{embedding vector}
}.
```

## 5. Token 序列的 Embedding

对于长度为 $T$ 的 token ID 序列

```math
(k_1,k_2,\dots,k_T),
```

逐个执行 embedding lookup 后得到

```math
X_{\mathrm{emb}}
=
\begin{bmatrix}
E_{k_1,:}\\
E_{k_2,:}\\
\vdots\\
E_{k_T,:}
\end{bmatrix}
\in
\mathbb R^{T\times d_e}.
```

因此，离散 token ID 序列

```math
(k_1,k_2,\dots,k_T)
```

被转换为连续向量序列

```math
X_{\mathrm{emb}}
\in
\mathbb R^{T\times d_e}.
```

这里得到的 token embedding 本身是 **context-independent representation**。

也就是说，对于同一个 token ID $k$，无论它出现在什么上下文中，embedding lookup 都会读取同一个向量

```math
E_{k,:}.
```

上下文相关的信息需要通过后续 Transformer 层进一步计算，并逐渐形成在 contextual hidden states 中。

因此，应当区分

```math
\boxed{
\text{token embedding}
}
```

与

```math
\boxed{
\text{contextual hidden state}
}.
```

前者主要提供 token 的初始连续表示，后者则依赖当前上下文。

## 6. Embedding Dimension 与 Model Dimension

需要区分 embedding dimension

```math
d_e
```

与 Transformer 的 model dimension

```math
d_{\mathrm{model}}.
```

其中，$d_e$ 表示 embedding lookup 输出向量的宽度，而 $d_{\mathrm{model}}$ 通常表示 Transformer residual stream 或 hidden states 的宽度。

很多 Transformer 模型直接令

```math
d_e=d_{\mathrm{model}},
```

但这并不是数学上的必要条件。

### 6.1 $d_e=d_{\mathrm{model}}$

当

```math
d_e=d_{\mathrm{model}}
```

时，

```math
X_{\mathrm{emb}}
\in
\mathbb R^{T\times d_{\mathrm{model}}},
```

token embedding 已经具有 Transformer 所需要的宽度，因此不需要额外的维度投影。

需要注意，**维度是否匹配**与**如何引入位置信息**是两个不同的问题。

有些模型使用 learned absolute positional embedding，并将其与 token embedding 相加；另一些模型使用 RoPE 等机制，在 attention 内部对 query 和 key 注入位置信息。

因此，不能把所有 Transformer 的输入统一写成

```math
\text{token embedding}
+
\text{position embedding}.
```

位置表示的具体形式将在后续章节单独讨论。

### 6.2 $d_e\neq d_{\mathrm{model}}$

如果

```math
d_e\neq d_{\mathrm{model}},
```

可以使用线性映射

```math
W_{\mathrm{in}}
\in
\mathbb R^{d_e\times d_{\mathrm{model}}}
```

将 token embedding 投影到模型宽度：

```math
\boxed{
H^{(0)}
=
X_{\mathrm{emb}}W_{\mathrm{in}}
}.
```

其中

```math
X_{\mathrm{emb}}
\in
\mathbb R^{T\times d_e},
```

所以

```math
H^{(0)}
\in
\mathbb R^{T\times d_{\mathrm{model}}}.
```

若线性层包含 bias，则

```math
H^{(0)}
=
X_{\mathrm{emb}}W_{\mathrm{in}}
+
b_{\mathrm{in}},
```

其中

```math
b_{\mathrm{in}}
\in
\mathbb R^{d_{\mathrm{model}}}.
```

这里的 $H^{(0)}$ 只表示完成维度映射后的初始连续表示；实际模型还可能通过其他机制加入位置信息。

### 6.3 参数量

如果直接令

```math
d_e=d_{\mathrm{model}},
```

那么输入 embedding matrix 包含

```math
\boxed{
K d_{\mathrm{model}}
}
```

个参数。

如果采用较小的 embedding dimension

```math
d_e\lt d_{\mathrm{model}}
```

并额外使用输入投影，那么忽略 bias 时，embedding 与输入投影一共包含

```math
\boxed{
K d_e
+
d_e d_{\mathrm{model}}
}
```

个参数。

这里第二项是

```math
d_e d_{\mathrm{model}},
```

而不是

```math
K d_e d_{\mathrm{model}}.
```

原因是 $W_{\mathrm{in}}$ 是所有 token 共享的一组参数，而不是为词表中的每个 token 分别定义一个投影矩阵。

因此，当词表规模 $K$ 很大、且

```math
d_e\ll d_{\mathrm{model}},
```

factorized embedding parameterization 有可能减少与输入 embedding 相关的参数量。

不过，这只是一种可选的架构设计。许多 decoder-only Transformer 仍然直接采用

```math
d_e=d_{\mathrm{model}}.
```

## 7. Embedding Space 的几何与语义边界

由于训练后的 embedding 是连续向量，因此可以分析 embedding space 中的几何关系。

设两个 token 的 embedding 为

```math
\mathbf u_i,\mathbf u_j
\in
\mathbb R^{d_e}.
```

一种常见度量是 Euclidean distance：

```math
\boxed{
d(\mathbf u_i,\mathbf u_j)
=
\|\mathbf u_i-\mathbf u_j\|_2
}.
```

Euclidean distance 同时受到向量方向和模长的影响。

对于非零向量，也可以使用 cosine similarity：

```math
\boxed{
\cos
(\mathbf u_i,\mathbf u_j)
=
\frac{
\mathbf u_i^\top\mathbf u_j
}{
\|\mathbf u_i\|_2
\|\mathbf u_j\|_2
}
}.
```

Cosine similarity 主要描述两个向量方向之间的关系，而对整体尺度相对不敏感。

训练后的 embedding 可能编码与以下因素有关的结构：

- token 的词法模式；
- 句法使用方式；
- token 共现关系；
- token 频率；
- 与 next-token prediction 目标相关的其他统计规律。

因此，一些在模型中具有相似使用模式的 token，可能在 embedding space 中表现出某些几何上的相似性。

但是，不能简单地将任意 embedding distance 或 cosine similarity 直接解释为人类语言学意义上的“语义距离”。

Embedding matrix 的目标并不是单独学习一个具有明确语言学解释的语义空间，而是作为整个语言模型的一部分共同优化 next-token prediction 等训练目标。

更重要的是，输入 token embedding 本身是静态的、context-independent 的。

同一个 token 在不同上下文中首先读取相同的

```math
E_{k,:},
```

而其真正依赖上下文的表示通常会在 Transformer 更深层的 hidden states 中逐渐形成。

因此，

```math
\boxed{
\text{复杂的上下文语义结构}
\not\equiv
\text{input embedding 本身}
}.
```

## 8. Special Token 与 Padding

如果 BOS、EOS、UNK 等 special token 被纳入模型词表，那么从 embedding layer 的角度看，它们和普通 token 使用完全相同的数学操作。

例如，若 EOS 的 token ID 为

```math
k_{\mathrm{EOS}},
```

那么其 embedding 为

```math
E_{k_{\mathrm{EOS}},:}.
```

Embedding layer 本身并不会因为这个 token 表示“序列结束”而采用不同的 lookup 规则。

Special token 的特殊功能来自 tokenizer、训练数据、模型结构以及生成过程等更高层约定。

此外，并非所有 tokenizer 或语言模型都一定定义 BOS、EOS、UNK 或 PAD。具体包含哪些 special token 取决于 tokenizer 和模型设计。

### Padding

如果模型定义了 PAD token，那么 PAD 同样对应一个 token ID，并可以对应 embedding matrix 中的一行。

但是，padding 的完整语义并不只由 embedding layer 决定。

在 Transformer 中，padding position 通常还需要结合 attention mask，使其不作为正常上下文参与 attention。

训练时也通常需要通过 loss mask 或类似 `ignore_index` 的机制，避免 padding target 对损失产生影响。

某些 embedding API 还允许指定类似 `padding_idx` 的参数，使对应 embedding row 不按照普通 token 的方式更新。

具体初始化和梯度行为应以实际使用框架的定义为准。

因此，需要区分

```math
\boxed{
\text{PAD token embedding}
}
```

和

```math
\boxed{
\text{padding position 的 masking 语义}
}.
```

前者属于 representation，后者还涉及 attention 和训练过程。
