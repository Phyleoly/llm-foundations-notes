# Token 到概率模型

## 1. 文本空间与 Token 序列

设原始文本空间为 $\mathcal S$，模型词表为

$$
\mathcal V=\{v_1,v_2,\dots,v_K\},
\qquad
K=|\mathcal V|.
$$

对于固定长度 $T$，token 序列属于笛卡尔积空间 $\mathcal V^T$。所有有限长度的 token 序列构成 Kleene star：

$$
\mathcal V^*
=
\bigcup_{T=0}^{\infty}\mathcal V^T,
\qquad
\mathcal V^0=\{\epsilon\},
$$

其中 $\epsilon$ 表示空序列。

Tokenizer 可以写成映射

$$
\tau:\mathcal S\rightarrow\mathcal V^*.
$$

对于文本 $s\in\mathcal S$，

$$
\tau(s)
=
(x_1,x_2,\dots,x_T),
\qquad
x_i\in\mathcal V.
$$

若 $x_i=v_k$，再通过索引映射

$$
\operatorname{id}:\mathcal V\rightarrow\{1,2,\dots,K\}
$$

得到

$$
\operatorname{id}(x_i)=k.
$$

因此，原始文本经过 tokenizer 后，被表示为有限词表上的离散 token 序列，并最终转换为整数 token ID 供模型计算。

> 本章将 tokenizer 视为一个给定的离散化映射。词表如何构造以及 BPE 等具体算法见 [Tokenization](../03_tokenization/README.md)。


## 2. 从确定序列到随机变量

给定一个观测到的 token 序列

$$
x_{1:T}
=
(x_1,x_2,\dots,x_T),
$$

定义对应的随机变量序列

$$
X_{1:T}
=
(X_1,X_2,\dots,X_T),
\qquad
X_i\in\mathcal V.
$$

其中，大写 $X_i$ 表示随机变量，小写 $x_i$ 表示其具体观测值。

对于固定长度 $T$，序列 $x_{1:T}$ 的联合概率为

$$
P(X_{1:T}=x_{1:T}),
$$

通常简写为

$$
p(x_{1:T}).
$$

语言建模的基本目标，就是在离散 token 序列空间上建立概率分布。


## 3. 自回归概率分解

根据概率的链式法则，

$$
\begin{aligned}
p(x_{1:T})
&=
p(x_1)
p(x_2\mid x_1)
p(x_3\mid x_{1:2})
\cdots
p(x_T\mid x_{1:T-1})
\\
&=
\prod_{i=1}^{T}
p(x_i\mid x_{<i}),
\end{aligned}
$$

其中

$$
x_{<i}
=
(x_1,x_2,\dots,x_{i-1})
=
x_{1:i-1}.
$$

约定

$$
x_{<1}=\epsilon,
$$

则首项也可以统一写成 $p(x_1\mid x_{<1})$。

最终得到

$$
\boxed{
p(x_{1:T})
=
\prod_{i=1}^{T}
p(x_i\mid x_{<i})
}
$$

链式法则本身是概率恒等式，并没有引入额外的独立性假设。自回归语言模型所做的是选择从左到右的顺序，对每一个条件概率

$$
p(x_i\mid x_{<i})
$$

进行参数化和学习。

于是，序列建模被转化为连续的 next-token prediction：

$$
x_{<i}
\longrightarrow
x_i.
$$


## 4. BOS 与序列起点

实际语言模型通常会引入起始 token

$$
\mathrm{BOS}.
$$

此时，

$$
p(x_1\mid\mathrm{BOS}),
$$

$$
p(x_2\mid\mathrm{BOS},x_1),
$$

并一般写为

$$
p(x_{1:T}\mid\mathrm{BOS})
=
\prod_{i=1}^{T}
p(x_i\mid\mathrm{BOS},x_{<i}).
$$

在理论推导中，通常将 BOS 视为初始上下文的一部分，仍简写为

$$
p(x_{1:T})
=
\prod_{i=1}^{T}
p(x_i\mid x_{<i}).
$$

BOS 的作用是为序列提供统一的起始上下文。


## 5. EOS 与变长序列

固定长度为 $T$ 时，前面的概率分解已经足够。但真实语言模型需要建模不同长度的序列，因此通常还会引入结束 token

$$
\mathrm{EOS}.
$$

完整序列可以表示为

$$
(x_1,x_2,\dots,x_T,\mathrm{EOS}).
$$

包含 BOS 时，其概率为

$$
\begin{aligned}
&p(x_{1:T},\mathrm{EOS}\mid\mathrm{BOS})
\\
&=
\left[
\prod_{i=1}^{T}
p(x_i\mid\mathrm{BOS},x_{<i})
\right]
p(\mathrm{EOS}\mid\mathrm{BOS},x_{1:T}).
\end{aligned}
$$

省略 BOS 后，

$$
p(x_{1:T},\mathrm{EOS})
=
\left[
\prod_{i=1}^{T}
p(x_i\mid x_{<i})
\right]
p(\mathrm{EOS}\mid x_{1:T}).
$$

EOS 使模型不仅学习“下一个 token 是什么”，还学习“序列是否应该在此结束”，因而参与了序列长度分布的建模。

BOS、EOS 等特殊 token 是否计入 $K$，取决于具体 tokenizer 和模型定义。后文将 $K$ 视为输出层实际对应的词表大小。


## 6. 参数化语言模型

真实数据可以看作来自未知分布 $p_{\mathrm{data}}$。语言模型使用参数化分布 $p_\theta$ 对其进行近似：

$$
p_\theta(x_{1:T})
=
\prod_{i=1}^{T}
p_\theta(x_i\mid x_{<i}),
$$

其中 $\theta$ 表示模型中所有可学习参数。

对于位置 $i$，模型根据上下文 $x_{<i}$ 定义下一个 token 在整个词表上的条件概率：

$$
p_\theta
\left(
X_i=v_k
\mid
X_{<i}=x_{<i}
\right),
\qquad
k=1,\dots,K.
$$

简写为

$$
p_\theta(X_i=v_k\mid x_{<i}).
$$

这些概率满足

$$
p_\theta(X_i=v_k\mid x_{<i})\ge 0,
$$

以及

$$
\sum_{k=1}^{K}
p_\theta(X_i=v_k\mid x_{<i})
=
1.
$$

因此，

$$
X_i\mid x_{<i}
\sim
\operatorname{Categorical}
\left(
p_{i,1},p_{i,2},\dots,p_{i,K}
\right),
$$

其中

$$
p_{i,k}
=
p_\theta(X_i=v_k\mid x_{<i}).
$$

语言模型的核心问题由此变为：**给定上下文，如何得到词表上的这 $K$ 个概率？**


## 7. 单个序列的似然与负对数似然

对于观测到的训练序列 $x_{1:T}$，模型赋予它的似然为

$$
\operatorname{Lik}(\theta;x_{1:T})
=
p_\theta(x_{1:T})
=
\prod_{i=1}^{T}
p_\theta(x_i\mid x_{<i}).
$$

最大似然训练要求

$$
\theta^\star
=
\arg\max_\theta
p_\theta(x_{1:T}).
$$

由于 $\log$ 严格单调递增，

$$
\arg\max_\theta
p_\theta(x_{1:T})
=
\arg\max_\theta
\log p_\theta(x_{1:T}).
$$

利用乘积的对数性质，

$$
\begin{aligned}
\log p_\theta(x_{1:T})
&=
\log
\prod_{i=1}^{T}
p_\theta(x_i\mid x_{<i})
\\
&=
\sum_{i=1}^{T}
\log p_\theta(x_i\mid x_{<i}).
\end{aligned}
$$

因此可以等价地最小化负对数似然：

$$
\boxed{
\mathcal L_{\mathrm{NLL}}
(\theta;x_{1:T})
=
-
\sum_{i=1}^{T}
\log
p_\theta(x_i\mid x_{<i})
}
$$

即

$$
\arg\max_\theta
\log p_\theta(x_{1:T})
=
\arg\min_\theta
\mathcal L_{\mathrm{NLL}}(\theta;x_{1:T}).
$$


## 8. 从单个序列到训练数据集

设训练集为

$$
\mathcal D
=
\left\{
x_{1:T_1}^{(1)},
x_{1:T_2}^{(2)},
\dots,
x_{1:T_N}^{(N)}
\right\},
$$

其中第 $n$ 个序列长度为 $T_n$。

在标准最大似然推导中，将训练序列视为独立观测样本，则

$$
p_\theta(\mathcal D)
=
\prod_{n=1}^{N}
p_\theta
\left(
x_{1:T_n}^{(n)}
\right),
$$

进一步展开为

$$
p_\theta(\mathcal D)
=
\prod_{n=1}^{N}
\prod_{i=1}^{T_n}
p_\theta
\left(
x_i^{(n)}
\mid
x_{<i}^{(n)}
\right).
$$

取对数得到

$$
\log p_\theta(\mathcal D)
=
\sum_{n=1}^{N}
\sum_{i=1}^{T_n}
\log
p_\theta
\left(
x_i^{(n)}
\mid
x_{<i}^{(n)}
\right).
$$

最大似然估计为

$$
\boxed{
\theta^\star
=
\arg\max_\theta
\sum_{n=1}^{N}
\sum_{i=1}^{T_n}
\log
p_\theta
\left(
x_i^{(n)}
\mid
x_{<i}^{(n)}
\right)
}
$$

等价地，

$$
\boxed{
\theta^\star
=
\arg\min_\theta
-
\sum_{n=1}^{N}
\sum_{i=1}^{T_n}
\log
p_\theta
\left(
x_i^{(n)}
\mid
x_{<i}^{(n)}
\right)
}
$$

实际训练时，通常按参与损失计算的有效 target token 数量进行平均。定义

$$
M
=
\sum_{n=1}^{N}T_n,
$$

则平均 token-level NLL 为

$$
\boxed{
\mathcal L_{\mathrm{NLL}}(\theta)
=
-\frac{1}{M}
\sum_{n=1}^{N}
\sum_{i=1}^{T_n}
\log
p_\theta
\left(
x_i^{(n)}
\mid
x_{<i}^{(n)}
\right)
}
$$

这里采用的是 **token-level averaging**：每个有效 target token 权重相同，而不是每个序列权重相同。

若存在 padding，通常通过 loss mask 将其排除；若 EOS 作为预测目标，则 EOS 也应计入有效 token 数 $M$。

至此，序列概率建模被转化为每个位置上的 $K$ 类条件概率预测。


## 9. 从隐藏状态到词表概率分布

对于位置 $i$，模型需要计算

$$
p_\theta(X_i=v_k\mid x_{<i}),
\qquad
k=1,\dots,K.
$$

首先，将上下文编码为 $d$ 维隐藏状态：

$$
h_i
=
f_\theta(x_{<i}),
\qquad
h_i\in\mathbb R^d.
$$

> 关于 token ID、embedding 与连续向量表示的关系，见 [Representation](../02_representation/README.md)。

随后通过输出线性层投影到词表空间：

$$
z_i
=
W_{\mathrm{out}}h_i+b_{\mathrm{out}},
$$

其中

$$
W_{\mathrm{out}}\in\mathbb R^{K\times d},
\qquad
b_{\mathrm{out}}\in\mathbb R^K,
$$

因此

$$
z_i
=
(z_{i,1},z_{i,2},\dots,z_{i,K})
\in\mathbb R^K.
$$

第 $k$ 个分量 $z_{i,k}$ 称为 **logit**。Logit 是未归一化分数，可以取任意实数：

$$
z_{i,k}\in\mathbb R.
$$

利用 Softmax 将 logits 映射为概率：

$$
p_i
=
\operatorname{softmax}(z_i),
$$

其中

$$
p_{i,k}
=
\frac{\exp(z_{i,k})}
{\sum_{j=1}^{K}\exp(z_{i,j})}.
$$

此时

$$
p_{i,k}>0,
\qquad
\sum_{k=1}^{K}p_{i,k}=1.
$$

于是

$$
\boxed{
p_\theta(X_i=v_k\mid x_{<i})
=
\frac{\exp(z_{i,k})}
{\sum_{j=1}^{K}\exp(z_{i,j})}
}
$$

整个过程可以写成

$$
\boxed{
x_{<i}
\longrightarrow
h_i
\longrightarrow
z_i
\longrightarrow
p_i
}
$$

即

$$
\text{context}
\rightarrow
\text{hidden state}
\rightarrow
\text{logits}
\rightarrow
\text{probabilities}.
$$

### Shifted Next-Token Prediction

在实际 decoder-only Transformer 中，更常见的下标约定是：

$$
h_i
=
f_\theta(x_{\le i}),
$$

位置 $i$ 的隐藏状态用于预测 $x_{i+1}$：

$$
p_\theta
\left(
X_{i+1}=v_k
\mid
x_{\le i}
\right)
=
\operatorname{softmax}
\left(
W_{\mathrm{out}}h_i+b_{\mathrm{out}}
\right)_k.
$$

这与理论记号

$$
p_\theta(x_i\mid x_{<i})
$$

只是索引方式不同，本质上都是 next-token prediction。

关于训练时如何使用真实历史 token，以及 input 与 target 如何进行位置对齐，见 [Teacher Forcing 与标签移位](./02_teacher_forcing_and_label_shifting.md)。


## 10. 负对数似然与交叉熵

考虑位置 $i$，设真实 token 为

$$
x_i^{(n)}=v_c,
\qquad
c\in\{1,\dots,K\}.
$$

模型预测分布为

$$
p_i
=
(p_{i,1},p_{i,2},\dots,p_{i,K}),
$$

其中

$$
p_{i,j}
=
p_\theta
\left(
X_i=v_j
\mid
x_{<i}^{(n)}
\right).
$$

该 token 的负对数似然为

$$
\ell_i^{(n)}
=
-\log p_{i,c}.
$$

将真实 token 写成 one-hot 目标分布

$$
y_i
=
(y_{i,1},y_{i,2},\dots,y_{i,K}),
$$

其中

$$
y_{i,j}
=
\begin{cases}
1, & j=c,\\
0, & j\neq c,
\end{cases}
$$

则目标分布与预测分布之间的交叉熵为

$$
\begin{aligned}
H(y_i,p_i)
&=
-\sum_{j=1}^{K}
y_{i,j}\log p_{i,j}
\\
&=
-\log p_{i,c}.
\end{aligned}
$$

因此，在 one-hot hard label 下，

$$
\boxed{
H(y_i,p_i)
=
-\log p_{i,c}
=
\ell_i^{(n)}
}
$$

即单个 token 上

$$
\boxed{
\mathrm{CE}
=
\mathrm{NLL}
}
$$

对所有有效 target token 求平均：

$$
\boxed{
\mathcal L_{\mathrm{CE}}(\theta)
=
\mathcal L_{\mathrm{NLL}}(\theta)
=
-\frac{1}{M}
\sum_{n=1}^{N}
\sum_{i=1}^{T_n}
\log
p_\theta
\left(
x_i^{(n)}
\mid
x_{<i}^{(n)}
\right)
}
$$

从分类角度看，每个位置是一个 $K$ 类分类问题；从概率建模角度看，它对应于最大化真实 token 的条件似然。

若采用 label smoothing、soft target 等目标，交叉熵不再等同于上述 one-hot NLL 形式。

将 Softmax 代入，

$$
p_{i,c}
=
\frac{\exp(z_{i,c})}
{\sum_{j=1}^{K}\exp(z_{i,j})},
$$

则单个 token 的损失为

$$
\begin{aligned}
\ell_i
&=
-\log p_{i,c}
\\
&=
-\log
\frac{\exp(z_{i,c})}
{\sum_{j=1}^{K}\exp(z_{i,j})}
\\
&=
-z_{i,c}
+
\log
\sum_{j=1}^{K}\exp(z_{i,j}).
\end{aligned}
$$

即

$$
\boxed{
\ell_i
=
-z_{i,c}
+
\operatorname{logsumexp}(z_i)
}
$$

训练的作用并不是单独让 $z_{i,c}$ 变大，而是提高真实 token 的 logit 相对于其他候选 token 的相对优势。

实际实现通常直接从 logits 计算 cross-entropy，而不是显式执行

$$
\operatorname{Softmax}
\rightarrow
\log,
$$

从而利用 `log-softmax` 或 `logsumexp` 获得更好的数值稳定性。

相关的稳定计算推导见 [LogSoftmax 与数值稳定性](./03_logsoftmax_numerical_stability.md)。

最终，整个计算过程可以概括为

$$
\boxed{
x_{<i}
\longrightarrow
h_i
\longrightarrow
z_i
\longrightarrow
p_i
\longrightarrow
-\log p_{i,c}
}
$$

或者

$$
\boxed{
\text{Context}
\rightarrow
\text{Hidden State}
\rightarrow
\text{Logits}
\rightarrow
\text{Token Distribution}
\rightarrow
\text{Cross-Entropy}
}
$$

而从概率建模的角度，

$$
\boxed{
\text{最大化训练数据的对数似然}
\quad\Longleftrightarrow\quad
\text{最小化平均 Token-level NLL}
}
$$

在 one-hot token supervision 下，又有

$$
\boxed{
\text{Token-level NLL}
=
\text{Cross-Entropy Loss}
}
$$



