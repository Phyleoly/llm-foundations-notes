# 离散 Token 表示与 Token Embedding

> 状态：已核对

> 前置内容：[文本空间与 Token 序列](../01_probability/01_token_probability_model.md#1-文本空间与-token-序列)。

## 1. 类别索引 Token ID

把「我喜欢你」输入模型时，tokenizer 先将它切成「我 / 喜欢 / 你」。假设 **词表（Vocabulary）** 按「我、喜欢、你、他」排序，并从 $1$ 开始编号，这句话就会变成 $(1,2,3)$。这里的 $3$ 指向词表的第三项「你」。

记词表为 $\mathcal V$，其中有 $K$ 个条目。每个条目有一个唯一的整数索引，记作 $\mathrm{id}$。

```math
\mathcal V = \{v_1,v_2,\dots,v_K\}, \qquad K=|\mathcal V|.
```

```math
\mathrm{id}: \mathcal V \rightarrow \{1,2,\dots,K\}.
```

如果某个输入条目是 $x=v_k$，它的索引就是 $\mathrm{id}(x)=k$。一段文本切分成 $T$ 个 token 后，可以表示为 ID 序列 $(k_1,k_2,\dots,k_T)$。序列的排列保留了 token 的先后顺序。

单独看两个 ID，即使 $k_1\lt k_2$，也只能说明一个条目排在另一个条目前面。差值 $|k_1-k_2|$ 衡量的是编号的间隔。「喜欢」和「你」的编号相邻，只是因为词表把它们放在了第二项和第三项。

下文手算时从 $1$ 开始编号，PyTorch 示例则用从 $0$ 开始的索引。编号和查找表的行序一一对应时，两种写法会选中相同的条目。具体切分方法见 [Tokenization](../03_tokenization/README.md)。

## 2. One-hot

### 2.1 用一个非零分量表示一个 token

既然编号可以定位类别，就可以把它用来选择向量中的一个坐标。给词表中的每个条目分配一个位置，属于哪个类别，就把对应的分量设为 $1$，其余分量设为 $0$。「你」的 ID 是 $3$，所以它的 one-hot 行向量是 $\mathbf e_3^\top=(0,0,1,0)$。

一般地，记第 $k$ 个 token 的 one-hot 列向量为 $\mathbf e_k$，第 $i$ 个分量为 $e_i^{(k)}$。各分量的取值由 $i$ 是否等于 $k$ 决定。

```math
\mathbf e_k = \begin{bmatrix} e_1^{(k)}\\ e_2^{(k)}\\ \vdots\\ e_K^{(k)} \end{bmatrix} \in \{0,1\}^K,
```

```math
e_i^{(k)} = \begin{cases} 1, & i=k,\\ 0, & i\neq k. \end{cases}
```

### 2.2 one-hot 缺陷

one-hot 为每个类别分配了一个独立的坐标轴。每个向量的 L2 范数都是 $1$，内积也只取决于两个向量的非零分量是否位于同一位置。

```math
\mathbf e_m^\top\mathbf e_n = \begin{cases} 1, & m=n,\\ 0, & m\neq n. \end{cases}
```

当 $m\neq n$ 时，两者的内积为 $0$。因此，展开欧氏距离的平方后，只剩下两个向量各自的范数平方。

```math
\begin{aligned} \|\mathbf e_m-\mathbf e_n\|_2^2 &= \|\mathbf e_m\|_2^2 + \|\mathbf e_n\|_2^2 - 2\mathbf e_m^\top\mathbf e_n \\ &= 1+1-0 \\ &= 2. \end{aligned}
```

对距离平方开方，再用同一个内积计算余弦相似度，可以得到

```math
\boxed{ \|\mathbf e_m-\mathbf e_n\|_2 = \sqrt 2 } \qquad (m\neq n).
```

```math
\frac{ \mathbf e_m^\top\mathbf e_n }{ \|\mathbf e_m\|_2 \|\mathbf e_n\|_2 } = 0, \qquad m\neq n.
```

所以，「我」到「你」的距离和「我」到「喜欢」的距离完全一样。**在 one-hot 空间里，任意两个不同的类别都等距。**

one-hot 的宽度等于词表大小 $K$。词表越大，显式构造稠密数组的成本就越高，尽管每个向量只有一个非零分量。这是高维稀疏表示的存储开销。

## 3. Embedding Matrix 把类别索引变成可学习的向量

### 3.1 词表有多少项，矩阵就有多少行

要让向量之间的远近可变，可以给每个词表条目分配一个可学习的向量，再沿用 ID 来定位它。取向量维度为 $d_e$，将 $K$ 个向量按词表顺序堆叠，就得到 **词嵌入矩阵（Embedding Matrix）** $E \in \mathbb R^{K\times d_e}$。

```math
E = \begin{bmatrix} E_{1,:}\\ E_{2,:}\\ \vdots\\ E_{K,:} \end{bmatrix}.
```

取出第 $k$ 行，得到的是一个 $d_e$ 维向量 $E_{k,:} \in \mathbb R^{d_e}$。因此，当输入条目为 $x=v_k$ 时，ID 就可以作为索引，从 **查找表（Lookup Table）** 中读出对应行。

```math
\boxed{ \mathrm{Emb}(x) = E_{k,:} }.
```

```math
\boxed{ \text{token ID }k \longrightarrow E_{k,:}\in\mathbb R^{d_e} }.
```

继续用四项词表，把每行设为两个分量。下表的坐标仅用于演算，实际模型会根据训练损失更新这些坐标。

| Token | ID | One-hot 行向量 $\mathbf e_k^\top$ | Embedding 行向量 $E_{k,:}$ |
| --- | ---: | --- | --- |
| 我 | 1 | $(1,0,0,0)$ | $(0.2,0.8)$ |
| 喜欢 | 2 | $(0,1,0,0)$ | $(0.6,0.3)$ |
| 你 | 3 | $(0,0,1,0)$ | $(0.1,0.9)$ |
| 他 | 4 | $(0,0,0,1)$ | $(0.4,0.7)$ |

词表固定后，token 和行号的对应关系也随之固定。一个 token 在序列里出现多次时，各位置读出的向量来自同一行，反向传播经由这些位置传回的梯度也会累加到该行。

### 3.2 一次查表，等价于 one-hot 左乘矩阵

对任意 ID $k$，embedding lookup 所做的就是取出第 $k$ 行。

```math
k \xrightarrow{\text{Embedding Lookup}} E_{k,:}.
```

```math
\boxed{ \mathrm{Emb}(v_k) = E_{k,:} }.
```

表中的「你」可以用来验证这件事。它的 one-hot 行向量左乘 $E$ 时，每个分量都充当相应矩阵行的权重，乘积里只留下系数为 $1$ 的第三行。

```math
\begin{aligned} \mathbf e_3^\top E &=0\,E_{1,:}+0\,E_{2,:}+1\,E_{3,:}+0\,E_{4,:}\\ &=E_{3,:}=(0.1,0.9). \end{aligned}
```

上面的 one-hot 也可以写成列向量。此时， $E^\top$ 的第 $k$ 列对应原矩阵的第 $k$ 行，右乘 $\mathbf e_k$ 就会选中这列，得到 $E^\top\mathbf e_k = E_{k,:}^\top$。将两边转置，就回到行向量的写法。

```math
\boxed{ \mathbf e_k^\top E = E_{k,:} }.
```

乘法的形状是 $(1\times K)$ 乘 $(K\times d_e)$，输出为 $(1\times d_e)$。按行展开时，第 $i$ 行乘上 $e_i^{(k)}$。由于只有第 $k$ 行的系数为 $1$，加权和就等于这一行本身。

```math
\begin{aligned} \mathbf e_k^\top E &= \sum_{i=1}^{K} e_i^{(k)}E_{i,:} \\ &= E_{k,:}. \end{aligned}
```

```math
\boxed{ \text{Embedding Lookup} \equiv \text{One-hot}\times E }.
```

实际实现可以直接用整数取行，省去显式构造 one-hot 数组的开销。

```math
\boxed{ \text{token ID} \longrightarrow \text{row lookup in }E \longrightarrow \text{embedding vector} }.
```

## 4. 从一串 ID 到一批向量

### 4.1 查表保留序列顺序，并在末尾增加一个维度

给定一条长度为 $T$ 的序列 $(k_1,k_2,\dots,k_T)$，按输入顺序对每个位置查表，再把拿到的向量依次叠成矩阵。

```math
X_{\mathrm{emb}} = \begin{bmatrix} E_{k_1,:}\\ E_{k_2,:}\\ \vdots\\ E_{k_T,:} \end{bmatrix} \in \mathbb R^{T\times d_e}.
```

输入 $(k_1,k_2,\dots,k_T)$ 有 $T$ 个整数，输出 $X_{\mathrm{emb}} \in \mathbb R^{T\times d_e}$ 有 $T$ 行，每行包含一个 token 的 $d_e$ 个分量。把开头的「我喜欢你」换成「你喜欢我」，ID 序列随之变成 $(3,2,1)$，拼出的矩阵如下。

```math
X_{\mathrm{emb}}= \begin{bmatrix} E_{3,:}\\ E_{2,:}\\ E_{1,:} \end{bmatrix} = \begin{bmatrix} 0.1 & 0.9\\ 0.6 & 0.3\\ 0.2 & 0.8 \end{bmatrix}.
```

这次输出有三行，对应句子的三个位置。查找表仍有四行，对应词表中的四个条目。处理一批序列时，各样本共用同一张表，批次维度和序列维度会原样保留。

| 输入 | ID 张量形状 | 查表后的形状 |
| --- | --- | --- |
| 一条长度为 $T$ 的序列 | $(T,)$ | $(T,d_e)$ |
| $B$ 条对齐到长度 $T$ 的序列 | $(B,T)$ | $(B,T,d_e)$ |

### 4.2 同一个「我」换了位置，会发生什么？

在「我喜欢你」和「你喜欢我」中，「我」分别在第一个和第三个位置。查表时，两处的 ID 相同，所以都读出词表的第一行，再放到各自对应的输出位置。

| 句子 | 「我」在序列中的位置 | 查到的向量 |
| --- | ---: | --- |
| 我 / 喜欢 / 你 | 第 1 个 | $E_{1,:}=(0.2,0.8)$ |
| 你 / 喜欢 / 我 | 第 3 个 | $E_{1,:}=(0.2,0.8)$ |

固定 $E$ 后，这个规律对任意 ID $k$ 都成立。ID 决定读取向量 $E_{k,:}$，所在位置决定该向量放到输出的第几行。查表得到的初始表示称为 $\boxed{ \text{token embedding} }$。

「我」在两句话里分别充当主语和宾语。这种区别来自词序和其他 token。后续 Transformer 层在计算 $\boxed{ \text{contextual hidden state} }$ 时会用到位置线索和可见上下文，于是两个「我」可以由相同的初始向量得到不同的隐藏状态。

### 4.3 用 PyTorch 验证

把这两条句子放在同一个 batch，查表应得到形状为 $(2,3,2)$ 的张量。代码同时核对直接查表与 one-hot 矩阵乘法的结果，以及两处「我」读出的向量。

```python
import torch
from torch import nn
from torch.nn import functional as F

# 代码从 0 编号，四行依次对应“我、喜欢、你、他”。
weights = torch.tensor([
    [0.2, 0.8],
    [0.6, 0.3],
    [0.1, 0.9],
    [0.4, 0.7],
])
embedding = nn.Embedding.from_pretrained(weights, freeze=False)
input_ids = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)

# 保留批次和序列维度，在末尾增加向量维度。
vectors = embedding(input_ids)
print(vectors.shape)  # torch.Size([2, 3, 2])

# 仅为核对等价性，在此显式构造 one-hot。
one_hot = F.one_hot(input_ids, num_classes=embedding.num_embeddings)
torch.testing.assert_close(one_hot.to(vectors.dtype) @ embedding.weight, vectors)
torch.testing.assert_close(vectors[0, 0], vectors[1, 2])
```

## 5. Embedding 维度如何接上 Transformer？

上节用两列的词表得到两维向量。一般地，查表输出的宽度记为 $d_e$，Transformer 残差流中每个位置的宽度记为 $d_{\mathrm{model}}$。当模型设定 $d_e=d_{\mathrm{model}}$ 时，查表得到的向量已满足宽度要求。两个维度也可以独立设置，再用投影连接。

### 5.1 查表宽度与模型宽度相同

当 $d_e=d_{\mathrm{model}}$ 时，查表结果已经具有形状 $X_{\mathrm{emb}} \in \mathbb R^{T\times d_{\mathrm{model}}}$，因此可以省去输入投影。

位置线索仍要通过模型所采用的位置机制引入。例如，采用可学习的绝对位置嵌入时，模型会给每个位置取一个相同宽度的向量，并在输入处相加。

```math
\text{token embedding} + \text{position embedding}.
```

使用 RoPE 的模型则会在注意力层中按位置旋转 query 和 key。具体方式见 [位置表示](./02_position_representation.md) 和 [RoFormer §3.2](https://arxiv.org/html/2104.09864v5#S3.SS2)。

### 5.2 查表宽度与模型宽度不同

当 $d_e\neq d_{\mathrm{model}}$ 时，让所有位置共享一个投影矩阵 $W_{\mathrm{in}} \in \mathbb R^{d_e\times d_{\mathrm{model}}}$，就能把查到的向量统一映射到模型宽度。

```math
\boxed{ H^{(0)} = X_{\mathrm{emb}}W_{\mathrm{in}} }.
```

查表结果的形状是 $X_{\mathrm{emb}} \in \mathbb R^{T\times d_e}$。右乘 $W_{\mathrm{in}}$ 时，两个矩阵内侧的 $d_e$ 对齐，输出变为 $H^{(0)} \in \mathbb R^{T\times d_{\mathrm{model}}}$。因此变化的是每个位置的宽度，仍然有 $T$ 个位置。

如果投影层带有偏置，它会给每一行加上同一个向量。其维度为 $b_{\mathrm{in}} \in \mathbb R^{d_{\mathrm{model}}}$，与投影后的向量宽度一致。

```math
H^{(0)} = X_{\mathrm{emb}}W_{\mathrm{in}} + b_{\mathrm{in}},
```

### 5.3 参数量怎么算？

两种方式的输出都有 $d_{\mathrm{model}}$ 列，输入端的参数量取决于有没有分开存储词表和投影。直接设 $d_e=d_{\mathrm{model}}$ 时，词表的每一行都有 $d_{\mathrm{model}}$ 个参数。整张表的参数量等于行数乘以列数。

```math
\boxed{ K d_{\mathrm{model}} }
```

如果选择 $d_e\lt d_{\mathrm{model}}$，词表中的每一行会变短，同时需要另存一个输入投影矩阵。参数量由这两部分相加得到。

```math
\boxed{ K d_e + d_e d_{\mathrm{model}} }
```

这里的投影成本为 $d_e d_{\mathrm{model}}$，因为所有词表条目共用一个矩阵。如果把这一项写成 $K d_e d_{\mathrm{model}}$，就等于把同一个投影重复计入 $K$ 次。

当词表很大、同时 $d_e\ll d_{\mathrm{model}}$ 时，词表矩阵节省的参数可能比新增的投影参数更多。ALBERT 使用了这样的 **分解式 embedding 参数化**。将它与 $d_e=d_{\mathrm{model}}$ 时的直接查表并列，参数量如下。 [ALBERT §3.1](https://arxiv.org/html/1909.11942v6#S3.SS1)

| 做法 | 查表输出形状 | 输入投影 | 参数量（忽略偏置） |
| --- | --- | --- | --- |
| 直接使用模型宽度 | $(T,d_{\mathrm{model}})$ | 无 | $K d_{\mathrm{model}}$ |
| 先查表，再投影 | $(T,d_e)$ | $W_{\mathrm{in}}$ | $K d_e+d_e d_{\mathrm{model}}$ |

## 6. Embedding 空间里，两个向量接近意味着什么？

### 6.1 欧氏距离和余弦相似度各看一件事

回到上面的二维词表。「我」和「你」的两个分量都只相差 $0.1$，「我」和「喜欢」则分别相差 $0.4$ 和 $0.5$。按欧氏距离衡量，前一对更接近。向量之间的远近现在由坐标决定，训练可以改变这些坐标。

把两行 embedding 写成列向量 $\mathbf u_i=E_{i,:}^{\top}$ 和 $\mathbf u_j=E_{j,:}^{\top}$，其中 $\mathbf u_i,\mathbf u_j \in \mathbb R^{d_e}$。**欧氏距离**衡量两个点相隔多远，向量的长度和方向都会影响结果。

```math
\boxed{ d(\mathbf u_i,\mathbf u_j) = \|\mathbf u_i-\mathbf u_j\|_2 }.
```

如果想比较方向，可以先将两个非零向量分别除以各自的范数，再计算内积。这就是 **余弦相似度**。把其中任一个向量乘以正数，改变的只是长度，归一化后的方向保持不变。

```math
\boxed{ \cos (\mathbf u_i,\mathbf u_j) = \frac{ \mathbf u_i^\top\mathbf u_j }{ \|\mathbf u_i\|_2 \|\mathbf u_j\|_2 } }.
```

取 $A=(1,0)$、 $B=(2,0)$、 $C=(1,1)$。从点 $A$ 到 $B$ 或 $C$ 都只移动一个单位。从原点看， $A$ 与 $B$ 同向，而 $A$ 与 $C$ 的夹角是 $45^\circ$。

| 向量对 | 欧氏距离 | 余弦相似度 |
| --- | ---: | ---: |
| $A$ 与 $B$ | $1$ | $1$ |
| $A$ 与 $C$ | $1$ | $\sqrt{2}/2$ |

### 6.2 向量相近可以提供什么线索？

前面的坐标是演算用的。实际模型会在预测下一 token 等训练目标下更新 embedding。训练后的向量接近，可能反映相似的句法角色或共现模式，也可能与分词粒度、词形或出现频率有关。要判断两个 token 是否近义，还得结合它们实际代表的文本片段和用法。

这里也能看出输入向量与上下文隐藏状态的区别。一个 token 可能有多种用法，但在每一次查表时，它都从同一行 $E_{k,:}$ 取向量。具体用法要等后续层结合上下文，才会反映在隐藏状态里。

```math
\boxed{ \text{复杂的上下文语义结构} \not\equiv \text{input embedding 本身} }.
```

## 7. 特殊 Token、Padding 与训练目标

### 7.1 特殊 Token 也通过同一张表取向量

第 4 节的两句话恰好等长。当一批序列的长度不同，需要先把它们对齐才能组成矩形张量。最常见的做法是用 **PAD** 补齐短序列。此外，词表还可能包含标记开始的 **BOS**、标记结束的 **EOS**，以及某些 tokenizer 用来表示未知片段的 **UNK**。

特殊 token 也有自己的 ID，查表的方式与普通条目相同。例如，输入 EOS 时，用它的 ID $k_{\mathrm{EOS}}$ 取出向量 $E_{k_{\mathrm{EOS}},:}$。当 EOS 出现在训练标签里，模型会学习何时预测它。实际生成时，解码程序根据配置的结束 ID 决定何时停止。

### 7.2 PAD 有向量，注意力仍要知道哪些位置是补齐的

把「我 / 喜欢 / 你」和「你」放在同一个 batch。加上 BOS 和 EOS 后，两条序列分别占 $5$ 和 $3$ 个位置，所以短序列需要在右侧补两个 PAD。

| 样本 | 位置 1 | 位置 2 | 位置 3 | 位置 4 | 位置 5 |
| --- | --- | --- | --- | --- | --- |
| 我 / 喜欢 / 你 | BOS | 我 | 喜欢 | 你 | EOS |
| 你 | BOS | 你 | EOS | PAD | PAD |

输入 ID 的形状是 $(2,5)$，查表结果为 $(2,5,d_e)$。补出来的两个位置也有向量，都来自 $E_{k_{\mathrm{PAD}},:}$。这里的 $\boxed{ \text{PAD token embedding} }$ 是查表结果。后续要用位置标记来确定 $\boxed{ \text{padding position 的 masking 语义} }$。

本例用 $1$ 表示有效位置，用 $0$ 表示补齐位置，两条序列的 **padding mask** 分别为 $(1,1,1,1,1)$ 和 $(1,1,1,0,0)$。对有效 query，注意力可以据此排除 PAD 对应的 key。

自回归模型还使用 **causal mask**，把当前位置之后的 token 遮住。这里采用右侧补齐，因此末尾的 PAD 全都落在有效 query 的未来，已经被因果约束挡住。

### 7.3 标签移位后，保留预测 EOS 的损失

注意力处理的是每个位置能读哪些输入，训练损失则取决于哪些预测目标有效。做下一 token 预测时，位置 $t$ 的输出对应位置 $t+1$ 的 token。移动标签后，短序列的四组配对如下。 [Teacher Forcing 与标签移位](../01_probability/02_teacher_forcing_and_label_shifting.md#3-label-shifting)

| 当前输入 | 下一个 token | 训练标签 |
| --- | --- | --- |
| BOS | 你 | 「你」的 ID |
| 你 | EOS | EOS 的 ID |
| EOS | PAD | 忽略 |
| 第一个 PAD | PAD | 忽略 |

BOS 所在位置学习预测「你」，「你」所在位置学习预测 EOS。后面两个目标来自补齐，可以把它们的类别索引设为 `-100`，再由 `CrossEntropyLoss(ignore_index=-100)` 跳过。若 EOS 的 ID 还兼作补齐值，应根据目标位置的有效标记来选择忽略哪些标签，使真实的 EOS 目标仍能参与训练。 [PyTorch CrossEntropyLoss 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html)

如果还要控制 PAD 行从查表操作接收的梯度，可以给 `nn.Embedding` 设置 `padding_idx`。这个设置使指定行从查表操作收到的梯度为零。注意力的可见范围和训练损失纳入的目标，仍由各自使用的 mask 和标签决定。 [PyTorch Embedding 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Embedding.html)
