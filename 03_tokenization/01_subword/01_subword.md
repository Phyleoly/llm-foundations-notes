# 1. Subword

Tokenizer 要决定两件事：

1. 词表 $\mathcal V$ 里放什么；
2. 文本怎么切成词表里的 token。

记词表大小为 $K=|\mathcal V|$，切分后的序列长度为 $T$。

一般来说：

- token 越大，$K$ 越大，$T$ 越小；
- token 越小，$K$ 越小，$T$ 越大。

Subword 就是在两边之间取折中。

## 三种粒度

| 粒度 | `playing` 的可能切分 | 特点 |
| --- | --- | --- |
| Word | `playing` | 序列短，但词表大，容易遇到 OOV |
| Character | `p l a y i n g` | 词表小，但序列太长 |
| Subword | `play ing` | 兼顾词表大小和序列长度 |

Subword 的做法：

- 高频词尽量整体保留；
- 低频词拆成更小、能复用的片段；
- 最差还能退回到基础字符或 byte。

例如：

```text
play
player  → play + er
playing → play + ing
```

注意：`play`、`ing` 可能刚好像词根和后缀，但 subword 本质上是统计单元，不保证符合语言学边界。

## 训练和编码要分开

### 训练

输入语料 $D$ 和目标词表大小 $K$，得到 tokenizer 参数 $\theta$：

```math
\theta=\operatorname{Train}(D,K).
```

$\theta$ 里可能有：

- vocabulary
- merge rules
- token probabilities
- normalization rules

BPE、WordPiece、Unigram 的主要区别就在训练方式。

### 编码

固定 tokenizer 后，再切新文本：

```math
\tau_\theta(x)=(t_1,t_2,\ldots,t_T),\qquad t_i\in\mathcal V.
```

同一个词换一个 tokenizer，切分结果可能完全不同。所以只说“这个词应该这样切”没有意义，还要说明模型和配置。

## 三条思路

- **BPE**：从小单元开始，反复合并高频相邻单元。
- **WordPiece**：按语料目标学习词表，编码时通常做最长匹配。
- **Unigram**：先准备大候选词表，再删除贡献小的 token。

SentencePiece 放在后面单独记。它更像一套训练和编码工具，可以使用 BPE 或 Unigram，不是同层级的第四种算法。

