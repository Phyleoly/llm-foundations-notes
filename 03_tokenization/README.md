# 第三章 Tokenization

Tokenization 把原始文本转成 token 序列：

```math
\tau:\mathcal S\rightarrow\mathcal V^*.
```

- $\mathcal S$：原始文本空间。
- $\mathcal V$：词表。
- $\mathcal V^*$：由词表 token 组成的有限序列。

## 精选阅读路线

1. [Tokenizer 的完整处理流程](./04_tokenizer_pipeline.md)：先区分文本规范化、切分模型、后处理与 token ID 映射。
2. [Subword 基础](./01_subword/01_subword.md) 与 [方法对比](./01_subword/06_comparison.md)：建立词表大小、序列长度和算法选择的视角。
3. [BPE 详解](./02_bpe/01_basic_bpe.md) 与 [Byte-level BPE](./03_byte_level_bpe.md)：跟踪合并规则，再扩展到 Unicode 和字节层面。
4. [词表、特殊 Token 与模型接口](./05_special_tokens_and_vocab.md)：连接 tokenizer 与模型输入输出的约定。

需要逐项比较子词方法时，可沿 [Subword 专题](./01_subword/README.md) 阅读 WordPiece、Unigram 与 SentencePiece。完整 BPE 推导以本章的 BPE 详解为准。

## 前后关系

- token 序列上的概率建模见 [Probability](../01_probability/README.md)。
- token ID 到向量的映射见 [Representation](../02_representation/README.md)；理解 tokenizer 的基本输入输出不要求先学完整 embedding 章节。
- tokenizer 的实验比较见 [Tokenizer 的训练与比较](./06_tokenizer_training_and_evaluation.md)。

## 全部笔记

以下索引由文章标题与状态生成，阅读顺序以上方路线为准。

<!-- notes:index:start -->
| 笔记 | 状态 |
| --- | --- |
| [1. Subword](./01_subword/01_subword.md) | 草稿 |
| [2. BPE](./01_subword/02_bpe.md) | 草稿 |
| [3. WordPiece](./01_subword/03_wordpiece.md) | 草稿 |
| [4. Unigram](./01_subword/04_unigram.md) | 草稿 |
| [5. SentencePiece](./01_subword/05_sentencepiece.md) | 草稿 |
| [6. 放到一起看](./01_subword/06_comparison.md) | 草稿 |
| [BPE（Byte Pair Encoding）](./02_bep/01_basic_bpe.md) | 已归档 |
| [BPE（Byte Pair Encoding）](./02_bpe/01_basic_bpe.md) | 草稿 |
| [Byte-level BPE 与 Unicode](./03_byte_level_bpe.md) | 提纲 |
| [Tokenizer 的完整处理流程](./04_tokenizer_pipeline.md) | 提纲 |
| [词表、特殊 Token 与模型接口](./05_special_tokens_and_vocab.md) | 提纲 |
| [Tokenizer 的训练与比较](./06_tokenizer_training_and_evaluation.md) | 提纲 |

<!-- notes:index:end -->

