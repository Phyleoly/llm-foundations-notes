# 03 Tokenization

Tokenization：把原始文本转成 token 序列。

```math
\tau:\mathcal S\rightarrow\mathcal V^*.
```

- $\mathcal S$：原始文本空间
- $\mathcal V$：词表
- $\mathcal V^*$：由词表 token 组成的有限序列

## 目前写到哪里

这轮只记 **Subword Tokenization**。

文本规范化、特殊 token、token ID 映射和完整 tokenizer pipeline 后面再补。

## 目录

```text
03_tokenization/
├── README.md
└── 01_subword/
    ├── README.md
    ├── 01_subword.md
    ├── 02_bpe.md
    ├── 03_wordpiece.md
    ├── 04_unigram.md
    ├── 05_sentencepiece.md
    ├── 06_comparison.md
    └── code/
        └── README.md
```

阅读顺序：

1. [Subword 基础](01_subword/01_subword.md)
2. [BPE](01_subword/02_bpe.md)
3. [WordPiece](01_subword/03_wordpiece.md)
4. [Unigram](01_subword/04_unigram.md)
5. [SentencePiece](01_subword/05_sentencepiece.md)
6. [方法对比](01_subword/06_comparison.md)

## 前后关系

- token 序列上的概率建模：[Probability](../01_probability/README.md)
- token ID 到向量的映射：[Representation](../02_representation/README.md)

