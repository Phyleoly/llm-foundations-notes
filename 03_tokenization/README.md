# 第三章 Tokenization

本章讨论原始文本如何被离散化为有限词表上的 token 序列。

## 建议目录

1. `01_vocabulary_and_subword.md`
   - character / word / subword
   - vocabulary size
   - OOV 问题
2. `02_bpe.md`
   - BPE 的基本思想
   - merge rule
   - vocabulary 构造
   - encode 过程
3. `03_byte_level_bpe.md`
   - byte-level 表示
   - byte-level BPE
4. `04_special_tokens.md`
   - BOS / EOS / PAD / UNK
   - special token 与词表

## 与其他章节的关系

概率章节只将 tokenizer 抽象为

```math
\tau:\mathcal S\rightarrow\mathcal V^*.
```

本章负责解释这个映射在实际 tokenizer 中如何构造。

- token 序列上的概率建模见 [Probability](../01_probability/README.md)。
- token ID 到连续向量的映射见 [Representation](../02_representation/README.md)。
