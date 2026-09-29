# 第二章 Representation

本章讨论离散 token 如何转换为模型内部使用的连续向量表示。

## 建议目录

1. `01_discrete_representation.md`
   - token ID
   - one-hot representation
   - one-hot 的局限
2. `02_token_embedding.md`
   - embedding matrix
   - embedding lookup
   - embedding dimension
3. `03_positional_representation.md`
   - 为什么需要位置信息
   - absolute / relative position 的基本概念
4. `04_output_projection_and_weight_tying.md`
   - hidden state 到 vocabulary logits
   - output projection
   - weight tying

## 与其他章节的关系

- token ID 的来源见 [Tokenization](../03_tokenization/README.md)。
- logits 与 token 概率之间的关系见 [Probability](../01_probability/README.md)。
- embedding 如何进入 Transformer 见 [Transformer](../04_transformer/README.md)。
