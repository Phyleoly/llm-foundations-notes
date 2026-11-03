# 第二章 Representation

本章讨论离散 token 如何转换为模型内部使用的连续向量表示。

## 精选阅读路线

1. [离散 Token 表示与 Token Embedding](./01_token_embedding.md)：先建立 token ID、one-hot 与 embedding lookup 的对应关系。
2. [位置表示](./02_position_representation.md)：理解位置或相对距离参与计算的不同方式。
3. [输出投影与 Weight Tying](./04_output_projection_and_weight_tying.md)：连接 hidden state、词表 logits 与输入输出权重。

需要深入研究隐式位置时，再读 [NoPos / NoPE 论文阅读与复现](./03_nope_reading_list.md)。这条研究支线依赖 self-attention 与 causal mask，不是阅读本章基础内容的前置条件。

## 与其他章节的关系

- token 序列的基本约定见 [Token 到概率模型](../01_probability/01_token_probability_model.md#1-文本空间与-token-序列)；具体切分算法见 [Tokenization](../03_tokenization/README.md)。
- logits 与 token 概率之间的关系见 [Probability](../01_probability/README.md)。
- embedding 如何进入 Transformer 见 [Transformer](../04_transformer/README.md)。

## 全部笔记

以下索引由文章标题与状态生成，阅读顺序以上方路线为准。

<!-- notes:index:start -->
| 笔记 | 状态 |
| --- | --- |
| [离散 Token 表示与 Token Embedding](./01_token_embedding.md) | 草稿 |
| [位置表示](./02_position_representation.md) | 草稿 |
| [NoPos / NoPE 论文阅读与复现](./03_nope_reading_list.md) | 草稿 |
| [输出投影与 Weight Tying](./04_output_projection_and_weight_tying.md) | 提纲 |

<!-- notes:index:end -->
