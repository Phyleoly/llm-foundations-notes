# 第四章 Transformer

本章讨论 decoder-only Transformer 如何根据上下文计算 hidden states，并最终产生 next-token logits。

## 精选阅读路线

1. [Attention 的基本计算](./01_attention.md) → [Self-Attention](./02_self_attention.md)：从加权平均走到同一序列内的 Q/K/V 计算。
2. [Causal Attention Mask](./04_causal_attention_mask.md) → [Multi-Head Attention](./03_multi_head_attention.md)：区分因果可见范围与多头表示。
3. [逐位置 MLP](./05_mlp.md) 与 [残差连接、LayerNorm 和 RMSNorm](./06_residual_and_layernorm.md)：补齐 block 中其余计算。
4. [完整前向过程](./07_decoder_only_transformer.md)：把各组件连接成从 token IDs 到 logits 的一条路径。

完成基础主线后，再阅读 [参数量、计算量与显存](./08_parameter_count_and_compute.md) 和 [高效 Attention](./09_efficient_attention.md)，理解实现成本。

## 与其他章节的关系

- 输入向量见 [Representation](../02_representation/README.md)。
- causal mask 与 Teacher Forcing 的训练语义见 [Teacher Forcing 与标签移位](../01_probability/02_teacher_forcing_and_label_shifting.md)。
- logits 到概率与损失见 [Probability](../01_probability/README.md)。
- 对内部表示与组件作用的观察和干预见 [Interpretability](../08_interpretability/README.md)。

## 全部笔记

以下索引由文章标题与状态生成，阅读顺序以上方路线为准。

<!-- notes:index:start -->
| 笔记 | 状态 |
| --- | --- |
| [Attention 的基本计算](./01_attention.md) | 草稿 |
| [Self-Attention](./02_self_attention.md) | 草稿 |
| [Multi-Head Attention 与 MQA / GQA](./03_multi_head_attention.md) | 提纲 |
| [Causal Attention Mask](./04_causal_attention_mask.md) | 草稿 |
| [逐位置 MLP 与门控前馈层](./05_mlp.md) | 提纲 |
| [残差连接、LayerNorm 与 RMSNorm](./06_residual_and_layernorm.md) | 提纲 |
| [Decoder-only Transformer 的完整前向过程](./07_decoder_only_transformer.md) | 提纲 |
| [参数量、计算量与显存](./08_parameter_count_and_compute.md) | 提纲 |
| [Attention 的计算与存储优化](./09_efficient_attention.md) | 提纲 |

<!-- notes:index:end -->
