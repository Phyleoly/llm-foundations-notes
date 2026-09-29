# 第四章 Transformer

本章讨论 decoder-only Transformer 如何根据上下文计算 hidden states，并最终产生 next-token logits。

## 建议目录

1. `01_attention.md`
2. `02_self_attention.md`
3. `03_multi_head_attention.md`
4. `04_causal_attention_mask.md`
5. `05_mlp.md`
6. `06_residual_and_layernorm.md`
7. `07_decoder_only_transformer.md`

## 与其他章节的关系

- 输入向量见 [Representation](../02_representation/README.md)。
- causal mask 与 Teacher Forcing 的训练语义见 [Teacher Forcing 与标签移位](../01_probability/02_teacher_forcing_and_label_shifting.md)。
- logits 到概率与损失见 [Probability](../01_probability/README.md)。
