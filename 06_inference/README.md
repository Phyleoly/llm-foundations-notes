# 第六章 Inference

本章讨论训练完成后，语言模型如何进行自回归生成。

## 建议目录

1. `01_autoregressive_generation.md`
2. `02_temperature.md`
3. `03_top_k_and_top_p.md`
4. `04_greedy_and_beam_search.md`
5. `05_kv_cache.md`

## 与训练的区别

训练阶段可以利用完整真实序列进行 Teacher Forcing；推理阶段没有未来真实 token，因此必须逐步生成并将已生成 token 作为后续上下文。

参见 [Teacher Forcing 与标签移位](../01_probability/02_teacher_forcing_and_label_shifting.md)。
