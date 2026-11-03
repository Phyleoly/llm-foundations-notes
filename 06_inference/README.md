# 第六章 Inference

本章讨论训练完成后，语言模型如何进行自回归生成。

## 精选阅读路线

1. [自回归生成的完整流程](./01_autoregressive_generation.md)：跟踪输入前缀、下一 token 分布与停止条件。
2. [Temperature](./02_temperature.md) → [Top-k 与 Top-p](./03_top_k_and_top_p.md)：区分概率变换与候选集合筛选。
3. [KV Cache 与增量解码](./05_kv_cache.md)：理解历史计算如何复用，以及位置和 mask 如何保持一致。

比较确定性搜索策略时读 [Greedy Decoding 与 Beam Search](./04_greedy_and_beam_search.md)；研究性能时读 [推理吞吐、延迟与内存](./06_inference_efficiency.md)。

## 与训练的区别

训练阶段可以利用完整真实序列进行 Teacher Forcing；推理阶段没有未来真实 token，因此必须逐步生成并将已生成 token 作为后续上下文。

参见 [Teacher Forcing 与标签移位](../01_probability/02_teacher_forcing_and_label_shifting.md)。

## 全部笔记

以下索引由文章标题与状态生成，阅读顺序以上方路线为准。

<!-- notes:index:start -->
| 笔记 | 状态 |
| --- | --- |
| [自回归生成的完整流程](./01_autoregressive_generation.md) | 提纲 |
| [Temperature 与概率分布](./02_temperature.md) | 提纲 |
| [Top-k 与 Top-p 采样](./03_top_k_and_top_p.md) | 提纲 |
| [Greedy Decoding 与 Beam Search](./04_greedy_and_beam_search.md) | 提纲 |
| [KV Cache 与增量解码](./05_kv_cache.md) | 提纲 |
| [推理吞吐、延迟与内存](./06_inference_efficiency.md) | 提纲 |

<!-- notes:index:end -->
