# KV Cache 与增量解码

> 状态：提纲

> 前置内容：[Multi-Head Attention 与 MQA / GQA](../04_transformer/03_multi_head_attention.md)、[Causal Attention Mask](../04_transformer/04_causal_attention_mask.md)、[自回归生成的完整流程](./01_autoregressive_generation.md)、[位置表示](../02_representation/02_position_representation.md)。

本篇要回答：怎样复用历史计算，使逐 token 生成避免重复计算已有前缀？

## 1. Prefill 与 Decode

- 分别跟踪整段 prompt 和单个新增 token 的 Q/K/V 形状。
- 解释哪些历史张量可复用、每层保存什么，以及输出 logits 的位置。

## 2. 缓存更新与位置一致性

- 画出新 K/V 追加到历史缓存的过程，核对 causal mask 的 query/key 长度。
- 说明 position IDs、RoPE 与已缓存内容的一致性；区分 padding 位置和真实上下文长度。

## 3. 缓存大小与头数

- 从层数、batch、长度、KV 头数、head dimension 和 dtype 推导内存估算。
- 比较 MHA、MQA 与 GQA 的缓存形状，区分缓存内存与模型权重内存。

## 4. 最小核对与边界

- 在相同模型状态下比较有 cache 与无 cache 的逐步 logits，并记录数值误差口径。
- 讨论多序列结束、beam 重排和上下文窗口变化时需要维护的状态。

## 阅读入口

- [Hugging Face Transformers：How caching works](https://huggingface.co/docs/transformers/cache_explanation)
