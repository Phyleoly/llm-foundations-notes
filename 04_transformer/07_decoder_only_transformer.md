# Decoder-only Transformer 的完整前向过程

> 状态：提纲

> 前置内容：[Multi-Head Attention 与 MQA / GQA](./03_multi_head_attention.md)、[Causal Attention Mask](./04_causal_attention_mask.md)、[逐位置 MLP 与门控前馈层](./05_mlp.md)、[残差连接、LayerNorm 与 RMSNorm](./06_residual_and_layernorm.md)、[位置表示](../02_representation/02_position_representation.md)、[输出投影与 Weight Tying](../02_representation/04_output_projection_and_weight_tying.md)。

本篇要回答：一批 token IDs 如何经过整个网络得到 next-token logits？

## 1. 输入与位置机制

- 定义 batch、序列长度与 model dimension，跟踪 token IDs 到初始向量。
- 选定一种位置机制并标出其作用点，避免默认所有模型都在输入端相加。

## 2. Decoder Block 与堆叠

- 按明确的 Pre-Norm 或 Post-Norm 约定画出完整 block。
- 跟踪 attention、MLP、残差与多层堆叠的形状，区分训练与推理开关。

## 3. 最终归一化与输出

- 连接最终 hidden states、输出投影与 vocabulary logits。
- 链接标签移位和 loss 的定义，区分某位置的输入 token 与其预测目标。

## 4. 贯穿例子与检查

- 用一条三到四 token 的序列画出逐层张量形状表，检查每一项参数的作用。
- 列出 padding、位置索引、特殊 token 与词表尺寸需要一致的条件。
