# Perplexity 与语言建模评估

> 状态：提纲

> 前置内容：[Token 到概率模型](../01_probability/01_token_probability_model.md)、[LogSoftmax 与数值稳定性](../01_probability/03_logsoftmax_numerical_stability.md)、[Batching、Padding、Packing 与 Loss Mask](../05_training/01_batching_padding_and_masking.md)。

本篇要回答：perplexity 如何聚合，什么情况下两个数值不能直接比较？

## 1. 从 NLL 到 Perplexity

- 推导平均 token NLL 与 perplexity 的关系，注明对数底数和计数单位。
- 区分先汇总损失再指数化与直接平均各 batch perplexity。

## 2. 有效 Token 与上下文

- 标出 BOS、EOS、padding 和被忽略 target 是否计入指标。
- 比较固定窗口切块与滑动窗口评估，说明边界上下文和重复计分的处理。

## 3. 可比性

- 固定 tokenizer、语料划分、上下文长度、预处理与归约方式。
- 讨论跨 tokenizer 比较时 token 单位的变化，以及按 byte 等单位报告的适用条件。

## 4. 最小例子与核对

- 对两个长度不同的序列手算语料级指标，验证分母。
- 检查评估模式与梯度开关；区分概率建模质量和下游任务表现。

## 阅读入口

- [Hugging Face Transformers：Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)
