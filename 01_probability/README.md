# 第一章 语言模型的概率建模

本章从 token 序列上的概率分布出发，建立自回归语言模型的训练目标，并进一步说明该目标在训练数据组织和数值计算中的实现方式。

## 目录

1. [Token 到概率模型](./01_token_probability_model.md)
   - 文本空间与 token 序列
   - 随机变量与序列概率
   - 自回归概率分解
   - BOS / EOS
   - 参数化语言模型
   - 最大似然与 NLL
   - hidden state、logits 与 Softmax
   - Cross-Entropy

2. [Teacher Forcing 与标签移位](./02_teacher_forcing_and_label_shifting.md)
   - Teacher Forcing
   - causal mask 与并行训练
   - label shifting
   - input / target alignment

3. [LogSoftmax 与数值稳定性](./03_logsoftmax_numerical_stability.md)
   - Softmax 的数值溢出问题
   - LogSoftmax
   - LogSumExp
   - subtract-max trick
   - Stable Softmax

## 本章主线

```text
Token Sequence
      │
      ▼
Autoregressive Factorization
      │
      ▼
Maximum Likelihood
      │
      ▼
NLL / Cross-Entropy
      │
      ├──────────────┐
      ▼              ▼
Teacher Forcing   LogSoftmax
Label Shifting    LogSumExp
      │              │
      └──────┬───────┘
             ▼
       Stable Training Loss
```

## 与后续章节的关系

本章只把 tokenizer 视为一个将文本映射到有限 token 序列的给定映射。词表如何构造以及 BPE 等具体算法将在 [Tokenization](../03_tokenization/README.md) 中讨论。

本章中的 hidden state 被视为模型根据上下文计算得到的连续表示。离散 token ID 如何映射到连续向量将在 [Representation](../02_representation/README.md) 中讨论。

hidden state 如何进一步由 Transformer 计算得到，则在 [Transformer](../04_transformer/README.md) 中展开。
