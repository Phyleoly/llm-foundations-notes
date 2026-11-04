# 表示探针与信息可解码性

> 状态：提纲

> 前置内容：[可解释性的问题与证据](./overview.md)、[Token Embedding](../02_representation/01_token_embedding.md)，以及监督学习的训练集、验证集和测试集划分。

本篇要回答：如何判断 hidden state 中能读出某种信息，又为什么这不等于模型使用了这些信息？

## 1. 冻结表示，再训练读取器

- 固定模型及 activation 的提取位置，定义样本、标签和读取的 token 位置。
- 从线性探针开始；比较更强探针时，同时报告参数量、训练预算与正则化。
- 区分从单个 token 表示读出信息和从整个序列汇总后读出信息。

## 2. 设计能排除捷径的对照

- 与多数类、输入 embedding、随机初始化模型等适合当前任务的基线比较。
- 根据研究问题按序列、模板、实体或长度划分数据，避免同一来源的相近样本泄漏。
- 用控制任务考察探针自己的记忆能力；说明随机标签如何生成。论文中的词类型随机映射与逐样本标签打乱并非同一设置。
- 分层比较时保持训练协议一致，报告多个种子的变动；必要时进一步学习最小描述长度等容量控制思路。

## 3. 接到 NoPE 的位置问题

- 输入：冻结 NoPE 模型的逐层表示；标签：token 的绝对位置。
- 先确定序列长度分布和有效位置，检查 padding、内容和模板是否直接泄漏标签。
- 分别报告同长度留出序列上的误差与跨长度误差，避免用训练长度内的结果宣称长度泛化。
- 将探针结果与 [NoPE 中的位置来源问题](../02_representation/03_nope_reading_list.md) 对照；模型是否依赖该信息还需另设干预。

## 4. 最小练习与完成标准

为位置探针写出数据划分、两个基线、一个控制任务及失败标准。若今后运行，保存逐层结果，并给出一段仅关于“在这些设置下的信息可解码性”的结论。

还应解释一个反例：假设某种信息可以从表示中读出，但输出层及后续计算完全忽略它，探针仍可能成功。实际使用问题转到 [Activation Patching](./activation_patching.md)。

## 来源

- [Hewitt 与 Liang，2019：Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/)：控制任务和 selectivity。
- [Haviv 等，2022：Transformer Language Models without Positional Encodings Still Learn Positional Information](https://aclanthology.org/2022.findings-emnlp.99/)：位置探针的具体研究场景；论文阅读统一在现有 NoPE 笔记维护。
