# NoPos / NoPE 论文阅读与复现

> 状态：草稿

> 前置内容：[位置表示](./02_position_representation.md)、[Self-Attention](../04_transformer/02_self_attention.md) 与 [Causal Attention Mask](../04_transformer/04_causal_attention_mask.md)，重点是 causal mask 与置换等变性的关系。

本篇要回答：没有显式位置编码时，模型能否获得位置信息；这些信息来自哪里；模型能否泛化到训练时没有见过的序列长度？

NoPos 与 NoPE 是相关论文使用的命名，都指不使用显式位置编码的设置。具体的模型结构、mask、初始化和训练条件仍需逐篇核对。

本页记录阅读入口、概念总结框架与候选复现目标，个人总结和实验结果留待后续补充。

## 1. 阅读顺序与进度

主线按 **Haviv 2022 → Chi 2023 → Kazemnejad 2023 → Wang 2024** 阅读，依次关注实证现象、机制解释、表示能力与长度泛化边界。机制分析补读 Zuo 2025，理论基础补读 Bhattamishra 2020，早期工作补读 Irie 2019。

| 论文 | 关注问题 | 阅读 | 个人总结 | 复现或证明复核 |
| --- | --- | --- | --- | --- |
| Haviv 等，2022 | NoPos 的性能与内部位置信息 | 待阅读 | 待补充 | 未开始 |
| Chi 等，2023 | 自注意力方差中的位置线索 | 待阅读 | 待补充 | 未开始 |
| Kazemnejad 等，2023 | 绝对 / 相对位置的表达能力与长度泛化 | 待阅读 | 待补充 | 未开始 |
| Wang 等，2024 | 长上下文中的注意力分散与温度调整 | 待阅读 | 待补充 | 未开始 |
| Zuo 等，2025 | 相邻位置表示的相似性 | 待阅读 | 待补充 | 未开始 |
| Bhattamishra 等，2020 | 仅有位置掩码时的计算能力 | 待阅读 | 待补充 | 未开始 |
| Irie 等，2019 | 自回归语言模型中的早期实证观察 | 待阅读 | 待补充 | 未开始 |

上表记录论文阅读、总结与论文复现进度，不把基础算子检查计作论文复现。当前实验入口和验证记录见 [NoPE 主题项目](../projects/nope/README.md)；项目发现入口统一登记在 [主题项目目录](../projects/README.md)。

## 2. 论文与复现切入点

### 2.1 Haviv 等，2022

[Transformer Language Models without Positional Encodings Still Learn Positional Information](https://aclanthology.org/2022.findings-emnlp.99/) — Findings of EMNLP 2022。

论文比较 NoPos 与显式位置编码的语言建模性能，并用冻结语言模型后的逐层探针检查绝对位置信息。阅读时关注探针、前缀打乱和双向模型对照分别支持什么结论；“可解码出位置”与“预测时如何使用位置”需要区分。

候选复现：先选择逐层位置探针，记录位置预测平均绝对误差随层数的变化，再考虑前缀打乱实验。作者资源：[adihaviv/nopos](https://github.com/adihaviv/nopos)。

### 2.2 Chi 等，2023

[Latent Positional Information is in the Self-Attention Variance of Transformer Language Models Without Positional Embeddings](https://aclanthology.org/2023.acl-short.102/) — ACL 2023，Short Papers。

论文从自注意力输出的方差收缩解释位置线索，并分析随机初始化、冻结模型与预训练模型。阅读时先核对方差的计算维度、输入与初始化假设，以及 LayerNorm 前后测量的位置。

候选复现：从随机初始化的小模型开始，绘制方差随 token 位置的变化，比较 causal mask 与无 mask 的设置，并检查不同随机种子下的现象。统计位置和归一化方式以原文为准。

### 2.3 Kazemnejad 等，2023

[The Impact of Positional Encoding on Length Generalization in Transformers](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4e85362c02172c0c6567ce593122d31c-Abstract-Conference.html) — NeurIPS 2023。

论文比较 NoPE、绝对位置编码、T5 相对位置编码、ALiBi 与 RoPE，并给出 NoPE 表达绝对和相对位置信息的理论构造。阅读时区分构造所需的假设、训练后观察到的注意力模式，以及特定数学 / 推理任务上的长度泛化结果。

候选复现：从论文中的一个任务开始，固定训练长度与训练预算，比较不同位置机制在更长测试序列上的表现，并记录 scratchpad 格式。作者代码：[McGill-NLP/length-generalization](https://github.com/McGill-NLP/length-generalization)。

### 2.4 Wang 等，2024

[Length Generalization of Causal Transformers without Position Encoding](https://aclanthology.org/2024.findings-acl.834/) — Findings of ACL 2024。

论文研究 NoPE 的上下文长度限制，将长度泛化失败与注意力分布分散联系起来，并通过调整注意力头的温度参数扩展上下文。阅读时关注温度的定义、调参数据与测试数据的区分，以及不同任务的评估方式。

候选复现：选择一个模型，对照温度调整前后的长序列困惑度、注意力熵与 passkey retrieval 表现。作者代码：[AntNLP/nope_head_scale](https://github.com/AntNLP/nope_head_scale)。

### 2.5 Zuo 等，2025

[Position Information Emerges in Causal Transformers Without Positional Encodings via Similarity of Nearby Embeddings](https://aclanthology.org/2025.coling-main.632/) — COLING 2025。

论文提出相邻位置的内部表示更相似这一机制假说，并在随机初始化与训练后的模型中研究该现象。阅读时将其与 Chi 等人的方差解释对照，检查表示取自哪一层、如何衡量相似性，以及现象成立的条件。

候选复现：绘制不同层的表示余弦相似度矩阵，再按位置距离汇总。分别检查随机初始化和训练后模型，并用无 causal mask 的模型作对照。

### 2.6 Bhattamishra 等，2020

[On the Computational Power of Transformers and Its Implications in Sequence Modeling](https://aclanthology.org/2020.conll-1.37/) — CoNLL 2020。

论文在其理想化计算模型与假设下，证明只使用位置掩码而不使用位置编码的 Transformer 也可具备图灵完备性。阅读时列出硬注意力、数值精度、模型结构及生成过程的假设，并区分表达能力与训练可学性。

候选复核：逐步重建证明中的关键构造，整理每个组件的作用；如实现简化示例，说明它覆盖了证明的哪一部分。

### 2.7 Irie 等，2019

[Language Modeling with Deep Transformers](https://www.isca-archive.org/interspeech_2019/irie19b_interspeech.html) — Interspeech 2019。

论文在面向语音识别的自回归语言建模中，观察到移除位置编码仍可保持或略微改善性能，并讨论可访问历史信息随位置增加所提供的线索。阅读时关注模型深度、位置编码消融，以及语言模型困惑度与语音识别词错误率的区别。

候选复现：先做语言模型中的有 / 无位置编码对照；如果只复现语言建模部分，应将范围与完整语音识别实验分开记录。

## 3. 顺序约束、隐式位置与显式位置

- 比较 causal mask 对“允许访问哪些位置”的限制，与位置机制对位置索引、相对位移等信息的引入。
- 讨论为什么“没有显式位置编码”不等于“模型完全无法获得位置信息”；将模型从因果结构中学习隐式位置的可能性列入说明。
- 区分结构性质与实际学习效果：破坏任意置换等变性，本身不能证明模型已经学会使用绝对位置或相对距离。
- 将阅读形成的结论与 [位置表示](./02_position_representation.md) 中的绝对位置、相对位置联系起来；避免直接断言 causal mask 可以完全替代位置表示，或所有 Transformer 都必须使用显式位置编码。

## 4. 后续总结与复现记录

每篇读完后补充以下内容，并更新上方进度：

- **问题与结论**：论文回答了什么，结论适用于哪些设置。
- **证据与推导**：对应的定理、假设、图表和对照实验；哪些是观察，哪些是机制假说。
- **个人理解与疑问**：与已有位置表示笔记的联系，尚未理解或需要验证的部分。
- **复现记录**：选定的图表或结论、代码版本、环境、模型与数据配置、训练 / 测试长度、随机种子、命令、日志和结果路径。
- **结果比较**：原文结果、自己的结果、配置差异与可能原因；小规模实验单独注明，不写成原规模复现。

正式开展论文复现前，需要确定目标图表、设置与计算预算。基础算子验证已开始；论文设置核对、方差测量与论文结果复现尚未完成。运行入口中的 smoke 检查只验证其明确列出的计算性质，不能据此宣称复现了论文结论。

隐式位置的概念总结与复现记录集中在本篇；显式位置机制见 [位置表示](./02_position_representation.md)，基础推导见 [Self-Attention](../04_transformer/02_self_attention.md) 与 [Causal Attention Mask](../04_transformer/04_causal_attention_mask.md)。
