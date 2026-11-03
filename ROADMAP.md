# Roadmap

本页记录近期目标与完成标准，不汇总文章完成度。文章状态以正文开头为准，完整目录见各章 README。

写作与维护基础已具备：[五类模板](./templates/README.md)、自动章节目录、文档检查和工具测试已落地；[NoPE 本地试点](./projects/nope/README.md) 已提供基础算子检查。下一阶段转向正文写作与实验协议，算子检查不代表论文复现完成。

## 下一批工作

### 1. 补齐 Attention 的计算与因果约束

在 [Self-Attention](./04_transformer/02_self_attention.md) 中补出同一输入生成 Q/K/V 的投影、形状与小矩阵计算；在 [Causal Attention Mask](./04_transformer/04_causal_attention_mask.md) 中展开加性 mask、固定 mask 下重排 token，以及同步重排输入和 mask 的区别。

完成标准：用同一组三 token 输入逐步核对分数、可见范围和输出，明确 causal、padding 与 loss mask 的作用位置，并将例子与未来输入扰动检查对应起来。

### 2. 完成小模型的前向主线

围绕 [Decoder-only Transformer 的完整前向过程](./04_transformer/07_decoder_only_transformer.md) 选定一套最小配置，连接 embedding、位置机制、attention、MLP、归一化与输出投影。优先补齐这条链路所需的定义和例子。

完成标准：提供可运行的前向例子，从 token IDs 跟踪到 `[batch, sequence, vocabulary]` logits，逐层核对形状，明确位置机制、归一化、权重共享与目标标签的对齐约定。

### 3. 为 NoPE 方差问题建立实验协议

沿 [NoPE 阅读路线](./02_representation/03_nope_reading_list.md) 选择 Chi 等人的一个具体结论，在 [本地专题](./projects/nope/README.md) 中核对输入分布、初始化、方差统计维度、LayerNorm 前后测量位置，以及 causal 与无 mask 的对照设置。

完成标准：关键设置能够追溯到原文，偏离原文的部分单独说明；按 [专题记录模板](./projects/nope/EXPERIMENT_RECORD.md) 确定配置、随机种子、预算、运行命令和产物格式。协议明确后开展小规模机制验证，保存多个种子的结果及图表生成命令，并写清结论边界。

## 后续顺序

前向主线完成后，再用同一组配置连接 [小数据训练](./05_training/08_training_loop_and_checkpoints.md)、[无 cache 的自回归生成](./06_inference/01_autoregressive_generation.md) 与 [Perplexity 评估](./07_evaluation/01_perplexity.md)。重点核对 loss 聚合、有效 token 计数和逐步生成过程，并保留可恢复的最小训练记录。

NoPE 继续作为本地试点维护；需要独立发布且远端地址确定后，再在同一 `projects/nope/` 路径接入 submodule。正式公开前的干净检出验证、代表性实验核对和许可证决策，按 [专题与公开规则](./STYLE_GUIDE.md#8-主题项目与历史实验) 完成。

完成或改变优先级后更新本页，不在这里复制文章状态与历史实验表。新增主题先归入现有知识章节；确有独立知识边界时再扩展章节。
