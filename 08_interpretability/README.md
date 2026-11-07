# 第八章 Interpretability

本章讨论：如何提出关于模型行为的解释，并用可检查的证据判断它是否反映了模型的实际计算。以 LLM 的机制可解释性（mechanistic interpretability）为主线，同时建立输入归因、表示探针等通用方法的概念边界。

学习目标是能够围绕一个具体行为，说明研究对象、测量指标、候选机制、干预方法和结论范围。第一轮可以从小模型和单个 next-token 预测开始。

## 从现有笔记接入

不必先读完训练、推理和评估全章。开始时检查自己是否能回答以下问题；需要时回到对应笔记补齐。

| 必要基础 | 对应入口 | 在可解释性中的用途 |
| --- | --- | --- |
| hidden state 如何投影成 logits，logits 如何变成概率 | [输出投影](../02_representation/04_output_projection_and_weight_tying.md)、[概率模型](../01_probability/01_token_probability_model.md) | 定义要解释的输出及 logit difference |
| Q/K/V、attention 权重和 head 输出分别是什么 | [Self-Attention](../04_transformer/02_self_attention.md)、[Multi-Head Attention](../04_transformer/03_multi_head_attention.md) | 区分信息路由、被传递的内容和输出贡献 |
| 残差、归一化、MLP 在哪里更新表示 | [残差与归一化](../04_transformer/06_residual_and_layernorm.md)、[完整前向](../04_transformer/07_decoder_only_transformer.md) | 确定读取或替换 activation 的准确位置 |

学习梯度归因时再补链式法则与梯度；学习探针时再补监督学习的数据划分与正则化；学习 SAE 时再补稀疏编码和重建目标。各篇正文只列自身所需的前置知识。

## 精选阅读路线

| 阶段 | 阅读入口 | 这一阶段要产出的东西 |
| --- | --- | --- |
| 1. 提出可检验的问题 | [可解释性的问题与证据](./overview.md) → [解释的评估](./evaluation_and_faithfulness.md) | 用一个例子区分行为观察、可解码信息、干预效应和机制假说 |
| 2. 读懂内部计算 | [残差流](./residual_stream.md) | 在 Pre-Norm 计算图上标出两次写入，解释早期写入如何被后续层读取或抵消 |
| 3. 定位候选信息 | [输入归因](./attribution_methods.md)、[表示探针](./probing_representations.md) | 分别说明“输出对什么敏感”和“表示中能读出什么”，列出各自对照 |
| 4. 验证局部作用 | [Activation Patching](./activation_patching.md) | 一份 clean / corrupted 对照协议，以及对恢复量的正确解释 |
| 5. 连接计算路径 | [回路与 Path Patching](./circuits.md) | 一个包含节点、边、干预证据和未解释部分的候选回路 |
| 6. 从组件走向特征 | [Superposition 与 SAE](./superposition_and_sae.md) | 区分 neuron、feature 和 SAE latent，并设计特征验证方法 |
| 拓展：改变模型行为 | [Activation Steering 与模型编辑](./steering_and_editing.md) | 区分干预成功、原有机制解释和修改后的副作用 |

读残差流时，可回到第四章查阅 [Attention 输出投影](../04_transformer/03_multi_head_attention.md#2-并行计算与输出投影) 和 [MLP 矩阵定义](../04_transformer/05_mlp.md#1-逐位置前馈计算)。要观察中间状态对应的词表分数，可接着读 [Logit Lens](./logit_lens.md)。

第三阶段的两种方法可以按问题选择，不要求先掌握所有归因算法再学习 patching。评估贯穿每个阶段；[论文阅读路线](./reading_list.md) 将原始文献接到这些问题上。

```mermaid
flowchart LR
    A[明确行为与指标] --> B[观察表示并提出假说]
    B --> C[设计干预和对照]
    C --> D[在留出样本上验证]
    D --> E[写出机制与适用范围]
    D -->|假说未被支持| B
```

## 第一轮怎么学

1. 先读 [概览](./overview.md)，写清一个待解释行为。例如：“在固定提示下，为什么模型更偏向候选 token A 而不是 B？”
2. 画出一个 block 的计算，标出 residual stream、attention head 输出、MLP 输出和归一化的位置。把自己的图与 [残差流笔记](./residual_stream.md) 对照。
3. 手算 [patching 的示意例子](./activation_patching.md#4-一个只用于手算的例子)，理解原始分数、恢复量和恢复率的区别。
4. 按该篇的最小实验协议，先确认一个小型预训练模型确实表现出目标行为，再缓存、替换和测量 activation。将实验记录放入实际开展的专题，使用 [实验记录模板](../templates/experiment_record.md)。

上述实验是学习计划，当前未执行，也不表示本仓已经提供可运行的可解释性实验。读论文时分别记录作者结果、自己的理解和实际复现范围。

## 与现有研究问题的关系

- [NoPE 阅读路线](../02_representation/03_nope_reading_list.md) 是一个自然切入点：位置能否被探针读出、位置线索来自哪里、模型预测是否依赖这些线索，是三个不同的问题。
- 第四章负责 Transformer 的原始计算定义；本章负责如何观测、干预和解释这些计算。
- 第七章负责 [模型任务评估](../07_evaluation/README.md)；本章的评估对象是“解释是否忠实、完整，以及能否跨样本成立”。

## 全部笔记

以下索引由文章标题与状态生成，阅读顺序以上方路线为准。

<!-- notes:index:start -->
| 笔记 | 状态 |
| --- | --- |
| [Activation Patching：用替换实验检验局部作用](./activation_patching.md) | 草稿 |
| [输入归因、梯度与局部解释](./attribution_methods.md) | 提纲 |
| [计算回路、Induction Heads 与 Path Patching](./circuits.md) | 提纲 |
| [解释的忠实性、完整性与实验评估](./evaluation_and_faithfulness.md) | 提纲 |
| [Logit Lens：读取中间层的词表预测](./logit_lens.md) | 草稿 |
| [可解释性的问题、对象与证据](./overview.md) | 草稿 |
| [表示探针与信息可解码性](./probing_representations.md) | 草稿 |
| [可解释性论文与实践阅读路线](./reading_list.md) | 草稿 |
| [残差流](./residual_stream.md) | 草稿 |
| [Activation Steering、模型编辑与解释边界](./steering_and_editing.md) | 提纲 |
| [Superposition、特征与稀疏自编码器](./superposition_and_sae.md) | 草稿 |

<!-- notes:index:end -->
