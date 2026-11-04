# Superposition、特征与稀疏自编码器

> 状态：提纲

> 前置内容：[表示探针](./probing_representations.md)、[残差流](./residual_stream_and_logit_lens.md)；需要理解向量空间、线性组合与重建损失。

本篇要回答：为什么单个 neuron 不一定对应一个概念，稀疏自编码器又试图恢复怎样的表示单元？

## 1. 区分 neuron、方向与 feature

- neuron 是选定表示中的一个坐标单元；某个属性也可能分布在多个坐标上。
- 明确本文所说的 feature 是哪种可检验属性或计算单元，不把人给出的短标签直接当作完整定义。
- 用 toy model 学习 superposition：在稀疏激活等条件下，模型可能用有限维度表示更多潜在特征，同时承受干扰。
- 区分分布式表示、单个 neuron 的多义性与 superposition，不能只凭一张高激活样例图认定三者等价。

## 2. SAE 的最小计算对象

以单个行向量 activation $h\in\mathbb R^d$ 为输入，设 SAE latent 的维度为 $m$。一个示意编码与解码为：

```math
f=\sigma(hW_{\mathrm{enc}}+b_{\mathrm{enc}}),
\qquad
\hat h=fW_{\mathrm{dec}}+b_{\mathrm{dec}}.
```

其中， $W_{\mathrm{enc}}\in\mathbb R^{d\times m}$， $f\in\mathbb R^m$， $W_{\mathrm{dec}}\in\mathbb R^{m\times d}$，两项偏置分别属于 $\mathbb R^m$ 和 $\mathbb R^d$。通常会使用大于输入维度的 latent 空间，同时要求每个样本只激活少量 latent。

- 从带稀疏惩罚的重建目标开始，明确 $\sigma$、惩罚项、decoder 归一化和输入预处理。
- 后续比较不同稀疏机制；上式只是示意，不覆盖所有 SAE 架构及其中心化约定。
- 区分 encoder 的检测方向与 decoder 的重建方向，不能默认二者相同。

## 3. 怎样评价一个学到的特征

| 检查对象 | 要记录的证据 |
| --- | --- |
| 重建与稀疏度 | 重建误差、每样本激活数量、dead latents，以及替换为重建 activation 后原模型的损失变化 |
| 语义解释 | 留出数据中的正例、反例和低激活样例，而不只挑最高激活文本 |
| 干预效应 | 抑制或增强特征后的目标变化、无关任务变化，以及与随机方向等对照的差别 |
| 稳定性 | 随机种子、字典规模和数据变化后，特征是否拆分、合并或消失 |

一个 latent 容易命名，不足以证明它是唯一真实概念；较好的重建也不保证所有重要计算都已被解释。

## 4. 最小练习与完成标准

先在二维纸面例子中画出三个不正交的候选特征方向，讨论同时激活时的干扰。随后设计一个小型 SAE 的评估表，分别列出重建、稀疏性、语义与干预指标。

使用预训练 SAE 时要核对所绑定的模型、层、activation 位置和预处理约定。当前仅建立学习框架，没有训练或加载 SAE。

## 来源

- [Elhage 等，2022：Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)：在可控模型中研究 superposition，区分实验设置与推广假说。
- [Bricken 等，2023：Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)：字典学习与特征分析；[作者研究简介](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning) 提供阅读入口。
