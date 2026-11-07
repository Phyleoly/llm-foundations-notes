# Superposition、特征与稀疏自编码器

> 状态：草稿

> 前置内容：[表示探针](./probing_representations.md)、[残差流](./residual_stream.md)；需要理解向量空间、线性组合与重建损失。

本篇要回答：为什么单个 neuron 不一定对应一个概念，稀疏自编码器又试图恢复怎样的表示单元？

## 1. 区分 neuron、方向与 feature

MLP 的一个隐藏单元可以对多种模式激活，一个特征也可以分布在表示的多个坐标上。以残差流为例，若假设特征沿线性方向表示，一个位置的状态可以近似分解为

```math
r\approx\sum_{j=1}^{m}s_jf_j+\varepsilon.
```

其中 $f_j\in\mathbb R^d$ 是候选特征方向， $s_j$ 是当前样本的特征强度， $\varepsilon$ 是重建后剩余的部分。同一个前向也可以按子层写入分解，那是由计算图确定的精确求和。这里的特征方向则要从数据中估计，并检验它们表示什么。[残差流的加性展开](./residual_stream.md#2-计算图中的残差流)

当特征数 $m\gt d$ 时，这些方向无法两两正交。Elhage 等在小型模型中发现，当输入特征稀疏、很少同时激活时，模型可以用有限维度表示更多特征，并借助非线性减轻干扰。这是受控条件下的叠加实例。对真实模型，还需要检验特征间是否存在干扰，以及学到的方向是否稳定。[Elhage 等，2022，Demonstrating Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)

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

Gemma Scope 在 Gemma 2 的多个观察点训练 SAE，其中包括 MLP 更新后的残差流。作者分别评估稀疏度、重建保真度和特征可解释性。对照其他模型时，要先找准对应的观察点，因为 Gemma 2 在 Attention 和 MLP 前后都包含归一化。[Lieberum 等，2024，第 3.1、4 节](https://arxiv.org/html/2408.05147)

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
