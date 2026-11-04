# Activation Steering、模型编辑与解释边界

> 状态：提纲

> 前置内容：[Activation Patching](./activation_patching.md)、[残差流](./residual_stream_and_logit_lens.md)；使用 SAE 特征时再读 [Superposition 与 SAE](./superposition_and_sae.md)。

本篇要回答：改变内部表示或参数后能控制模型行为，能够为原有机制提供什么证据？

## 1. 区分三种修改对象

| 方法 | 修改什么 | 首先要回答的问题 |
| --- | --- | --- |
| Activation patching | 用另一运行中的值替换当前 activation | 此处的信息差异是否影响指定行为 |
| Activation steering | 在选定位置加入、移除或缩放表示方向 | 这种操作能否稳定改变行为，以及代价是什么 |
| 模型编辑 | 修改参数或持久化的模型组件 | 编辑目标能否泛化，其他行为是否受到影响 |

上述区分描述操作方式，不预先保证可解释性或编辑质量。

## 2. Steering 的实验变量

- 定义方向来自对比样本、探针、SAE 还是其他估计方法；构造方向的数据与评估数据分开。
- 写清干预层、token 位置、强度、是否每个生成步重复施加，以及方向的归一化方式。
- 记录目标效果、语言质量与非目标行为，比较零强度、随机方向和其他合适的基线。
- 干预成功说明该操作能够影响行为；解释模型自然运行时的机制，还需检查该方向原本如何出现、何时被使用。

## 3. 模型编辑与因果定位

- 以事实关联编辑为例，分别理解定位实验、权重更新算法和编辑评估。
- 区分让模型接受新关联、在改写问题上泛化，以及保持无关知识；每一项都需要单独的测量。
- 编辑位置有效不自动证明事实仅储存在该位置，也不等于自然检索路径已被完整定位。

## 4. 最小练习与完成标准

为一个小型模型上的可测行为设计强度扫描，提前列出一个目标指标和一个非目标指标。解释“可以控制”和“忠实解释原机制”的证据差别。当前未执行 steering 或编辑实验。

## 来源

- [Turner 等：Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248)：activation addition 的研究入口，初稿发布于 2023 年。
- [Meng 等，2022：Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)：causal tracing 与 ROME；阅读时分别记录定位方法和编辑结果。
