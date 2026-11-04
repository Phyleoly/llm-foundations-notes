# 可解释性论文与实践阅读路线

> 状态：草稿

> 前置内容：[本章导读](./README.md)、[可解释性的问题与证据](./overview.md)。各阶段的计算基础按对应笔记补齐。

本篇要回答：如何按研究问题组织可解释性阅读，并把论文变成可以核对的笔记和小实验？

以下是学习路线，不是穷尽文献的综述。论文列入本页不表示已经精读或复现；写作时使用 [论文阅读模板](../templates/paper_reading.md)，把原文结论和自己的理解分开。

## 1. 第一轮主线

按下表顺序阅读；先掌握每篇的问题与关键证据，再选择一个图或方法深入。年份以初次公开年份为主，正式会议版本有需要时另注。

| 顺序 | 原始资料 | 首轮阅读重点 | 阅读产出 |
| --- | --- | --- | --- |
| 1 | [Elhage 等，2021：A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | Transformer Overview、QK / OV 分解，注明 attention-only 等简化条件 | 用仓库的行向量约定重画一个 head 的信息流 |
| 2 | [Hewitt 与 Liang，2019：Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/) | 控制任务为什么能帮助识别探针的记忆能力 | 为一个位置探针设计基线与控制任务 |
| 3 | [Heimersheim 与 Nanda，2024：How to use and interpret activation patching](https://arxiv.org/abs/2404.15255) | 第 2—4 节：实验设计、证据解释、指标 | 解释恢复与破坏方向的差别，填写一份 patching 协议 |
| 4 | [Wang 等，2022：Interpretability in the Wild](https://arxiv.org/abs/2211.00593) | IOI 任务、候选回路与评估 | 把节点、路径、干预和结论一一对应 |
| 5 | [Elhage 等，2022：Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) | 定义、toy model 的数据与目标、稀疏性和干扰 | 说明为什么 feature 不必与单个 neuron 对齐 |
| 6 | [Bricken 等，2023：Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | 字典学习、特征分析与局限；可先读 [作者简介](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning) | 制作一张包含反例和干预验证的特征评估表 |

初学时先完成前三项和一个有对照的局部实验，再扩展到完整回路与 SAE。这个顺序是本仓的学习安排，不是论文之间的严格依赖。

## 2. 按疑问补读

| 遇到的问题 | 阅读入口 | 需要带走的问题 |
| --- | --- | --- |
| 梯度归因相对什么基线 | [Sundararajan 等，2017：Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html) | 参考点、路径与归因总和对应什么 |
| 黑盒模型怎样做局部解释 | [Ribeiro 等，2016：LIME](https://arxiv.org/abs/1602.04938)、[Lundberg 与 Lee，2017：SHAP](https://arxiv.org/abs/1705.07874) | 邻域和特征缺失如何定义；代理解释适用于什么范围 |
| 为什么不能只看 attention 图 | [Jain 与 Wallace，2019](https://aclanthology.org/N19-1357/)、[Wiegreffe 与 Pinter，2019](https://aclanthology.org/D19-1002/) | 双方对解释的定义、检验方式和模型设置有何差别 |
| 解释图是否依赖模型本身 | [Adebayo 等，2018：Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292) | 应随机化什么，图像实验中的结论怎样合理迁移到文本 |
| 逐层词表读取是否可靠 | [Belrose 等：Tuned Lens](https://arxiv.org/abs/2303.08112) | 读取器增加了什么可学习映射和验证要求 |
| 换指标或对照为什么得到不同位置 | [Zhang 与 Nanda：Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042) | 指标与 corruption 方法的选择如何影响结论，ICLR 2024 |
| 复制模式与 in-context learning 有何联系 | [Olsson 等，2022：In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | 哪些是相关观察、哪些有直接干预证据 |
| 模型说出的理由是否忠实 | [Turpin 等，2023：Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388) | 如何检测输入偏差对答案和解释的不同影响 |

## 3. 后续拓展

- [Turner 等：Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248)：研究 activation 干预的可控性，连接 [steering 笔记](./steering_and_editing.md)。
- [Meng 等，2022：Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)：区分因果定位、参数编辑和泛化评估。
- [Ameisen 等，2025：Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)：学习 replacement model 与 attribution graph；把图作为待验证的机制描述，检查近似误差和原模型中的干预结果。

这些内容放在局部干预和基础回路之后学习。大模型上的成功案例需要按原文的模型、输入和方法条件理解。

## 4. 实践入口与记录要求

[TransformerLens 官方文档](https://transformerlensorg.github.io/TransformerLens/) 可用于学习 activation 缓存与 hook 干预。工具入口于 2026-11-04 核对；开始实验时再固定实际库版本、模型 revision、tokenizer、缓存名称和任何权重转换选项。

当前未新增研究环境或安装依赖。实验逐步从张量形状与自替换检查，走到 [最小 patching 协议](./activation_patching.md#6-第一份实验协议)，再决定是否需要完整的专题实现。

每次精读至少记录：论文要解释的行为、模型与数据条件、关键图表、干预及对照、结论边界，以及自己仍未理解的问题。实际复现记录使用 [实验模板](../templates/experiment_record.md)，不把工具运行成功等同于复现论文。
