# 计算回路、Induction Heads 与 Path Patching

> 状态：提纲

> 前置内容：[残差流](./residual_stream.md)、[Activation Patching](./activation_patching.md)。

本篇要回答：如何把若干有作用的组件连接成一个可检验的计算机制？

## 1. 从组件集合到计算图

- 定义研究行为、输入分布、节点粒度和边的含义：节点可以是 head、MLP 或特征，边表示具体的信息传递关系。
- 区分“这些组件分别影响输出”与“它们通过这条路径共同实现算法”。
- 明确回路外组件如何处理，以及被简化或尚未解释的计算。

## 2. 两个阅读案例

| 案例 | 阅读时要重建的内容 | 边界 |
| --- | --- | --- |
| Induction heads | 在形如 `[A][B] ... [A]` 的输入中，候选机制如何定位前一个 A 后面的 B 并促进预测 B | 检查模型层数、位置机制和训练条件；不把这一模式当作所有 in-context learning 的完整解释 |
| Indirect Object Identification（IOI） | 在给定句式中，哪些组件帮助区分重复出现的主体与目标接收者 | 结论依赖模型和任务模板，需要用新名字、新模板及失败案例复查 |

先画出论文中的最小信息流，再标注每一条边对应的观察或干预。区分作者已验证的路径与自己画图时补出的连接。

## 3. Path Patching 与近似定位

- 对一个 sender → receiver 假说，说明改动了什么、哪些中间量被固定、哪些下游计算会重算；不同 path patching 实现的协议可能不同。
- 与直接替换整个组件输出比较，解释为什么总效应不能自动分解成某一条边的作用。
- 将 attribution patching 等局部梯度近似作为候选筛选方法；对关键组件再做真实替换，并检查近似误差。
- 手工选出的回路也需要评估完整性与冗余；自动发现方法仍依赖指标、粒度和搜索空间。

## 4. 最小练习与完成标准

先为一个重复序列复制假说画三至五个节点的候选图。对每条边写出可反驳的干预预测，并列出一个可能的替代路径。

开展实验后，对完整模型、保留候选回路的模型和合适的对照报告同一行为指标；具体标准见 [解释的评估](./evaluation_and_faithfulness.md)。当前未执行回路复现。

## 来源

- [Olsson 等，2022：In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)：复制模式及不同证据之间的关系。
- [Wang 等，2022：Interpretability in the Wild](https://arxiv.org/abs/2211.00593)：GPT-2 small 的 IOI 回路及 faithfulness、completeness、minimality 评估。
- [Ameisen 等，2025：Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)：进阶阅读，关注 replacement model、attribution graph 与原模型干预验证之间的区别。
