# Temperature 与概率分布

> 状态：提纲

> 前置内容：[LogSoftmax 与数值稳定性](../01_probability/03_logsoftmax_numerical_stability.md)、[自回归生成的完整流程](./01_autoregressive_generation.md)。

本篇要回答：温度怎样改变采样分布，又有哪些变化不能仅靠温度实现？

## 1. 温度的定义

- 写出正温度下 logits 缩放与 Softmax 的关系，说明归一化发生在什么位置。
- 区分 attention 内部的缩放与输出分布的 sampling temperature。

## 2. 分布变化与极限

- 用固定 logits 比较不同温度下的概率与熵。
- 讨论接近零、较大温度和并列最大值；区分数学极限与接口对零的特殊处理。

## 3. 与选择策略的关系

- 分析正温度缩放对 argmax 的影响，再比较 greedy 与随机采样。
- 说明温度本身不改变词表候选集合，截断策略留到 top-k / top-p。

## 4. 最小实验与记录

- 选择一个小词表，画出温度变化时各 token 的概率。
- 固定随机种子和采样次数，区分单次输出差异与分布层面的差异。
