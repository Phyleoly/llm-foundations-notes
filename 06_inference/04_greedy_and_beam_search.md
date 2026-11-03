# Greedy Decoding 与 Beam Search

> 状态：提纲

> 前置内容：[Token 到概率模型](../01_probability/01_token_probability_model.md)、[自回归生成的完整流程](./01_autoregressive_generation.md)。

本篇要回答：逐步选最大概率 token，与寻找高概率完整序列有什么区别？

## 1. 序列打分

- 从条件概率乘积转为 log-probability 求和，明确 EOS 是否计入。
- 讨论不同长度序列的可比性与长度归一化约定。

## 2. Greedy Decoding

- 写出逐步 argmax 的选择过程，并用小搜索树构造局部选择不等于全局最优的例子。
- 记录并列最大值、停止条件与确定性设置。

## 3. Beam Search

- 定义 beam width、候选扩展、保留和完成序列的处理。
- 跟踪两轮 beam 更新，区分搜索空间扩大与找到全局最优的保证。

## 4. 比较与边界

- 用相同概率树比较 greedy、beam 与随机采样的目标。
- 说明序列似然、任务指标和开放式生成质量之间不能直接画等号。
