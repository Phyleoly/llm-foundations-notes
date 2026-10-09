# 4. Unigram

> 先准备一个较大的候选词表，再逐步删除贡献较小的 token。

这个方向和 BPE 相反：

- BPE：从小词表开始加；
- Unigram：从大词表开始删。

## 概率模型

每个 token $t\in\mathcal V$ 有一个概率 $p(t)$。

对一种切分

```math
z=(t_1,t_2,\ldots,t_m),
```

使用独立性近似：

```math
p(z)=\prod_{i=1}^{m}p(t_i).
```

这个假设很简单，但足够用来比较同一字符串的不同切分。

## 训练

简化流程：

1. 生成一个较大的候选词表；
2. 估计每个 token 的概率；
3. 计算删除某个 token 后，语料损失会增加多少；
4. 删除影响较小的一批 token；
5. 重新估计概率；
6. 词表缩到目标大小后停止。

基础字符通常不能随便删，否则有些文本会无法编码。

实际实现还会涉及：

- EM；
- 候选剪枝；
- character coverage；
- 必保留 token。

## 编码：找最优路径

同一个字符串可能有多种合法切分。确定性编码选择概率最大的那一条：

```math
z^*=\arg\max_z\prod_{t\in z}p(t)
=\arg\min_z\sum_{t\in z}-\log p(t).
```

可以把它看成图上的最短路径问题，用动态规划或 Viterbi 求解。

## 小例子

`unigram` 可能有多种切分：

```text
uni + gram
un + i + gram
u + n + i + g + r + a + m
```

Unigram 比较的是整条路径的概率，不是简单选择最长 token。

## Subword Regularization

训练语言模型时，不一定每次都用唯一的最优切分。也可以从多种合理切分中采样，让模型不要过度依赖一种固定分词。

要区分：

- Viterbi：确定性最优切分；
- sampling：按概率采样切分。

## 记住

优点：

- 能比较完整切分路径；
- 支持概率分词；
- 支持 subword regularization。

问题：

- 训练比 BPE 复杂；
- 初始候选词表会影响结果；
- token 独立只是一种近似。

## 代码以后补

```python
vocab = initialize_vocab(corpus)
probs = fit_probabilities(corpus, vocab)
vocab = prune_vocab(corpus, vocab, probs, target_size)
tokens = viterbi_encode(text, vocab, probs)
```

测试重点：多路径得分、动态规划回溯、必保留字符、剪枝后能否继续编码、采样能否复现。

