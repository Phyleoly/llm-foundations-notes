# 第七章 Evaluation

本章讨论如何评价语言模型的概率建模质量与任务表现。

## 精选阅读路线

1. [Perplexity 与语言建模评估](./01_perplexity.md)：核对有效 token、损失聚合和可比性。
2. [任务评估与实验协议](./02_task_evaluation.md)：说明模型输出如何变成可复查的任务分数。
3. [评估的局限、污染与结论边界](./04_evaluation_limitations.md)：检查分数支持哪些结论、还缺哪些证据。

需要选择任务集合时，再用 [Benchmark 的覆盖范围与选择](./03_benchmarks.md) 比较能力覆盖与协议差异。

## 与概率建模的关系

Perplexity 与 token-level NLL 直接相关，因此建议先阅读 [Token 到概率模型](../01_probability/01_token_probability_model.md)。具体评估还依赖上下文、tokenizer 和数据划分，不能脱离协议解释数值。

模型表现与解释质量是不同的评估对象。输入归因、探针或回路解释的忠实性与完整性见 [解释的评估](../08_interpretability/evaluation_and_faithfulness.md)。

## 全部笔记

以下索引由文章标题与状态生成，阅读顺序以上方路线为准。

<!-- notes:index:start -->
| 笔记 | 状态 |
| --- | --- |
| [Perplexity 与语言建模评估](./01_perplexity.md) | 提纲 |
| [任务评估与实验协议](./02_task_evaluation.md) | 提纲 |
| [Benchmark 的覆盖范围与选择](./03_benchmarks.md) | 提纲 |
| [评估的局限、污染与结论边界](./04_evaluation_limitations.md) | 提纲 |

<!-- notes:index:end -->
