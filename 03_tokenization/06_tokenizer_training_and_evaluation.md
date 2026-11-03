# Tokenizer 的训练与比较

> 状态：提纲

> 前置内容：[Subword 方法对比](./01_subword/06_comparison.md)、[Tokenizer 的完整处理流程](./04_tokenizer_pipeline.md)。

本篇要回答：怎样判断 tokenizer 的压缩程度、覆盖能力与模型使用成本？

## 1. 语料与训练配置

- 记录语种、领域、清洗方式、采样比例、训练与评估划分。
- 控制词表规模、特殊 token、预处理与未知字符策略，保存训练产物。

## 2. 覆盖与切分效率

- 定义 tokens per byte / character、token fertility 与未知输入处理的测量口径。
- 按语言、代码、数字和罕见字符分组比较，避免只报告一个平均值。

## 3. 与模型成本的关系

- 讨论词表规模对 embedding / 输出投影的影响，以及序列长度对训练和推理的影响。
- 区分 tokenizer 本身的指标与训练后语言模型的任务表现。

## 4. 最小实验设计

- 选一小组固定文本，导出切分结果、长度和 round-trip 检查记录。
- 实现入口沿用 Subword 的 code 目录；记录版本、随机种子与运行配置。
