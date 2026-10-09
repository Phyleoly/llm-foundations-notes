# 5. SentencePiece

> SentencePiece 更像 tokenizer 工具和训练框架，不是与 BPE、WordPiece、Unigram 同层级的第四种算法。

它常用的底层模型包括：

- BPE
- Unigram

所以只说“用了 SentencePiece”信息不够，还要说明 `model_type`。

## 它主要解决什么

一些 tokenizer 会先按空格切词，再在词内部学习 subword。这个假设不适合所有语言。

SentencePiece 直接把输入看成 Unicode 字符序列，并把空格也编码进去。常见的空格标记是 `▁`：

```text
Hello world → ▁Hello + ▁world
```

这样 token 序列里仍然保留着恢复空格的信息。

## 基本流程

1. 文本规范化；
2. 显式表示空格；
3. 选择 BPE 或 Unigram；
4. 训练指定大小的词表；
5. 保存 model 和 vocabulary；
6. 用同一个 model 做 encode/decode。

实验中至少要记录：

```text
model_type
vocab_size
character_coverage
normalization_rule
special_tokens
random_seed
```

## 和前面几种方法的关系

| 名称 | 角色 |
| --- | --- |
| BPE | 合并算法 |
| WordPiece | 词表学习与最长匹配方案 |
| Unigram | 概率词表与概率切分模型 |
| SentencePiece | 规范化、训练、编码和解码框架 |

## 记住

优点：

- 可以直接处理原始文本；
- 不强依赖空格分词；
- 训练、编码和解码接口统一；
- 使用 Unigram 时可以做采样切分。

问题：

- 规范化可能改变原文本；
- `▁` 等内部符号需要单独解释；
- 工具名不能说明底层算法。

## 代码以后补

后面加一个最小 encode/decode round trip，重点检查：

- 空格是否正确恢复；
- 规范化是否符合预期；
- 特殊 token ID 是否固定；
- 同一模型保存再加载后结果是否一致。

