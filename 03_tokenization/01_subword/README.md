# Subword Tokenization

这一部分只记一个问题：

> 怎样在词表大小、序列长度和生词处理能力之间做平衡？

## 文件

| 文件 | 重点 |
| --- | --- |
| [01_subword.md](01_subword.md) | 为什么需要 subword |
| [02_bpe.md](02_bpe.md) | 从小词表开始，不断合并 |
| [03_wordpiece.md](03_wordpiece.md) | 学词表，编码时做最长匹配 |
| [04_unigram.md](04_unigram.md) | 从大词表开始，不断删减 |
| [05_sentencepiece.md](05_sentencepiece.md) | 直接在原始文本上训练 tokenizer |
| [06_comparison.md](06_comparison.md) | 放到一起比较 |

## 看每种算法时记什么

1. 初始词表从哪里来？
2. 训练时是合并 token，还是删除 token？
3. 用什么分数选下一步？
4. 新文本怎么切？
5. 遇到未知字符怎么办？

代码以后统一放进 [`code/`](code/README.md)。正文只保留思路和最小例子。

