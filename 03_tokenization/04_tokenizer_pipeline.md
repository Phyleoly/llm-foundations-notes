# Tokenizer 的完整处理流程

> 状态：提纲

> 阅读位置：可作为本章的阅读起点。

本篇要回答：文本从输入到 token IDs，经过了哪些彼此独立的步骤？

## 1. 文本规范化与预切分

- 定义 normalization 与 pre-tokenization 的职责，并说明是否改变原始文本。
- 用空格、大小写、标点和组合字符记录不同处理顺序的结果。

## 2. 子词模型与后处理

- 把 BPE / WordPiece / Unigram 放进流水线，区分训练 tokenizer 与使用 tokenizer。
- 说明后处理如何插入约定符号；词表和特殊 token 的细节链接到相应笔记。

## 3. 编码、解码与偏移

- 画出文本、片段、子词、token ID、解码文本的对应关系。
- 说明 offset mapping 与原始文本对齐需要满足哪些假设。

## 4. 最小例子与边界

- 选择一段含中文、空格和标点的文本，逐步展示中间结果。
- 区分可逆编码与不可逆规范化，避免把完整 tokenizer 等同于 BPE 算法。

## 阅读入口

- [Hugging Face Tokenizers：The tokenization pipeline](https://huggingface.co/docs/tokenizers/pipeline)
