# 词表、特殊 Token 与模型接口

> 状态：提纲

> 前置内容：[Tokenizer 的完整处理流程](./04_tokenizer_pipeline.md)、[离散 Token 表示与 Token Embedding](../02_representation/01_token_embedding.md)。

本篇要回答：tokenizer 与模型之间依赖哪些必须一致的约定？

## 1. 词表与 ID 映射

- 记录字符串 token、token ID、词表大小与有效 ID 范围。
- 说明重新排列词表或更换 tokenizer 为什么需要检查模型参数的对应关系。

## 2. 特殊 Token 的约定

- 分别记录 BOS、EOS、PAD、UNK 的用途，以及模型是否实际定义它们。
- 区分真实特殊 token 与文本中看起来相似的普通字符串。

## 3. 扩展词表与参数同步

- 说明添加 token 后输入 embedding、输出投影与配置需要检查哪些尺寸。
- 讨论新增向量的初始化与后续训练，区分有参数行和已经学会使用该 token。

## 4. 最小例子与边界

- 列出一组带 BOS / EOS / PAD 的 IDs，并标注文本内容、控制符号与填充位置。
- mask 与 loss 的处理链接到 Training，避免从 PAD ID 直接推断屏蔽语义。
