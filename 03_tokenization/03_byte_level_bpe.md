# Byte-level BPE 与 Unicode

> 状态：提纲

> 前置内容：[BPE 详解](./02_bpe/01_basic_bpe.md)。

本篇要回答：为什么以字节为基础符号仍然能编码多语言文本，token 又为何不一定对应完整字符？

## 1. Unicode 与 UTF-8

- 区分 Unicode 码点、UTF-8 字节与用户看到的字符。
- 选英文、中文和 emoji，逐层记录文本、字节和符号表示。

## 2. 字节层面的 BPE

- 说明初始字节符号与后续 merge 的关系，承接已有 BPE 训练和编码规则。
- 区分 byte-level BPE 与字符级模型的 byte fallback，核对具体实现的差异。

## 3. 解码与可逆性

- 跟踪 token 对应字节的拼接顺序，再讨论 UTF-8 解码。
- 区分单 token 解码与整段解码，记录规范化对 round trip 的影响。

## 4. 最小例子与检查

- 用含多字节字符的短文本展示一个字符被多个 token 覆盖的情形。
- 列出未知输入、空白、边界符号与特殊 token 的处理约定。
