# Code（之后补）

这轮不写实现，只先把位置留出来。

## 目录设想

```text
code/
├── bpe.py
├── wordpiece.py
├── unigram.py
├── sentencepiece_demo.py
├── common.py
└── tests/
    ├── test_bpe.py
    ├── test_wordpiece.py
    ├── test_unigram.py
    └── test_roundtrip.py
```

## 尽量统一接口

```python
model = train(corpus, vocab_size, **config)
tokens = encode(text, model)
text = decode(tokens, model)
```

模型至少要能导出：

- vocabulary
- 训练配置
- BPE merge rules 或 Unigram token probabilities
- 未知字符策略
- special token 配置

正文只放最小运行示例。完整实现和测试都放在这里，避免笔记里夹太长的代码。

