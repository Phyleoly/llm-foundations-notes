# 写作与组织规范

## 1. 文件与标题

- 目录：`NN_topic/`
- 文件：`NN_topic_name.md`
- 每个文件只保留一个 H1。
- H1 使用自然语言标题，例如 `# Teacher Forcing 与标签移位`。

## 2. 术语

建议全文统一以下写法：

| 概念 | 推荐写法 |
|---|---|
| token | token |
| tokenizer | tokenizer |
| token ID | token ID |
| hidden state | hidden state |
| logit / logits | 单个为 logit，向量或复数为 logits |
| Softmax | Softmax |
| LogSoftmax | LogSoftmax |
| LogSumExp | LogSumExp |
| Cross-Entropy | Cross-Entropy |
| Negative Log-Likelihood | NLL / 负对数似然 |
| Teacher Forcing | Teacher Forcing |
| label shifting | label shifting / 标签移位 |
| causal mask | causal mask |
| next-token prediction | next-token prediction |

## 3. 内容边界

一个概念只在一个位置完整展开：

- 概率目标、NLL、Cross-Entropy：`01_probability/`
- embedding 与连续表示：`02_representation/`
- BPE 等 tokenizer 算法：`03_tokenization/`
- attention 与 Transformer block：`04_transformer/`
- 优化与训练工程：`05_training/`
- decoding 与 KV cache：`06_inference/`
- perplexity 与 benchmark：`07_evaluation/`

其他章节需要使用该概念时，只保留最小必要定义并链接到主文档。

## 4. 引用规则

优先使用相对路径：

```markdown
[负对数似然与交叉熵](./01_probability/01_token_probability_model.md#10-负对数似然与交叉熵)
```

同目录文件：

```markdown
[LogSoftmax 与数值稳定性](./01_probability/03_logsoftmax_numerical_stability.md)
```

## 5. 公式与代码

- 行内公式使用 `$...$`；
- 简单的独立公式使用 `$$...$$`；
- 多行公式中若包含独立成行的 `=` 或 `-`，使用 ```` ```math ```` 围栏，避免被 GFM 识别为 Setext 标题；
- GitHub 不接受的 `\operatorname{name}` 应改用经过确认的 `\mathrm{name}`，不要对未知命令做批量替换；
- 多行推导优先使用 `aligned`；
- 代码标识符使用反引号，例如 `CrossEntropyLoss`；
- 公式与正文之间保留自然过渡句，避免连续堆叠公式。

提交前运行：

```bash
python scripts/check_markdown_math.py .
```

仅在需要应用确定性修复时运行：

```bash
python scripts/check_markdown_math.py --fix .
```

## 6. 推荐文档结构

```markdown
# 标题

> 前置内容：[...] (...)

一句话说明本节解决的问题。

## 1. ...

...

## 2. ...

...

## 小结

给出本篇最重要的关系式或概念边界，并链接后续内容。
```
