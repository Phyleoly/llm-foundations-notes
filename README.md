# LLM Foundations Notes

一套围绕大语言模型基础原理整理的学习笔记。

## 目录

### 01. [语言模型的概率建模](./01_probability/README.md)

从 token 序列上的概率分布出发，推导自回归分解、最大似然、NLL 与 Cross-Entropy，并说明 Teacher Forcing、Label Shifting 与数值稳定计算。

- [Token 到概率模型](./01_probability/01_token_probability_model.md)
- [Teacher Forcing 与标签移位](./01_probability/02_teacher_forcing_and_label_shifting.md)
- [LogSoftmax 与数值稳定性](./01_probability/03_logsoftmax_numerical_stability.md)

### 02. [Representation](./02_representation/README.md)

讨论离散 token 如何映射为连续向量，以及 embedding、位置表示与输出投影之间的关系。

### 03. [Tokenization](./03_tokenization/README.md)

讨论文本如何被离散化为 token，包括词表、BPE、byte-level BPE 与特殊 token。

### 04. [Transformer](./04_transformer/README.md)

讨论 Transformer 如何根据上下文计算 hidden states 与 next-token logits。

### 05. [Training](./05_training/README.md)

讨论 batching、padding、优化器、学习率、梯度累积与混合精度等训练机制。

### 06. [Inference](./06_inference/README.md)

讨论自回归生成、temperature、top-k / top-p、KV cache 等推理机制。

### 07. [Evaluation](./07_evaluation/README.md)

讨论 perplexity、任务评估与模型评测方法。

## 知识主线

```text
Raw Text
   │
   ├──────── Tokenization ────────┐
   │                              ▼
   │                           Token IDs
   │                              │
   │                        Representation
   │                              ▼
   │                           Embeddings
   │                              │
   │                         Transformer
   │                              ▼
   │                            Logits
   │                              │
   └──── Probability / Training ──┤
                                  ▼
                         Next-token Distribution
                                  │
                             Cross-Entropy
                                  │
                               Training
                                  │
                               Inference
                                  │
                              Evaluation
```

> 目录编号表示主要阅读顺序，不表示严格的理论依赖。例如第一章会先把 tokenizer 视为给定映射，具体的 BPE 等算法在第三章再展开。

## 文件命名

统一使用：

```text
NN_topic_name.md
```

约定：

- 文件名使用小写英文和 `snake_case`；
- 章节目录使用两位数字前缀；
- 文档 H1 可以使用中文；
- 文件间引用统一使用相对路径；
- 图片等静态资源统一放入 [`assets/`](./assets/README.md)。

详细写作约定见 [STYLE_GUIDE.md](./STYLE_GUIDE.md)。
