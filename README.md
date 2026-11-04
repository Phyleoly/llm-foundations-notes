# LLM Foundations Notes

一套面向已有深度学习基础的 LLM 系统笔记。项目以个人学习与实验积累为主，持续公开推导、疑问与核对过程。

## 目录

### 01. [语言模型的概率建模](./01_probability/README.md)

从 token 序列上的概率分布出发，推导自回归分解、最大似然、NLL 与 Cross-Entropy，并说明 Teacher Forcing、Label Shifting 与数值稳定计算。

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

### 08. [Interpretability](./08_interpretability/README.md)

讨论如何解释模型的表示与计算：从输入归因、表示探针和残差流，进入 activation patching、计算回路、稀疏特征及解释的验证。

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

> 现有目录编号保留为稳定路径，不表示严格的理论依赖。阅读顺序以各章的精选路线为准。例如第一章会先把 tokenizer 视为给定映射，具体的 BPE 等算法在第三章再展开。

可解释性是一条与基础主线并行的学习路线：理解 hidden states、logits 和 Transformer block 后，即可进入 [第八章](./08_interpretability/README.md)，按“明确行为 → 观察表示 → 提出假说 → 干预验证”学习。训练完整模型不是开始学习的前提；[NoPE 的位置问题](./02_representation/03_nope_reading_list.md) 可以用来练习区分信息可解码性与实际使用。

## 阅读与积累

各章 README 提供人工精选路线和从文章生成的完整索引。基础概念在正文中展开，论文阅读与实验通过 [主题项目入口](./projects/README.md) 连接；当前项目入口与历史结果记录各有用途，引用结果时应保留对应版本。

[近期计划](./ROADMAP.md) 只记录下一步工作。写作、目录组织、模板选择与归档规则统一放在 [STYLE_GUIDE.md](./STYLE_GUIDE.md)；新增文档从 [模板目录](./templates/README.md) 开始，图片等静态资源见 [assets](./assets/README.md)。

项目许可尚待选择；当前不声明统一的内容或代码许可。

## 仓库布局

```text
llm-foundations-notes/
├── 01_probability/              概率建模
├── 02_representation/           连续表示与位置机制
├── 03_tokenization/             文本切分与词表
├── 04_transformer/              网络组件与完整前向
├── 05_training/                 训练目标的实际优化
├── 06_inference/                生成与推理计算
├── 07_evaluation/               评估协议与结论边界
├── 08_interpretability/         表示、计算机制与解释证据
├── projects/                    专题项目入口
│   └── nope/                    NoPE 本地试点，尚非 submodule
├── templates/                   五类写作起点，README 说明用法
│   ├── note.md                  概念笔记
│   ├── paper_reading.md         论文阅读
│   ├── chapter_readme.md        章节导读
│   ├── topic_readme.md          专题入口
│   └── experiment_record.md     实验记录
├── assets/                      主笔记使用的图示与资源
├── scripts/                     创建、索引与检查工具
├── tests/                       文档工具的测试
├── reproductions/NoPE/README.md  旧地址的迁移入口
├── ROADMAP.md                   近期计划
└── STYLE_GUIDE.md               写作与维护规则
```

## 文档检查

文档工具使用 Python 3.11 或更新版本，依赖与专题研究环境分开。建议先创建虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活；Linux/macOS 使用 `source .venv/bin/activate`。随后安装文档工具依赖并检查主仓内容：

```bash
python -m pip install -r requirements-docs.txt
python scripts/notes.py check
```

新增笔记与刷新章节索引：

```bash
python scripts/notes.py new 04_transformer/attention_example.md --title "Attention 的最小例子"
python scripts/notes.py index
python scripts/notes.py index --check
```

论文阅读、章节导读、专题入口和实验记录的起点见 [模板目录](./templates/README.md)。工具检查内部文件、图片、可解析锚点、文章元信息与索引一致性；它不判断理论或实验结论是否正确。

完整写作流程、检查边界，以及原数学检查器和显式 `--fix` 用法，统一见 [新增与维护流程](./STYLE_GUIDE.md#7-新增与维护流程)。

文档检查也会由 GitHub Actions 在 push 和 pull request 时运行。静态检查不能替代 GitHub 页面上的实际渲染验证。
