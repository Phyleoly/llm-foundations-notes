# NoPE 专题试点

本目录提供一个可以独立运行的 NoPE 基础算子检查入口，当前位于主仓的 `projects/nope/`，仍是本地试点，尚无独立远端或 submodule。后续拆分以整个 NoPE 专题为单位。

## 当前范围

当前实现是 **LayerNorm → Multi-Head Attention**，输入和输出均为 `[batch, sequence, hidden]`。它不引入显式位置编码，默认使用 causal mask，并提供无 mask 对照。Dropout 固定为 0。

运行入口只检查以下事项：

- 前向输出形状与输入一致，输出及反向梯度有限。
- 改动未来输入后，已有前缀位置的输出在指定误差内保持一致。
- 同一模型去掉 causal mask 后，前缀输出会响应这次未来输入扰动。

当前扰动是对序列后半段输入取负。这样可避免小维度时独立随机替换在 LayerNorm 后仍落到相同方向，导致无 mask 对照失效。

这些检查证明当前代码的基础运行和因果访问语义。它们没有验证位置表示能力、方差随位置的规律或长度泛化，也不构成 Chi 等人论文的实验复现。`chi_2023_variance` 是原有草稿的目录名，保留以便后续继续核对原文；当前模块还不是完整 Transformer 或语言模型。

## 环境与运行

使用 Python 3.11 或更新版本。本次本地验证环境为 Python 3.11.14、PyTorch 2.9.1+cu126，所有检查使用 CPU；CUDA 构建版本不代表本次执行使用了 GPU。

以下命令从本目录执行，也适用于以后独立克隆的专题仓库：

```bash
python -m venv .venv
```

激活新环境后安装 CPU 依赖：

```bash
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python smoke.py
python -m unittest discover -s tests -v
```

Windows PowerShell 可用 `.venv\Scripts\Activate.ps1` 激活；Linux/macOS 使用 `source .venv/bin/activate`。已有可用 PyTorch 环境时可以直接运行，无须修改全局环境。

默认 smoke 配置为 CPU、float32、seed 11、batch 2、长度 16、隐藏维度 32、4 个头。可以显式改变规模或设备：

```bash
python smoke.py --device cpu --sequence-length 8 --hidden-dim 16 --num-heads 4
python smoke.py --dtype float64 --output outputs/smoke-float64.json
```

输出为 JSON，包括环境、实际配置、输入生成方式、源码 SHA256、数值误差和检查结果。`--output` 只写入新文件，拒绝覆盖已有记录；不指定该参数时仅输出到终端。运行失败会返回非零退出码。

模块调用见 [模型实现](chi_2023_variance/src/model.py)：`NoPEModel(config)(inputs, causal=True)`。输入需为三维张量，device、dtype 和最后一维应与模型一致。`ModelConfig.seq_length` 是 smoke 输入长度配置，模块本身根据输入的实际长度构造 mask。

## 已执行的验证

2026-11-02 在上述 CPU 环境运行了 6 项测试和默认 smoke。当前默认输出保存在 [CPU smoke 记录 v2](reports/smoke_cpu_v2.json)，生成命令为 `python smoke.py --output reports/smoke_cpu_v2.json`，其源码哈希对应记录生成时的实现。

[早期 CPU 开发记录](reports/smoke_cpu.json) 保留了修改扰动方式之前的结果与源码哈希，不作为当前实现的验证依据。两份记录均来自未提交的本地开发代码，正式发布时还需补充可获取的代码提交。

测试包含输出和梯度、未来信息隔离、无 mask 对照、输入轴约定、最小合法配置下的多种子回归，以及从专题目录执行命令并读取配置和验证证据。测试通过不代表论文复现完成。

## 后续研究与记录

开始 Chi 方差实验前，需要独立核对原文的输入分布、初始化、模型结构、方差统计维度和 LayerNorm 前后测量位置。实验目标和预算明确后，再增加配置、结果和分析。

每项正式实验使用本专题维护的 [实验记录模板](EXPERIMENT_RECORD.md)，记录实际代码提交和配置、环境、数据与产物标识、生成命令及结论边界。该模板随本专题演化，不依赖主仓模板更新。未提交的开发运行可用源码哈希识别，但发布时还需要可获取的代码提交。

代码、测试、配置和小型结果摘要进入版本管理。大数据、权重和运行目录保存在 Git 之外，记录获取方式及完整性标识。

## 拆为独立仓库时

本目录内部链接和运行命令不依赖主笔记目录。建立专题远端并确认所需提交可获取后，主仓将在同一 `projects/nope/` 路径用 submodule 替换当前普通目录，专题入口保持稳定。

目前没有 submodule 指针。已有报告继续保留实际运行时的证据；本次目录迁移没有改写报告或源码哈希。许可条款仍需在公开前决定。
