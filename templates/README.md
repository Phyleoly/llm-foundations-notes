# 写作模板

选择最接近当前工作的模板，替换说明文字，删除未使用的内容。普通笔记从自由正文开始；可选模块放在 HTML 注释中，需要时移到注释外即可。

| 模板 | 放在哪里 | 如何使用 |
| --- | --- | --- |
| [普通笔记](./note.md) | 内容所属章节下的无编号英文文件名 | 创建命令默认使用，也可手工复制；保留标题、草稿状态和本篇问题，自由组织正文 |
| [论文阅读](./paper_reading.md) | 所属知识章节；专题内的深入讨论由专题自行组织 | 使用 `--template paper_reading` 或手工复制；分别记录作者结论、原文证据和自己的理解 |
| [章节入口](./chapter_readme.md) | 主仓章节的 `README.md` | 手工复制并填写导读、精选路线；保留唯一一对自动索引标记，再运行 `index` |
| [专题入口](./topic_readme.md) | `projects/<topic>/README.md`，适用于本地试点和之后接入的独立仓库 | 手工复制到对应专题，说明独立运行方式、实际进展和结果入口 |
| [实验记录](./experiment_record.md) | 专题内例如 `reports/run_id.md` | 每次记录复制一份，使用执行状态；链接实际配置与产物，不重复手抄全部参数 |

## 创建笔记

以下命令在主仓库根目录运行，目标文件已存在时会拒绝覆盖：

```bash
python scripts/notes.py new 04_transformer/example.md --title "示例笔记"
python scripts/notes.py new 04_transformer/paper_example.md --title "论文阅读示例" --template paper_reading
```

手工复制同样有效；复制后将 `{{ title }}` 替换为实际标题。填写具体问题和正文后，刷新目录并检查：

```bash
python scripts/notes.py index
python scripts/notes.py check
```

新章节需先创建章节目录及其 `README.md`，再添加文章。章节入口、专题入口和实验记录不使用笔记的“草稿／已核对”状态；实验记录默认标为“未执行”，实际运行后更新。

## 模板与规则的关系

写作、状态、引用和维护规则统一见 [STYLE_GUIDE](../STYLE_GUIDE.md)。模板提供起点，不要求每篇文章保留全部模块，也不替代内容核对。

专题可以复制实验模板并按研究问题裁剪，之后由专题维护自己的记录要求。NoPE 已有 [专题实验记录模板](../projects/nope/EXPERIMENT_RECORD.md)；无需为了同步主仓模板而重写已完成的实验记录。
