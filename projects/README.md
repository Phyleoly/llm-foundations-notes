# 专题项目

本目录连接知识笔记与需要独立运行的研究专题。专题范围、版本和接入规则统一见 [写作与组织规范](../STYLE_GUIDE.md#8-主题项目与历史实验)。

## 当前入口

| 专题 | 当前状态 | 入口 |
| --- | --- | --- |
| NoPE | 主仓内的本地运行试点；尚无独立远端或 submodule | [独立运行说明](./nope/README.md)、[论文阅读入口](../02_representation/03_nope_reading_list.md) |

NoPE 已统一放在 `projects/nope/`；[原地址](../reproductions/NoPE/README.md) 只保留迁移说明。基础算子检查不代表 Chi 2023 的方差结论已经复现。

## 使用入口

当前可直接进入 [NoPE 目录](./nope/README.md)，按其 README 安装专题依赖并运行检查。创建新的专题入口时，可从 [专题 README 模板](../templates/topic_readme.md) 开始。

将来接入独立仓库后，普通阅读仍可从本页开始；需要执行专题时，才初始化对应 submodule：

```bash
git submodule update --init -- projects/nope
```

上述命令适用于未来接入后的仓库，当前尚未注册该 submodule。历史记录与当前运行入口的区别、发布核对及许可事项见 [专题规则](../STYLE_GUIDE.md#8-主题项目与历史实验)。
