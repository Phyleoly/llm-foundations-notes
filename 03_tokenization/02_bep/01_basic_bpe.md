# BPE（Byte Pair Encoding）

BPE（Byte Pair Encoding，字节对编码）最初由 Philip Gage 于 1994 年提出，是一种通过反复替换高频相邻字节对实现压缩的数据压缩算法 [1]。

Sennrich et al.（2016）将 BPE 应用于神经机器翻译，通过学习高频字符组合构建子词词表，缓解低频词与未登录词（Out-of-Vocabulary，OOV）问题 [2]。

随后，Radford et al.（2019）在 GPT-2 中采用 Byte-level BPE，使 tokenizer 能够以字节作为基础符号处理 Unicode 文本 [3]。

**核心思想：从最小符号单元开始，反复合并语料中出现频率最高的相邻符号对，逐步构建更大的 token。**

BPE 在字符级和词级表示之间取得折中：

- 高频字符组合逐渐合并为较大的 token。
- 低频词可以保留为多个较小的子词单元。
- 通过控制合并次数，间接控制词表大小。

需要注意，现代 NLP 中的 BPE 与最初的数据压缩算法在目标和具体实现上存在差异：前者主要学习适合文本建模的子词词表与合并规则，后者主要利用替换减少数据存储空间。

## 1. 基本原理

### 1.1 为什么需要 BPE？


BPE 属于 Subword Tokenization，通过数据驱动的方式自动学习常见子词片段。

例如：

```text
lower → l o w e r
      → lo w e r
      → low e r
```

随着合并次数增加，常见的符号组合逐渐形成更大的 token。

但 BPE 学到的是**统计上高频的相邻符号组合**，不一定对应语言学意义上的词根、词缀或完整单词。

Bostrom and Durrett（2020）指出，BPE 的贪心构造方式可能产生不符合形态学结构的子词划分，并在其实验中发现 Unigram 分词能够匹配或超过 BPE 的下游表现 [7]。

### 1.2 初始符号

BPE 的初始符号并不一定是字节。

常见实现包括：

| 类型 | 初始符号 | 说明 |
|---|---|---|
| Character-level BPE | Unicode 字符 | 实现直观 |
| Byte-level BPE | 字节（0～255） | 可覆盖任意 UTF-8 文本 |

对于 Byte-level BPE，基础字节符号通常有 256 个。通过后续合并，可以构造对应多个字节的 token [3]。

这里首先采用 **Character-level BPE** 解释训练和编码算法，再讨论 Byte-level BPE 的实现差异。


## 2. BPE 的数学定义

设经过预处理后的训练语料由符号序列集合 $S$ 组成。

每个符号序列 $s$ 对应出现频率 $f(s)$。

对于相邻符号对 $(a,b)$，其加权出现频率定义为：

```math

c(a,b)=\sum_{s\in S}f(s)\cdot n_s(a,b)
```

其中：

- $S$：当前训练语料的符号序列集合。
- $f(s)$：符号序列 $s$ 的出现次数。
- $n_s(a,b)$：相邻符号对 $(a,b)$ 在序列 $s$ 中的出现次数。
- $c(a,b)$：相邻符号对 $(a,b)$ 的总加权频率。

BPE 每一轮选择：

```math
(a^*,b^*)=\arg\max_{(a,b)}c(a,b)
```
随后将其合并为新符号：


```math
t_{\mathrm{new}}=a^*b^*
```

这是一种贪心合并策略，每轮选择当前频率最高的符号对，不回溯之前的选择 [2]。

因此，早期的合并会改变后续符号对的出现频率，最终结果具有路径依赖性。

## 3. BPE 训练（Training）

### 3.1 算法流程

BPE 训练主要包括以下步骤：

1. **初始化（Initialization）**：将训练语料拆分为初始符号序列，构建基础词表。
2. **统计频率（Pair Counting）**：计算所有相邻符号对的加权出现频率。
3. **选择符号对（Pair Selection）**：找到出现频率最高的相邻符号对。
4. **执行合并（Pair Merging）**：将其合并为一个新 token，并更新当前符号序列。
5. **更新规则（Rule Update）**：将新 token 加入词表，将合并规则追加到有序列表。
6. **重复迭代（Iteration）**：直到达到预设合并次数或其他停止条件。

### 3.2 训练伪代码

```text
Algorithm: BPE_Train(corpus, max_merges)

Input:
    corpus      : 训练语料及其频率
    max_merges  : 最大合并次数

Output:
    vocab       : token 词表
    merges      : 有序合并规则列表

1. sequences, freqs = initialize(corpus)
2. vocab = get_initial_vocab(sequences)
3. merges = []

4. for i in range(max_merges):

5.     # 统计相邻符号对的加权频率
6.     pair_freq = count_pairs(sequences, freqs)

7.     # 不存在可合并的符号对
8.     if not pair_freq:
9.         break

10.    # 获取当前出现频率最高的符号对
11.    best_pair = select_best_pair(pair_freq)

12.    # 合并所有非重叠匹配位置
13.    sequences = merge_pair(sequences, best_pair)

14.    # 更新词表
15.    new_token = concat(best_pair)
16.    vocab.add(new_token)

17.    # 保存合并规则及其顺序
18.    merges.append(best_pair)

19. return vocab, merges
```

其中：

| 数据结构 | 作用 |
|---|---|
| `sequences` | 保存当前语料的符号序列 |
| `freqs` | 保存各序列的出现频率 |
| `pair_freq` | 保存相邻符号对的加权频率 |
| `vocab` | 保存基础 token 和新 token |
| `merges` | 保存有序合并规则 |

同频时应采用确定性的选择规则，以保证训练结果可复现。

### 3.3 停止条件

BPE 常见停止条件包括：

- 合并次数达到 `max_merges`。
- 词表达到目标 `vocab_size`。
- 不存在可合并的相邻符号对。
- 最高频符号对低于 `min_frequency`。
- 达到其他预设约束，例如 token 最大长度。

`max_merges` 与 `vocab_size` 并不是相同的概念。

设初始词表大小为 $V_0$，实际成功执行 $K$ 次合并。

如果每次合并都引入一个此前不存在的 token，则：

```math
|V_{\mathrm{final}}|=V_0+K
```
如果还需要添加 $R$ 个不与普通 token 重复的特殊 token，则：

```math
|V_{\mathrm{final}}|=V_0+K+R
```
这是理想情况下的词表规模关系。实际实现还需要考虑重复 token、特殊 token 和提前停止等问题。

## 4. BPE 训练示例

假设训练语料为：

```text
low     × 5
lower   × 2
```

### 4.1 初始化

将单词拆分为字符序列：

```text
l o w       × 5
l o w e r   × 2
```

初始词表：

```text
vocab = {l, o, w, e, r}
```

### 4.2 第 1 轮合并

统计相邻符号对：

| 符号对 | 出现频率 |
|---|---:|
| `(l, o)` | 7 |
| `(o, w)` | 7 |
| `(w, e)` | 2 |
| `(e, r)` | 2 |

`(l, o)` 与 `(o, w)` 的出现频率相同。

假设同频时按照字典序升序选择，优先合并：

```text
(l, o) → lo
```

合并后：

```text
lo w       × 5
lo w e r   × 2
```

更新词表：

```text
vocab = {l, o, w, e, r, lo}
```

### 4.3 第 2 轮合并

重新统计相邻符号对：

| 符号对 | 出现频率 |
|---|---:|
| `(lo, w)` | 7 |
| `(w, e)` | 2 |
| `(e, r)` | 2 |

最高频符号对为 `(lo, w)`。

执行合并：

```text
(lo, w) → low
```

更新语料：

```text
low       × 5
low e r   × 2
```

假设只执行两轮合并，则最终词表为：

```text
vocab = {
    l, o, w, e, r,
    lo, low
}
```

有序合并规则为：

```text
merges = [
    (l, o),    # rank = 0
    (lo, w)    # rank = 1
]
```

**训练完成后，不仅需要保存词表，还需要保存合并规则及其先后顺序。**

## 5. BPE 编码（Encoding）

训练阶段根据相邻符号对的出现频率学习合并规则，而编码阶段不再重新统计频率，而是根据学习到的 **Merge Rank** 执行合并。

Berglund and van der Merwe（2023）对实际 BPE tokenizer 的规则语义进行了形式化研究，并分析了不同合并规则定义与编码行为之间的关系 [5]。

### 5.1 Merge Rank

假设训练得到：

```text
merges = [
    (l, o),
    (lo, w)
]
```

将其转换为：

```python
merge_rank = {
    ("l", "o"): 0,
    ("lo", "w"): 1
}
```

`rank` 越小，合并规则优先级越高。

### 5.2 编码流程

1. 将输入片段转换为初始符号序列。
2. 查找当前相邻符号对中可以应用的规则。
3. 选择 `rank` 最小的合并规则。
4. 执行合并并更新序列。
5. 重复执行，直到没有可应用的规则。
6. 将结果映射为 token ID。

### 5.3 编码伪代码

```text
Algorithm: BPE_Encode(text, merges)

Input:
    text    : 待编码文本片段
    merges  : 有序合并规则

Output:
    tokens  : 编码后的 token 序列

1. tokens = initialize_symbols(text)
2. merge_rank = build_rank(merges)

3. while True:

4.     candidates = get_valid_pairs(tokens, merge_rank)

5.     if not candidates:
6.         break

7.     best_pair = min_rank_pair(candidates)

8.     tokens = merge_pair(tokens, best_pair)

9. return tokens
```

### 5.4 编码示例

输入：

```text
lower
```

初始符号序列：

```text
l o w e r
```

按照训练得到的合并规则：

```text
Step 1: (l, o) → lo

lo w e r

Step 2: (lo, w) → low

low e r
```

最终分词结果：

```text
["low", "e", "r"]
```

### 5.5 BPE 与最长匹配的区别

BPE 编码并不是简单地在词表中寻找最长的 token，而是按照训练得到的合并规则优先级执行合并 [5]。

例如：

```text
rank 0: (a, b)  → ab
rank 1: (b, c)  → bc
rank 2: (a, bc) → abc
```

对于输入：

```text
a b c
```

首先执行：

```text
(a, b) → ab
```

得到：

```text
ab c
```

此时不存在 `(ab, c)` 对应的规则，因此停止。

结果为：

```text
["ab", "c"]
```

虽然词表中包含 `abc`，但不会直接将其作为最终结果。

这说明 **BPE 的分词结果不仅由最终词表决定，还由合并规则及其优先级决定。**

## 6. BPE 的边界情况

### 6.1 重叠符号对（Overlapping Pairs）

考虑：

```text
a a a
```

相邻符号对 `(a, a)` 出现两次：

```text
位置 (0, 1)
位置 (1, 2)
```

因此：

```text
count(a, a) = 2
```

但本轮合并时，如果采用从左到右的非重叠匹配：

```text
a a a → aa a
```

实际只执行一次合并。

因此：

**符号对的统计频率，不一定等于本轮实际合并的次数。**

该问题会直接影响增量统计的正确性。

### 6.2 同频处理（Tie-breaking）

当多个符号对具有相同最高频率时，需要明确处理策略。

常见方式：

- 根据符号对字典序。
- 根据首次出现顺序。
- 根据预定义的符号 ID 顺序。

为了使基础实现与优化实现可比较，建议采用统一且确定性的规则：

```text
优先级 1：频率降序
优先级 2：符号对字典序升序
```

### 6.3 词边界（Word Boundaries）

Pre-tokenization 决定 BPE 允许在哪些范围内进行合并。

例如：

```text
hello world
```

如果预切分结果为：

```text
["hello", "world"]
```

则通常不会跨越两个片段的边界执行 BPE 合并。

SentencePiece 采用了不同于传统预切词方案的设计，可以将空格信息编码进文本表示，并直接从原始句子学习子词模型 [4]。

因此，预切分方式不仅影响编码结果，也影响训练阶段能够学习的合并规则。

### 6.4 未知字符（Unknown Symbols）

Character-level BPE 的初始字符表通常来自训练语料。

当输入包含未出现过的字符时，需要额外的未知字符处理策略。

Byte-level BPE 则可以将任意 UTF-8 文本表示为字节序列，从而避免因未见过某个 Unicode 字符而无法表示输入的问题 [3]。

### 6.5 可逆性（Round-trip）

理想情况下，编码与解码应满足：

```math
\mathrm{decode}(\mathrm{encode}(x))=x
```
但该性质依赖具体实现。

如果文本经过不可逆规范化，例如统一转换为小写，则可能无法恢复原始输入。

对于 Byte-level BPE，还需要确保解码时正确拼接 token 对应的字节序列，然后再执行 UTF-8 解码。

单个 token 对应的字节序列不一定构成完整的 UTF-8 字符。