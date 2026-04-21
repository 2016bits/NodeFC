# 完整检索流程

本文按当前仓库里的真实代码实现，整理从 `BM25` 初始召回开始，到文档切句、构造证据图、添加语义边，再到最终通过检索图为每条 `claim` 选出证据句子的完整流程。

对应的核心脚本主要有：

- `scripts/bm25_retrieve.py`
- `scripts/split_sentence.py`
- `scripts/construct_graph.py`
- `scripts/add_semantic_edge.py`
- `scripts/search_graph_decomposition_aware.py`
- `scripts/search_graph_decomposition_aware_modules/*.py`
- `scripts/utils/convert_id2text.py`
- `scripts/utils/construct_verify_data.py`

需要先说明一点：

- 从 `BM25` 到最终证据句的这条链路里，最终检索脚本 `search_graph_decomposition_aware.py` 还依赖一个上游的 `claim decomposition` 文件。
- 也就是说，图检索阶段默认不是直接把整条 `claim` 一次性拿去搜，而是先读取已经分解好的 `atomic_facts`，再做分事实检索和组装。
- 这份文档会把这一步也纳入整体流程里说明，但它的生成不在本文覆盖的几个脚本内。

---

## 1. 总览

整条链路可以概括成 6 个阶段：

1. 用 `BM25` 从语料库里为每条 `claim` 召回一批候选文档。
2. 把候选文档切成句子，得到句子节点 `sent_nodes`。
3. 对句子做三元组抽取，构造每条 `claim` 自己的局部证据图。
4. 再给句子节点补一层语义相似边，得到更完整的检索图。
5. 读取 `claim decomposition`，按 `atomic fact` 的依赖顺序，在图上做分事实检索、修复和重排。
6. 把各个 `fact` 的候选句组装成最终 `top_evidences`，也就是这条 `claim` 的证据句集合。

从数据流上看，核心中间文件通常是：

- `bm25_[SPLIT].json`
- `bm25_sentnodes_[SPLIT].json`
- `bm25_nodes_[SPLIT].json`
- `bm25_edges_[SPLIT].json`
- `bm25_semantic_edges_[SPLIT].json`
- `nodefc_decomposition_aware_*.json`

---

## 2. 第一步：BM25 初始召回

脚本：

- `scripts/bm25_retrieve.py`

典型命令：

```bash
python scripts/bm25_retrieve.py --split dev --topk 10
```

### 2.1 输入

脚本默认读取：

- `--in_path /mnt/data/yangjun/data/HOVER/data/converted_data/[SPLIT]_full.json`

每条样本至少包含：

- `id`
- `claim`
- `gold_evidence_list`
- `label`
- `num_hops`

同时需要一个 Pyserini/Lucene 索引：

- `--index_path /mnt/data/yangjun/data/HOVER/corpus/index`

### 2.2 检索逻辑

脚本做的事情很直接：

- 初始化 `LuceneSearcher(args.index_path)`
- 设置 `BM25` 参数为 `k1=0.9, b=0.4`
- 用整条 `claim` 作为查询
- 对每条 `claim` 取 `topk` 个命中文档
- 把每个命中的 `docid / score / text(contents)` 保存下来

也就是说，这一步还没有图结构，只有“claim -> topk 文档”的粗召回结果。

### 2.3 输出

输出文件默认是：

- `./data/plan2/top[K]/bm25_[SPLIT].json`

README 里的简化版一般会写成：

- `data/bm25_[SPLIT].json`

单条结果结构大致是：

```python
{
    "id": ...,
    "claim": ...,
    "gold_evidence": ...,
    "label": ...,
    "num_hops": ...,
    "retrieved_docs": [
        {
            "docid": ...,
            "score": ...,
            "text": ...
        }
    ]
}
```

这一步的作用是先把检索范围从整个语料库缩小到一小批候选文档。

---

## 3. 第二步：把候选文档切成句子节点

脚本：

- `scripts/split_sentence.py`

典型命令：

```bash
python scripts/split_sentence.py --split dev
```

### 3.1 输入

默认读取：

- `data/bm25_[SPLIT].json`

也就是上一步的 BM25 输出。

### 3.2 切句逻辑

这一步的目标是把文档级候选变成句子级候选。

脚本做了几件事：

- 对每篇召回文档取出 `docid`、`score`、`text`
- 调用 `split_into_sentences(title, text)`
- 先把文档标题 `title/docid` 作为第一个“句子”加入
- 再用 `nltk.sent_tokenize(text)` 切正文
- 做一次空白归一化
- 过滤掉长度太短的句子，阈值是 `len(s) > 10`

这意味着：

- 图里的句子节点不只包含正文句，也包含“标题句”
- 后续如果标题本身很有辨识度，它也可能进入检索候选

### 3.3 输出

默认输出：

- `data/bm25_sentnodes_[SPLIT].json`

每条 `claim` 输出一组 `sent_nodes`：

```python
{
    "id": ...,
    "claim": ...,
    "label": ...,
    "sent_nodes": [
        {
            "sid": "s{qid}_{sid_counter}",
            "docid": title,
            "doc_rank": rank,
            "doc_score": score,
            "sent_idx": sent_idx,
            "sentences": sent
        }
    ]
}
```

这里的关键字段是：

- `sid`：句子节点 id，后面所有图检索基本都围绕它转
- `doc_rank`：这句话来自 BM25 的第几个文档
- `doc_score`：原始文档 BM25 分数
- `sent_idx`：句子在该文档内的位置

这一步之后，整个系统的基本检索单位已经从“文档”变成“句子”。

---

## 4. 第三步：构造局部证据图

脚本：

- `scripts/construct_graph.py`

典型命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/construct_graph.py --split dev
```

### 4.1 输入

默认读取：

- `./data/bm25_sentnodes_[SPLIT].json`

### 4.2 核心思路

这一步会为每条 `claim` 单独构建一个局部异构图，而不是构建一个全局图。

图的来源是：

- 句子节点已经有了
- 现在再从每个句子里抽取三元组
- 根据三元组构造实体节点、关系节点和各种连边

### 4.3 三元组抽取

脚本内部使用的是：

- `Babelscape/rebel-large`

具体做法：

- 对每个 `sent_node["sentences"]` 做文本归一化
- 批量送入 `REBEL` 模型
- 解析输出中的三元组 `(head, relation, tail)`
- 对头实体、关系、尾实体做规范化
- 过滤坏 span，比如包含 `<triplet>`、`<subj>` 这种异常输出

### 4.4 节点构造

对一条 `claim` 内的所有句子，脚本会去重构建：

- `entity_nodes`
- `relation_nodes`

去重方式是按规范化后的 `norm` 字段做哈希。

所以这里的图是“样本内局部图”：

- 同一条 `claim` 的不同句子共享一套实体节点和关系节点
- 但不同 `claim` 之间的图互相独立

### 4.5 边构造

脚本会构造三类边：

1. `sn_edges`
   - 句子到实体：`sid -> eid`
   - 一个句子里只要出现某个头实体或尾实体，就连一条边

2. `sr_edges`
   - 句子到关系：`sid -> rid`
   - 一个句子里只要抽到某个关系，就连一条边

3. `nrn_edges`
   - 实体-关系-实体结构：`head_eid -> rid -> tail_eid`
   - 本质上保留了三元组结构

### 4.6 输出

输出成两个文件：

- `./data/bm25_nodes_[SPLIT].json`
- `./data/bm25_edges_[SPLIT].json`

节点文件结构：

```python
{
    "id": ...,
    "sent_nodes": ...,
    "entity_nodes": [{"eid": ..., "name": ..., "norm": ...}],
    "relation_nodes": [{"rid": ..., "name": ..., "norm": ...}]
}
```

边文件结构：

```python
{
    "id": ...,
    "sn_edges": [{"sid": ..., "eid": ...}],
    "sr_edges": [{"sid": ..., "rid": ...}],
    "nrn_edges": [{"head_eid": ..., "rid": ..., "tail_eid": ...}]
}
```

这一步完成之后，系统已经不再只是“句子列表”，而是有了：

- 句子层
- 实体层
- 关系层
- 三元组结构层

也就是后面异构图检索的主体。

---

## 5. 第四步：给句子补语义边

脚本：

- `scripts/add_semantic_edge.py`

典型命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/add_semantic_edge.py --split dev
```

### 5.1 输入

默认读取：

- `./data/bm25_nodes_[SPLIT].json`

### 5.2 核心思路

前一步的图能表达“结构连接”，但还缺少“语义近邻”。

所以这一步会在句子节点之间再加一层 `S-S` 语义边。

### 5.3 具体做法

脚本内部做的是：

- 用 `sentence-transformers/all-MiniLM-L6-v2` 编码当前样本里的全部句子
- 向量做 `normalize_embeddings=True`
- 对每条 `claim` 的句子集合单独建一个 `HNSW` 索引
- 对每个句子找 `topk+1` 个最近邻
- 跳过自己
- 计算相似度 `sim = 1 - cosine_distance`
- 去重后保留无向边
- 只保留 `sim >= min_sim` 的边，默认阈值 `0.25`

也就是说，这里不是全语料级的语义图，而是“每条 claim 的候选句内部的局部语义图”。

### 5.4 输出

默认输出：

- `./data/bm25_semantic_edges_[SPLIT].json`

结构为：

```python
{
    "id": ...,
    "semantic_edges": [
        {"sid1": ..., "sid2": ..., "sim": ...}
    ]
}
```

到这里，针对每条 `claim`，我们手里已经有了一个完整的局部检索图：

- 句子节点 `S`
- 实体节点 `N`
- 关系节点 `R`
- 句子-实体边 `S-N`
- 句子-关系边 `S-R`
- 实体-关系-实体边 `N-R-N`
- 句子-句子语义边 `S-S`

---

## 6. 第五步之前的前置条件：claim decomposition

最终检索脚本：

- `scripts/search_graph_decomposition_aware.py`

除了图文件之外，还会读取：

- `--decomposition_path ./data/plan4.2/dev_2_decomposed_0_4000.json`

这说明最终检索不是单轮 `claim -> evidence`，而是：

- 先把 `claim` 拆成若干 `atomic_facts`
- 再按依赖关系逐个 `fact` 检索
- 最后再把这些 `fact` 的证据句装配回去

根据代码的使用方式，这些 `atomic_facts` 至少会包含：

- `id`
- `text`
- `rely_on`
- `critical`
- `constraint`

其中：

- `rely_on` 表示这个 `fact` 依赖哪些上游 `fact`
- `critical` 表示这个 `fact` 是否更关键
- `constraint` 用于编码数字、时间、数量等额外约束

---

## 7. 第五步：在检索图上做 decomposition-aware 检索

脚本：

- `scripts/search_graph_decomposition_aware.py`

典型命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/search_graph_decomposition_aware.py \
  --split dev \
  --decomposition_path ./data/plan4.2/dev_2_decomposed_0_4000.json \
  --nodes_path ./data/plan1/bm25_nodes_[SPLIT].json \
  --edges_path ./data/plan1/bm25_edges_[SPLIT].json \
  --semantic_edges_path ./data/plan1/bm25_semantic_edges_[SPLIT].json \
  --out_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000.json \
  --device cuda
```

这一阶段是整条流程最核心的部分。

---

## 8. 单条 claim 进入图检索前，会先做什么

### 8.1 读取并对齐图数据

主脚本会把三类文件按 `id` 对齐：

- `id2node`
- `id2edge`
- `id2semantic`

如果某条 `claim` 缺少节点或边数据，会直接跳过。

### 8.2 构建上下文 `context`

函数：

- `build_example_context`

这一步把原始节点和边转成方便检索时使用的索引结构，例如：

- `sid_list`
- `sent_texts`
- `sid2meta`
- `sid2eids`
- `sid2rids`
- `sid2keywords`
- `sid2numbers`
- `sid2time_tokens`
- `sid2quantity_tokens`
- `eid2name / eid2norm`
- `rid2name / rid2norm`

它的作用是让后面的各种打分函数可以快速查：

- 某句子有哪些实体
- 某句子有哪些关系
- 某句子有没有数字、时间、数量词
- 某句子来自哪个文档、在文档里排第几句

### 8.3 构建 sentence bank

函数：

- `encode_sentence_bank`

做法：

- 用 `SentenceTransformer` 对当前样本的全部句子编码
- 得到一个局部的句子向量库

后续所有 dense 检索，都是在这个局部 bank 上进行，而不是回到全库重搜。

### 8.4 构建语义相似查表

函数：

- `build_semantic_sim_map`

这一步把 `semantic_edges` 转成一个快速查询的字典：

- key 是 `(sid1, sid2)`
- value 是它们的最大语义相似度

后面做冗余控制、bridge 打分时都会查它。

### 8.5 构建最终异构检索图

函数：

- `build_hetero_graph`

图里的节点类型是：

- `S::{sid}`
- `N::{eid}`
- `R::{rid}`

图里的边包括：

- `S <-> N`
- `S <-> R`
- `N <-> R <-> N`
- `S <-> S`

其中：

- `S-S` 边来自前面构造好的语义边
- 权重由 `w_sn / w_sr / w_nrn / w_ss` 控制

这就是后面 `PPR` 扩展真正运行的“最终检索图”。

---

## 9. 先给整条 claim 生成全局入口

虽然最终是按 `atomic fact` 检索，但主脚本会先为整条 `claim` 生成一份全局入口，用作兜底和补充。

### 9.1 claim 句子入口 `claim_entry_s`

函数：

- `semantic_entry_from_bank`

做法：

- 用 biencoder 编码整条 `claim`
- 和当前样本的所有句子向量做点积相似度
- 取前 `claim_entry_k`

得到的是：

- 哪些句子从语义上最像整条 `claim`

### 9.2 claim 实体入口 `claim_entry_n`

函数：

- `entity_entry_n`

做法：

- 用 `spaCy NER` 从 `claim` 中抽实体
- 和图里的实体节点做规范化匹配
- 再用字符串包含关系做一层补充匹配

得到的是：

- 哪些实体节点最可能是整条 `claim` 的核心入口

---

## 10. 对 atomic facts 做拓扑排序

函数：

- `topological_sort_facts`
- `build_fact_graph_stats`

由于 `atomic_facts` 带有 `rely_on` 依赖关系，所以主脚本不会乱序检索，而是：

- 先拓扑排序
- 保证父 `fact` 先检索
- 子 `fact` 后检索

同时还会计算一批图统计量：

- `depth_map`
- `max_depth`
- `critical_count`
- `fact_count`
- `children`

这些统计量后面会影响：

- `fact` 的角色判断
- 候选预算
- bridge helper 预算
- 最终组装时的文档预算

---

## 11. 第六步：对每个 atomic fact 做局部检索

核心函数：

- `retrieve_one_fact`

这一步是“最终检索图得到 claim 证据句子”的真正主体。

系统不是一次选句，而是先为每个 `fact` 找到一小批高质量候选句，再在最后统一组装。

---

## 12. 每个 fact 先构建自己的画像 `fact_profile`

函数：

- `build_fact_profile`

对每个 `fact`，系统会抽取：

- `keywords`
- `salient_keywords`
- `entity_surface_keywords`
- `relation_keywords`
- `numbers`
- `time_tokens`
- `quantity_tokens`
- `constraint_text`
- `negation_mode`
- `entry_n`
- `entry_r`

这里的含义是：

- `entry_n`：这个事实关心哪些实体
- `entry_r`：这个事实关心哪些关系
- `constraint_text`：如果这个事实带数字、时间、数量约束，就把约束转成检索信号

---

## 13. 给每个 fact 判定角色

函数：

- `infer_fact_role`

角色分为三类：

1. `anchor`
   - 这类事实含有明显数值、时间、数量约束

2. `verify`
   - 通常是关键事实、叶子事实，要求句子对事实本身提供直接支持

3. `bridge`
   - 更偏向链路连接，负责把父事实和子事实串起来

角色不同，后面的候选预算、排序权重、覆盖条件都会不同。

---

## 14. 为每个 fact 构造入口点

当前实现不是只用一种召回，而是融合多种信号。

### 14.1 句子入口 `entry_s`

`entry_s` 由四部分加权融合：

1. `base_entry_s`
   - 用 `fact["text"]` 在句子 bank 上做 dense 检索

2. `lexical_entry_s`
   - 遍历全部句子，按以下特征打分
   - 实体匹配
   - 关系匹配
   - 关键词重合
   - target match
   - constraint 一致性
   - 否定匹配
   - binding 匹配
   - 与父事实支持集的 bridge 特征

3. `constraint_entry_s`
   - 专门处理数字、时间、数量约束
   - 同时结合词面匹配和 constraint 的 dense 相似度

4. `dependency_entry_s`
   - 如果有父事实，或者当前是 `bridge fact`
   - 则额外强化那些更容易与父事实支撑链路闭合的句子

最后通过 `merge_score_maps` 融合成真正的 `entry_s`。

### 14.2 实体入口 `entry_n`

实体入口来自两部分：

- 当前 `fact_profile["entry_n"]`
- 父事实支撑集中传播下来的 dependency seed entities

### 14.3 关系入口 `entry_r`

关系入口也来自两部分：

- 当前 `fact_profile["entry_r"]`
- 父事实支撑集中传播下来的 dependency seed relations

这样设计的意义是：

- 当前事实不会只看自己
- 还会继承已经找到的上游证据，形成真正的 dependency-aware 检索

---

## 15. 候选预算是动态的

`retrieve_one_fact` 里并不是固定取一个 `topk`。

候选预算会根据这些因素动态变大：

- 当前事实是不是 `critical`
- 它在 DAG 里的深度
- 它是不是 `bridge fact`
- 它是否依赖多个父事实
- 整条 `claim` 的依赖链是不是足够深

所以复杂链路上的事实，会拿到更大的候选搜索空间。

---

## 16. 初始候选召回

函数：

- `select_sentence_candidates`
- `filter_anchor_candidates`

从 `entry_s` 排序结果里取出候选句时，系统会把多种 component score 一起保留下来：

- `recall_score`
- `dense_score`
- `lexical_score`
- `constraint_score`
- `dependency_score`
- `ppr_score`

如果角色是 `anchor`，还会先做一次约束一致性过滤，尽量把数字和时间完全不匹配的句子提前剔掉。

---

## 17. Cross-Encoder 重排

函数：

- `build_fact_rerank_query_text`
- `rerank_cross_encoder`

系统不是直接拿 `fact["text"]` 去重排，而是先构造一个更适合当前事实的 rerank query。

### 17.1 verify / anchor

会优先使用：

- 实体名
- 关系词
- `fact text`
- `constraint text`

### 17.2 bridge 或有父事实的 fact

会额外把父事实提供的：

- 实体线索
- 关系线索

也拼到 query 里。

这一步的目标是让 reranker 更清楚：

- 当前事实要验证什么
- 当前事实需要连接哪条已有证据链

---

## 18. 候选精排：把“像不像”变成“能不能支持这个 fact”

函数：

- `enrich_fact_candidates`
- `compute_direct_support_pass`

对每个重排后的候选句，系统会再算一大批细粒度特征，包括：

- `semantic_relevance`
- `entity_pair_score`
- `relation_match_score`
- `keyword_overlap`
- `time_quantity_consistency`
- `negation_compatibility`
- `context_independence`
- `background_penalty`
- `binding_score`
- `bridge_score`
- `doc_rank_bonus`

然后得到三类关键分数：

1. `direct_support_score`
   - 这个句子能不能直接支撑当前事实

2. `bridge_support_score`
   - 这个句子能不能帮助把当前事实和父事实的证据链串起来

3. `fact_score`
   - 这个句子对当前事实整体有多好

同时还会打上：

- `support_type = direct_support / bridge_support / candidate`

这一步之后，系统拿到的已经不是普通“相关句”，而是“对某个事实承担什么作用”的候选句。

---

## 19. 覆盖状态判断

函数：

- `build_fact_coverage_summary`
- `compute_fact_coverage_status`

对每个 `fact`，系统会判断：

- 有没有 direct support
- 有没有 bridge support
- 父事实是否已经 covered
- 依赖闭包是否成立
- 当前 fact 是否 covered
- 当前 fact 是否 fully covered

这里有一条非常关键的规则：

- `verify`、`anchor`、`critical fact` 默认要求 direct support
- `bridge fact` 允许主要靠 bridge support
- 但只要涉及依赖关系，就还要看 dependency closure 是否闭合

也就是说：

- 不是“句子相关”就算成功
- 必须在依赖链里真的站得住

---

## 20. 如果第一次没找到合适证据，会进入 repair

当前实现有两种 repair。

### 20.1 Direct Repair

函数：

- `build_targeted_direct_repair_candidates`

触发条件：

- 当前 `fact` 需要 direct support
- 但当前候选里还没有足够好的 direct support

做法：

- 构造更聚焦的 direct query
- 再做一次 lexical recall
- 再做一次 dense recall
- 如果有约束，再叠加 constraint recall
- 然后重新 rerank 和精排

### 20.2 Bridge Repair

函数：

- `build_targeted_bridge_repair_candidates`
- `build_chain_completion_entries`
- `ppr`

触发条件：

- 当前 fact 和父事实之间的链路还没有真正闭合
- 或者 bridge fact 还没有形成有效连接

它的做法更像“在图上补链路”：

1. 从父事实的支撑结果里提取 dependency seeds
2. 把父句子、父实体、父关系、binding 要求、高分候选一起变成扩展入口
3. 在最终异构检索图上运行 `PPR`
4. 用 `lexical + dense + PPR` 三路信号重新召回桥接句
5. 再做一次 cross-encoder 重排和精排

这里的 `PPR` 是整条流程里“图检索”最直接的体现：

- 入口点不再只来自当前 fact 文本
- 还来自父事实已经找到的证据链
- 分数会沿着 `S-N-R-S` 和 `S-S` 语义边传播
- 从而把真正能补全链路的句子抬上来

---

## 21. 每个 fact 的输出是什么

`retrieve_one_fact` 最终会为每个 `fact` 返回：

- `entry_s`
- `entry_n`
- `entry_r`
- `fact_profile`
- `coverage_summary`
- `support_profile`
- `candidates`
- `expanded`

其中最重要的几个字段是：

- `coverage_summary`
  - 说明当前事实有没有被覆盖，缺什么，还需不需要 repair

- `support_profile`
  - 把当前事实已经找到的支撑句、实体、关系概括出来
  - 供子事实继续使用

- `candidates`
  - 当前事实最终保留下来的 top 候选句

---

## 22. 第七步：把所有 fact 的候选句组装成最终证据句集合

核心函数：

- `aggregate_top_evidences`

这一步不是简单地做全局 `top-k`，而是做一个 coverage-driven assembly。

目标是：

- 尽量覆盖更多事实
- 尽量保证依赖链闭合
- 控制文档数和冗余
- 最终输出一组真正可用的证据句

---

## 23. 先构造全局 sentence pool

函数：

- `build_sentence_support_pool`

它会把所有 `fact` 的候选句合并成一个全局池子，来源包括：

- 每个 `fact` 的 `top candidates`
- `coverage_summary` 里的 `direct_winners`
- `coverage_summary` 里的 `bridge_winners`
- `coverage_summary` 里的 `direct_candidates`
- `coverage_summary` 里的 `bridge_candidates`

对每个句子，pool 里会保留：

- 它支持哪些 `fact`
- 对每个 `fact` 的支持细节
- 最好的 `aggregate_score`
- 最好的 `fact_score`

这一步相当于把“分事实候选”压成“全局候选池”。

---

## 24. 动态文档预算

函数：

- `compute_dynamic_doc_budget`

最终选句时，系统并不想无限制地跨文档拼句子，所以会先算一个动态文档预算。

预算会随着这些因素增加：

- `fact_count` 更大
- `max_depth` 更深
- `critical_count` 更多
- 候选句覆盖到的文档数更多

这样复杂 claim 会允许更多文档参与，而简单 claim 会被限制得更紧。

---

## 25. Hard Two-Stage Selection

`aggregate_top_evidences` 的第一阶段是按拓扑顺序做硬选择。

### 25.1 对 verify / anchor fact

策略是：

- 优先选 direct winners
- 如果和父事实之间链路还不闭合，再补 bridge helper
- 只有当前事实真的 `covered`，才会把它记为已完成

### 25.2 对 bridge fact

策略是：

- 不要求 direct winner
- 直接从 bridge candidates 里选 primary bridge sentence
- 如果还不够，再补 helper bridge sentences

换句话说，组装阶段仍然保留“事实角色”这个概念，而不是把所有句子一视同仁。

---

## 26. 选择结果会被重新评估

函数：

- `evaluate_selected_set`

系统会对当前选中的句子集合算一个整体 `utility`，主要考虑：

- 覆盖了多少 `facts`
- 覆盖了多少 `fully covered facts`
- 覆盖了多少 `critical facts`
- dependency coverage 有多少
- cross-document bridge 有多少
- 冗余有多大
- 是否超出文档预算

所以最终选择不是“单句分数最大”，而是“整组句子放在一起是否最有用”。

---

## 27. Rescue Pass：补救未覆盖事实

函数：

- `rescue_uncovered_facts`

如果 hard two-stage 之后仍然有一些事实没被覆盖，系统会继续做补救：

- 优先考虑 `critical fact`
- 再优先更深层的 fact
- 从该 fact 的 direct/bridge 候选中试探性加入一句
- 每加入一句都重新评估整组 `utility`
- 只有增益足够大才保留
- 直到达到 `max_evidence` 或没有明显增益

这一步的意义是：

- 最终输出不是一次拍板
- 而是“先硬选，再补救”

---

## 28. 最终输出：claim 的证据句子在哪里

主检索脚本输出文件里，最重要的字段是：

- `top_evidences`

每个元素本身就已经是最终证据句，包含：

- `sid`
- `text`
- `docid`
- `doc_rank`
- `score`
- `fact_score`
- `support_type`
- `supporting_facts`
- `support_details`

也就是说：

- 如果你想直接拿“这条 claim 的最终证据句文本”，就看 `top_evidences[*].text`

除此之外，输出里还有：

- `entry_sids / entry_nids / entry_rids`
- `reranked_candidates`
- `fact_traces`
- `assembly_summary`

其中：

- `fact_traces` 能看到每个 atomic fact 最后选中了哪些句子
- `assembly_summary` 能看到整个组装阶段的覆盖情况

---

## 29. 把最终句子导出成更直观的文本结果

如果希望只保留文本级结果，可以继续跑：

```bash
python scripts/utils/convert_id2text.py \
  --in_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000.json \
  --node_path ./data/plan1/bm25_nodes_[SPLIT].json \
  --out_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000_text.json \
  --split dev
```

这个脚本会把：

- `entry_sids` 转成 `entry_semantic_texts`
- `entry_nids` 转成 `entry_entity_texts`
- `top_evidences[*].text` 汇总成 `top_evidence_texts`

因此：

- 如果你只是想看每条 `claim` 的最终证据句文本，`top_evidence_texts` 就是最方便的字段

---

## 30. 如果还要给验证器使用

可以再把最终证据句整理成验证输入：

```bash
python scripts/utils/construct_verify_data.py \
  --evidence_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000_text.json \
  --raw_path ./data/plan1/bm25_[SPLIT].json \
  --out_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000_verifying_data.json \
  --split dev \
  --max_evidence 5
```

输出里会包含：

- `claim`
- `gold_evidence`
- `retrieved_evidence`
- `num_hops`
- `label`

其中：

- `retrieved_evidence` 就是最终送去验证阶段的证据句列表

---

## 31. 一条 claim 从头到尾是怎样流动的

可以把整条链路压缩成一句更直观的话：

1. 先用 `BM25` 从全库召回一批文档。
2. 把这些文档切成句子，形成句子节点。
3. 用三元组抽取把句子、实体、关系连成一个局部异构图。
4. 再用句向量近邻把句子之间补上语义边。
5. 读取上游分解好的 `atomic facts`，按依赖顺序逐个 fact 检索。
6. 对每个 fact 融合 dense、lexical、constraint、dependency 和图传播信号找候选句。
7. 如果 direct support 不够，就做 direct repair；如果依赖链没闭合，就在图上做 PPR 驱动的 bridge repair。
8. 最后再把所有 fact 的候选句按覆盖率、依赖闭包、文档预算和冗余统一组装成 `top_evidences`。

所以，从实现上说，最终证据句并不是直接从 `BM25` 结果里“挑出来”的，而是：

- 先经过句子化
- 再经过结构化图构建
- 再经过语义边增强
- 再经过 decomposition-aware 的图检索与链路修复
- 最后才被 assembly 模块选成最终证据句

---

## 32. 一套可复现的端到端命令

如果按这条流程顺着跑，命令大致是：

```bash
python scripts/bm25_retrieve.py --split dev
python scripts/split_sentence.py --split dev
CUDA_VISIBLE_DEVICES=0 python scripts/construct_graph.py --split dev
CUDA_VISIBLE_DEVICES=0 python scripts/add_semantic_edge.py --split dev
CUDA_VISIBLE_DEVICES=0 python scripts/search_graph_decomposition_aware.py \
  --split dev \
  --decomposition_path ./data/plan4.2/dev_2_decomposed_0_4000.json \
  --nodes_path ./data/plan1/bm25_nodes_[SPLIT].json \
  --edges_path ./data/plan1/bm25_edges_[SPLIT].json \
  --semantic_edges_path ./data/plan1/bm25_semantic_edges_[SPLIT].json \
  --out_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000.json \
  --device cuda
python scripts/utils/convert_id2text.py \
  --in_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000.json \
  --node_path ./data/plan1/bm25_nodes_[SPLIT].json \
  --out_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000_text.json \
  --split dev
python scripts/utils/construct_verify_data.py \
  --evidence_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000_text.json \
  --raw_path ./data/plan1/bm25_[SPLIT].json \
  --out_path ./data/plan4.2/nodefc_decomposition_aware_dev_0_4000_verifying_data.json \
  --split dev \
  --max_evidence 5
```

如果你当前只关心“claim 最终对应哪些证据句”，最直接看两个位置：

- `nodefc_decomposition_aware_*.json` 里的 `top_evidences[*].text`
- `*_text.json` 里的 `top_evidence_texts`

---

## 33. 最后一句总结

当前仓库里的完整检索流程，本质上是一个：

“先用 BM25 缩小文档范围，再把候选文档转成句子级异构证据图，接着补上句间语义边，最后在 decomposition-aware 的图检索与组装框架中，为每个 atomic fact 找到 direct support 或 bridge support，最终合成为 claim 的证据句集合”的多阶段检索系统。
