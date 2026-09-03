# RAG 选型现状与优化分析

> 记录日期：2026-09-03。基于对代码的全面探查 + 金标评估最新报告（`rag_eval/report_gold_qfixed.json`）。
> 结论部分结合了项目历史实测记录（多项改动已被受控实验否决，见文末）。

---

## 一、当前选型清单

### 1. 向量数据库

| 链路 | 选型 | 出处 |
|---|---|---|
| 主链路 | LlamaIndex 默认 **SimpleVectorStore**（JSON 落盘，无独立向量库，105MB 向量文件全内存） | `rag_llamaindex.py:665` |
| 旧链路 | 本地 **FAISS IndexFlatIP**（精确内积） | `durian__inference_api.py:1857` |
| 外部 | RAGFlow（HTTP API 走隧道，dataset `7c684b1e8b9711f1bc9dddf1410424b0`） | `.env:48-55` |

### 2. 向量索引算法

两条自研链路都是**暴力精确检索**（Flat / SimpleVectorStore），无 HNSW/IVF，无 ANN 参数。

### 3. 分块策略

- 源管道 `rag_rechunk.py:39-41`：**900 字 / overlap 150**，按句边界切（优先 `。.` `；;`，退回 `，,`/空格）
- 特例：2024 版马来文手册 `MY_DOA_Pakej_Teknologi_Durian_2024` → **300/50**（信息密度高，切细）
- 进索引再切一刀：`SentenceSplitter` **512/80**（`rag_llamaindex.py:556`）
- 表格 / qa_pair / 泰文双语**整块绕过切分器**；表格截 2400 字符
- 泰文 legacy chunk 入库做**中文译文 + 泰文原文双语拼接**（`rag_llamaindex.py:594-605`）

### 4. 分块数量

- 索引 **8375 nodes**（text 7322 / qa_pair 1000 / table 41 / table_broken 12）
- 源 chunks：`rag_build/chunks.jsonl` 8351 行
- 旧 FAISS 链路仅 337 chunks / 6 个 PDF（已过时）

### 5. 召回算法

- **自研 hybrid**（`rag_llamaindex.py:1180-1314`）：
  - dense：`similarity_top_k = max(12, candidate_limit)`
  - BM25：自实现全量内存扫描，**k1=1.5, b=0.75**；分词为拉丁/数字词 + 中文 2-3gram + **泰文 3-4gram**
  - candidate_limit = `max(36, top_k×8)`，最终 top_k=5
- **泰文查询整体跳过 dense**（判据：泰文字符数 ≥ 中文×3，`rag_llamaindex.py:1190`）——known-item 口径实测 Hit@3 +4.3pp，McNemar p<0.0001
- 后过滤：`evidence_matches_query`（min_score 0.42）或 `bm25 > 1.2`；填空模式走邻居扩展（±1）+ BM25 权重 1.35 + 锚点覆盖 ≥0.6
- API 侧参数：top_k=6, candidate_top_k=36, min_score=0.35（`durian__inference_api.py:222-232`）

### 6. RRF 配置

- **k = 60.0**，dense/BM25 权重均 1.0（`rag_llamaindex.py:1194,1223`）
- 去重键是**归一化文本前 900 字符**（`re.sub(r"[\W_]+","",…)`），**不是 node_id**（`:809,1198`）
- 已两次踩坑并记录在案：
  1. 只改单侧文本 → 同一 node 在两路裂成两条，RRF 分数劈半掉出 Top-K
  2. 近似文本互相碰撞 → 不同 node 共享键被合并，静默不可达（修复带来中文口径 Hit@3 +13.8pp）

### 7. 向量化模型

| 用途 | 模型 | 维度 | 多语言 | 备注 |
|---|---|---|---|---|
| 主 embedding | **BAAI/bge-small-zh-v1.5**（本地 CPU，4 层） | 512 | 否 | 泰文 token 中位 66.7% UNK |
| 旧链路 | e5-small | 384 | 英文 | `model_e5/` |
| Rerank | **BAAI/bge-reranker-v2-m3**（XLM-R 24 层） | — | 多语言 | **只接在 API 侧**，评估主链路未用 |
| 翻译备用 | opus-mt-th-en（本地 308MB） | — | 泰→英 | 仅离线管道+备用 |

### 8. 查询侧处理

- 语种检测：泰文判据"泰文字符数 ≥ 中文×3"；zh/th/ms/en 四语判别器
- 泰文查询 → 跳过 dense 通路
- 术语别名扩展：`glossary.json` 330 条词条（14 个 section）
- 意图检测：code/disease/variety/planting 四类规则，影响后过滤阈值
- 填空/原文精确模式：BM25 加权 1.35 + 邻居扩展 + 锚点覆盖
- 查询侧泰文声调符号修复 `normalize_thai`
- 查询翻译：**已被受控实验全面否决**（见文末）

### 9. 当前指标（2026-09-03，top_k=5）

| 口径 | Hit@1 | Hit@3 | Hit@5 | nDCG@5 |
|---|---|---|---|---|
| 全量（1247 条） | 72.01% | 78.11% | 79.63% | 0.7631 |
| 核心口径（987 条，可达∩同语种） | 89.36% | 96.45% | 97.87% | 0.9421 |

分语种 Hit@3（同语种金标）：en 100%（108）/ zh 98.18%（604）/ th 97.84%（139）/ **ms 82.73%（139，短板）**

可达率 97.11%（1211/1247）；独立判据 Top1 垃圾率 1.28%；P50 164ms / P95 251ms。

---

## 二、更优选型分析

> 优先级排序。注意：多项"显而易见"的改进已被本项目受控实验否决（见第三节），以下只列未被否决的方向。

### 值得做

**1. 把 bge-reranker-v2-m3 接进主评估链路（最高性价比）**
模型已在本地、多语言、API 侧已验证可用。Cross-encoder rerank 是当前未启用的最大杠杆，直接瞄准 Hit@1——Hit@5 已 79.6% 而 Hit@1 仅 72.0%，说明 gap 主要在排序而非召回。代价：CPU 上 batch 4 × 16 候选，延迟需实测（当前 P95 251ms）。

**2. embedding 换 bge-m3（中期，解决结构性缺陷）**
bge-small-zh 对 th/ms 是根本性不匹配，现在靠"泰文跳 dense + 双语拼接"两个补丁绕过。bge-m3 dense+sparse 多语言输出能统一链路、删掉语种特判，主要收益落在马来文短板。代价：568M 参数 CPU 推理慢、全量重建索引、所有历史结论需重测——建议只做独立分支实验，不动生产。

**3. 去重键从文本改为 node_id**
文本去重键已造成两类静默事故（RRF 劈裂、近似碰撞吞 node）。node_id 才是正确语义。改后需全量金标回归验证（注意 `_dense_retrieve` 与 `_build_lexical_node_cache` 两侧同步的老坑不再适用，但合并行为本身会变化）。

**4. 分块管道去冗余**
900/150 切完又被 512/80 二次切，源管道的句边界和重叠实际被抵消。两个方向二选一：
- 统一为 parent-child 检索（小块检索、大块回传，LlamaIndex 原生支持）
- 砍掉一层，只保留一处切分

**5. 马来文短板专项**
ms Hit@3 82.7% 显著低于其他语种。已知 dense 降权不显著已撤回，待金标扩到 400+ 再议；reranker（第 1 项）对 ms 可能同样有效，优先验证。

### 不用动

- **索引算法/向量库**：8375 条向量，暴力精确检索就是最优解（ANN 在这个量级只损失召回不省时间）。SimpleVectorStore JSON 落盘够用。唯一问题是三条链路并存（faiss 旧链 337 chunks 已过时、RAGFlow 外部）——维护性上应考虑归一，而非换库。
- **RRF k=60**：标准值。权重调优空间已被 dense 分档实验证明很小。真要调，k 扫 10-100 是零成本脚本活，但预期边际收益 <0.5pp。
- **BM25 k1/b**：自实现 + 字符 gram 是为泰文无分词定制的，工作正常；k1/b 扫参属低优先级。

### 已被实验否决，不要再提

1. **泰文查询侧翻译**：opus-mt-th-en 离线翻译受控实验全面否决（丢词根因 + dense 侧塌缩，跨语言向量鸿沟须双侧同语）。见 `docs/RAG_多语言查询处理.md`。
2. **马来文 dense 降权**：McNemar 不显著（ns@3 p=0.0654）且最优点在 0.3/0.4/0.5 间漂移，不值得引入永久分语种分支。见 memory `project_dense_weight_per_lang.md`。

---

## 三、方法论提醒（来自项目历史教训）

- **「现状」≠「基线」**：在已上线的改动上做 A/B，基线臂必须显式还原改动，否则逐题零差异是必然。
- **评估判据与被评估对象共享实现时指标必然虚高**：任何"完美"指标都要独立判据交叉验证。
- **口径打架时**：优先选"与该语种真实使用场景对齐、且没有自命中污染"的口径。泰文用 known-item，FAQ raw 指标会被自命中严重虚高。
- **改文本规范化必须两侧同步**（`_dense_retrieve` + `_build_lexical_node_cache`），且先查 `_normalized_dedupe_key` 是否受影响（非 Mn 字符的改动才危险）。
- **before/after 必须同一索引版本**：先看 `docstore.json` mtime。
