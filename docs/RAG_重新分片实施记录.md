# RAG 重新分片：实施记录与文件清洗流程

> 执行日期：2026-09-02
> 涉及文件：`rag_rechunk.py`（新建）、`rag_llamaindex.py`（改造）、`rag_eval.py`（补独立判据）
> 前置方案：`docs/RAG_表格图片向量化方案.md`、`docs/RAG_表格图片向量化_实施流程.md`
> 评估体系：`docs/RAG_评估体系.md`

---

## 一、验收结果

| 指标 | 目标 | before | after | 判定 |
|---|---|---|---|---|
| **语料可达率** | ≥85% | 54.5% | **91.8%** | ✅ +37.3pp |
| **Top1 垃圾率**（独立判据） | ≤2% | 27% | **0.8%** | ✅ |
| 证据垃圾率（独立判据） | ≤3% | — | **0.5%** | ✅ |
| variety 类垃圾率 | ≤3% | 20.0% | **0.0%** | ✅ |
| numeric 类垃圾率 | ≤3% | 10.4% | **0.0%** | ✅ |
| P95 延迟 | ≤400ms | 252ms | 258ms | ✅ |
| 空结果率 | ≤5% | 0.0% | 0.0% | ✅ |

**核心口径纯检索能力**（可达 ∩ 同语种）：

```
Hit@1   30.8% → 77.8%   (+47.0pp)
Hit@3   61.5% → 88.9%   (+27.4pp)
Hit@5   69.2% → 88.9%   (+19.7pp)
MRR     0.481 → 0.815
```

索引组成变化：

| 分组 | before | after | 变化 | 说明 |
|---|---|---|---|---|
| 文献(MinerU) | 3262 | **5391** | +2129 | 抢救回被误丢的内容 |
| PDF | 643 | 1338 | +695 | 新采集 7 份权威 PDF |
| FAQ | 2476 | 1000 | −1476 | 问答对不再被 splitter 劈成 2.5 份 |
| 泰文 DOCX | 1287 | 1185 | −102 | 去重 |
| **合计** | 7668 | **8914** | +1246 | 索引文本总量 +20.9% |
| 含残渣 node | 1558 | **0** | — | |

---

## 二、五项改动与各自提升的指标

### 改动 C（先讲，因为它是最大功臣）：修 `is_noise_text()` 的黑名单误杀

原实现只要命中**一个**版权页特征词，就把整块 chunk 丢弃（`rag_llamaindex.py:186`）：

```python
front_matter = ["本书由", "主编", "副研究员", ..., "出版"]
if any(m in compact for m in front_matter):   # 命中 1 个就丢
    return True
```

正文里正常讨论"出版""主编"的段落全部被误杀。改为：

```python
hits = sum(1 for m in front_matter if m in compact)
if hits >= 2 and len(compact) < 220:   # 需 ≥2 个特征词，且文本短（真版权页都很短）
    return True
```

**归因实测：这一条贡献了新增可达的 110/110。** 我做了严格分解 —— 把旧索引文本套上新清洗规则，看那些"新增可达"的金标锚点能否匹配：

```
① 清洗生效（旧索引已有内容，残渣挡住匹配）:   0 条 (0%)
② 内容挽回（旧索引压根没有，被 noise 过滤丢了）: 110 条 (100%)
```

这**纠正了我原本的假设** —— 我以为可达率提升主要来自清洗让锚点能匹配，实际全部来自内容挽回。

→ 提升 **语料可达率 54.5% → 91.8%**

### 改动 A：区分「键:标量」与「键:字符串」

原 `clean_text()` 对残渣键一律只删键名、保留值，而 `bbox` 走的分支同样只删键名，导致 `'bbox': [48, 136, 63, 149]` 的**坐标数字全留在正文**。新管道拆成两条规则：

```python
_RE_KV_SCALAR   = ...  # 键:数字/布尔/None → 整对删除
_RE_KV_KEYONLY  = ...  # 键:字符串        → 只删键名，保留值（正文在这里）
```

→ 提升 **证据垃圾率**、**Top1 垃圾率**

### 改动 B：删 MinerU 的 type **值**（不只是键名）

最大的漏网。打平后 `'text'`、`'inline_equation'`、`'title'` 这些 type 的**值**变成孤立带引号字符串留在正文。实测 **62.2% 的 chunk 含这类残留**：

```
'text' 12025 次 / 'inline_equation' 2682 / 'title' 941 / 'image' 414 / 'list' 199
```

→ 提升 **垃圾率**

### 改动 D：表格/QA 整块绕过 splitter

原 `rebuild_index()` 走 `from_documents(transformations=[splitter])`，所有 Document 无差别过 `chunk_size=512`，表格必被腰斩、表头与数据行分家。新增 `load_rechunk_nodes()`：

```python
if block_type in ("table", "table_broken", "image", "qa_pair"):
    nodes.append(TextNode(text=index_text, metadata=meta))     # 整块，不切
else:
    for nd in splitter.get_nodes_from_documents([doc]): ...    # 只有正文才切
```

现在 1044 个整块 node。副作用可见：FAQ 从 2476 node 降到 1000（每个问答对不再被劈开）。

→ 提升 **核心口径 Hit@1 30.8% → 77.8%**

### 改动 E：残表只留纯 caption

表头已丢的残表不可恢复，因此**绝不让裸数值进 `index_text`**，只保留 caption，且 caption 须通过五道校验：长度 8–90、文字占比 >0.55、无引号无尖括号、必须以 `表/图/Table/Figure` + 编号开头、编号后描述 ≥8 字、数字占比 ≤0.3。

→ 提升 **variety 类垃圾率 20.0% → 0.0%**、**numeric 类 10.4% → 0.0%**

---

## 三、独立判据：为什么需要，是什么

### 起因：我的评估在循环论证

改造后评估报**垃圾率 0.0%**。我没有采信 —— `rag_eval.evidence_defect()` 的正则与 `rag_rechunk` 的清洗正则**同源**，用清洗规则检验清洗结果必然满分。

回测证实：独立判据下真实值是 **27%**。

### 设计思路

换一个维度：不问"有没有 MinerU 残渣特征"，而问"**这段文字像不像人写的句子**"。四条规则，与清洗管道零共享（`rag_eval.py:independent_defect`）：

| 规则 | 判据 | 抓什么 |
|---|---|---|
| `struct_symbol` | `[{}]` 或 `\[\s*\d+\s*,\s*\d+` | 花括号、`[48, 136` 这类坐标数组开头 |
| `stray_quotes` | 单/双引号计数 ≥4 | 孤立引号。人写的句子不会这样 |
| `number_run` | `(?:\b\d+(?:\.\d+)?\b[\s,\|]+){4,}` | 连续 4 个以上纯数字 token |
| `low_text_ratio` | 文字字符占比 <0.5 | 汉字+字母+泰文占比过低 |

**关键在于它们不检测任何 MinerU 特有关键字。** 清洗管道再怎么优化针对 `bbox`/`para_blocks` 的正则，都无法让这四条自动通过 —— 必须真的把文本变成通顺句子才行。

### 它抓到了什么

原判据说干净、独立判据说脏的，是这两类：

```
⚠️ stray_quotes   单果重 '1.6 kg' ，果壳棕黄色，果刺稀疏...
⚠️ number_run     13, 488, 409, 547 ，回填压肥时，将表土和较好的肥料放于...
```

性质与改造前不同：语义完整可读，只是**引号和坐标尾巴**，属瑕疵而非垃圾。但既然暴露就补了两条规则（`strip_quote_noise`）：

```python
_RE_QUOTED_VALUE  # 剥离 '3 kg' → 3 kg，循环 3 次处理嵌套
_RE_BARE_BBOX     # 删裸坐标；要求 ≥2 个逗号分隔整数，且后面不接单位
                  # （避免误删正文里的"3, 5, 7 年生"这类合法列举）
```

修完独立判据 27% → **0.8%**。

### 已固化

`independent_defect()` 进了评估工具，报告中标为 **★★ 最可信的负向指标**，`compare` 纳入 `ind_defect_ratio` / `ind_top1_defect_rate`。以后每次改造自动交叉验证。

> **方法论沉淀**：评估判据与被评估对象共享实现时，指标必然虚高。任何"完美"的指标都要用独立实现的判据交叉验证。这与之前"低到反常的指标先怀疑指标"是同一条原则的两面。

---

## 四、完整的文件清洗流程（项目实际实现）

入口：`python rag_rechunk.py build` → 产出 `rag_build/chunks.jsonl`

```
┌─ 来源一：clean_chunks.jsonl（9402 条历史数据）
│    └→ group_of() 按 doc 分三组
│         ├─ FAQ（1000）      ─→ 仅压缩空白，原样保留，block_type=qa_pair
│         ├─ 泰文 DOCX（651）  ─→ 原样保留 + 标记 needs_translation
│         └─ 文献（7751）      ─→ 进入抢救清洗链
│
└─ 来源二：raw/pdfs/ + rag_pdfs/（13 个 PDF）
     └→ 逐页 PyMuPDF 解析
          ├─ find_tables() → 表格转 Markdown（整块，不切分）
          ├─ get_text("blocks") → 正文（排除表格区域，防重复）
          └─ get_images() → 仅计数，图片留到下一轮做 VLM 描述
                    ↓
              全局去重 → rag_build/chunks.jsonl
```

### 4.1 文献抢救链（核心，7 步顺序执行）

`strip_residue()` 的执行顺序有讲究，不能调换：

```
① _RE_BBOX          删 {'bbox': [48,136,63,149]} 整段
② _RE_KV_SCALAR     删「键:数字/布尔」整对        ← 改动 A
③ _RE_KV_KEYONLY    删「键:字符串」的键名，保留值  ← 改动 A
④ _RE_TYPE_VALUE    删 'text' 'title' 等 type 值  ← 改动 B
⑤ _RE_BODY_WORD     删 image_body / table_body 裸词
⑥ HTML 清理          _RE_HTML_TABLE → _RE_HTML_ANY → _RE_ENTITY → _RE_URL
⑦ normalize_units() LaTeX 归一化
⑧ 标点/空白规整      _RE_STRUCT_PUNCT → _RE_ORPHAN_PUNCT → _RE_WS
⑨ strip_quote_noise() 删裸 bbox 坐标 + 剥离引号包裹  ← 独立判据发现后补
```

**为什么 ② 必须在 ③ 之前**：若先执行只删键名的 ③，`'bbox': [...]` 的值就变成裸数字留下了，② 再也无法识别它属于哪个键。

**为什么 ⑨ 放在最后**：此时结构标记已清除，剩下的引号基本都是 span 值的包裹符，可以安全剥离；提前执行会误删正文里的合法引号。

`normalize_units()` 处理的 LaTeX 残留（实测频次）：

```
mathrm 1673 / circ 316 / sim 205 / times 85 / cdot 83 / pm 70 / mu 70
```

其中有一条上下文敏感的规则：MinerU 偶尔把温度的度数符号输出成 `\%`，只在前后 28 字符内出现"温度/低温/高温/积温/摄氏/temperature"时才改成 `℃`，否则保留为百分号。这避免把真实的湿度百分比改错。

### 4.2 质量分级 `grade()`

清洗后分三档，决定后续处理路径：

| 等级 | 判据 | 处置 |
|---|---|---|
| `junk` | 清洗后 <60 字符 / 文字占比 <0.55 / 仍含残渣标记 / **数字占比 >0.35** | 直接丢弃（1756 条） |
| `table_fragment` | 原文含 `<td>`/`<table>` | 走 caption 抢救（657 条） |
| `usable` | 其余 | 切分入库（5338 条） |

注：原本还有个 `salvaged_short` 宽容等级，实测那些 <40 字符的碎片全是 `"cks': 'image', 'image_body'"` 这类残渣尾巴，无检索价值，已取消该等级。

"数字占比 >0.35 判 junk"是关键一条 —— 清洗后仍以数字为主的，是被截断的数据行而非正文。

### 4.3 残表 caption 抢救 `extract_caption()`

历史数据的表格被硬截断在 500 字符，表头不可恢复。唯一还有检索价值的是 caption：

```
正则匹配  (?:表|图|附表|Table|Fig(?:ure)?)\s*[\dIVX]{1,3}(?:[-–—.]\d{1,3})?[\s.:：]{0,3}(...)
截断边界  连续空格 / 小数 / 括号引用 / 2位以上数字 —— 切掉后面的数据部分
五道校验  长度 8–90、文字占比 >0.55、无引号、无尖括号、
          必须以「表/图/Table」+编号开头、编号后描述 ≥8 字、数字占比 ≤0.3
```

通过校验的写成：

```json
{
  "block_type": "table_broken",
  "index_text": "表 3 榴莲开花期多元线性回归分析",
  "display_text": "表 3 ...\n（注：该表格在历史数据中已损坏，表头丢失，数值不可引用）",
  "metadata": {"damaged": true, "raw_fragment": "原始残片前1200字符"}
}
```

657 条残片中只有 **12 条**通过校验，645 条因无可用 caption 被丢弃。这个通过率很低，但符合设计意图：**宁可不召回，不可召回残表**。

### 4.4 PDF 表格误检过滤 `table_to_markdown()`

PyMuPDF 的 `find_tables()` 会把页眉、摘要、目录误判为表格。实测误检样例：

```
Frontiers p1:  |ang R|, Wei|              ← 作者行
JAD p1:        |ARTICLE INFO|ABSTRACT...  ← 摘要块
MY_DOA p3:     |Col1|Col2| KANDUNGAN      ← 目录
```

六道过滤：

```
1. 行数 ≥3（表头+分隔+至少1数据行）
2. 首行不以 Col\d 开头
3. 列数 ≥3（单列不算表）
4. 占位列头 ColN 占比 ≤0.4（过半说明表结构未被识别）
5. 各行列数一致性 ≥0.8（误检的文本块列数不齐）
6. 单元格去重后 >2 个（目录误检的典型特征是同一 cell 重复）
7. 单元格合并文字占比 ≥0.25（全是数字符号则等同裸数值表）
```

原始检出 57 个表，过滤后保留 **32 个**。

### 4.5 正文与表格去重

同一页既提表格又提正文时，正文会包含表格内容造成重复。用 bbox 包含关系排除：

```python
blocks = page.get_text("blocks")
keep = [b[4] for b in blocks if not any(
    b[0] >= t[0]-4 and b[1] >= t[1]-4 and b[2] <= t[2]+4 and b[3] <= t[3]+4
    for t in table_bboxes)]
```

### 4.6 PDF 级别的两道前置检查

```
① 全文 SHA1 去重 —— LKE_Standard_V1.0.pdf 与 -20260623015226-1.pdf 内容完全相同，
   跳过 1 个副本
② 文本层稀疏检测 —— 总文本 < 20×页数 判为扫描件，标记待 OCR 不入库
   （TH_ACFS_TAS_Durian_TH_81.pdf 37 页 0 字符，纯扫描）
```

### 4.7 切分策略 `split_text()`

正文 900 字符窗口、150 重叠，**按句边界切**：

```
优先切点：。 . ； ; \n     （位置需 > size*0.5）
退化切点：， , 空格          （位置需 > size*0.4）
都不满足则硬切
```

表格、图片、QA 对**不走这里**，整块保留。

### 4.8 统一输出 schema

```json
{
  "id": "sha1(doc:page:seq:text_sha1)",
  "doc": "...", "source_file": "...", "page": 45,
  "block_type": "text | table | table_broken | qa_pair",
  "provenance": "legacy_faq | legacy_thai | legacy_lit_salvage | pdf_parse",
  "index_text": "供向量化",
  "display_text": "供展示/喂 LLM",
  "text": "= index_text，向后兼容旧读取路径",
  "metadata": {
    "text_sha1": "...", "table_markdown": "...", "caption": "...",
    "damaged": true, "original_text": "...", "src_lang": "th",
    "needs_translation": true, "category": "...", "question": "..."
  }
}
```

### 4.9 索引侧的一个坑

`load_rechunk_nodes()` 里 metadata 必须分两套。第一次实现时把 `display_text` 统一放进 metadata，正文 node 直接报错：

```
ValueError: Metadata length (1027) is longer than chunk size (512)
```

原因是 `SentenceSplitter` 会把 metadata 序列化长度算进 `chunk_size`。修法：

- **整块 node**（table/qa_pair）不走 splitter，可以携带 `display_text`、`table_markdown`、`original_text` 等长字段
- **正文 node** 走 splitter，metadata 只保留 `doc`/`source_file`/`page`/`block_type`/`provenance` 等短字段；正文的 display_text 与 index_text 相同，无需单独存

同时所有非检索字段进 `excluded_embed_metadata_keys`，避免 metadata 被算进向量。

---

## 五、有意接受的代价

### 5.1 丢失可达 15 条

新增可达 110 条，同时**丢失 15 条**。抽样：

```
- [numeric] 马来西亚和澳大利亚榴莲叶片营养诊断中氮磷钾正常值范围分别是多少？
- [general] 如何根据染色剂特性选择榴莲种子活力检测方案？
```

这些金标的原文本就在表格里。改造前它们以裸 `<td>` 形式在索引中（锚点能字面匹配），改造后被判为不可恢复残表、只留 caption，锚点不再匹配。

**这是有意的取舍**：那些数值的表头已丢，召回它们等于让模型把数字配错列名。净收益 +95 条，换来数值幻觉风险归零。

### 5.2 全量口径 Hit@5 几乎未动（3.9% → 3.5%）

不是失败，是**指标失效**：94% 的金标是"中文提问 + 英文/泰文原文"，锚点匹配跨语言无效。核心可比子集反而从 13 条缩到 9 条。

真正的证据在核心口径 +47pp 和垃圾率归零。这个缺陷的根治方案是离线翻译（见 `docs/RAG_跨语言检索方案.md`）。

---

## 六、顺带找回的丢失文献

之前报 `499 canceled` 失败的后台调研 agent，**在死前下载了两个文件**：

| 文件 | 页数 | 说明 |
|---|---|---|
| `MY_DOA_Pakej_Teknologi_Durian_2012.pdf` | 62 | **Pakej Teknologi = pt** —— 正是丢失清单里 741 chunk 的 `pt_durian_2012`，马来西亚农业部技术手册 |
| `TH_ACFS_TAS_Durian_TH_81.pdf` | 37 | 泰国 ACFS 榴莲标准，纯扫描件待 OCR |

另外 `Durio_bibliographic_review_IPGRI_1997.pdf` 第一次下载截断成 0 页（287KB），重试后拿到完整 196 页（661KB）。这也说明**下载后必须校验页数，不能只看 `%PDF` 文件头**。

`raw/pdfs/` 现有 7 份权威 PDF，贡献 1325 个 chunk。

---

## 七、复现与回滚

```bash
# 完整重跑
python rag_rechunk.py build              # 清洗 → rag_build/chunks.jsonl
python rag_llamaindex.py rebuild         # 重建 FAISS 索引
python rag_eval.py audit-gold            # 可达性预检（必须重跑）
python rag_eval.py baseline --tag after_rechunk
python rag_eval.py compare --before before_rechunk --after after_rechunk

# 只看清洗统计不写文件
python rag_rechunk.py stats

# 回滚
rm -rf rag_llamaindex_storage
cp -r rag_llamaindex_storage.bak-20260902 rag_llamaindex_storage
```

---

## 八、遗留项

1. **1 条退化题需人工复核**：`v2_cleaned_0040`（榴莲种子营养袋种植的基质配比、播种深度）
2. **`TH_ACFS_TAS_Durian_TH_81.pdf` 待 OCR**：37 页纯扫描，泰国官方标准，价值高
3. **623 张图片未处理**：当前只计数，未做 VLM 描述（原计划阶段 3）
4. **核心口径仅 9 条**：需靠离线翻译扩到百条量级才能作为可信验收依据
5. **`rag_pdfs/` 与 `raw/pdfs/` 并存**：目前 `parse_pdfs()` 同时扫两个目录靠 SHA1 去重，
   后续应统一到 `raw/pdfs/`
