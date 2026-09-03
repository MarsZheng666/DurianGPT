# RAG 表格图片向量化 —— 实施流程

> 范围：**仅本地 LlamaIndex 链路**（`rag_llamaindex.py` + FAISS），不改动 RAGFlow
> 制定日期：2026-09-01
> 方案依据：`docs/RAG_表格图片向量化方案.md`（表格走 T1+T2，图片走 I1+I5）

---

## 一、前置事实核查结果（决定流程起点）

开工前已实地核查三处，结论如下：

| 检查位置 | 命令/方式 | 结果 |
|---|---|---|
| 本地 `rag_pdfs/` | `ls` | 只有 6 个文件：LKE_Standard ×2、SOP_V0 ×4，共 337 chunk |
| 远端 `durian-server` | `find / -xdev -name "*MinerU*" -o -name "ocr_texts*"` | **无任何结果**；`find /home /data /opt /srv -iname "*durian*.pdf"` 同样为空 |
| RAGFlow dataset `7c684b1e…` | `GET /api/v1/datasets/{id}/documents` | **total=6**，与本地 `rag_pdfs/` 完全相同，无那 22 篇文献 |

**结论：22 篇文献 + 3 份泰文 DOCX 的源文件已彻底丢失，MinerU 原始 JSON 也不存在。**

补充两个关键判断：

1. **打平数据无法还原**。8264/9402 个 chunk 恰好被硬截断在 499/500 字符，且每个 `item_index` 只对应 1 个 chunk（分布 `{1: 9402}`），没有重叠片段可供拼接。表格 HTML 被切在 `Amount of N application (k` 这类位置，后半截不存在于任何地方。
2. **上一轮用的是 MinerU 在线服务**。图片存的是 `cdn-mineru.openxlab.org.cn/result/2025-10-05/...` 临时 CDN 链接而非本地文件，这是图片彻底丢失的直接原因。**新流程必须用本地 MinerU，图片落本地磁盘。**

### 可以保留的干净数据

按来源分组统计污染分布：

| 分组 | chunk 数 | 含解析残渣 | 含表格 HTML | 含图片 URL | 处置 |
|---|---|---|---|---|---|
| 文献 | 7751 | 6869 | 695 | 618 | **全部废弃重建** |
| FAQ（`FAQ_{en,zh,th,ms}_N`） | 1000 | 0 | 0 | 0 | **原样保留** |
| 泰文 DOCX ×3 | 651 | 0 | 0 | 0 | **原样保留** |

污染 100% 集中在文献部分。FAQ 是 `type: qa_pair` 的结构化问答（含 `question`/`answer`/`category` 字段），与 MinerU 无关，不受影响。

### 需重新采集的清单

**文献 22 篇**（按原 chunk 数排序，反映内容体量与优先级）：

| # | 原 chunk | 标题 |
|---|---|---|
| 1 | 1468 | Boosting Durian Productivity |
| 2 | 1461 | Durio_Bibliographic review |
| 3 | 1041 | 中国榴莲栽培技术 |
| 4 | 741 | pt_durian_2012 |
| 5 | 405 | A review of durian plant-bat pollinator interactions |
| 6 | 324 | Fruit Quality and Antioxidant Content in Durian cv. 'Monthong' in Different Maturity Stages |
| 7 | 323 | Propagation and Cultivation Techniques of Bentara Durian |
| 8 | 267 | A method for durian precise fertilization based on improved RBF neural network algorithm |
| 9 | 250 | Advances in Production Technology of Durian |
| 10 | 183 | Fertilizer practices and soil properties in durian orchards during the fruit-bearing stage |
| 11 | 159 | Physicochemical Properties of Soil Cultivated with (Durian) |
| 12 | 148 | Effect of 1-methylcyclopropene (1-MCP) on … |
| 13 | 144 | 榴莲成熟度的判断与保鲜技术研究进展_侯双迪 |
| 14 | 133 | durian-protocol（原文件名为 UUID：C2BC4F73-5912-4924-92F63FAE52E5D9A6） |
| 15 | 128 | Durian-Production-Guide-koronadal-new |
| 16 | 121 | Climatic variables study on generative characters of (durian) |
| 17 | 115 | 榴莲果实品质与矿质元素的灰色关联度和通径分析_朱振忠 |
| 18 | 109 | Durian mortality and yield determinants in Vietnam Mekong Delta |
| 19 | 90 | Effective pollination period in durian and the factors regulating it |
| 20 | 75 | Durian Floral Differentiation and Flowering Habit |
| 21 | 39 | 金枕榴莲种植技术及大数据分析溯源_单友成 |
| 22 | 27 | 榴莲树种植的注意事项 |

**泰文 DOCX 3 份**（源文件丢失但 chunk 干净，可暂不重采）：
`การผลิตทุเรียนภาคใต้ตอนล่างn.docx`(326) / `คู่มือการผลิตทุเรียนคุณภาพ-จังหวัดชุมพร.docx`(251) / `การผลิตทุเรียน.docx`(74)

> 注意 #14 原文件名是 UUID，无法从名字反查原文；需从其 chunk 内容判断是什么文档。#11/#12/#16 标题被截断，同样需靠内容确认。

### 运行环境

| 项 | 状态 |
|---|---|
| 本地 Python | 3.12.13（`.venv`） |
| 本地已装 | pymupdf 1.28.2、pypdf 6.16.2、faiss-cpu 1.15.0、llama-index-core 0.14.24、torch 2.13.0、pillow 12.3.0 |
| 本地待装 | `mineru`、`pandas`、`lxml`、`openai`（调 qwen-vl-max 用） |
| 远端 GPU | RTX 5090 32GB，**已用 31733/32607 MiB（vLLM 占满）** |

**MinerU 跑在哪：** 远端 GPU 无余量，方案是本地 Mac CPU 跑（慢，22 篇可接受，一次性成本），**不去抢 vLLM 的显存**。若嫌慢再单独申请停 vLLM 的时间窗，不在本流程内擅自操作。

---

## 二、目标产物与目录结构

```
durian-training/
├── raw/                          # 新增：原始素材，纳入备份，不进 git（大文件）
│   ├── pdfs/                     # 重新采集的 22 篇源 PDF ← 唯一真源
│   └── mineru/<doc_stem>/         # 本地 MinerU 输出
│       ├── <stem>_content_list.json
│       ├── <stem>_middle.json
│       └── images/*.jpg          # 图片落本地磁盘（关键：不再用 CDN 链接）
│
├── rag_build/                    # 新增：中间产物，可重跑，可删
│   ├── blocks.jsonl              # 阶段2产物：结构化块（text/table/image）
│   ├── image_captions.jsonl      # 阶段3产物：VLM 描述缓存（幂等，避免重复付费）
│   ├── table_summaries.jsonl     # 阶段4产物：表格摘要缓存
│   └── chunks.jsonl              # 最终入库 chunk（替代 clean_chunks.jsonl 的文献部分）
│
├── clean_chunks.jsonl            # 保留，但只取 FAQ(1000)+DOCX(651) 共 1651 条
└── rag_llamaindex_storage/       # FAISS 索引（重建前先备份）
```

### chunk 目标 schema

```json
{
  "id": "sha1(doc:block_id:sub_index)",
  "doc": "中国榴莲栽培技术.pdf",
  "source_file": "中国榴莲栽培技术.pdf",
  "page": 45,
  "block_type": "text | table | table_row | image",

  "index_text": "用于向量化的文本",
  "display_text": "用于展示/喂 LLM 的文本",

  "metadata": {
    "caption": "表 3-2 不同榴莲品种果实品质比较",
    "section": "第三章 品种选择",
    "table_markdown": "| 品种 | ... |",
    "table_html": "<table>...</table>",
    "image_path": "raw/mineru/xxx/images/abc.jpg",
    "parent_id": "整表 node 的 id（table_row 才有）",
    "caption_model": "qwen-vl-max",
    "text_sha1": "..."
  }
}
```

**`index_text` 与 `display_text` 必须分离** —— 这是整个改造的结构性关键。当前 `rag_llamaindex.py:929-935` 把 `text`/`index_text`/`original_text`/`display_text` 全赋成同一个值，导致"为检索优化文本"和"为展示保留原格式"两个目标无法并存。

---

## 三、实施阶段

### 阶段 0：备份与基线固化

**动作**

1. 备份现有索引与 chunk（改造失败要能原样回退）：
   ```
   cp -r rag_llamaindex_storage rag_llamaindex_storage.bak-20260901
   cp clean_chunks.jsonl clean_chunks.jsonl.bak-20260901
   ```
2. 建立**召回基线**：整理 30-50 条真实问题（覆盖品种/病害/施肥/成熟度/表格数值/图片识别六类），跑当前 `retrieve()` 记录 top-5 结果与分数，存 `tests/rag_baseline_20260901.json`。
3. `.gitignore` 追加 `raw/`、`rag_build/`、`*.bak-*`。

**验收**：基线文件生成，且能用一条命令复跑对比。

> 没有基线就无法证明改造是改好了还是改坏了。这一步不能跳。

---

### 阶段 1：重新采集源文献

**动作**

1. 按第一节清单逐篇找回 PDF，落到 `raw/pdfs/`。文件名统一为清单里的标题（去掉 `_MinerU__时间戳`），中文标题保留中文。
2. `#11/#12/#14/#16` 四篇标题不全或为 UUID 的，先从 `clean_chunks.jsonl` 取该 `doc` 的前若干 chunk 读内容，确认是什么文献再检索。
3. 建立 `raw/pdfs/MANIFEST.md`：记录每篇的标题、来源 URL/出处、采集日期、SHA256。**这是防止本次事故重演的核心措施** —— 上一轮就是因为没有真源清单，丢了才发现无从追溯。
4. 泰文 DOCX 3 份：暂不重采，其 651 条 chunk 直接沿用。

**验收**：`raw/pdfs/` 有 22 个 PDF，MANIFEST 完整，SHA256 可校验。

**风险**：部分文献可能是付费/内部资料找不回。**采集不全不阻塞后续阶段** —— 拿到几篇就先跑几篇，管道验证不依赖数量。缺失的在 MANIFEST 里标 `MISSING` 并记录原 chunk 数，作为已知覆盖缺口。

---

### 阶段 2：本地 MinerU 解析 + 结构化落地

这是整个改造的核心，也是上一轮出事的地方。

**动作**

1. 安装本地 MinerU（先跑 1 篇小文献验证版本与输出格式，再批量）：
   ```
   .venv/bin/pip install mineru pandas lxml
   ```
   > MinerU 2.x CLI 形如 `mineru -p <pdf> -o <outdir>`，输出 `*_content_list.json` + `images/`。**具体参数与输出字段名以实际安装版本为准**，先单篇验证后再固化到脚本，不凭记忆写死。

2. 批量解析 → `raw/mineru/<doc_stem>/`。**务必确认图片是落到本地 `images/` 目录，而不是又变成 CDN 链接。** 这是本阶段第一优先的验证项。

3. 新建 `ingest_mineru.py`，把 MinerU 输出转成 `rag_build/blocks.jsonl`。**核心原则：按 block 的 `type` 分流，绝不 `str()` 整个结构。**

   | MinerU block type | 处置 |
   |---|---|
   | `text` | 取纯文本，按标题层级记录 `section` |
   | `table` | 取 `spans[].html`（**实测键名是 `html`，不是 `table_body`**）→ 用 pandas/lxml 转 Markdown；`table_caption` 存 `caption` |
   | `image` | 取本地 `image_path` + `img_caption` → 只写 metadata，text 位置留占位符 |
   | `equation` | 取 latex，套用现有 `normalize_ocr_units()` 归一化 |
   | `discarded` | 丢弃（页眉页脚） |

4. 图片过滤规则（防装饰图污染向量库）：尺寸 < 100×100 px、宽高比 > 8:1（分隔线）、同一 hash 在多页重复出现 ≥ 3 次（logo/水印）→ 标记 `skip: true`，不送 VLM。

5. 表格降级兜底：若 `html` 解析失败或表格为空，保留 `caption` + 原始 HTML 进 metadata，`index_text` 只用 caption，**绝不把裸 `<td>` 数字塞进 `index_text`**（宁可不召回，不可召回残表）。

**验收**（硬指标，用脚本断言）

- `blocks.jsonl` 中含 `'bbox'` / `'para_blocks'` / `'spans'` 的记录数 = **0**
- 表格块数 > 0，且每个表格块的 `table_markdown` 能被 pandas 成功读回
- 图片块的 `image_path` 全部指向 `raw/mineru/**/images/` 下**真实存在**的文件
- 抽 10 个表格块人工核对：表头、行列数、数值与原 PDF 一致

**这一步不通过，后面全部不要做。**

---

### 阶段 3：图片 VLM 描述（方案 I1）

**动作**

1. 复用 `qwen_vl_max_client.py`（已接好 qwen-vl-max，只是从未进 ingest 链路）。
2. 对每张未被 skip 的图片，送入：**图片本身 + 图注 + 前后各约 200 字正文 + 所属章节标题**。
   > 不给上下文，VLM 只会输出"一片绿色的叶子"；给了上下文才能定位到"炭疽病病斑"。这是方案文档里第 2 条坑。
3. Prompt 要求领域化结构输出，不要散文：类别（品种特征/病害症状/农事操作/器械设备/数据图表）、可见对象、关键视觉特征、与图注的对应关系。
4. **结果写入 `rag_build/image_captions.jsonl`，以图片 SHA256 为 key 做幂等缓存**。重跑管道不重复调 API。
5. 数据图表类图片（折线/柱状图）额外要求 VLM 读出坐标轴含义与趋势，但**明确禁止编造具体数值**。
6. 失败重试 3 次后标 `caption_failed: true`，降级为方案 I5（仅占位符 + 图注），不阻塞流程。

**入库文本拼装顺序**（权重从高到低）：图注 → VLM 描述 → 上下文摘要。

**验收**

- 覆盖率：非 skip 图片的描述生成成功率 ≥ 95%
- 抽 20 张人工检查：描述与图片内容相符，无幻觉数值，无"无法识别"类空话
- 缓存有效性：二次运行 API 调用数 = 0

---

### 阶段 4：表格摘要（方案 T2）+ chunk 组装

**动作**

1. 对数值型表格（列中数字占比 > 50%）调本地 LLM（vLLM `durian-lora`，走 `127.0.0.1:8010`）生成一段自然语言摘要，缓存进 `rag_build/table_summaries.jsonl`。
   - `index_text` = caption + 摘要
   - `display_text` = caption + 完整 Markdown 表
   - 这就是父子索引：向量库存摘要，命中后取原表喂 LLM
2. 非数值型表格（对照表、名录）跳过摘要，`index_text` 直接用 caption + Markdown。
3. 长表（> 30 行）额外做**行级双写**（方案 T3）：每行独立成 `block_type: table_row` 的 chunk，**每行复制一份表头**，`parent_id` 指向整表 node。整表 node 同时保留，用于聚合类问题。
4. 正文按语义边界切分，沿用 `SentenceSplitter(chunk_size=512, chunk_overlap=80)`。
5. 组装 `rag_build/chunks.jsonl`，与保留的 1651 条（FAQ + DOCX）合并。

**验收**

- 表格 chunk 的 `index_text` 中不含裸 HTML 标签
- 每个 `table_row` 都能通过 `parent_id` 找到整表
- 摘要抽检 15 条：数值与原表一致，无编造

---

### 阶段 4.5：非中文语料离线翻译成中文

**为什么需要这一步**

索引里 **17% 是泰文 node**（1287 个，来自 3 份手册的 651 个 chunk），而 embedding 是
中文单语 `bge-small-zh-v1.5`。实测：

- 纯泰文句子在该模型分词后是 **100% `[UNK]`**
- 真实泰文 chunk 的 UNK 率 43-62%，**存活 token 全是数字和标点**（如
  `['50','5','.','5','-','6','.','5','1',',','600']`）
- 泰文语义完全丢失，向量由残存数字决定 —— 这是伪信号，还会在**数值类问题**上
  被数字巧合命中，与表格 HTML 碎片属同一类故障叠加

**为什么选翻译而不是换多语言 embedding**

实测 `bge-reranker-v2-m3`（XLM-R 多语言）的跨语言对齐分数：英文同义句 5.640、
中文 5.621、马来文 2.038、**泰文 -1.961**。对齐质量随语言资源量呈数量级差距。
换模型是把对齐负担交给模型实时承担，而它在泰语上的能力只有英语的零头；
离线翻译是在入库阶段消灭跨语言问题，检索时纯中文对纯中文，不受资源不均衡影响。

完整论证见 `docs/RAG_跨语言检索方案.md`。

**动作**

1. 翻译对象：泰文 3 份手册全部 chunk + 英文文献正文。FAQ（2476 node）与
   SOP/LKE（643 node）已是中文，跳过。
2. 翻译器优先用远端 vLLM `durian-lora`（`127.0.0.1:8010`），零额外成本；
   泰文若质量不佳改用 qwen-vl-max 所在 API 通道。
3. **幂等缓存**：以 `text_sha1` 为 key 存 `rag_build/translations.jsonl`，
   与图片 VLM 描述缓存同一套机制，重跑不重复付费。
4. **必须保留原文**：`metadata.original_text` + `metadata.src_lang`，不可覆盖。
   泰文源 docx 已丢失，现存 chunk 是**仅存副本**。
5. 先固化榴莲领域术语表（品种名、病害名、农事操作）作为 prompt 的一部分，
   避免同一术语在不同 chunk 里译法不一致而割裂检索。

**验收**

- 泰文/英文 chunk 的 `index_text` 为中文，`metadata.original_text` 完整保留原文
- 抽检 20 条：术语译法一致，数值与单位未被改动
- 缓存有效性：二次运行翻译调用数 = 0
- 重跑 `rag_eval.py audit-gold`：`zh→th`(21) 与 `zh→en`(220) 金标应转为同语种，
  **核心口径样本量从 13 条扩大到百条量级**

> 这一步有个额外收益：它不只改善检索，还顺带修掉了评估体系当前最大的缺陷
> —— 核心口径样本过小（见 `docs/RAG_评估体系.md` 第五节）。

---

### 阶段 5：索引重建（表格/图片不参与切分）

**动作**

1. 改 `rag_llamaindex.py`。当前 `rebuild_index()`（`:349-377`）走 `VectorStoreIndex.from_documents(docs, transformations=[splitter])`，会把**所有** Document 无差别过一遍 splitter，表格必被切断。改为：
   - 正文 → 过 `SentenceSplitter`
   - `table` / `table_row` / `image` → **直接构造 `TextNode`，绕过 splitter**
   - 最后 `VectorStoreIndex(nodes=all_nodes)` 手工建索引
2. `load_extra_chunk_documents()`（`:258-345`）适配新 schema：向量化读 `index_text`，展示读 `display_text`，metadata 全量透传。
3. **收紧 `clean_text()`（`:158`）**：数据干净后，擦 bbox/image_path 的正则不再需要，保留 `normalize_ocr_units()` 即可。继续留着会误伤正常文本里的 `type:`、`content:` 等字样。
4. **重审 `is_noise_text()`（`:186`）**：
   - parser-marker 计数分支可以删（数据源头已干净）
   - `front_matter` 黑名单（`主编`/`出版`/`责任编辑` 等）**有误杀风险** —— 正文里讨论"出版"就会被整块丢掉。改为只在文档前 3 页生效
5. `retrieve_for_backend()`（`:906-937`）不再把四个字段赋同值：`index_text` 给检索侧，`display_text` 给前端，图片 chunk 额外透传 `image_path` 供回显。
6. 重建索引并持久化。

**验收**

- 表格 node 在 docstore 中是完整一块，未被截断
- 图片 chunk 可被检索命中，且 metadata 带可访问的 `image_path`
- **跑阶段 0 的基线对比**：整体召回不退化，表格/图片类问题召回显著提升
- `image__vector_store.json` 仍为空是**预期的**（走 I1 文本路线，不做 CLIP 图向量）；在代码或本文档中明确注明，消除架构歧义

---

### 阶段 6：验证与收尾

1. 六类问题全量回归，与基线逐条对比，产出对比报告。
2. 重点验证防幻觉：问表格数值，检查回答是否与原表一致、是否正确引用来源。
3. 前端验证图片回显链路是否通。
4. 更新 `README.md` 记录新 ingest 流程与目录约定。
5. `image__vector_store.json` 空壳去向定论（方案 P4）：明确"当前不做图向量检索"，或删除该文件。

---

## 四、执行顺序与依赖

```
阶段0 备份+基线 ──┬→ 阶段1 采集文献 → 阶段2 MinerU结构化落地 ─┬→ 阶段3 图片VLM描述 ──┐
                  │                        (硬闸门)            ├→ 阶段4 表格摘要 ────┤
                  └→ 可与阶段1并行                              └→ 阶段4.5 离线翻译 ──┤
                                                                                      ↓
                                                        阶段5 索引重建 → 阶段6 验证
```

- 阶段 0 必须最先做
- 阶段 2 是**硬闸门**，验收不过不进入阶段 3/4/4.5
- 阶段 3（图片描述）、阶段 4（表格摘要）、阶段 4.5（离线翻译）三者互不依赖，**可并行**
  —— 都是"调模型 + 幂等缓存"的同构任务，建议共用一套缓存与重试框架
- 阶段 1 采集不全不阻塞：拿到几篇先跑几篇
- 阶段 4.5 不依赖阶段 1 的采集结果 —— 泰文那 651 个 chunk 已在手，可最先启动

---

## 五、风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 部分文献找不回 | 知识覆盖缺口 | MANIFEST 标 `MISSING` + 原 chunk 数，明示缺口；不阻塞管道 |
| 本地 CPU 跑 MinerU 太慢 | 阶段 2 拖长 | 先跑 3 篇高价值（#1/#2/#3）验证效果；申请停 vLLM 的时间窗需**单独确认**，不擅自操作远端 |
| MinerU 版本输出字段与预期不符 | 脚本写死会全崩 | 先单篇验证输出结构，再固化字段名 |
| VLM 描述质量不稳 | 图片召回噪声 | 缓存 + 抽检；不合格的降级为 I5 占位符 |
| 表格摘要引入幻觉 | 数值错误 | `display_text` 始终携带原表，LLM 以原表为准；摘要仅用于检索 |
| 改坏检索 | 线上问答退化 | 阶段 0 备份 + 基线对比；`rag_llamaindex_storage.bak-*` 可整目录回滚 |
| 翻译术语不一致 | 同一术语多种译法，割裂检索 | 先固化领域术语表进 prompt；抽检 20 条核对译法一致性 |
| 翻译改动数值 | 数值失真，幻觉风险 | 验收专项检查数值与单位未被改动；原文永久留在 `metadata.original_text` 可回溯 |
| 泰文原文再丢一次 | 不可恢复（源 docx 已丢） | `metadata.original_text` 绝不覆盖；翻译是新增字段而非替换 |
| 新数据又丢一次 | 重蹈覆辙 | `raw/` 纳入备份策略；MANIFEST 记录 SHA256 与来源；**禁止再用 MinerU 在线服务的 CDN 链接作为图片唯一存储** |

---

## 六、本次事故的根因与预防

上一轮的问题不是"技术选型错"，而是**两个工程习惯缺失**：

1. **把解析器的中间产物 `str()` 之后当最终数据存**，且没保留原始输出 → 结构信息不可逆丢失，且 500 字符截断让残余数据连修复都做不到。
   → 今后原则：**原始解析输出必须原样落盘保存**（`raw/mineru/`），派生产物（`rag_build/`）都可重算。
2. **依赖外部临时 CDN 链接作为图片的唯一存储**，且没有源文件清单 → 链接过期即数据消失，且丢了无从追溯。
   → 今后原则：**源文件 + MANIFEST（含 SHA256 与出处）是不可再生资产，必须备份。**
