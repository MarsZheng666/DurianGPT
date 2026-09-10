# 榴莲多语言 AI Agent——阶段与任务总规划

> 依据：[durian_agent_architecture.md](durian_agent_architecture.md)（66 节设计文档）全文拆解，无遗漏。
> 任务编号与会话任务系统（Task #1–#66）一一对应；每任务独立可验证。
> 执行约定：每个任务「代码 + unittest 验证通过 + 独立 commit + push GitHub」。
> 状态图例：✅ 已完成（附提交哈希）｜🔲 待办｜◐ 部分内核已提前落地（见附录 A）

## 总览

| 阶段 | 主题 | 任务数 | 状态 | 对应文档 |
|---|---|---|---|---|
| 阶段一 | 主链路（语义→路由→基础 RAG→图→API） | 21 | **✅ 全部完成**（2026-09-10） | §65 第一阶段 |
| 阶段二 | 检索增强（术语库/改写/重排/证据闭环/生成/引用/多模态建库） | 11 | **✅ 全部完成**（2026-09-10） | §65 第二阶段 |
| 阶段三 | ReAct 与工具层（九工具+PolicyGate+确认接口） | 12 | **✅ 全部完成**（2026-09-10） | §65 第三阶段 |
| 阶段四 | 记忆（Checkpoint/摘要/Token 预算/五级降级） | 5 | 🔲 待办 | §65 第四阶段 |
| 阶段五 | 权限/评估/运维（权限五项/错误处理/六指标/存储/服务拆分/E2E） | 17 | 🔲 待办 | §65 第五阶段 |
| 合计 | | **66** | 44 完成 / 22 待办 | |

---

## 阶段一：主链路（21 任务，✅ 已完成）

代码全部在新包 `durian_agent/`（不触碰线上遗留 `durian_langgraph.py` / `durian__inference_api.py`）。
验证方式：`tests/test_agent_*.py`，unittest，共 190 项全绿。

| # | 模块 | 任务 | 文档 | 提交 |
|---|---|---|---|---|
| 7 | 编排 | 定义 AgentState 完整状态结构（§4 全 23 字段） | §4 | a3f122c |
| 8 | 语义 | 通用 InputNormalize（NFKC 坑防护/温度/单位） | §11 | beb0a73 |
| 2 | 语义 | 泰语归一化（词典优先+tokenizer，legacy 修复 verbatim 迁移+parity） | §11 | 836adb6 |
| 12 | 语义 | 实体归一化 alias→canonical（四语收敛） | §9 | c42ef04 |
| 5 | 语义 | 20 个 Intent 分类（LLM 主力+规则兜底+60 条多语言样例集） | §7 | 9b0161b |
| 14 | 语义 | 生育阶段识别（10 阶段，最长优先词表） | §8 | 99b8685 |
| 19 | 语义 | risk_features 8 项风险标志（宁多勿漏） | §6 | 89dc0ff |
| 3 | 语义 | SemanticParse→Canonical Schema（§46 约束，两层合成） | §5/§6/§46 | f986f12 |
| 13 | 路由 | RuleRouter 三路路由（§61 严格落地，策略数据化） | §12/§13/§15/§61 | 1c39a58 |
| 35 | SIMPLE | SimpleAgent（可选 RAG 注入） | §14 | 8f5ca21 |
| 17 | 建库 | 文档解析与结构识别（五类块） | §18 | b1d1731 |
| 20 | 建库 | 正文递归分块（300-800/overlap 10-20%，句边界） | §19 | f23a275 |
| 16 | 建库 | 表格处理 Markdown 化（表头/单位/行列/说明保留） | §20 | d8c27bb |
| 21 | 建库 | Chunk 实体识别与 Metadata（§22 十键，真实语料接入） | §22 | 91ef238 |
| 25 | 建库 | Milvus+BM25 双索引（milvus-lite+纯 Python BM25，§40 过滤验证） | §18/§58 | 6eb9271 |
| 23 | 检索 | 四路召回（实体级跨语言扩展；dense 垃圾命中 0.3 下限） | §25 | f0cd763 |
| 27 | 检索 | Weighted RRF（§26 公式，chunk_id 融合键防裂块） | §26 | 190f661 |
| 1 | 编排 | LangGraph 图拓扑（§44 实为 19 节点+§45 全边分支） | §3/§44/§45 | 784eeb6 |
| 9 | 编排 | ContextInit 节点（缺省补齐/最小权限/HumanMessage 追加） | §3/§44 | a84a004 |
| 6 | 接入 | Conversation Gateway（角色白名单/thread 注册/语言校验） | §2 | bb63f11（补丁 67db09d） |
| 4 | 接入 | POST /api/chat（§60 契约+thread_id 回传扩展） | §60 | 06cb0b3 |

文档入库提交：09872b7（架构设计文档，任务拆分依据）。

---

## 阶段二：检索增强（11 任务，✅ 已完成，2026-09-10）

| # | 模块 | 任务 | 文档 | 提交 |
|---|---|---|---|---|
| 24 | 检索 | Query Build 跨语言扩展完整验收（§23 三输入；不整句翻译） | §9/§23 | 94bfca1 |
| 30 | 检索 | Evidence Check 五项判断（实体/症状/数值/冲突/单点） | §29 | 8b78765 |
| 28 | 检索 | Query Rewrite 两层（§47 禁令 + 数字护栏） | §24/§47 | 5128794 |
| 29 | 检索 | Evidence Retry 循环验收（改写救回/耗尽拒答/计数簿记） | §29/§63 | fe0517e |
| 32 | 检索 | Cross Encoder Reranker（本地 bge-reranker-v2-m3） | §27 | 46e13a7 |
| 26 | 检索 | Reranker 阈值（默认 0.3 可配置；驱动重试链） | §28 | 831f4d9 |
| 31 | 检索 | Final Generation 完整验收（§48 全禁令/无证据不调 LLM） | §30/§48 | 0dfdd27 |
| 33 | 检索 | 引用一一对应（[chunk_id] 解析，未引用不进 sources） | §31 | 0dfdd27 |
| 15 | 语义 | 术语库 canonical_id 体系（glossary_v2） | §10 | 97f0e29 |
| 18 | 建库 | 图片处理 VLM 三要素（通道依赖注入，测试零费用） | §21 | 6fe750c |
| 22 | 建库 | 扫描件 OCR+版面恢复+分块（tesseract/vlm 双通道） | §21 | fc149f6 |

数据记账（诚实声明）：术语库实际 **206 概念 / 527 别名**（zh199/en203/th102/ms23），
文档「1330+」与实测不符；§10 建议区间为 300-500 概念 + 1000-2000 别名，
别名数偏低——扩充需批量四语翻译（LLM 通道确认后进行）。

---

## 阶段三：ReAct 与工具层（12 任务，✅ 已完成，2026-09-10）

| # | 模块 | 任务 | 文档 | 提交 |
|---|---|---|---|---|
| 43 | ReAct | 工具层基座 + UserContextTool（ToolRegistry/Context；§39 角色表） | §17/§39 | 856c9cf（补丁 ccbab19） |
| 40 | ReAct | WeatherTool + Provider 体系（依赖注入，InMemory mock） | §17 | 2bcd043 |
| 37 | ReAct | SoilSensorTool（土壤湿度/地温/pH） | §17 | 899c9ae |
| 41 | ReAct | OrchardAssetTool（园区/地块/设备/人员） | §17 | 721bc22 |
| 34 | ReAct | AlarmTool（告警过滤查询） | §17 | 20ff5ee |
| 45/42/44 | ReAct | Task 工具三件套（查询/创建+§42幂等/更新，§41 确认闸在 Registry 强制拦截） | §17/§41/§42 | e5c6fc5 |
| 36 | ReAct | AgricultureRagTool（RAG 闭环封装，证据回写状态） | §16/§17 | 677fb69 |
| 38 | ReAct | ToolPolicyGate（§16/§64：专业结论无证据→强制 RAG，一次性护栏） | §16/§64 | 03993a2 |
| 39 | ReAct | ReAct 循环 + 图四节点接线（文本 JSON 协议，步数上限，敏感操作拦截收尾） | §17/§49 | 2b77749 |
| 10 | 接入 | POST /api/chat/confirm（§41 闭环：批准执行/取消/403/一次性） | §41/§60 | 10a86c1 |

关键实测坑（详见 #39 提交说明）：LangGraph 按节点函数签名注解过滤输入
state（AgentState 注解会滤掉 GraphState 运行时键，分支永远不可见）；
证据覆盖只要求内容实体槽位（orchard/plot 是定位符）。

| # | 模块 | 任务 | 文档 |
|---|---|---|---|
| 39 | ReAct | ReAct 循环（Reason→Tool→Observation 迭代） | §17/§49 |
| 38 | ReAct | ToolPolicyGate（专业结论前强制 AgricultureRagTool） | §16/§64 |
| 36 | ReAct | AgricultureRagTool | §17 |
| 40 | ReAct | WeatherTool | §17 |
| 37 | ReAct | SoilSensorTool | §17 |
| 41 | ReAct | OrchardAssetTool | §17 |
| 34 | ReAct | AlarmTool | §17 |
| 45 | ReAct | TaskQueryTool | §17 |
| 42 | ReAct | TaskCreateTool（含幂等） | §17 |
| 44 | ReAct | TaskUpdateTool | §17 |
| 43 | ReAct | UserContextTool | §17 |
| 10 | 接入 | POST /api/chat/confirm 确认接口 | §60 |

工具数据源（天气/传感器/资产/工单）需要业务侧对接或 mock 先行；图内 reactAgent→toolPolicy→permissionCheck→toolNode 链路已在 #1 占位打通。

---

## 阶段四：记忆与长会话（5 任务，🔲 待办）

| # | 模块 | 任务 | 文档 |
|---|---|---|---|
| 56 | 记忆 | Checkpoint 持久化（thread_id 隔离，业务会话 ID） | §32 |
| 51 | 记忆 | 记忆结构组装（System+Summary+窗口+业务状态+Query 五段式） | §33 |
| 55 | 记忆 | 滑动窗口与摘要（保留 6 类信息/剔除 3 类） | §35 |
| 57 | 记忆 | Token Budget 分配（10/15/25/35/5/10） | §34 |
| 48 | 记忆 | Token 膨胀五级降级（Level 0-4） | §36 |

图的 memoryCompress 节点已占位（#1）；thread 隔离已有 MemorySaver 验证，持久化换 SqliteSaver。

---

## 阶段五：权限/评估/运维（17 任务，🔲 待办）

| # | 模块 | 任务 | 文档 |
|---|---|---|---|
| 49 | 权限 | Tool Permission（按角色注册工具，Worker 3/Manager 7/Admin 更多） | §39 |
| 53 | 权限 | Data Permission（metadata filter：tenant/role_scope/orchard_scope） | §40 |
| 54 | 权限 | 敏感操作二次确认（Validation→展示→确认→执行） | §41 |
| 47 | 权限 | Task Tool 幂等（idempotency_key） | §42 |
| 52 | 权限 | 多用户隔离（五维+checkpoint key=tenantId:userId:threadId） | §37 |
| 46 | 错误 | 9 种错误分类与处理策略 | §50 |
| 50 | 错误 | Retry 原则（LLM 1-2/Tool 1-2/Rewrite 2，写操作靠幂等） | §51 |
| 59 | 评估 | 全链路 Trace 日志（15 项字段） | §52 |
| 61 | 评估 | RAG 离线指标（Recall@K/Hit/MRR/NDCG，四语种分报） | §53 |
| 58 | 评估 | 生成质量指标（5 项，重点 Faithfulness/Abstention） | §54 |
| 60 | 评估 | 多语言检索评估集（1200+，语言×intent×difficulty；#5 的 60 条样例是种子） | §55 |
| 63 | 评估 | 语料解析覆盖率（正文/表格/图片/扫描页分层） | §56 |
| 64 | 评估 | 长对话成功率 | §57 |
| 11 | 接入 | POST /api/rag/search 独立 RAG 接口 | §60 |
| 62 | 基建 | 存储层（Milvus/关系库 8 表/Redis/Checkpoint store） | §58 |
| 66 | 基建 | 三服务拆分（Agent/RAG/BusinessTool） | §59 |
| 65 | E2E | 端到端复杂任务场景回归（§43 示例 16 步全链路） | §43 |

Data Permission 的 array_contains 过滤已在 #25 VectorIndex 验证通过（manager-only 文档不泄漏给 worker），阶段五补租户维与全链路。

---

## 附录 A：已提前落地的内核（已于阶段二全部正式落地，本附录仅存档）

| 任务 | 已落地部分 | 所在提交 | 待补内核 |
|---|---|---|---|
| #24 Query Build | build_queries（original/canonical/expanded 三查询） | f0cd763 | 完整验收+§23 说明的 Schema 深度利用 |
| #28 Query Rewrite | 图 rewriteQuery 占位（扩展查询重试） | 784eeb6 | 规则层+LLM 层（§47 禁猜清单） |
| #29 Evidence Retry | 图 evidence 三分叉+retry 循环+fallback 拒答 | 784eeb6 | LLM 改写内核 |
| #30 Evidence Check | 占位（有融合证据即充分） | 784eeb6 | 五项判断（实体/条件/数值/冲突/单点） |
| #31 Final Generation | rag/generation.py（§48 prompt+证据块） | 784eeb6 | 诱导补剂量被拒等完整验收 |
| #32/#26 Reranker+阈值 | 图 rerank 节点占位（透传 RRF 序） | 784eeb6 | Cross Encoder（本地 bge-reranker-v2-m3）+阈值调优 |
| #33 引用 | rag_sources 图+API 通路 | 784eeb6/06cb0b3 | 引用与实际使用证据一一对应校验 |

## 附录 B：执行边界（全程有效）

1. **远端禁改**：durian-server 上的服务与数据绝不触碰；隧道端口 8010/19380 不用于推理。
2. **LLM 通道**：OpenAICompatibleLLM 只认 `DURIAN_AGENT_LLM_*` 显式配置；测试一律 FakeLLM 离线跑。
3. **嵌入**：本地 `models/bge-small-zh-v1.5`（依赖注入可替换）。
4. **遗留代码**：`durian_langgraph.py`（用户 WIP）与 `durian__inference_api.py`（线上）不动。
5. 每任务完成标准：代码 + 测试通过 + 独立 commit + push `origin main`。
