# Durian Agent — 榴莲种植园多语言 AI Agent

面向马来西亚榴莲种植园智慧运营场景的多语言（中文 / English / ไทย / Bahasa Melayu）领域 AI Agent。

> 核心理念：**多语言模型负责理解，规则负责高风险路由，RAG 负责专业证据，ReAct 负责复杂任务编排，权限系统负责能力边界，Memory 负责多轮连续性。**
> 不采用「所有请求都丢给 Agent」或「所有请求都强制 RAG」的极端设计，在确定性与灵活性之间分层处理。

## 核心能力

1. 榴莲种植专业知识问答（证据约束生成，带引用）
2. SOP、操作规程、内部文档查询
3. 病虫害 / 施肥 / 灌溉 / 花果管理等专业问题辅助判断
4. 园区、地块、设备、人员等资产查询
5. 自然语言创建、查询、操作工单（敏感操作二次确认）
6. 天气、土壤、传感器等实时数据查询
7. 多轮对话与长会话记忆（滑动窗口 + 摘要 + 五级降级）
8. 角色 / 租户 / 园区三级权限隔离

## 系统架构

```text
┌──────────────────────────────────────────────┐
│                 Client Layer                  │
└───────────────────────┬──────────────────────┘
                        ↓
┌──────────────────────────────────────────────┐
│           Conversation Gateway                │
│   tenant / user / role / thread / language    │
└───────────────────────┬──────────────────────┘
                        ↓
┌──────────────────────────────────────────────┐
│             LangGraph Agent Layer             │
│  InputNormalize → SemanticParse → RuleRouter  │
│            ↓          ↓          ↓            │
│        MUST_RAG   SIMPLE    COMPLEX_TASK      │
│      RAG Pipeline  SimpleAgent  ReAct Agent   │
│                     ↓         ToolPolicyGate  │
└───────────┬───────────────────┬──────────────┘
            ↓                   ↓
┌─────────────────────┐ ┌──────────────────────┐
│    RAG Service      │ │ Business Tool Service │
│ 四路召回/RRF/重排/   │ │ Weather/Sensor/Asset │
│ 证据判断/约束生成    │ │ Task/Alarm/RAG/User   │
└──────────┬──────────┘ └──────────┬───────────┘
           ↓                       ↓
┌──────────────────────────────────────────────┐
│                Knowledge Layer                │
│  milvus-lite / BM25 / 术语库 / 金标评估集     │
└───────────────────────┬──────────────────────┘
                        ↓
┌──────────────────────────────────────────────┐
│            Observability Layer                │
│   15 字段全链路 Trace / 离线指标 / 审计留痕   │
└──────────────────────────────────────────────┘
```

## 三条执行路径

路由输入不是原始文本，而是**语言无关的 Canonical Schema**（20 意图 / 10 生育阶段 / 7 实体槽位 / 8 风险标志），路由策略全部数据化、可审计。

### MUST_RAG — 高风险问题（农药 / 剂量 / 诊断 / 法规 / 病虫害实体在场）

```text
四路召回（Original/Canonical Dense + Original/Expanded BM25）
  → Weighted RRF（0.20/0.30/0.15/0.35）
  → Cross Encoder 重排（阈值 0.3，低于不进上下文）
  → Evidence Check 五项判断（实体覆盖/症状词/数值/冲突/单点）
  → 充分：带 [chunk_id] 引用的证据约束生成
  → 不足：改写重查（上限 2 次）→ 仍不足：诚实拒答，禁止自由发挥
```

### SIMPLE — 普通问题

直接 LLM 回答；可选 RAG 注入（证据充分才进上下文）。上下文按 §33 五段式组装（System + 摘要 + 滑动窗口 + 业务状态 + 问题），分节 Token 预算。

### COMPLEX_TASK — 复杂任务（实时数据 / 业务操作 / 多工具依赖）

ReAct 循环（文本 JSON 协议，步数上限 6）：Reason → Select Tool → Action → Observation 迭代。九个工具：`agriculture_rag` / `weather` / `sensor` / `asset_query` / `alarm_query` / `task_query` / `task_create` / `task_update` / `user_context`。

安全闸：
- **ToolPolicyGate**——想下专业农艺结论而状态中无有效证据时，强制注入 `agriculture_rag` 取证（一次性护栏防死循环）
- **敏感操作**——工单创建/更新在工具层强制拦截为草稿，经 `/api/chat/confirm` 用户确认后以幂等键执行
- **双层工具权限**——ReAct prompt 只列该角色可见的工具 + 执行层强制拦截

## 技术选型

| 层 | 选型 | 为什么 |
|---|---|---|
| 编排 | **LangGraph** 1.2.10 | 显式节点/边，19 节点全拓扑可审计；条件分支清晰 |
| 向量库 | **milvus-lite**（嵌入式） | 单文件本地库，无服务器/Docker；原生 metadata 过滤支撑租户/角色/园区三级数据权限（array_contains 实测隔离） |
| BM25 | **纯 Python 自实现**（~80 行，k1=1.5/b=0.75） | 万级 chunk 毫秒级，无需外置服务；分词逻辑可控（多语混排） |
| 嵌入 | **bge-small-zh-v1.5**（本地） | 语料以中文为主；本地推理零成本、数据不出境 |
| 重排 | **bge-reranker-v2-m3**（本地 CrossEncoder） | 实测相关/无关判别 0.80/0.00；RRF 解决「找得到」，重排解决「排得准」 |
| LLM | **Provider 注入**（OpenAI 兼容） | 显式配置 `DURIAN_AGENT_LLM_*`，缺失即报错；测试一律 FakeLLM 离线零成本 |
| VLM / OCR | 依赖注入通道 | VLM 三要素描述（图片描述/图中文字/农业语义）；OCR 双通道（tesseract/VLM），本机缺失时清晰报错不静默降级 |
| 关系库 | SQLite 八表 | 部署最简；换 PG 仅换 DDL 方言 |
| 缓存 | RedisLike 进程内实现 | 与 redis-py 同接口，生产按同签名切换 |

**刻意不用的**：LlamaIndex/LangChain 检索框架（管线自研，行为可控）、整句翻译式跨语言（实测泰文在中文 embedding 下语义塌缩，改用**实体级别名扩展**的四路召回兜底）。

## 知识库

```text
语料 8351 chunks，四来源：
  泰国 legacy docx 651 / 文献抢救 4961 / FAQ 1000 / PDF 解析 1739
语言：zh 6700+ / th 792 / en 347 / ms 155
```

- `rag_build/`：`chunks.jsonl`（含 §22 十键 Metadata）+ 双语对照产物 + 术语库
- 术语库 `glossary_v2.json`：206 canonical 概念 / 527 四语别名（zh199/en203/th102/ms23），canonical_id 确定性可复现
- 泰文语料声调符号损坏修复（两套互不相容的 legacy 方案，逐文档判别，幂等）
- `rag_eval/gold.jsonl`：1247 条四语金标评估集（语言×意图×难度维度）
- `raw/` `rag_pdfs/`：源文档；`clean_chunks.jsonl`：丢失文献的唯一幸存解析产物（不可再生）

## 关键设计决策

| 决策 | 理由 |
|---|---|
| 高风险结论必须证据支撑，检索失败**拒答**而非自由发挥 | 农药/剂量幻觉的代价 >> 多一次检索 |
| 风险标志识别**宁多勿漏**（OR 合并规则层与 LLM 层） | 漏检=幻觉，误检只多一次检索 |
| 跨语言靠实体级别名扩展，不整句翻译 | 整句翻译引入噪声与语义漂移；实体归一是确定性的 |
| 摘要先摘后剪，总量未超阈值**零压缩零丢失** | 先摘后剪不丢内容；平时零成本 |
| 幂等键贯穿工单创建与确认重放 | Agent retry 与网络 retry 都不会重复建单 |
| 长会话五级降级（L0 完整 → L4 仅问题+业务状态） | Token 膨胀不拖垮可用性；各级上下文量单调不增 |

## 快速开始

```bash
# 环境：Python 3.12
pip install -r requirements.txt

# 全套测试（561 项，离线，不调用任何外部服务）
python -m unittest discover tests -p "test_agent_*.py"

# 启动 API（离线模式：无 LLM 时接口契约仍完整）
uvicorn durian_agent.api.app:create_app --factory --host 127.0.0.1 --port 8001

# 接真实 LLM（任一 OpenAI 兼容端点；不配置则相关路径诚实降级）
export DURIAN_AGENT_LLM_BASE_URL=...
export DURIAN_AGENT_LLM_MODEL=...
export DURIAN_AGENT_LLM_API_KEY=...
```

### API 一览（§60 契约）

```http
POST /api/chat          # {thread_id, message, language} → {answer, route, sources, pending_confirmation, thread_id}
POST /api/chat/confirm  # {thread_id, confirmation_id, approved} → 敏感操作执行/取消（一次性，403 会话校验）
POST /api/rag/search    # {query, semantic_schema?, top_k} → {documents, evidence_sufficient}
```

请求头：`X-User-Id` / `X-Role`（worker/manager/admin）/ `X-Tenant-Id`。

## 项目结构

```text
durian_agent/
├── graph.py            # LangGraph 图：19 节点 + 全边分支（§44/§45）
├── state.py            # AgentState §4 全 23 字段
├── router.py           # 三路规则路由（策略数据化）
├── normalize.py        # 四语输入归一化（NFKC 坑防护）
├── thai.py             # 泰语修复 + 词典优先分词
├── glossary.py         # 术语库（v1 扁平 + v2 canonical_id 体系）
├── llm.py              # LLM Provider 注入（FakeLLM/OpenAI 兼容）
├── errors.py / retry.py    # §50 九类错误目录 + 瞬态-only 重试
├── observability.py    # §52 15 字段全链路 Trace
├── semantic/           # Intent/生育阶段/实体归一/风险标志/SemanticParse
├── rag/                # 建库/四路召回/RRF/重排/证据判断/改写/生成/VLM/OCR
├── agent/              # SimpleAgent（五段式上下文）
├── tools/              # 九工具 + ReAct 引擎 + PolicyGate + Provider 体系
├── memory/             # Checkpoint/滑动窗口摘要/Token 预算/五级降级
├── evaluation/         # 检索指标/生成质量/评估集/覆盖率/长对话成功率
├── storage/            # 八表 SQLite + RedisLike
├── api/                # Gateway / chat / confirm / rag-search / 确认存储
└── services.py         # 三服务拆分（Agent/RAG/BusinessTool）
```

## 质量与验收

- **66 个任务全部完成**（架构文档 66 节无遗漏拆解，五阶段实施，每任务「代码+测试+独立 commit」）
- **561 项单元/集成测试全绿**（unittest，全离线，外部依赖全部替身注入）
- **端到端回归**：§43 十六步场景（天气→土壤→强制取证→四路召回→工单确认→总结），真实 milvus-lite + BM25 组件
- 离线指标：Recall@K / Hit@1/3 / MRR / NDCG 四语种分报；生成质量五项（重点 Faithfulness / Abstention Accuracy）
- 已知边界（诚实记账）：术语库别名数低于文档建议区间（扩充需批量四语翻译）；图片/扫描语料为 0（管线已备）；阈值权重待评估集跑批调优

## 文档

- [架构设计（66 节）](docs/durian_agent_architecture.md) — 唯一设计事实源
- [任务规划](docs/durian_agent_task_plan.md) — 66 任务五阶段全景与提交索引
- [会话隔离机制](docs/会话隔离机制.md) / [对话架构双路径方案](docs/对话架构双路径方案.md) / [本地Demo异常处理与超时机制](docs/本地Demo异常处理与超时机制.md)
