# 榴莲种植园多语言 AI Agent 架构与流程设计方案

## 1. 项目目标

面向马来西亚榴莲种植园智慧运营场景，构建一个支持中文、英文、泰文、马来语的专业领域 AI Agent。

系统主要支持以下能力：

1. 榴莲种植专业知识问答；
2. SOP、操作规程、内部文档查询；
3. 病虫害、施肥、灌溉、花果管理等专业问题辅助判断；
4. 园区、地块、设备、人员等资产查询；
5. 自然语言创建、查询和操作工单；
6. 天气、土壤、传感器等实时数据查询；
7. 多轮对话和长会话记忆；
8. 不同角色之间的数据和工具权限隔离。

设计原则：

- 高风险农业专业结论必须有知识库证据支撑；
- 简单问题不强制进入复杂 Agent 流程；
- 复杂问题由 ReAct 动态编排工具；
- 多语言输入统一映射到语言无关的农业语义空间；
- 原始语言保留，不通过统一翻译成中文解决多语言问题；
- 工具权限与知识数据权限分别控制；
- 长会话不能因为 Token 膨胀导致整体不可用。

---

## 2. 总体架构

整体划分为六层：

```text
┌──────────────────────────────────────────────┐
│                 Client Layer                 │
│ Web / App / IM / Plantation Operation UI    │
└───────────────────────┬──────────────────────┘
                        ↓
┌──────────────────────────────────────────────┐
│              Conversation Gateway            │
│ Auth / User / Role / Thread / Language       │
└───────────────────────┬──────────────────────┘
                        ↓
┌──────────────────────────────────────────────┐
│             LangGraph Agent Layer            │
│                                              │
│ Normalize → Semantic NLU → Rule Router       │
│                   ↓                          │
│      MUST_RAG / SIMPLE / COMPLEX_TASK        │
│                                              │
│ RAG Pipeline        ReAct Agent              │
│                      ↓                       │
│          Tool Policy / Permission Gate       │
└───────────────┬──────────────────────────────┘
                ↓
┌──────────────────────────────────────────────┐
│                  Tool Layer                  │
│ Agriculture RAG                             │
│ Weather Tool                                │
│ Soil / Sensor Tool                          │
│ Orchard Asset Tool                          │
│ Task / Work Order Tool                      │
│ Alarm Tool                                  │
└───────────────────────┬──────────────────────┘
                        ↓
┌──────────────────────────────────────────────┐
│                Knowledge Layer               │
│ Milvus / BM25 / Metadata / Terminology       │
│ SOP / Agriculture Docs / Asset DB / Task DB │
└───────────────────────┬──────────────────────┘
                        ↓
┌──────────────────────────────────────────────┐
│             Observability Layer              │
│ Trace / Evaluation / Metrics / Audit         │
└──────────────────────────────────────────────┘
```

---

## 3. LangGraph 主流程

推荐将流程拆成明确节点，而不是把整个 ReAct 循环塞进一个 Node。

```text
START
  ↓
ContextInit
  ↓
InputNormalize
  ↓
SemanticParse
  ↓
RuleRouter
  ↓
┌───────────────────┬─────────────────────┬─────────────────────┐
│ MUST_RAG          │ SIMPLE              │ COMPLEX_TASK        │
↓                   ↓                     ↓
RAG Pipeline        SimpleAgent           ReActAgent
↓                   ↓                     ↓
AnswerCompose       AnswerCompose         ToolPolicyGate
                                            ↓
                                      Tool Execution
                                            ↓
                                        ReActAgent
                                            ↓
                                      AnswerCompose
                                            ↓
                                      Safety / Grounding
                                            ↓
                                           END
```

---

## 4. LangGraph State 设计

```java
public class AgentState {

    // ===== 会话基础信息 =====
    String threadId;
    String userId;
    String role;
    String language;

    // ===== 用户输入 =====
    String originalQuery;
    String normalizedQuery;

    // ===== 多语言语义理解 =====
    SemanticSchema semantic;

    // ===== 路由 =====
    RouteType route;

    // ===== RAG =====
    List<SearchQuery> searchQueries;
    List<RetrievedDocument> retrievedDocs;
    List<RetrievedDocument> rerankedDocs;

    int retrievalCount;
    boolean evidenceSufficient;

    // ===== ReAct =====
    List<Message> messages;
    List<ToolCallRecord> toolCalls;

    // ===== 记忆 =====
    String historySummary;
    List<Message> recentMessages;

    // ===== 权限 =====
    Set<String> allowedTools;
    Set<String> allowedKnowledgePartitions;

    // ===== 输出 =====
    String finalAnswer;

    // ===== 错误和降级 =====
    ErrorType lastError;
    int retryCount;
    DegradeLevel degradeLevel;
}
```

---

## 5. 多语言语义识别层

不采用多语言关键词规则堆叠，而采用：

```text
中文 / English / ไทย / Bahasa Melayu
                 ↓
          Multilingual NLU
                 ↓
          Canonical Schema
```

LLM 只负责回答“用户在问什么”，不负责最终决定是否必须 RAG。

---

## 6. Semantic Schema

```json
{
  "language": "ms",
  "intent": "irrigation_decision",

  "entities": {
    "cultivar": "CULTIVAR_D197",
    "orchard": "ORCHARD_03",
    "plot": null,
    "fertilizer": null,
    "pesticide": null,
    "disease": null,
    "pest": null
  },

  "growth_stage": "flowering",

  "plant_parts": [
    "leaf"
  ],

  "symptoms": [
    "leaf_yellowing"
  ],

  "environment": {
    "rainfall": "high",
    "soil_moisture": null,
    "temperature": null,
    "humidity": null,
    "drainage": "unknown"
  },

  "risk_features": {
    "diagnosis_requested": false,
    "dosage_requested": false,
    "pesticide_related": false,
    "regulation_related": false,
    "weather_dependent": true,
    "realtime_data_required": true,
    "domain_knowledge_required": true,
    "business_action_required": false
  }
}
```

---

## 7. 榴莲领域 Intent

```text
disease_diagnosis
pest_diagnosis
disease_control
pest_control
fertilization
irrigation
nutrient_diagnosis
flowering_management
fruit_management
pruning
soil_management
weather_risk
harvest
variety_query
general_knowledge
asset_query
alarm_query
task_create
task_query
task_update
```

---

## 8. 生育阶段

```text
seedling
vegetative
pre_flowering
flowering
fruit_set
fruit_development
pre_harvest
harvest
post_harvest
unknown
```

---

## 9. 领域实体体系

Canonical Entity 与多语言术语表共用同一套标准 ID。

例如：

```text
CULTIVAR_D197
canonical_name = Musang King

aliases:
zh:
  猫山王

en:
  Musang King
  D197
  Mao Shan Wang

ms:
  Raja Kunyit

th:
  对应泰语写法
```

统一语义识别：

```text
alias → canonical_id
```

RAG Query Expansion：

```text
canonical_id → multilingual aliases
```

---

## 10. 多语言领域术语库

建议第一版：

```text
约 300～500 个 canonical concept
约 1000～2000 条 multilingual alias
```

当前可以维护：

```text
1330+ 四语种领域术语/别名
```

主要覆盖：

```text
榴莲品种
病害
虫害
病原体
农药
有效成分
肥料
营养元素
生育阶段
植株部位
症状
土壤指标
天气指标
园区作业
设备
任务类型
```

数据结构：

```json
{
  "canonical_id": "DISEASE_017",
  "type": "disease",
  "canonical_name": "Durian Stem Canker",

  "aliases": {
    "zh": ["榴莲茎干溃疡病"],
    "en": ["stem canker", "durian stem canker"],
    "ms": ["..."],
    "th": ["..."]
  },

  "related_entities": [
    "PATHOGEN_PHYTOPHTHORA_PALMIVORA"
  ]
}
```

---

## 11. 输入文本归一化

InputNormalize 节点处理：

```text
Unicode normalization
空格处理
标点处理
大小写处理
数字标准化
单位标准化
泰语特殊字符规范化
```

泰语额外：

```text
Thai text
↓
基础 normalization
↓
领域词典优先匹配
↓
Thai tokenizer
↓
domain-aware tokens
```

专业术语优先作为完整 token 保留。

---

## 12. Rule Router

Router 不直接使用原始语言。

输入：

```text
Intent
+
Entity
+
Growth Stage
+
Risk Features
```

输出：

```text
MUST_RAG
SIMPLE
COMPLEX_TASK
```

---

## 13. MUST_RAG 规则

默认 MUST_RAG：

```text
病害诊断
虫害诊断
农药推荐
农药剂量
施肥剂量
具体农艺阈值
法规
标准
SOP
食品安全
出口要求
专业诊断
```

示例：

```java
if (semantic.risk.pesticideRelated) {
    return MUST_RAG;
}

if (semantic.risk.dosageRequested) {
    return MUST_RAG;
}

if (semantic.risk.regulationRelated) {
    return MUST_RAG;
}

if (semantic.risk.diagnosisRequested) {
    return MUST_RAG;
}
```

---

## 14. SIMPLE 路径

适用于：

```text
什么是榴莲坐果？
为什么需要修剪？
滴灌和喷灌有什么区别？
把刚才的内容总结成三点
```

普通解释可直接 LLM；如模型认为需要知识，可调用 RAG。

---

## 15. COMPLEX_TASK 路由

以下情况进入 ReAct：

```text
需要多个工具
需要实时数据
需要园区资产
需要业务操作
需要根据前一个工具结果决定下一步
```

例如：

> 根据未来三天天气和当前土壤湿度，判断 3 号园猫山王今天是否需要灌水，如果有积水风险则创建巡检任务。

---

## 16. Complex Task 中的专业知识约束

增加 `ToolPolicyGate`：

```text
当前步骤是否涉及专业农业结论？
       ↓
      YES
       ↓
是否已有有效 RAG Evidence？
   ↓            ↓
  YES          NO
   ↓            ↓
继续推理     强制 Agriculture RAG
```

涉及以下内容时必须先获得专业证据：

```text
灌溉阈值
施肥剂量
病害诊断
农药
生育期管理
```

---

## 17. ReAct 工具集

```text
AgricultureRagTool
WeatherTool
SoilSensorTool
OrchardAssetTool
AlarmTool
TaskQueryTool
TaskCreateTool
TaskUpdateTool
UserContextTool
```

ReAct 循环：

```text
Reason
↓
Select Tool
↓
Action
↓
Observation
↓
Reason
↓
Next Tool
```

---

## 18. RAG 建库流程

```text
原始文档
  ↓
Document Parser
  ↓
结构识别
  ↓
正文 / 标题 / 表格 / 图片 / 扫描页
  ↓
分别解析
  ↓
结构化 Chunk
  ↓
实体识别
  ↓
Metadata
  ↓
Embedding
  ↓
Milvus + BM25 Index
```

---

## 19. 文档解析与分块

正文：

```text
标题
→ 子标题
→ 段落
→ 递归分块
```

Chunk 建议：

```text
300～800 tokens
```

Overlap：

```text
10%～20%
```

---

## 20. 表格处理

```text
原始表格
↓
整块解析
↓
Markdown
↓
保留：
表头
单位
行列关系
说明文字
```

---

## 21. 图片和扫描件

图片：

```text
Image
↓
VLM
↓
图片描述
+
图中文字
+
农业语义信息
```

扫描件：

```text
Scan PDF
↓
OCR
↓
版面恢复
↓
Chunk
```

---

## 22. Chunk Metadata

```json
{
  "chunk_id": "...",
  "document_id": "...",
  "language": "en",
  "title": "...",
  "section": "...",

  "entities": [
    "CULTIVAR_D197",
    "STAGE_FLOWERING"
  ],

  "domain": "fertilization",
  "source_type": "sop",

  "role_scope": [
    "worker",
    "manager"
  ],

  "orchard_scope": [
    "ORCHARD_03"
  ]
}
```

---

## 23. 多语言 RAG 检索流程

```text
Original Query
      +
Canonical Schema
      +
Terminology Dictionary
      ↓
Cross-lingual Query Expansion
```

不把所有问题统一翻译成中文。

---

## 24. Query Rewrite

采用：

> 规则优先 + LLM 补充

第一层规则：

```text
文本标准化
+
实体标准化
+
术语别名扩展
+
单位标准化
```

第二层：

当用户表达口语化或首轮检索不足时，调用 LLM Rewrite。

LLM 允许：

```text
重组已有信息
补充 canonical terms
提高检索表达
```

LLM 禁止：

```text
猜病害
猜农药
猜剂量
新增用户没有提供的事实
```

---

## 25. 四路召回

```text
1. Original Dense
2. Canonical Dense
3. Original BM25
4. Expanded BM25
```

流程：

```text
                    Query
                      ↓
             Semantic Normalize
                      ↓
             Terminology Expand
                      ↓
 ┌────────────────────┼────────────────────┐
 ↓                    ↓                    ↓
Original Dense   Canonical Dense    Original BM25
                                           ↓
                                    Expanded BM25
 └────────────────────┬────────────────────┘
                      ↓
                     RRF
```

---

## 26. Weighted RRF

公式：

```text
score(d)
=
Σ wi / (k + rank_i(d))
```

初始化权重可参考：

```text
Original Dense      0.20
Canonical Dense     0.30
Original BM25       0.15
Expanded BM25       0.35
```

最终必须基于离线评估集调优。

---

## 27. Reranker

```text
4 路召回
↓
RRF Top30~50
↓
Cross Encoder Reranker
↓
Top3~5
```

RRF 解决“找得到”，Reranker 解决“排得准”。

---

## 28. Reranker Threshold

阈值含义：

```text
文档相关性低于该值
→ 不进入最终上下文
```

阈值必须基于验证集调优，不能固定拍脑袋。

---

## 29. Evidence Check

除了 reranker score，还判断：

```text
是否覆盖关键实体
是否覆盖用户核心条件
是否覆盖数值条件
是否存在来源冲突
TopK 是否只有单点相关
```

如果证据不足：

```text
Evidence Check
   ↓
 insufficient
   ↓
Query Rewrite
   ↓
再次召回
```

最大检索次数建议：

```text
2～3 次
```

---

## 30. Final Generation

LLM 输入：

```text
System Prompt
+
User Question
+
Canonical Semantic Schema
+
Top-K Evidence
+
Source Metadata
```

Prompt 明确：

```text
只根据提供的证据回答专业结论

不得自行补充：
剂量
农药
法规
阈值
```

---

## 31. 引用设计

内部返回：

```json
{
  "answer": "...",
  "sources": [
    {
      "document_id": "DOC001",
      "chunk_id": "CHUNK_78",
      "title": "Durian Fertilization SOP",
      "section": "4.2"
    }
  ]
}
```

---

## 32. 长会话 Memory

LangGraph Checkpoint 按：

```text
thread_id
```

隔离。

thread_id 是业务会话 ID，不是 Java Thread ID。

---

## 33. 记忆结构

```text
System Prompt
+
Long-term Summary
+
Recent Sliding Window
+
Current Business State
+
Current Query
```

---

## 34. Token Budget

示例：

```text
System         10%
Summary        15%
Recent History 25%
RAG            35%
Current Query   5%
Output Reserve 10%
```

---

## 35. 滑动窗口与摘要

最近 N 轮保留原文。

超过 Token threshold 的旧消息进入 Summary。

摘要保留：

```text
用户业务上下文
当前讨论主题
重要事实
已确认参数
未完成任务
工具执行结果
```

不保留：

```text
无意义寒暄
重复回答
大量 Tool 原始输出
```

---

## 36. Token 膨胀失败降级

```text
Level 0:
完整上下文

Level 1:
缩小滑动窗口

Level 2:
删除重复和低价值工具输出

Level 3:
只保留摘要 + 最近关键消息

Level 4:
只保留当前问题 + 必要业务状态
```

用于处理：

```text
token overflow
summary failure
model context error
timeout
```

---

## 37. 多用户隔离

按以下维度隔离：

```text
user_id
thread_id
tenant_id
role
orchard_scope
```

Checkpoint key：

```text
tenantId:userId:threadId
```

---

## 38. 权限体系

权限分两层：

```text
Tool Permission
+
Data Permission
```

---

## 39. Tool Permission

Worker：

```text
AgricultureRAG
AssetQuery
TaskQuery
```

Manager：

```text
AgricultureRAG
AssetQuery
TaskQuery
TaskCreate
TaskUpdate
Weather
Sensor
```

Admin：

```text
更多管理工具
```

Agent 只注册当前用户允许的工具。

---

## 40. Data Permission

RAG 检索加 metadata filter：

```text
tenant_id = IOI
AND
role_scope contains manager
AND
orchard_scope contains ORCHARD_03
```

Milvus 可通过 partition + metadata 联合实现。

---

## 41. 敏感操作二次确认

对于：

```text
创建任务
修改任务
删除任务
大规模操作
```

流程：

```text
Agent 生成待执行操作
↓
Validation
↓
展示给用户
↓
用户确认
↓
Tool Execute
```

---

## 42. Task Tool 幂等

Task Create 必须加入：

```text
request_id / idempotency_key
```

避免 Agent retry 和网络 retry 导致重复工单。

---

## 43. 完整复杂任务示例

用户：

```text
未来三天会下雨吗？
结合现在的土壤湿度和猫山王开花期要求，
判断今天需不需要灌水。
如果有积水风险，帮我安排排水巡检。
```

流程：

```text
User
↓
InputNormalize
↓
SemanticParse
↓
COMPLEX_TASK
↓
ReAct
↓
WeatherTool
↓
SoilSensorTool
↓
PolicyGate
↓
AgricultureRAG
↓
4 路召回
↓
RRF
↓
Reranker
↓
Evidence Check
↓
返回专业证据
↓
ReAct 综合判断
↓
AssetTool
↓
Task 草稿
↓
Permission Check
↓
用户确认
↓
TaskCreateTool
↓
Final Answer
```

---

## 44. LangGraph Node 推荐

```text
contextInitNode
normalizeNode
semanticParseNode
routeNode
ragQueryBuildNode
retrieveNode
rrfNode
rerankNode
evidenceCheckNode
rewriteQueryNode
simpleAgentNode
reactAgentNode
toolPolicyNode
toolNode
permissionCheckNode
confirmationNode
memoryCompressNode
answerNode
fallbackNode
```

---

## 45. Graph Edge

```text
START
→ contextInit
→ normalize
→ semanticParse
→ route
```

Route：

```text
MUST_RAG
→ ragQueryBuild

SIMPLE
→ simpleAgent

COMPLEX_TASK
→ reactAgent
```

RAG：

```text
ragQueryBuild
→ retrieve
→ rrf
→ rerank
→ evidenceCheck
```

Evidence：

```text
sufficient
→ answer

insufficient AND retrievalCount < max
→ rewriteQuery
→ retrieve

insufficient AND max reached
→ fallback
```

ReAct：

```text
reactAgent
→ toolPolicy
```

有 tool call：

```text
toolPolicy
→ permissionCheck
→ toolNode
→ reactAgent
```

结束：

```text
reactAgent
→ answer
```

---

## 46. SemanticParse Prompt 约束

```text
You are a multilingual semantic parser for a durian plantation system.

You do NOT answer the user's question.

You only extract structured semantics.

Do not diagnose diseases.

Do not infer pesticide, dosage or treatment.

Normalize multilingual agricultural entities to canonical IDs when possible.

Return JSON only.
```

---

## 47. Query Rewrite Prompt

```text
Rewrite the query only for retrieval.

You may:
- preserve original meaning
- add canonical agricultural terminology
- resolve aliases already provided
- make implicit references explicit using conversation context

You must NOT:
- diagnose
- introduce diseases not stated or identified
- invent dosage
- invent pesticides
- add unsupported facts
```

---

## 48. RAG Generation Prompt

```text
Answer agricultural professional claims only from the supplied evidence.

If evidence is insufficient:
state that the current knowledge base does not provide enough evidence.

Do not use model prior knowledge to invent:
- dosage
- pesticide recommendations
- thresholds
- legal requirements
- SOP requirements
```

---

## 49. ReAct System Prompt

```text
For complex plantation tasks, dynamically use available tools.

If the task requires agricultural professional judgment and no validated agricultural evidence exists in the current state, call AgricultureRagTool before forming a professional conclusion.

Never execute a tool the user's role is not authorized to access.

Sensitive write operations require confirmation.
```

---

## 50. Error Handling

```text
RAG_NO_RESULT
RAG_LOW_CONFIDENCE
TOOL_TIMEOUT
TOOL_PERMISSION_DENIED
TOOL_INVALID_ARGUMENT
LLM_TIMEOUT
LLM_TOKEN_OVERFLOW
SUMMARY_FAILED
TASK_CREATE_FAILED
```

示例：

```text
RAG_NO_RESULT
→ Rewrite

LLM_TOKEN_OVERFLOW
→ MemoryDegrade

TOOL_TIMEOUT
→ Retry once
→ fallback

PERMISSION_DENIED
→ Direct response
```

---

## 51. Retry 原则

```text
LLM retry       1～2
Tool retry      1～2
RAG rewrite     2
```

写操作不盲目自动重试，必须依赖幂等键。

---

## 52. 可观测性

记录：

```text
trace_id
thread_id
user_id
route
intent
tool sequence
retrieval latency
retrieval source
RRF rank
reranker score
evidence decision
token input/output
memory compression count
fallback count
final success/failure
```

---

## 53. RAG 指标

离线：

```text
Recall@K
Hit@1
Hit@3
MRR
NDCG
```

中文、英文、泰文、马来语分别统计。

---

## 54. 生成质量指标

```text
Answer Correctness
Faithfulness
Citation Accuracy
Evidence Coverage
Abstention Accuracy
```

高风险场景重点关注：

```text
Faithfulness
Abstention Accuracy
```

---

## 55. 多语言检索评估集

1200+ 数据建议按以下维度构建：

```text
语言：
zh / en / th / ms

Intent：
病害
虫害
施肥
灌溉
SOP
园区操作
任务
资产

Difficulty：
Easy
Alias
Cross-language
Professional term
Long query
Colloquial
```

示例：

```json
{
  "query": "...",
  "language": "th",
  "intent": "disease_diagnosis",
  "gold_document_ids": ["..."],
  "gold_chunk_ids": ["..."]
}
```

---

## 56. 有效语料解析覆盖率

定义：

```text
成功解析并进入可检索索引的有效知识单元
/
人工标注的有效知识单元总数
```

按以下类型分层评估：

```text
正文
表格
图片
扫描页
```

---

## 57. 长对话成功率

定义：

```text
成功请求数
/
所有有效用户请求数
```

失败包括：

```text
token overflow
模型上下文异常
摘要失败导致请求失败
未正常生成响应
```

---

## 58. 数据库建议

Milvus：

```text
vector index
metadata
partition
```

关系数据库：

```text
users
roles
permissions
orchards
plots
tasks
tool_audit
conversation_metadata
```

Redis：

```text
session hot state
short cache
idempotency
rate limit
```

Checkpoint：

```text
LangGraph checkpoint store
```

---

## 59. 服务拆分建议

第一版不必微服务化过度。

建议：

```text
Agent Service
RAG Service
Business Tool Service
```

Agent Service：

```text
LangGraph
Semantic NLU
Router
ReAct
Memory
```

RAG Service：

```text
Query Normalize
Query Expansion
Dense
BM25
RRF
Rerank
Evidence
```

Business Tool：

```text
Weather
Sensor
Asset
Task
Alarm
```

---

## 60. API 设计

### 对话接口

```http
POST /api/chat
```

Request：

```json
{
  "thread_id": "xxx",
  "message": "...",
  "language": "auto"
}
```

Response：

```json
{
  "answer": "...",
  "route": "COMPLEX_TASK",
  "sources": [],
  "pending_confirmation": null
}
```

### Confirmation API

```http
POST /api/chat/confirm
```

```json
{
  "thread_id": "...",
  "confirmation_id": "...",
  "approved": true
}
```

### RAG API

```http
POST /api/rag/search
```

```json
{
  "query": "...",
  "semantic_schema": {},
  "user_scope": {},
  "top_k": 5
}
```

返回：

```json
{
  "documents": [],
  "evidence_sufficient": true
}
```

---

## 61. 伪代码：主 Router

```java
RouteType route(SemanticSchema s) {

    if (s.risk.businessActionRequired
            || s.risk.realtimeDataRequired) {
        return COMPLEX_TASK;
    }

    if (s.risk.pesticideRelated
            || s.risk.dosageRequested
            || s.risk.regulationRelated
            || s.risk.diagnosisRequested) {
        return MUST_RAG;
    }

    return SIMPLE;
}
```

---

## 62. 伪代码：RAG

```java
RagResult ragSearch(
        String originalQuery,
        SemanticSchema semantic,
        UserScope scope) {

    QueryBundle bundle =
        queryBuilder.build(originalQuery, semantic);

    List<Document> d1 =
        denseOriginal.search(bundle.original());

    List<Document> d2 =
        denseCanonical.search(bundle.canonical());

    List<Document> d3 =
        bm25Original.search(bundle.original());

    List<Document> d4 =
        bm25Expanded.search(bundle.expanded());

    List<Document> fused =
        weightedRrf.fuse(d1, d2, d3, d4);

    List<Document> reranked =
        reranker.rerank(originalQuery, fused);

    EvidenceResult evidence =
        evidenceChecker.check(
            originalQuery,
            semantic,
            reranked
        );

    return new RagResult(
        reranked,
        evidence
    );
}
```

---

## 63. 伪代码：Evidence Retry

```java
while (state.retrievalCount < MAX_RETRIEVAL) {

    RagResult result = ragSearch(...);

    if (result.evidenceSufficient()) {
        state.evidenceSufficient = true;
        break;
    }

    state.normalizedQuery =
        queryRewrite.rewrite(
            state.originalQuery,
            state.semantic,
            result.documents()
        );

    state.retrievalCount++;
}

if (!state.evidenceSufficient) {
    return abstain();
}
```

---

## 64. 伪代码：ReAct Policy

```java
if (agentWantsToProduceProfessionalConclusion()) {

    if (!state.evidenceSufficient) {

        return ToolCall.of(
            "AgricultureRagTool",
            buildRagArguments(state)
        );
    }
}
```

---

## 65. MVP 开发顺序

第一阶段：

```text
Semantic Schema
Rule Router
Basic RAG
Milvus
BM25
RRF
```

第二阶段：

```text
Terminology Dictionary
Four-way Retrieval
Reranker
Evidence Check
```

第三阶段：

```text
ReAct
Weather
Sensor
Asset
Task
```

第四阶段：

```text
Memory
Checkpoint
Token Degradation
```

第五阶段：

```text
Permission
Audit
Evaluation
Observability
```

---

## 66. 最终系统主线

```text
四语种用户输入
      ↓
多语言文本归一化
      ↓
领域意图 + 实体 + 生育阶段 + 风险识别
      ↓
统一农业语义 Schema
      ↓
规则路由
      ↓
┌──────────────┬───────────────┬────────────────┐
│ MUST_RAG     │ SIMPLE        │ COMPLEX TASK   │
│              │               │                │
│ 强制专业检索 │ 普通问答      │ ReAct 编排     │
│              │               │                │
│              │               │ Policy Gate    │
└──────┬───────┴───────────────┴───────┬────────┘
       ↓                               ↓
跨语言 Query Expansion             多业务 Tool
       ↓                               ↓
4 路 Dense/BM25                   Agriculture RAG
       ↓                           Weather
Weighted RRF                      Sensor
       ↓                           Asset
Reranker                          Task
       ↓                               ↓
Evidence Check                    ReAct Loop
       ↓                               ↓
Answer / Rewrite                Final Answer
       │
       ↓
Memory + Checkpoint
       ↓
Permission + Audit
```

核心设计原则：

> 多语言模型负责理解，规则负责高风险路由，RAG 负责专业证据，ReAct 负责复杂任务编排，权限系统负责能力边界，Memory 负责多轮连续性。

整个系统不采用“所有请求都丢给 Agent”或者“所有请求都强制 RAG”的极端设计，而是在确定性与灵活性之间做分层处理。
