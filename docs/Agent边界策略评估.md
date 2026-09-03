# Agent 边界策略评估

> 评估对象：`durian_langgraph.py`（编排层）+ `durian__inference_api.py`（adapter/域实现层）
> 日期：2026-09-02

这个 agent 的「边界」有三层：**证据边界**（不许编）、**话题边界**（只管榴莲）、**记忆边界**（只存用户明说的话）。

---

## 一、现状：已有的边界机制

### 1. 证据边界（做得最扎实的一层）

| 机制 | 位置 | 作用 |
|---|---|---|
| 追问检索词剔除 AI 旧回答 | `durian_langgraph.py:187-204, 946-954` | 未引用的回答不能变成检索输入，防止幻觉自我强化 |
| 历史白名单 | `durian_langgraph.py:1048-1056` | 只有**带证据**的 assistant 消息才能进入后续上下文 |
| 精确问答强制摘抄 | `durian_langgraph.py:1295-1338` | 编号/名称类问题只从检索原文抄，查不到就明说（1321-1331） |
| 强制检索不可绕过 | `durian_langgraph.py:1067-1073` | 路由只能改写查询，不能跳过知识库 |
| 无资料加免责声明 | `durian_langgraph.py:1345-1353, 1369-1377` | 兜底前缀「参考资料不足，仅作一般性判断」 |
| 拒绝扩写无据追问 | `durian_langgraph.py:1354-1366` | 用户说「再详细点」但查不到依据时，直接拒答 |

### 2. 话题/上下文边界

- 精确问答不继承历史，防止串味（`durian_langgraph.py:834-841`）
- 无历史的「追问」纠正为新话题（`durian_langgraph.py:856-863`）
- 闲聊/打招呼走固定回复，不进模型生成（`durian_langgraph.py:1275-1288`）
- 该澄清时追问而不是猜（`durian_langgraph.py:1289-1294`）

### 3. 记忆边界

- 只存**用户显式表达**的偏好/档案，注释明确「never infer model facts」（`durian_langgraph.py:291`）
- 敏感词拦截（`durian_langgraph.py:85-96`）：密码、验证码、身份证、银行卡、API key 等一律不存（271, 316-320）
- 疑问句不存（302-303）、单条截断 400 字符（265）、按 user_id 隔离（354, 396, 427）

---

## 二、越界风险清单

### 高风险

**1. 域外话题没有结构性硬边界，只靠 prompt 软约束**
路由的 intent 只有 `greeting / casual / new_topic / follow_up`（`durian_langgraph.py:854`），**不存在「域外拒绝」这一类**。域外拒绝只存在于生成端 `SYSTEM_PROMPTS` 的一句话指令里（`durian__inference_api.py:395-396`「只回答榴莲相关问题……应说明只能回答榴莲相关内容」）。模型忽略该指令时无结构性兜底。详见第三节。

**2. 农艺建议的下游安全风险**
「一般性判断」模式下模型可能输出农药剂量、用药方案等有现实危害的建议，前缀免责声明不能兜底。这类内容没有专门的关键词拦截或降级（`chemical_guidance_rules` 仅是 prompt 规则，`durian__inference_api.py:5817`）。

### 中风险

**3. 检索内容无注入过滤**
`evidence_text` 来自 RAGFlow/外部搜索，直接拼进 final_messages（`durian_langgraph.py:1168-1173`），没有指令过滤。知识库文档若被投毒可注入系统提示。知识库是内部的可降低风险，但链路上确实无防护。

**4. user_id 完全信任上游**
记忆的读写删全部按 user_id 隔离，但编排层对 user_id 零校验（`durian_langgraph.py:432-434` 只判非空）；adapter 层 `get_current_user` 默认值就是 `admin`（`durian__inference_api.py:603`）。前端若可伪造 `x_user_id` header，就能读写删**别人的**长期记忆——横向越权。

**5. 边界强度依赖 adapter 的分类器**
`is_exact_query` / `is_casual_query` / `is_contextual_exact_query` 全部由 adapter 提供（`durian_langgraph.py:802-815`），编排层没有兜底判定。分类器误判 = 硬规则整体失效。

### 低风险

**6. 记忆删除的词项交集匹配过宽**（`durian_langgraph.py:419-423`）：「忘记我说过 X」会删掉所有与 X 有词项重叠的记忆，可能误删相近但无关的记忆。

**7. 记忆提取的正则启发式**：「我在使用 XXX」这类句式（`durian_langgraph.py:283-285`）会被存为 profile，敏感词表只覆盖凭据类，不含健康/隐私类信息。有 400 字符截断兜底。

**8. 异常信息直接推给前端**（`durian_langgraph.py:1487`）：`str(exc)` 可能带出内部路径、SQL 等实现细节。

---

## 三、话题边界方案深入分析

### 现有结构：四道闸，只有前两道是结构性的

```
用户输入
  │
  ├─ 闸1 入口拦截（结构化，确定性规则）
  │    is_greeting_only        api:4087  精确问候词表，4 语言，不进 RAG 不继承上下文
  │    classify_casual_message api:4112  ≤24 字符的致谢/辱骂 → 固定回复，不进领域模型
  │    classify_user_intent    api:4183  greeting / explicit_follow_up / new_topic 强规则
  │
  ├─ 闸2 LLM 语义路由（可选，默认关闭）
  │    route_context_with_llm  api:4307  规则不确定时 LLM 判断是否依赖上下文
  │    · DURIAN_ENABLE_LLM_CONTEXT_ROUTER 默认 False（api:4364）→ 未开启时直接兜底 new_topic
  │    · 失败降级 new_topic（api:4445）
  │    · 输出被后处理约束：greeting/new_topic 强制不带历史（api:4429-4435）
  │
  ├─ 闸3 检索域限定（结构化，词表 + 过滤）
  │    scope_web_query_to_durian api:6720  外搜 query 强制追加 durian/榴莲
  │    evidence_is_durian_scoped api:6739  web 结果必须含榴莲标记词，否则丢弃
  │    should_use_rag_for_query  api:16009 无域词不走 RAG（stream2 路径）
  │    is_smalltalk_query        api:16059 ≤4 字符无域词当寒暄（stream2 路径）
  │
  └─ 闸4 生成端约束（软约束，prompt 一句话）
       SYSTEM_PROMPTS api:393-422 「唯一角色是榴莲专家……只回答榴莲相关问题」
       （域内/域外、消费者/种植者的细分也全在这一段里）
```

### 关键结论

**1. 话题边界的真正防线 = 词表覆盖度 + 模型自觉，而不是路由结构。**
路由器（闸1/闸2）只区分 `greeting / new_topic / follow_up`，**不做域内/域外判断**。域外问题会一路走到闸3：词表过滤会把证据清空 → `source_missing` → 免责声明 + 模型生成，此时唯一拦截是 `SYSTEM_PROMPT` 那句话。模型被绕过（越狱、长上下文稀释指令）时无兜底。

**2. LLM 语义路由默认关闭，意味着实际运行的是「短词表 + new_topic 兜底」。**
`DURIAN_ENABLE_LLM_CONTEXT_ROUTER` 未开启时（api:4364-4371），规则不认识的输入一律 new_topic，完全不区分域内域外。这解释了为什么域外问题畅通到生成端。

**3. 数字标识符的「故意豁免」与检索域限定自相矛盾。**
`classify_user_intent` 见到任何数字标识符直接判 new_topic（api:4198，注释明说允许 HTTP 429、CVE-2026-1234 这类非榴莲编号）；`is_smalltalk_query` 也放行（api:16076-16079）。但 `scope_web_query_to_durian` 会把 "HTTP 429" 改写成 "HTTP 429 durian" 去搜，`evidence_is_durian_scoped` 再滤掉不含榴莲词的结果——**入口故意放行、检索端又强制 scoped，非榴莲编号查询大概率检索失败**，最后落到 exact_not_found 或免责声明。设计意图（能问 HTTP 429）和执行（搜不到）不一致，值得重新对齐。

**4. 词表是双向风险。**
- 漏出：词表外的域外表达（超过 24 字符、无域词）漏到生成端，靠 prompt 拦。
- 误伤：词表外的**域内**表达（方言、新品种名、口语化描述）会被当成寒暄或查不到资料，正确问题得到错误回答。域词表（api:16035-16046、16082-16093）需要随业务持续维护。

**5. 与编排层的衔接缺口。**
`durian_langgraph.py` 的固定回复分支（greeting/casual/clarify/exact/grounded/soft_no_source）全部**不经过 SYSTEM_PROMPT**，只有最后的流式生成分支（1367+ → `stream_generate`）才带上域约束。软约束只在自由生成那条路上生效；不过其余分支本身是白名单式固定文案，无越界内容，可接受。

### 改进建议（按性价比排序）

1. **在 route 节点增加 `out_of_domain` intent + 固定拒答回复**（类似 clarification 的处理，`durian_langgraph.py:1289-1294` 的模式）。可以用一个轻量分类器或直接复用 LLM 路由器加一个输出枚举值。这是把「模型自觉」升级为「结构拦截」的最小改动。
2. **对齐标识符豁免与检索 scoped 的矛盾**：要么承认非榴莲编号可查（放开 scoped 过滤的标识符白名单路径），要么在入口就明确拒答，避免「放行→搜不到→免责声明」的空转。
3. **给 LLM 语义路由开默认开关评估**：它同时能改善追问识别和域外识别，代价是每次路由多一次小模型调用（max_tokens=96, temperature=0）。
4. **域外固定回复也四语言化**：如果加了 `out_of_domain`，拒答文案要像 greeting_reply 一样覆盖 zh/en/ms/th。

---

## 四、结论

**证据边界**是这套设计里真正下功夫的地方——「无据不答、无据不扩写、无据不加历史」形成闭环，基本没有越界口子。

**话题边界**是三道结构性闸门 + 一道 prompt 软闸：入口拦截做得很细（四语言词表、辱骂处理、标识符豁免），检索域限定扎实；但**路由层不做域内/域外判断**，域外拒绝完全押在生成端 SYSTEM_PROMPT 一句话上，且语义路由默认关闭。建议优先补 `out_of_domain` 结构化拦截。

**记忆边界**整体克制，主要风险在 user_id 信任链（adapter 层 `get_current_user` 默认 `admin`）。
