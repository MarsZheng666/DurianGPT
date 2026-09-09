"""durian_agent — 榴莲种植园多语言 AI Agent（docs/durian_agent_architecture.md 的实现包）。

分层（对应架构文档 §59 服务划分）：
- state        AgentState 全局状态（§4）
- normalize    多语言输入归一化（§11）
- semantic     SemanticParse / Intent / 生育阶段 / 实体归一化 / risk_features（§5-§10）
- router       RuleRouter 三路路由（§12-§15, §61）
- rag          建库 / 四路召回 / Weighted RRF / Reranker / Evidence（§18-§29）
- agent        SimpleAgent / ReAct / ToolPolicyGate（§14, §16-§17）
- memory       Checkpoint / 滑动窗口摘要 / Token 降级（§32-§36）
- api          /api/chat 等接口（§60）
"""

__version__ = "0.1.0"
