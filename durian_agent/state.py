"""AgentState — LangGraph 全局状态定义（架构文档 §4）。

设计要点：
- 全字段 total=False（可选），LangGraph 各节点按需写入自己的键；
- messages 使用 add_messages reducer，由 LangGraph 管理追加/去重；
- 其余键默认"最后写入者胜出"（LastValue），与节点化拆分的设计一致；
- init_state() 是 ContextInit 节点（任务 #9）使用的状态工厂，
  提供安全默认值：最小权限 role=worker、降级级别 0、计数器归零。
"""

from __future__ import annotations

from enum import IntEnum
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

# ═══════════════════════════ 基础枚举 ═══════════════════════════

#: 路由结果（§12）：规则路由器的三路输出
RouteType = Literal["MUST_RAG", "SIMPLE", "COMPLEX_TASK"]

#: 错误类型（§50），共 9 种
ErrorType = Literal[
    "RAG_NO_RESULT",              # → Query Rewrite 重查
    "RAG_LOW_CONFIDENCE",         # → 视证据判断决定重查或拒答
    "TOOL_TIMEOUT",               # → Retry once → fallback
    "TOOL_PERMISSION_DENIED",     # → 直接回应无权限
    "TOOL_INVALID_ARGUMENT",      # → 提示参数问题
    "LLM_TIMEOUT",                # → Retry 1~2 次
    "LLM_TOKEN_OVERFLOW",         # → MemoryDegrade 降级
    "SUMMARY_FAILED",             # → 降级到低级别记忆
    "TASK_CREATE_FAILED",         # → 幂等键重试或报告失败
]


class DegradeLevel(IntEnum):
    """Token 膨胀失败降级级别（§36），Level 0 → Level 4 逐级收缩上下文。"""

    FULL_CONTEXT = 0        # 完整上下文
    SHRINK_WINDOW = 1       # 缩小滑动窗口
    DROP_TOOL_OUTPUT = 2    # 删除重复和低价值工具输出
    SUMMARY_ONLY = 3        # 只保留摘要 + 最近关键消息
    MINIMAL = 4             # 只保留当前问题 + 必要业务状态


# ═══════════════════════════ 子结构 ═══════════════════════════


class RetrievedDocument(TypedDict, total=False):
    """检索结果的最小结构；四路召回各自标注来源（§25）。"""

    chunk_id: str
    document_id: str
    text: str
    score: float
    source_route: str        # dense_original / dense_canonical / bm25_original / bm25_expanded
    rrf_rank: int            # Weighted RRF 融合后的名次（§26）
    rerank_score: float      # Cross Encoder 重排分（§27）
    metadata: Dict[str, Any] # Chunk Metadata（§22）


class ToolCallRecord(TypedDict, total=False):
    """ReAct 工具调用留痕（§17），供审计与 Trace（§52）。"""

    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    status: str              # ok / denied / error / timeout


# ═══════════════════════════ AgentState ═══════════════════════════


class AgentState(TypedDict, total=False):
    """LangGraph 全流程共享状态，字段与架构文档 §4 一一对应。"""

    # ===== 会话基础信息 =====
    thread_id: str           # 业务会话 ID，非系统线程 ID（§32）
    user_id: str
    role: str                # worker / manager / admin（§39）
    language: str            # zh / en / th / ms

    # ===== 用户输入 =====
    original_query: str
    normalized_query: str    # InputNormalize 输出（§11）

    # ===== 多语言语义理解 =====
    semantic: Dict[str, Any] # Canonical Semantic Schema（§6，任务 #3 细化）

    # ===== 路由 =====
    route: RouteType

    # ===== RAG =====
    search_queries: List[str]        # Query Build / Rewrite 产物（§23-§24）
    retrieved_docs: List[RetrievedDocument]  # 四路召回原始结果（§25）
    reranked_docs: List[RetrievedDocument]   # RRF+Reranker 后的最终证据（§26-§27）

    retrieval_count: int             # 已检索次数，Retry 上限 2~3（§29）
    evidence_sufficient: bool        # Evidence Check 结论（§29）

    # ===== ReAct =====
    messages: Annotated[List[AnyMessage], add_messages]
    tool_calls: List[ToolCallRecord]

    # ===== 记忆 =====
    history_summary: str             # Long-term Summary（§33, §35）
    recent_messages: List[AnyMessage]  # Recent Sliding Window（§35）

    # ===== 权限 =====
    allowed_tools: set               # Tool Permission（§39）
    allowed_knowledge_partitions: set  # Data Permission（§40）

    # ===== 输出 =====
    final_answer: str

    # ===== 错误和降级 =====
    last_error: ErrorType
    retry_count: int
    degrade_level: DegradeLevel


# §4 全部 23 个字段（测试与文档对齐用的冻结清单）
STATE_FIELDS: frozenset[str] = frozenset({
    "thread_id", "user_id", "role", "language",
    "original_query", "normalized_query",
    "semantic", "route",
    "search_queries", "retrieved_docs", "reranked_docs",
    "retrieval_count", "evidence_sufficient",
    "messages", "tool_calls",
    "history_summary", "recent_messages",
    "allowed_tools", "allowed_knowledge_partitions",
    "final_answer",
    "last_error", "retry_count", "degrade_level",
})


def init_state(
    thread_id: str = "",
    user_id: str = "",
    role: str = "worker",
    language: str = "zh",
    original_query: str = "",
) -> AgentState:
    """ContextInit 用的状态工厂：安全默认值（最小权限、零计数、Level 0）。"""
    return AgentState(
        thread_id=thread_id,
        user_id=user_id,
        role=role,                # 默认最小权限，由网关按账号覆写
        language=language,
        original_query=original_query,
        normalized_query=original_query,
        semantic={},
        search_queries=[],
        retrieved_docs=[],
        reranked_docs=[],
        retrieval_count=0,
        evidence_sufficient=False,
        messages=[],
        tool_calls=[],
        history_summary="",
        recent_messages=[],
        allowed_tools=set(),      # 空 = 无工具权限，权限系统就绪前保持最小暴露
        allowed_knowledge_partitions=set(),
        final_answer="",
        retry_count=0,
        degrade_level=DegradeLevel.FULL_CONTEXT,
    )
