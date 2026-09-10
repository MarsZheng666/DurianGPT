"""记忆结构组装（架构文档 §33，任务 #51）——五段式上下文。

    System Prompt
    + Long-term Summary          （#55 滑动窗口溢出摘要）
    + Recent Sliding Window      （最近 N 条原文）
    + Current Business State     （本轮业务状态：路由/实体/待确认操作等）
    + Current Query

每段经 #57 TokenBudget 分节截断（business_state 计入 summary 预算——
同为记忆上下文；RAG 证据文本走 rag 节，由生成侧单独注入）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from langchain_core.messages import AnyMessage

from durian_agent.memory.budget import TokenBudget

_SECTION_MARKERS = {
    "summary": "【会话摘要】",
    "history": "【最近对话】",
    "business_state": "【当前业务状态】",
    "query": "【当前问题】",
}


def render_messages(messages: Sequence[AnyMessage]) -> str:
    """最近对话窗口 → 文本（角色标注）。"""
    lines = []
    for msg in messages:
        role = {"HumanMessage": "用户", "AIMessage": "助手",
                "ToolMessage": "工具"}.get(type(msg).__name__, "消息")
        lines.append(f"[{role}] {str(msg.content)[:300]}")
    return "\n".join(lines)


def build_context(
    *,
    system_prompt: str,
    query: str,
    summary: str = "",
    recent_messages: Sequence[AnyMessage] = (),
    business_state: str = "",
    budget: Optional[TokenBudget] = None,
) -> str:
    """五段式组装。空段自动省略；各段按 §34 预算独立截断。"""
    budget = budget or TokenBudget()

    parts = [budget.fit("system", system_prompt)]

    if summary:
        parts.append(_SECTION_MARKERS["summary"]
                     + budget.fit("summary", summary))

    history_text = render_messages(recent_messages)
    if history_text:
        parts.append(_SECTION_MARKERS["history"]
                     + budget.fit("history", history_text))

    if business_state:
        # 业务状态计入 summary 预算（同为记忆上下文；与摘要合计不超 15%）
        parts.append(_SECTION_MARKERS["business_state"]
                     + budget.fit("summary", business_state))

    parts.append(_SECTION_MARKERS["query"] + budget.fit("query", query))
    return "\n\n".join(parts)


def business_state_from_graph_state(state: Dict[str, Any]) -> str:
    """从图状态提取本轮业务状态（§33 的 Current Business State）。"""
    semantic = state.get("semantic") or {}
    entities = semantic.get("entities") or {}
    active_entities = {k: v for k, v in entities.items() if v}
    lines = []
    if active_entities:
        lines.append("涉及实体：" + "，".join(f"{k}={v}"
                                             for k, v in active_entities.items()))
    if semantic.get("growth_stage") and semantic["growth_stage"] != "unknown":
        lines.append(f"生育阶段：{semantic['growth_stage']}")
    pending = state.get("pending_confirmation")
    if pending:
        lines.append(f"待确认操作：{pending.get('tool')} "
                     f"{pending.get('args', {}).get('title', '')}".strip())
    return "\n".join(lines)
