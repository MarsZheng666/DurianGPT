"""Token 膨胀五级降级（架构文档 §36，任务 #48）。

    Level 0  完整上下文
    Level 1  缩小滑动窗口（窗口减半）
    Level 2  删除重复和低价值工具输出
    Level 3  只保留摘要 + 最近关键消息
    Level 4  只保留当前问题 + 必要业务状态

触发（§36）：token overflow / summary failure / model context error /
timeout。错误驱动的事件式降级由错误处理策略（#46，阶段五）调用
next_level_on_error 完成升级；本模块提供各级别的确定性收缩实现。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from durian_agent.state import DegradeLevel

#: L3 保留的「最近关键消息」条数
CRITICAL_RECENT = 2


def shrink_window(messages: Sequence[AnyMessage],
                  factor: float = 0.5) -> List[AnyMessage]:
    """L1：窗口减半（保最近的）。"""
    keep = max(2, int(len(messages) * factor))
    return list(messages[-keep:])


def drop_tool_outputs(messages: Sequence[AnyMessage]) -> List[AnyMessage]:
    """L2：剔除 ToolMessage 原始输出（保留一行标记防上下文断裂）。"""
    out: List[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            out.append(ToolMessage(content="[工具输出已省略]",
                                   tool_call_id=msg.tool_call_id))
        else:
            out.append(msg)
    return out


def critical_messages(messages: Sequence[AnyMessage],
                      count: int = CRITICAL_RECENT) -> List[AnyMessage]:
    """L3：最近关键消息——对话轮次（Human/AI），不含原始工具输出
    （工具输出在 L2 已被剔除，L3 不会因大 ToolMessage 反而变大）。"""
    conversation = [m for m in messages
                    if isinstance(m, (HumanMessage, AIMessage))]
    if len(conversation) <= count:
        return conversation
    return conversation[-count:]


def apply_degrade(
    level: int,
    *,
    recent_messages: Sequence[AnyMessage],
    summary: str,
    business_state: str,
    query: str,
) -> Dict[str, Any]:
    """按级别收缩上下文各部分（与 #51 build_context 的入参对齐）。"""
    level = int(level)
    if level <= 0:
        return {"recent_messages": list(recent_messages), "summary": summary,
                "business_state": business_state, "query": query,
                "degrade_level": DegradeLevel.FULL_CONTEXT}
    if level == 1:
        return {"recent_messages": shrink_window(recent_messages),
                "summary": summary, "business_state": business_state,
                "query": query, "degrade_level": DegradeLevel.SHRINK_WINDOW}
    if level == 2:
        return {"recent_messages": drop_tool_outputs(recent_messages),
                "summary": summary, "business_state": business_state,
                "query": query, "degrade_level": DegradeLevel.DROP_TOOL_OUTPUT}
    if level == 3:
        return {"recent_messages": critical_messages(recent_messages),
                "summary": summary, "business_state": business_state,
                "query": query, "degrade_level": DegradeLevel.SUMMARY_ONLY}
    # Level 4：只留当前问题 + 必要业务状态
    return {"recent_messages": [], "summary": "",
            "business_state": business_state, "query": query,
            "degrade_level": DegradeLevel.MINIMAL}


#: 错误 → 降级目标（§36 触发清单；事件式调用在 #46/#50）
_ERROR_LEVEL_MAP = {
    "LLM_TOKEN_OVERFLOW": None,      # 逐级 +1（可能连续溢出需多级降）
    "SUMMARY_FAILED": 3,             # 摘要不可用 → 直接摘要级以下的保底组合
    "LLM_TIMEOUT": None,             # 超时属重试域（#50），不降级
}


def next_level_on_error(error_type: Optional[str],
                        current_level: int) -> int:
    """错误驱动的级别推进。返回新级别（不高于 4）。"""
    if error_type in ("LLM_TIMEOUT", "TOOL_TIMEOUT"):
        return current_level          # 超时走重试，不动上下文
    if error_type == "SUMMARY_FAILED":
        return max(current_level, _ERROR_LEVEL_MAP[error_type])
    if error_type in ("LLM_TOKEN_OVERFLOW", "RAG_NO_RESULT"):
        return min(current_level + 1, int(DegradeLevel.MINIMAL))
    # 未知错误：保守降一级
    return min(current_level + 1, int(DegradeLevel.MINIMAL))


def estimate_context_tokens(parts: Dict[str, Any]) -> int:
    """降级判定辅助：各部分合计的保守 token 估计。"""
    from durian_agent.memory.budget import estimate_tokens

    total = estimate_tokens(parts.get("summary", ""))
    total += estimate_tokens(parts.get("query", ""))
    total += estimate_tokens(parts.get("business_state", ""))
    total += sum(estimate_tokens(str(m.content))
                 for m in parts.get("recent_messages") or [])
    return total


def degrade_until_fits(
    parts: Dict[str, Any],
    token_limit: int,
    max_level: int = int(DegradeLevel.MINIMAL),
) -> Dict[str, Any]:
    """从当前级别逐级降，直到上下文估计量低于限额（或到 L4）。"""
    level = int(parts.get("degrade_level", 0)) \
        if "degrade_level" in parts else 0
    current = apply_degrade(level, **_kwargs_of(parts))
    while (estimate_context_tokens(current) > token_limit
           and int(current["degrade_level"]) < max_level):
        level = int(current["degrade_level"]) + 1
        current = apply_degrade(level, **_kwargs_of(parts))
    return current


def _kwargs_of(parts: Dict[str, Any]) -> Dict[str, Any]:
    return {k: parts.get(k, "" if k != "recent_messages" else [])
            for k in ("recent_messages", "summary", "business_state", "query")}
