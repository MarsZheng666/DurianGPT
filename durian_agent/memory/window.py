"""滑动窗口与摘要（架构文档 §35，任务 #55）。

- 最近 N 条消息保留原文（N=16，约 8 轮），更早的进 Summary；
- 摘要保留六类信息：用户业务上下文 / 当前讨论主题 / 重要事实 /
  已确认参数 / 未完成任务 / 工具执行结果；
- 剔除三类：无意义寒暄 / 重复回答 / 大量 Tool 原始输出；
- 两层实现：llm_summarize（§35 prompt，LLM 可用时）与
  rule_summarize（确定性规则，离线兜底——LLM 失败不阻塞会话）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from durian_agent.llm import LLMProvider

#: 最近保留原文的消息条数（约 8 轮）
MAX_RECENT_MESSAGES = 16

#: 触发摘要的字符阈值（token 的保守代理；超阈才真正压缩，平时零成本）
SUMMARY_CHAR_THRESHOLD = 6000

SUMMARY_SYSTEM_PROMPT = """Summarize this durian plantation conversation for future context.

KEEP (at most, concisely):
- user's business context (orchards, plots, cultivars they grow)
- current discussion topic
- important facts established (conclusions, evidence found)
- confirmed parameters (dosage, dates, task details)
- unfinished tasks or open questions
- key tool results (weather readings, sensor values, task ids)

DROP:
- greetings and small talk
- repeated answers
- verbose raw tool output

Answer in Chinese. Be concise: bullet points, under 300 characters.
"""


def split_window(
    messages: Sequence[AnyMessage],
    max_recent: int = MAX_RECENT_MESSAGES,
) -> Dict[str, List[AnyMessage]]:
    """切分：{"recent": 最近 N 条, "overflow": 更早的}。"""
    msgs = list(messages)
    if len(msgs) <= max_recent:
        return {"recent": msgs, "overflow": []}
    return {"recent": msgs[-max_recent:], "overflow": msgs[:-max_recent]}


_SMALLTALK = {"谢谢", "你好", "好的", "嗯", "好", "ok", "okay", "hi", "hello"}


def _is_smalltalk(text: str) -> bool:
    stripped = text.strip().lower()
    return stripped in _SMALLTALK or 0 < len(stripped) <= 1


def rule_summarize(
    messages: Sequence[AnyMessage],
    previous_summary: str = "",
) -> str:
    """确定性摘要（离线兜底）：问题/结论留要点，寒暄与重复剔除，
    工具输出限幅；与既有摘要合并并封顶。"""
    lines: List[str] = []
    seen_answers = set()
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = str(msg.content).strip()
            if text and not _is_smalltalk(text):
                lines.append(f"用户问：{text[:60]}")
        elif isinstance(msg, AIMessage):
            text = str(msg.content).strip()
            key = text[:40]
            if text and not _is_smalltalk(text) and key not in seen_answers:
                seen_answers.add(key)
                lines.append(f"回答要点：{text[:80]}")
        elif isinstance(msg, ToolMessage):
            lines.append(f"工具结果：{str(msg.content)[:60]}")

    if not lines:
        return previous_summary

    merged = (f"{previous_summary}\n" if previous_summary else "") + "\n".join(lines)
    return merged[:SUMMARY_CHAR_THRESHOLD]


def llm_summarize(
    messages: Sequence[AnyMessage],
    previous_summary: str,
    llm: LLMProvider,
) -> Optional[str]:
    """LLM 摘要（§35 六保留三剔除）。失败返回 None（调用方退规则层）。"""
    transcript = "\n".join(
        f"[{type(m).__name__}] {str(m.content)[:300]}" for m in messages)
    user = (f"Previous summary:\n{previous_summary or '（无）'}\n\n"
            f"New messages:\n{transcript}")
    try:
        result = llm.complete(SUMMARY_SYSTEM_PROMPT, user)
    except Exception:
        return None
    return (result or "").strip() or None


def summarize_overflow(
    overflow: Sequence[AnyMessage],
    previous_summary: str = "",
    llm: Optional[LLMProvider] = None,
) -> str:
    """压缩溢出消息：LLM 层（可用时）→ 规则层兜底。"""
    if not overflow:
        return previous_summary
    if llm is not None:
        summarized = llm_summarize(overflow, previous_summary, llm)
        if summarized:
            return summarized[:SUMMARY_CHAR_THRESHOLD]
    return rule_summarize(overflow, previous_summary)


def compress(
    messages: Sequence[AnyMessage],
    previous_summary: str = "",
    llm: Optional[LLMProvider] = None,
    max_recent: int = MAX_RECENT_MESSAGES,
    char_threshold: int = SUMMARY_CHAR_THRESHOLD,
) -> Dict[str, Any]:
    """§35 主入口：总闸 + 窗口切分 + 溢出摘要。

    返回 {"recent", "history_summary", "trimmed_ids"}——
    trimmed_ids 供 LangGraph RemoveMessage 收缩 messages 通道，
    防止长会话无限膨胀（§1：Token 膨胀不拖垮可用性）。

    总闸语义：历史总量未超阈值时**完全不压缩**（原文全保留，零成本、
    零丢失）；超阈值才做「溢出摘要 + 裁剪」——先摘后剪，不丢内容。
    """
    msgs = list(messages)
    total_chars = sum(len(str(m.content)) for m in msgs)
    if total_chars < char_threshold:
        return {"recent": msgs, "history_summary": previous_summary,
                "trimmed_ids": []}

    window = split_window(msgs, max_recent)
    overflow = window["overflow"]
    summary = summarize_overflow(overflow, previous_summary, llm) if overflow \
        else previous_summary

    trimmed_ids = [m.id for m in overflow if getattr(m, "id", None)]
    return {"recent": window["recent"], "history_summary": summary,
            "trimmed_ids": trimmed_ids}
