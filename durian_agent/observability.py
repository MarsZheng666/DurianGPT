"""全链路 Trace（架构文档 §52，任务 #59）。

§52 要求记录的 15 项字段（TraceRecord）：
    trace_id / thread_id / user_id / route / intent / tool_sequence /
    retrieval_latency / retrieval_source / rrf_rank / reranker_score /
    evidence_decision / token_input / token_output / memory_compressions /
    final_success

来源：大部分字段可从图最终状态推导（路由/意图/工具序列/检索来源与
名次/重排分/证据判定/记忆压缩/成败）；retrieval_latency 由图 invoke
计时（含检索的整体耗时口径，节点级打点留待生产 APM）。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceRecord:
    trace_id: str = ""
    thread_id: str = ""
    user_id: str = ""
    tenant_id: str = ""
    route: str = ""
    intent: str = ""
    tool_sequence: List[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    retrieval_sources: List[str] = field(default_factory=list)
    rrf_ranks: Dict[str, int] = field(default_factory=dict)
    reranker_scores: Dict[str, float] = field(default_factory=dict)
    evidence_decision: str = ""
    token_input: int = 0
    token_output: int = 0
    memory_compressions: int = 0
    fallback_count: int = 0
    final_success: bool = False
    last_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tool_sequence(state: Dict[str, Any]) -> List[str]:
    """从消息历史提取工具调用序列（AIMessage 的 action JSON）。"""
    records = state.get("tool_calls") or []
    if records:
        return [str(r.get("tool_name", "")) for r in records
                if r.get("tool_name")]
    sequence: List[str] = []
    for msg in state.get("messages") or []:
        content = str(getattr(msg, "content", ""))
        if getattr(msg, "__class__", None).__name__ != "AIMessage":
            continue
        content = content.strip()
        if not content.startswith("{"):
            continue
        try:
            parsed = json.loads(content)
            action = parsed.get("action") or {}
            if isinstance(action, dict) and action.get("tool"):
                sequence.append(str(action["tool"]))
        except (json.JSONDecodeError, ValueError):
            continue
    return sequence


def record_from_state(
    state: Dict[str, Any],
    *,
    latency_ms: float = 0.0,
    trace_id: Optional[str] = None,
) -> TraceRecord:
    """从图最终状态构建 TraceRecord。"""
    from durian_agent.memory.budget import estimate_tokens

    semantic = state.get("semantic") or {}
    docs = state.get("reranked_docs") or []
    final_answer = state.get("final_answer", "")
    reranked = {str(d.get("chunk_id")): d for d in docs}

    record = TraceRecord(
        trace_id=trace_id or f"tr-{uuid.uuid4().hex[:12]}",
        thread_id=state.get("thread_id", ""),
        user_id=state.get("user_id", ""),
        tenant_id=state.get("tenant_id", ""),
        route=state.get("route", ""),
        intent=str(semantic.get("intent", "")),
        tool_sequence=_tool_sequence(state),
        retrieval_latency_ms=round(latency_ms, 1),
        retrieval_sources=sorted({
            str(d.get("source_route", ""))
            for d in state.get("retrieved_docs") or []
            if d.get("source_route")}),
        rrf_ranks={cid: int(d.get("rrf_rank", 0))
                   for cid, d in list(reranked.items())[:10]},
        reranker_scores={cid: round(float(d.get("rerank_score", 0.0)), 3)
                         for cid, d in list(reranked.items())[:10]
                         if d.get("rerank_score") is not None},
        evidence_decision=(
            "sufficient" if state.get("evidence_sufficient") else
            ("insufficient" if docs else "no_result")),
        token_input=estimate_tokens(
            str(state.get("normalized_query", ""))
            + "\n".join(str(d.get("text", "")) for d in docs)
            + str(state.get("history_summary", ""))),
        token_output=estimate_tokens(final_answer),
        memory_compressions=int(state.get("memory_compressions", 0)),
        fallback_count=int(state.get("retry_count", 0)),
        final_success=bool(final_answer) and not state.get("last_error"),
        last_error=str(state.get("last_error", "") or ""),
    )
    return record


class TraceCollector:
    """进程内 trace 收集（append-only；持久化对接留待生产 APM）。"""

    def __init__(self, max_records: int = 1000):
        self.records: List[TraceRecord] = []
        self._max = max_records

    def add(self, record: TraceRecord) -> None:
        self.records.append(record)
        if len(self.records) > self._max:
            self.records = self.records[-self._max:]

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(r.to_dict(), ensure_ascii=False)
                         for r in self.records)


def timed_invoke(graph, query: str, **kwargs) -> Dict[str, Any]:
    """带计时的图调用：结果挂 last_trace（图实例属性）。"""
    start = time.perf_counter()
    result = graph.invoke(query, **kwargs)
    latency_ms = (time.perf_counter() - start) * 1000
    graph.last_trace = record_from_state(result, latency_ms=latency_ms)
    return result
