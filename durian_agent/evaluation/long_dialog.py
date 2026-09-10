"""长对话成功率（架构文档 §57，任务 #64）。

    成功率 = 成功请求数 / 所有有效用户请求数

失败口径（§57）：token overflow / 模型上下文异常 / 摘要失败导致请求失败 /
未正常生成响应。

实现：LongConversationHarness——用图实例跑 N 轮会话（checkpoint 续接），
按 §57 口径分类成败，产出成功率报告。离线验收用 FakeLLM + MemorySaver；
生产压测换真 LLM/SqliteSaver 即可。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from durian_agent.errors import (
    LLM_TOKEN_OVERFLOW,
    SUMMARY_FAILED,
)

_FAILURE_MARKERS = {
    LLM_TOKEN_OVERFLOW: ("token", "overflow", "上下文长度", "context length"),
    SUMMARY_FAILED: ("摘要失败", "summary fail"),
    "MODEL_CONTEXT_ERROR": ("context window", "最大上下文"),
    "NO_RESPONSE": ("未能完成", "未生成", "不可用", "无法解析"),
}


def classify_failure(answer: str) -> Optional[str]:
    """按 §57 失败口径对最终回答分类（None=成功）。"""
    text = answer or ""
    if not text.strip():
        return "NO_RESPONSE"
    for error, markers in _FAILURE_MARKERS.items():
        if any(marker in text for marker in markers):
            return error
    return None


def classify_state_failure(state: Dict[str, Any]) -> Optional[str]:
    """从图状态补充判定：last_error（如 RAG_NO_RESULT 是诚实拒答，
    属于成功响应而非请求失败——§57 口径是「请求级」成败）。"""
    # 诚实拒答（RAG_NO_RESULT + 拒答文案）是正常生成 → 成功
    return None


class LongConversationHarness:
    """多轮会话模拟器。"""

    def __init__(self, graph, turns: Sequence[str], *, thread_id: str,
                 role: str = "worker", user_id: str = "anonymous",
                 tenant_id: str = "default"):
        self.graph = graph
        self.turns = list(turns)
        self.thread_id = thread_id
        self.role = role
        self.user_id = user_id
        self.tenant_id = tenant_id

    def run(self) -> Dict[str, Any]:
        results = []
        for index, message in enumerate(self.turns):
            try:
                state = self.graph.invoke(
                    message, thread_id=self.thread_id, role=self.role,
                    user_id=self.user_id, tenant_id=self.tenant_id)
                answer = state.get("final_answer", "")
                failure = classify_failure(answer)
            except Exception as exc:              # 请求级异常（溢出等）
                answer = ""
                text = f"{type(exc).__name__}: {exc}"
                failure = next(
                    (error for error, markers in _FAILURE_MARKERS.items()
                     if any(m in text for m in markers)), "MODEL_CONTEXT_ERROR")
            results.append({"turn": index + 1, "answer": answer,
                            "failure": failure})
        return self.report(results)

    @staticmethod
    def report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        failures = [r["failure"] for r in results if r["failure"]]
        success = total - len(failures)
        by_type: Dict[str, int] = {}
        for failure in failures:
            by_type[failure] = by_type.get(failure, 0) + 1
        return {
            "total_requests": total,
            "successful": success,
            "success_rate": round(success / total, 4) if total else 0.0,
            "failures_by_type": by_type,
            "turns": results,
        }
