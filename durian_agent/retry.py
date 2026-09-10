"""Retry 原则（架构文档 §51，任务 #50）。

    LLM retry       1～2
    Tool retry      1～2
    RAG rewrite     2
    写操作不盲目自动重试，必须依赖幂等键（§42，已落地）

- 只对**瞬态**错误重试（timeout 类）；权限不足/参数错误立即失败
  （重试也不会有不同的结果）；
- RAG rewrite 上限即图 max_retrievals=2（#29 已落地）；
- 写操作的安全性由幂等键保证（#42/#47 已落地），重试等幂等。
"""

from __future__ import annotations

from typing import Callable, TypeVar

from durian_agent.errors import (
    TOOL_PERMISSION_DENIED,
    TOOL_TIMEOUT,
    classify_exception,
)

T = TypeVar("T")

#: §51 上限（1~2 取 2；RAG rewrite 2）
DEFAULT_POLICY = {"llm": 2, "tool": 2, "rag_rewrite": 2}

#: 不重试的错误（结果确定，重试无意义）
NO_RETRY_ERRORS = {TOOL_PERMISSION_DENIED, "TOOL_INVALID_ARGUMENT",
                   "LLM_TOKEN_OVERFLOW"}


class RetryPolicy:
    def __init__(self, llm: int = 2, tool: int = 2, rag_rewrite: int = 2):
        self.limits = {"llm": llm, "tool": tool, "rag_rewrite": rag_rewrite}

    def allows(self, kind: str, attempted: int) -> bool:
        return attempted < self.limits.get(kind, 0)

    def max_attempts(self, kind: str) -> int:
        return self.limits.get(kind, 0)


def with_retry(
    fn: Callable[[], T],
    *,
    kind: str,
    policy: RetryPolicy | None = None,
    on_error: Callable[[str], None] | None = None,
) -> T:
    """瞬态错误重试执行。非瞬态（权限/参数/溢出）立即抛出。"""
    policy = policy or RetryPolicy()
    attempted = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            error = classify_exception(exc, domain=kind)
            if on_error:
                on_error(error)
            attempted += 1
            if error in NO_RETRY_ERRORS or not policy.allows(kind, attempted):
                raise
