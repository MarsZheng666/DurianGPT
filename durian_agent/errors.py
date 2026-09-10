"""错误分类与处理策略（架构文档 §50，任务 #46）。

九种错误的处理策略目录（数据化，可审计）：

    RAG_NO_RESULT          → rewrite（改写重查，上限见 RetryPolicy）
    RAG_LOW_CONFIDENCE     → evidence_check（交证据判断决定重查或拒答）
    TOOL_TIMEOUT           → retry_once → fallback
    TOOL_PERMISSION_DENIED → direct_reply（直接回应无权限，不重试）
    TOOL_INVALID_ARGUMENT  → direct_reply（提示参数问题）
    LLM_TIMEOUT            → retry（1~2 次）
    LLM_TOKEN_OVERFLOW     → memory_degrade（#48 逐级降）
    SUMMARY_FAILED         → memory_degrade（→ L3）
    TASK_CREATE_FAILED     → idempotent_retry（幂等键重试或报告失败）

异常 → 错误类型 → 策略 的解析链在 classify_exception / resolve_strategy。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# ══════════════════ §50 错误类型 ══════════════════

RAG_NO_RESULT = "RAG_NO_RESULT"
RAG_LOW_CONFIDENCE = "RAG_LOW_CONFIDENCE"
TOOL_TIMEOUT = "TOOL_TIMEOUT"
TOOL_PERMISSION_DENIED = "TOOL_PERMISSION_DENIED"
TOOL_INVALID_ARGUMENT = "TOOL_INVALID_ARGUMENT"
LLM_TIMEOUT = "LLM_TIMEOUT"
LLM_TOKEN_OVERFLOW = "LLM_TOKEN_OVERFLOW"
SUMMARY_FAILED = "SUMMARY_FAILED"
TASK_CREATE_FAILED = "TASK_CREATE_FAILED"

#: §50 全部九种（与 state.ErrorType 字面量一致）
ALL_ERROR_TYPES = (
    RAG_NO_RESULT, RAG_LOW_CONFIDENCE, TOOL_TIMEOUT,
    TOOL_PERMISSION_DENIED, TOOL_INVALID_ARGUMENT,
    LLM_TIMEOUT, LLM_TOKEN_OVERFLOW, SUMMARY_FAILED, TASK_CREATE_FAILED,
)

#: 处理动作（供路由/降级/重试组件消费）
ACTION_REWRITE = "rewrite"
ACTION_EVIDENCE_CHECK = "evidence_check"
ACTION_RETRY = "retry"
ACTION_RETRY_ONCE = "retry_once"
ACTION_MEMORY_DEGRADE = "memory_degrade"
ACTION_DIRECT_REPLY = "direct_reply"
ACTION_IDEMPOTENT_RETRY = "idempotent_retry"

#: §50 策略目录
ERROR_STRATEGIES: Dict[str, Dict[str, Any]] = {
    RAG_NO_RESULT: {"action": ACTION_REWRITE, "retry_budget": "rag_rewrite"},
    RAG_LOW_CONFIDENCE: {"action": ACTION_EVIDENCE_CHECK, "retry_budget": None},
    TOOL_TIMEOUT: {"action": ACTION_RETRY_ONCE, "retry_budget": "tool",
                   "then": "fallback"},
    TOOL_PERMISSION_DENIED: {"action": ACTION_DIRECT_REPLY, "retry_budget": None},
    TOOL_INVALID_ARGUMENT: {"action": ACTION_DIRECT_REPLY, "retry_budget": None},
    LLM_TIMEOUT: {"action": ACTION_RETRY, "retry_budget": "llm"},
    LLM_TOKEN_OVERFLOW: {"action": ACTION_MEMORY_DEGRADE, "retry_budget": None},
    SUMMARY_FAILED: {"action": ACTION_MEMORY_DEGRADE, "retry_budget": None},
    TASK_CREATE_FAILED: {"action": ACTION_IDEMPOTENT_RETRY,
                         "retry_budget": "tool"},
}

_TIMEOUT_RE = re.compile(r"timeout|timed?\s*out", re.IGNORECASE)
_OVERFLOW_RE = re.compile(
    r"maximum\s+context|token.{0,10}limit|context.{0,10}(length|window)"
    r"|too\s+many\s+tokens", re.IGNORECASE)


def classify_exception(exc: BaseException,
                       domain: str = "llm") -> str:
    """异常 → 错误类型（启发式：超时/溢出/权限/参数）。"""
    message = f"{type(exc).__name__}: {exc}"
    if _TIMEOUT_RE.search(message):
        return TOOL_TIMEOUT if domain == "tool" else LLM_TIMEOUT
    if _OVERFLOW_RE.search(message):
        return LLM_TOKEN_OVERFLOW
    text = str(exc)
    if "权限不足" in text or "PermissionError" in type(exc).__name__:
        return TOOL_PERMISSION_DENIED
    if "参数错误" in text or isinstance(exc, (ValueError, TypeError)):
        if domain == "tool":
            return TOOL_INVALID_ARGUMENT
    return LLM_TIMEOUT if domain == "llm" else TOOL_TIMEOUT


def resolve_strategy(error_type: str) -> Dict[str, Any]:
    """错误类型 → 处理策略（未知类型保守降级）。"""
    return ERROR_STRATEGIES.get(
        error_type, {"action": ACTION_RETRY_ONCE, "retry_budget": None})


def should_degrade_memory(error_type: str) -> bool:
    return resolve_strategy(error_type)["action"] == ACTION_MEMORY_DEGRADE
