"""Query Rewrite 两层机制（架构文档 §24/§47，任务 #28）。

第一层（规则，离线可用）：
    文本标准化 + 实体标准化 + 术语别名扩展 + 单位标准化
    ——复用 #8 normalize_input 与 #24 build_queries。

第二层（LLM，触发条件：口语化表达或首轮检索不足）：
    遵守 §47 约束：
      You may:  保留原意 / 补 canonical 术语 / 解析已提供的别名 /
                用会话上下文把隐指变显式
      You must NOT: 诊断 / 引入未提及的病害 / 编造剂量 / 编造农药 /
                    添加无依据事实

安全护栏（本实现补充）：LLM 改写结果若引入原查询没有的**数字**，
视为违反"不得编造剂量"，丢弃改写、退回规则层结果——宁保守。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from durian_agent.llm import LLMProvider
from durian_agent.rag.recall import build_queries

# ══════════════════ §47 prompt ══════════════════

REWRITE_SYSTEM_PROMPT = """Rewrite the query only for retrieval.

You may:
- preserve original meaning
- add canonical agricultural terminology
- resolve aliases already provided
- make implicit references explicit using conversation context

You must NOT:
- diagnose
- introduce diseases not stated or identified
- invent dosage
- invent pesticides
- add unsupported facts

Return the rewritten query text only (no explanation, no JSON).
"""

#: 已提供的上下文（别名表/已检文档缺失信息）附在 user 消息里
_CONTEXT_HEADER = "Known canonical entities (use these, do not invent new ones):\n"
_DOCS_HEADER = "\nPreviously retrieved documents (insufficient, rewrite to search better):\n"


def rule_rewrite(query: str, semantic: Optional[Dict[str, Any]] = None) -> str:
    """第一层规则改写（§24 四要素）：标准化 + 实体标准名 + 术语别名扩展。

    用 expanded 形态（原查询 + 实体标准名 + 四语别名）：重试轮次的查询
    必须与首轮不同——实体已用标准名书写时 canonical 会与原查询相同，
    白白消耗一次检索机会。
    """
    queries = build_queries(query, semantic)
    return queries["expanded"] or queries["canonical"]


def _numbers_in(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _rewrite_introduces_numbers(original: str, rewritten: str) -> bool:
    """护栏：改写引入了原查询没有的数字 = 疑似编造剂量。"""
    return bool(_numbers_in(rewritten) - _numbers_in(original))


def llm_rewrite(
    query: str,
    semantic: Optional[Dict[str, Any]],
    docs: Optional[List[Dict[str, Any]]],
    llm: LLMProvider,
) -> Optional[str]:
    """第二层 LLM 改写。返回 None 表示不可用/被护栏拒绝（调用方退规则层）。"""
    user_parts = [query]
    entities = (semantic or {}).get("entities") or {}
    known = [f"{slot}={name}" for slot, name in entities.items() if name]
    if known:
        user_parts.append(_CONTEXT_HEADER + "\n".join(known))
    if docs:
        summary = "\n".join(
            f"- {str(d.get('text', ''))[:120]}" for d in docs[:3])
        user_parts.append(_DOCS_HEADER + summary)

    try:
        rewritten = llm.complete(REWRITE_SYSTEM_PROMPT, "\n".join(user_parts))
    except Exception:
        return None
    rewritten = (rewritten or "").strip().strip('"').strip("'")
    if not rewritten or rewritten == query:
        return None
    if _rewrite_introduces_numbers(query, rewritten):
        return None   # §47 护栏：不得编造剂量
    return rewritten


def rewrite_query(
    query: str,
    semantic: Optional[Dict[str, Any]] = None,
    docs: Optional[List[Dict[str, Any]]] = None,
    llm: Optional[LLMProvider] = None,
) -> str:
    """两层合成入口：LLM 层（可用时）→ 规则层兜底。"""
    if llm is not None:
        rewritten = llm_rewrite(query, semantic, docs, llm)
        if rewritten:
            return rewritten
    return rule_rewrite(query, semantic)
