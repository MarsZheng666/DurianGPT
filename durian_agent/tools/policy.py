"""ToolPolicyGate（架构文档 §16/§64，任务 #38）。

    当前步骤是否涉及专业农业结论？
           ↓ YES
    是否已有有效 RAG Evidence？
       ↓ NO          ↓ YES
    强制 Agriculture RAG   继续推理

专业结论判定（§16 清单）：灌溉阈值 / 施肥剂量 / 病害诊断 / 农药 /
生育期管理——由 #19 的 risk_features（domain_knowledge_required 等
标志）与诊断防治类 intent 共同给出；语义判据不足时**偏向强制侧**
（漏检代价是无证据的专业结论，误检只多一次检索）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: §16 必须先取证据的专业类别对应的意图/标志
_PROFESSIONAL_INTENTS = {
    "disease_diagnosis", "pest_diagnosis", "nutrient_diagnosis",
    "disease_control", "pest_control",
    "fertilization", "irrigation", "flowering_management",
    "fruit_management", "soil_management",
}

_PROFESSIONAL_RISK_FLAGS = (
    "pesticide_related", "dosage_requested", "diagnosis_requested",
    "regulation_related",
)


def involves_professional_conclusion(semantic: Optional[Dict[str, Any]]) -> bool:
    """§16 清单命中即视为专业结论（宁多勿漏）。"""
    if not semantic:
        return False
    intent = semantic.get("intent") or "unknown"
    if intent in _PROFESSIONAL_INTENTS:
        return True
    for secondary in semantic.get("secondary_intents") or []:
        if secondary in _PROFESSIONAL_INTENTS:
            return True
    risks = semantic.get("risk_features") or {}
    return any(risks.get(flag) for flag in _PROFESSIONAL_RISK_FLAGS)


def has_valid_evidence(state: Dict[str, Any]) -> bool:
    return bool(state.get("evidence_sufficient")) and bool(state.get("reranked_docs"))


def gate_final_answer(
    semantic: Optional[Dict[str, Any]],
    state: Dict[str, Any],
    query: str = "",
) -> Optional[Dict[str, str]]:
    """§64 伪代码落地：想下专业结论且无有效证据 → 强制 AgricultureRagTool。

    返回 None 表示放行；返回 {"tool": "agriculture_rag", "args": {...}}
    表示拦截并注入的强制工具调用。
    """
    if not involves_professional_conclusion(semantic):
        return None
    if has_valid_evidence(state):
        return None
    return {
        "tool": "agriculture_rag",
        "args": {"query": query or str((semantic or {}).get("normalized_query")
                                       or "")},
    }
