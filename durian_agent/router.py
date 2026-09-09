"""RuleRouter（架构文档 §12/§13/§15/§61，任务 #13）。

路由是**纯数据驱动 + 少量顺序逻辑**，不碰原始语言文本：
输入 Canonical Schema（intent/entities/growth_stage/risk_features），
输出 MUST_RAG / SIMPLE / COMPLEX_TASK。

判定顺序严格对齐 §61 伪代码：
1. business_action_required || realtime_data_required → COMPLEX_TASK
   （先判 COMPLEX：即使同时命中 MUST_RAG 标志，专业证据由 COMPLEX 路径的
    ToolPolicyGate（任务 #36）在推理中强制补齐，安全闭环不丢）；
2. pesticide/dosage/regulation/diagnosis 任一标志，或 intent/secondary
   落在诊断防治类标签 → MUST_RAG（§13 高风险清单）；
3. 其余 → SIMPLE（§14：普通解释直接 LLM，可选 RAG）。

补充 §15 的映射：需要园区资产/告警/工单工具的意图，裸 LLM 无法回答，
归入 COMPLEX_TASK（ReAct 路径）。

策略全部是**可审计的数据表**（对齐双路径定案：改策略不改代码）；
剂量/浓度数字兜底已在 #19 规则层（语言无关），此处不重复。
"""

from __future__ import annotations

from typing import Any, Dict

from durian_agent.state import RouteType

#: §13 默认 MUST_RAG 的意图（诊断/防治类；primary 与 secondary 都查）
MUST_RAG_LABELS = frozenset({
    "disease_diagnosis",
    "pest_diagnosis",
    "nutrient_diagnosis",
    "disease_control",
    "pest_control",
})

#: §15：必须调用业务工具的意图（裸 LLM 无法回答 → ReAct 路径）
TOOL_REQUIRED_LABELS = frozenset({
    "asset_query",
    "alarm_query",
    "task_create",
    "task_query",
    "task_update",
})

#: 触发 COMPLEX_TASK 的风险标志（§61 第一条）
COMPLEX_RISK_FLAGS = ("business_action_required", "realtime_data_required")

#: 触发 MUST_RAG 的风险标志（§61 第二条 / §13 清单）
MUST_RAG_RISK_FLAGS = (
    "pesticide_related",
    "dosage_requested",
    "regulation_related",
    "diagnosis_requested",
)


def route(semantic: Dict[str, Any]) -> Dict[str, Any]:
    """输入 SemanticParse 的 Canonical Schema，输出 {"route", "reason"}。"""
    risks = semantic.get("risk_features") or {}
    intent = semantic.get("intent") or "unknown"
    secondary = semantic.get("secondary_intents") or []
    all_intents = [intent] + list(secondary)

    # ── 1. COMPLEX_TASK（§61 顺序：先于 MUST_RAG）──
    hit_complex_flag = [flag for flag in COMPLEX_RISK_FLAGS if risks.get(flag)]
    if hit_complex_flag:
        return {
            "route": "COMPLEX_TASK",
            "reason": f"risk_flags={hit_complex_flag}（实时数据/业务操作→ReAct 编排）",
        }
    hit_tool = [label for label in all_intents if label in TOOL_REQUIRED_LABELS]
    if hit_tool:
        return {
            "route": "COMPLEX_TASK",
            "reason": f"intent={hit_tool} 需要业务工具（资产/告警/工单→ReAct）",
        }

    # ── 2. MUST_RAG（§13 高风险清单）──
    hit_rag_flag = [flag for flag in MUST_RAG_RISK_FLAGS if risks.get(flag)]
    if hit_rag_flag:
        return {
            "route": "MUST_RAG",
            "reason": f"risk_flags={hit_rag_flag}（高风险结论须证据支撑）",
        }
    hit_diag = [label for label in all_intents if label in MUST_RAG_LABELS]
    if hit_diag:
        return {
            "route": "MUST_RAG",
            "reason": f"intent={hit_diag} ∈ 诊断防治类标签（§13 默认强制 RAG）",
        }

    # 病害/虫害实体在场（术语表确定性识别）：涉及植保专业结论，
    # 无 LLM 打标时规则层判不出 disease_control 意图——按宁多勿漏强制 RAG
    entities = semantic.get("entities") or {}
    if entities.get("disease") or entities.get("pest"):
        hit = [slot for slot in ("disease", "pest") if entities.get(slot)]
        return {
            "route": "MUST_RAG",
            "reason": f"entities={hit} 涉及植保专业内容（§13 专业诊断，安全侧）",
        }

    # ── 3. SIMPLE ──
    return {
        "route": "SIMPLE",
        "reason": f"intent={intent} 普通问题，直接 LLM（可选 RAG）",
    }
