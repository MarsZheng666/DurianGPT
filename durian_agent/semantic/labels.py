"""领域标签集（架构文档 §7/§8）——LLM 打标签的封闭目标集。

设计（对齐双路径架构定案）：LLM 把任意语言的问句映射到这套固定标签，
路由决策变成查策略表（纯数据、可审计）。规则层只做高置信兜底，
禁止为追求覆盖而堆逐语言关键词表（现有代码已有漂移实例）。

注：文档 §6 示例中出现 intent "irrigation_decision"，不在 §7 清单内，
按 §7 为准并将其作为 "irrigation" 的别名收编（INTENT_ALIASES）。
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

#: §7 的 20 个领域 Intent（顺序即文档顺序）
INTENTS: Tuple[str, ...] = (
    "disease_diagnosis",
    "pest_diagnosis",
    "disease_control",
    "pest_control",
    "fertilization",
    "irrigation",
    "nutrient_diagnosis",
    "flowering_management",
    "fruit_management",
    "pruning",
    "soil_management",
    "weather_risk",
    "harvest",
    "variety_query",
    "general_knowledge",
    "asset_query",
    "alarm_query",
    "task_create",
    "task_query",
    "task_update",
)

#: LLM 输出兜底标签（不在 §7 内，路由层按保守策略处理）
UNKNOWN_INTENT = "unknown"

#: 文档内部不一致处收编的别名（§6 示例 → §7 正名）
INTENT_ALIASES = {
    "irrigation_decision": "irrigation",
}

_VALID = frozenset(INTENTS) | {UNKNOWN_INTENT}


def is_valid_intent(intent: str) -> bool:
    return intent in _VALID


def normalize_intent(intent: str) -> str:
    """LLM 输出的 intent 清洗：去空白、别名收编；非法值返回 unknown。"""
    if not isinstance(intent, str):
        return UNKNOWN_INTENT
    cleaned = intent.strip()
    cleaned = INTENT_ALIASES.get(cleaned, cleaned)
    return cleaned if cleaned in _VALID else UNKNOWN_INTENT


def normalize_secondary_intents(intents: Iterable) -> List[str]:
    """secondary intents 清洗：逐个归一，去重，剔除 unknown，保序。"""
    seen = set()
    result = []
    for item in intents or ():
        normalized = normalize_intent(item)
        if normalized != UNKNOWN_INTENT and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
