"""SemanticParse（架构文档 §5/§6/§46，任务 #3）：Multilingual NLU → Canonical Schema。

两层合成：
- LLM 层（主力）：一次结构化调用产出完整 Schema（language/intent/
  entities/stage/plant_parts/symptoms/environment/risk），
  prompt 严格遵守 §46 约束——只提取语义，不回答、不诊断、不推断农药剂量；
- 规则层（确定性 + 兜底）：实体归一化（#12）、生育阶段（#14）、
  intent 兜底（#5 规则层）、风险标志（#19）。

合成策略（合并方向均偏向安全侧）：
- entities：规则层命中的槽位**覆盖** LLM（术语表归一是确定性的），
  其余槽位（如 pesticide）保留 LLM 输出；
- risk_features：两层 OR 合并（#19 merge_risk_features）；
- intent：LLM 优先，非法/缺失时规则兜底，再不行 unknown；
- growth_stage：LLM 优先（能理解"谢花后两周"这类上下文），非法时规则层。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from durian_agent.llm import LLMProvider
from durian_agent.normalize import normalize_input
from durian_agent.semantic.entities import detect_entities
from durian_agent.semantic.intent import IntentClassifier, _extract_json
from durian_agent.semantic.labels import (
    INTENTS,
    UNKNOWN_INTENT,
    normalize_intent,
    normalize_secondary_intents,
)
from durian_agent.semantic.risks import detect_risk_features, merge_risk_features
from durian_agent.semantic.stages import (
    GROWTH_STAGES,
    UNKNOWN_STAGE,
    detect_growth_stage,
    normalize_growth_stage,
)

#: §6 entities 的 7 个槽位
ENTITY_SLOTS = ("cultivar", "orchard", "plot", "fertilizer", "pesticide", "disease", "pest")

#: 植株部位封闭集（§6 示例为 leaf；全集为本项目定义，LLM 输出据此过滤）
PLANT_PARTS = (
    "root", "trunk", "stem", "branch", "leaf", "flower",
    "fruit", "seed", "shoot", "whole_plant",
)

#: 环境条件键（§6）
ENVIRONMENT_KEYS = ("rainfall", "soil_moisture", "temperature", "humidity", "drainage")

# ══════════════════ §46 prompt ══════════════════

SEMANTIC_PARSE_SYSTEM_PROMPT = """You are a multilingual semantic parser for a durian plantation system.

You do NOT answer the user's question.

You only extract structured semantics.

Do not diagnose diseases.

Do not infer pesticide, dosage or treatment.

Normalize multilingual agricultural entities to canonical IDs when possible.

Return JSON only.

The user query may be in Chinese, English, Thai, or Malay. Extract this schema:
{{
  "language": "zh|en|th|ms",
  "intent": "<one of: {intents}>",
  "secondary_intents": ["<up to 2 labels from the same set>"],
  "entities": {{
    "cultivar": null|string, "orchard": null|string, "plot": null|string,
    "fertilizer": null|string, "pesticide": null|string,
    "disease": null|string, "pest": null|string
  }},
  "growth_stage": "<one of: {stages}>",
  "plant_parts": ["<one of: {parts}>"],
  "symptoms": ["<short descriptive phrases, e.g. leaf_yellowing>"],
  "environment": {{
    "rainfall": null|string, "soil_moisture": null|string, "temperature": null|string,
    "humidity": null|string, "drainage": null|string
  }},
  "risk_features": {{
    "diagnosis_requested": bool, "dosage_requested": bool, "pesticide_related": bool,
    "regulation_related": bool, "weather_dependent": bool, "realtime_data_required": bool,
    "domain_knowledge_required": bool, "business_action_required": bool
  }}
}}

Notes:
- intent is what the user is ASKING about (e.g. intent "disease_diagnosis" means the user
  requests a diagnosis) — extracting this intent is NOT diagnosing.
- Use null for unknown entity slots and empty lists when nothing applies.
""".format(
    intents=", ".join(INTENTS),
    stages=", ".join(GROWTH_STAGES),
    parts=", ".join(PLANT_PARTS),
)

# ══════════════════ 校验与清洗 ══════════════════


def _clean_entities(raw: Any) -> Dict[str, Optional[str]]:
    entities: Dict[str, Optional[str]] = {slot: None for slot in ENTITY_SLOTS}
    if isinstance(raw, dict):
        for slot in ENTITY_SLOTS:
            value = raw.get(slot)
            entities[slot] = value.strip() if isinstance(value, str) and value.strip() else None
    return entities


def _clean_str_list(raw: Any, allowed: Optional[tuple] = None) -> List[str]:
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            value = item.strip()
            if allowed is None or value in allowed:
                result.append(value)
    return result


def _clean_environment(raw: Any) -> Dict[str, Optional[str]]:
    env = {key: None for key in ENVIRONMENT_KEYS}
    if isinstance(raw, dict):
        for key in ENVIRONMENT_KEYS:
            value = raw.get(key)
            env[key] = value.strip() if isinstance(value, str) and value.strip() else None
    return env


def _clean_risk(raw: Any) -> Dict[str, bool]:
    if not isinstance(raw, dict):
        return {}
    return {key: bool(raw.get(key)) for key in raw
            if key in ("diagnosis_requested", "dosage_requested", "pesticide_related",
                       "regulation_related", "weather_dependent", "realtime_data_required",
                       "domain_knowledge_required", "business_action_required")}


# ══════════════════ SemanticParser ══════════════════


class SemanticParser:
    """LLM 主力 + 规则合成的 Canonical Schema 解析器（§5/§6）。"""

    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm
        self._classifier = IntentClassifier(llm)

    def parse(self, query: str) -> Dict[str, Any]:
        """返回 §6 Canonical Schema（全键齐备，可直接进 RuleRouter）。"""
        normalized = normalize_input(query)

        llm_schema = self._parse_with_llm(query)

        # ── intent：LLM 优先，规则兜底 ──
        if llm_schema is not None and normalize_intent(str(llm_schema.get("intent", ""))) != UNKNOWN_INTENT:
            intent = normalize_intent(str(llm_schema.get("intent", "")))
            secondary = normalize_secondary_intents(llm_schema.get("secondary_intents") or [])
        else:
            fallback = self._classifier._fallback(normalized)
            intent = fallback["intent"]
            secondary = fallback["secondary_intents"]

        # ── entities：规则层命中的槽位覆盖 LLM（确定性归一）──
        entities = _clean_entities(llm_schema.get("entities") if llm_schema else None)
        entities.update(detect_entities(normalized))

        # ── growth_stage：LLM 优先（可理解上下文），非法时规则层 ──
        stage_raw = str((llm_schema or {}).get("growth_stage", "") or "")
        stage = normalize_growth_stage(stage_raw)
        if stage == UNKNOWN_STAGE:
            stage = detect_growth_stage(normalized)

        # ── plant_parts / symptoms / environment：LLM 输出清洗 ──
        plant_parts = _clean_str_list(
            (llm_schema or {}).get("plant_parts"), allowed=PLANT_PARTS)
        symptoms = _clean_str_list((llm_schema or {}).get("symptoms"))
        symptoms.extend(self._rule_symptoms(normalized))
        environment = _clean_environment((llm_schema or {}).get("environment"))

        # ── risk_features：两层 OR（错分偏向安全侧）──
        risks = merge_risk_features(
            _clean_risk((llm_schema or {}).get("risk_features")),
            detect_risk_features(normalized, intent=intent),
        )

        language = str((llm_schema or {}).get("language", "")).strip()

        return {
            "language": language,
            "intent": intent,
            "secondary_intents": secondary,
            "entities": entities,
            "growth_stage": stage,
            "plant_parts": plant_parts,
            "symptoms": self._dedupe(symptoms),
            "environment": environment,
            "risk_features": risks,
        }

    # ──────────── 内部 ────────────

    def _parse_with_llm(self, query: str) -> Optional[Dict[str, Any]]:
        if self.llm is None:
            return None
        try:
            raw = self.llm.complete(SEMANTIC_PARSE_SYSTEM_PROMPT, query)
            return _extract_json(raw)
        except Exception:
            return None  # 错误处理策略属任务 #56，这里先降级到规则层

    def _rule_symptoms(self, text: str) -> List[str]:
        """规则层症状：术语表 symptom 类目（泰文键→中文标准名）。

        值用中文标准名（叶片发黄）而非 §6 示例的英文 snake_case：
        证据语料是中文，Evidence Check（#30）比对时中文更直接命中。
        """
        from durian_agent.glossary import load_glossary_categorized

        symptoms: List[str] = []
        categories = load_glossary_categorized()
        entries = categories.get("symptom", {})
        value = normalize_input(text)
        for alias, canonical in entries.items():
            if alias in value:
                symptoms.append(canonical)
        return symptoms

    @staticmethod
    def _dedupe(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result
