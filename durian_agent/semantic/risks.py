"""risk_features 风险标志识别（架构文档 §6，任务 #19）。

8 个标志是 RuleRouter（§13）的核心输入，决定问题是否强制走 RAG：
- 漏检（False Negative）的代价是模型幻觉农药/剂量 → 高风险；
- 误检（False Positive）的代价只是多一次检索 → 低成本。

因此规则层的设计原则是**错分偏向安全侧**：命中即置位（宁多勿漏），
与 LLM 层（任务 #3）的输出做 OR 合并。数字+单位模式是语言无关的
安全兜底（继承双路径 demo 已验证的剂量正则并扩充）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from durian_agent.normalize import normalize_input
from durian_agent.semantic.labels import UNKNOWN_INTENT

#: 需要领域知识支撑的意图（§7 中排除业务操作/资产/告警/通识类）
_PROFESSIONAL_INTENTS = frozenset({
    "disease_diagnosis", "pest_diagnosis", "disease_control", "pest_control",
    "fertilization", "irrigation", "nutrient_diagnosis", "flowering_management",
    "fruit_management", "pruning", "soil_management", "weather_risk",
    "harvest", "variety_query",
})

#: 诊断类意图 ⇒ diagnosis_requested（由 intent 派生，确定性最高）
_DIAGNOSIS_INTENTS = frozenset({"disease_diagnosis", "pest_diagnosis", "nutrient_diagnosis"})

#: 业务操作类意图 ⇒ business_action_required
_BUSINESS_INTENTS = frozenset({"task_create", "task_update"})

RISK_FEATURE_KEYS = (
    "diagnosis_requested", "dosage_requested", "pesticide_related",
    "regulation_related", "weather_dependent", "realtime_data_required",
    "domain_knowledge_required", "business_action_required",
)

# 语言无关兜底：数字 + 农事单位/稀释倍数（demo 已验证的 backstop 扩充）。
# 结尾用 (?![a-z0-9]) 而不是 \b：\b 在 CJK 前不成立（"ml可" 无边界）。
_DOSE_RE = re.compile(
    r"\d+\s*(?:ml|kg|ppm|กรัม|มิลลิลิตร|ลิตร|克|毫升|升|g|l)(?![a-z0-9])"
    r"|浓度|剂量|稀释|dosage|dilution",
    re.IGNORECASE,
)

_PESTICIDE_RE = re.compile(
    r"农药|杀虫剂|杀菌剂|除草剂|喷药|用药"
    r"|pesticides?|insecticides?|fungicides?|herbicides?|spray"
    r"|ยาฆ่าแมลง|ยากำจัด"
    r"|racun perosak|racun serangga|racun kulat",
    re.IGNORECASE,
)

_REGULATION_RE = re.compile(
    r"法规|标准|规程|sop|出口|食品安全|检疫"
    r"|regulations?|standards?|export|food safety"
    r"|มาตรฐาน"
    r"|piawaian|eksport|keselamatan makanan",
    re.IGNORECASE,
)

_WEATHER_RE = re.compile(
    r"天气|下雨|降雨|下雨了|雨季|台风"
    r"|weather|rain|forecast"
    r"|ฝน|อากาศ|พยากรณ์"
    r"|hujan|cuaca",
    re.IGNORECASE,
)

_REALTIME_RE = re.compile(
    r"传感器|实时|当前土壤|现在的|此刻"
    r"|sensors?|real[- ]?time|live data|soil moisture"
    r"|เซ็นเซอร์|ความชื้นในดินตอนนี้"
    r"|sensor|kelembapan tanah sekarang",
    re.IGNORECASE,
)

_BUSINESS_ACTION_RE = re.compile(
    r"创建|新建|帮我建|帮我安排|安排(?:一次|一个|巡检|排水|施肥)"
    r"|create.{0,20}task|schedule.{0,12}(?:task|inspection)"
    r"|สร้าง.{0,8}งาน|เปิดงาน"
    r"|buat.{0,10}tugasan|jadualkan",
    re.IGNORECASE,
)


def detect_risk_features(
    text: str,
    intent: str = UNKNOWN_INTENT,
) -> Dict[str, Any]:
    """规则层风险标志：8 个键全量返回，命中即 True（宁多勿漏）。

    intent 可选——已知时派生 diagnosis/business/domain_knowledge
    三个标志（确定性最高）；未知时这三个标志退化为关键词匹配或保守 False。
    """
    value = normalize_input(text)

    def _hit(pattern: re.Pattern) -> bool:
        return bool(pattern.search(value))

    diagnosis = intent in _DIAGNOSIS_INTENTS or _hit(
        re.compile(r"是什么病|得了什么|什么病害|什么虫|缺什么|怎么诊断"
                   r"|diagnose|what (?:disease|pest|deficiency)"
                   r"|เป็นโรคอะไร|โรคอะไร"
                   r"|penyakit apa|kenapa", re.IGNORECASE)
    )
    business = intent in _BUSINESS_INTENTS or _hit(_BUSINESS_ACTION_RE)
    domain = intent in _PROFESSIONAL_INTENTS

    return {
        "diagnosis_requested": diagnosis,
        "dosage_requested": _hit(_DOSE_RE),
        "pesticide_related": _hit(_PESTICIDE_RE),
        "regulation_related": _hit(_REGULATION_RE),
        "weather_dependent": _hit(_WEATHER_RE),
        "realtime_data_required": _hit(_REALTIME_RE),
        "domain_knowledge_required": domain,
        "business_action_required": business,
    }


def merge_risk_features(*layers: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """多层标志 OR 合并（规则层 × LLM 层），错分统一偏向安全侧。"""
    merged = {key: False for key in RISK_FEATURE_KEYS}
    for layer in layers:
        if not layer:
            continue
        for key in RISK_FEATURE_KEYS:
            if bool(layer.get(key)):
                merged[key] = True
    return merged
