"""实体归一化（架构文档 §9，任务 #12）：alias → canonical 标准名。

规则层（确定性、离线可用）：
- 术语表别名最长优先匹配（四语别名 → 中文标准名，同一实体的
  猫山王 / Musang King / D197 / Raja Kunyit / หมอนทอง 收敛到同一标准名）；
- 园区/地块编号模式识别（ORCHARD_03 / 3号园 / สวนที่ 3 / petak 5 → 标准编码）。

边界：
- pesticide 槽位暂不填充——术语表当前没有农药类目，LLM 层补足，
  canonical_id 体系（CULTIVAR_D197 形式）在阶段二任务 #15 落地，
  此处 canonical = 中文标准名（在术语表内稳定唯一）。
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Tuple

from durian_agent.glossary import load_glossary_categorized
from durian_agent.normalize import normalize_input

# 术语表类目 → §6 实体槽位
ENTITY_SLOT_BY_CATEGORY = {
    "variety_thai": "cultivar",
    "variety_malaysia": "cultivar",
    "variety_indonesia": "cultivar",
    "disease": "disease",
    "pest": "pest",
    "nutrient": "fertilizer",
}

# 园区/地块编号：多语言书写模式 → 标准编码 ORCHARD_n / PLOT_n
_ORCHARD_PATTERNS = [
    re.compile(r"orchard\s*[-_]?\s*(\d+)", re.IGNORECASE),   # ORCHARD_03 / orchard 3
    re.compile(r"(\d+)\s*号园"),                              # 3号园
    re.compile(r"สวน\s*(?:ที่\s*)?(\d+)"),                    # สวนที่ 3
]
_PLOT_PATTERNS = [
    re.compile(r"plot\s*[-_]?\s*(\d+)", re.IGNORECASE),       # PLOT_12
    re.compile(r"(\d+)\s*号(?:地块|地块)"),                    # 5号地块
    re.compile(r"แปลง\s*(?:ที่\s*)?(\d+)"),                   # แปลงที่ 2
    re.compile(r"petak\s*(\d+)", re.IGNORECASE),              # petak 5 (ms)
]


@lru_cache(maxsize=1)
def _alias_index() -> Tuple[Tuple[str, str, str], ...]:
    """(匹配键小写, 标准名, 槽位) 列表，按键长降序（最长优先）。

    匹配键包含两类：
    - 术语表别名（en/th 为主；disease/pest/nutrient 类目暂无中文别名，
      阶段二任务 #15 扩充）；
    - 标准名自身（炭疽病/疫霉/钾……中文查询直接命中，零成本补中文覆盖）。
    拉丁键 ≥2 字符；CJK 单字标准名（氮/磷/钾）信息密度高，允许。
    """
    entries: List[Tuple[str, str, str]] = []
    for category, aliases in load_glossary_categorized().items():
        slot = ENTITY_SLOT_BY_CATEGORY.get(category)
        if not slot:
            continue
        for alias, canonical in aliases.items():
            for key in {alias, canonical}:
                folded = key.casefold()
                if len(folded) >= 2 or any("\u4e00" <= ch <= "\u9fff" for ch in folded):
                    entries.append((folded, canonical, slot))
    entries.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple(entries)


def detect_entities(text: str) -> Dict[str, str]:
    """从输入中抽取可确定性归一的实体槽位。

    返回 §6 entities 的子集；同一槽位取最长别名的命中。
    输入先过 normalize_input(fold_case=True)，全半角/大小写差异不漏。
    """
    value = normalize_input(text, fold_case=True)
    result: Dict[str, str] = {}

    for alias, canonical, slot in _alias_index():
        if slot in result:
            continue
        if alias in value:
            result[slot] = canonical

    for patterns, prefix, slot in (
        (_ORCHARD_PATTERNS, "ORCHARD", "orchard"),
        (_PLOT_PATTERNS, "PLOT", "plot"),
    ):
        if slot in result:
            continue
        for pattern in patterns:
            m = pattern.search(value)
            if m:
                result[slot] = f"{prefix}_{int(m.group(1))}"
                break

    return result


# ══════════ v2：canonical_id 识别（§9，任务 #15）══════════

#: v2 type → §6 实体槽位（与 ENTITY_SLOT_BY_CATEGORY 对齐）
_SLOT_BY_TYPE = {
    "cultivar": "cultivar",
    "disease": "disease",
    "pest": "pest",
    "nutrient": "fertilizer",
}


@lru_cache(maxsize=1)
def _v2_id_index() -> Tuple[Tuple[str, str, str], ...]:
    """(匹配键小写, canonical_id, 槽位) 按键长降序——从 glossary_v2 构建。"""
    from durian_agent.glossary import load_glossary_v2

    data = load_glossary_v2()
    entries: List[Tuple[str, str, str]] = []
    for entry in (data or {}).get("entries", []):
        slot = _SLOT_BY_TYPE.get(entry.get("type", ""))
        if not slot:
            continue
        forms = {entry["canonical_name"]}
        for lang_aliases in entry.get("aliases", {}).values():
            forms.update(lang_aliases)
        for form in forms:
            folded = form.casefold()
            if len(folded) >= 2 or any("\u4e00" <= ch <= "\u9fff" for ch in folded):
                entries.append((folded, entry["canonical_id"], slot))
    entries.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple(entries)


def detect_entities_with_ids(text: str) -> Dict[str, str]:
    """§9 正名形态：alias → canonical_id（如 D197 → CULTIVAR_xxx）。

    与 detect_entities 同规则（最长优先、同一槽位取最长命中），
    值为 canonical_id 而非标准名。
    """
    value = normalize_input(text, fold_case=True)
    result: Dict[str, str] = {}
    for alias, canonical_id, slot in _v2_id_index():
        if slot in result:
            continue
        if alias in value:
            result[slot] = canonical_id
    for patterns, prefix, slot in (
        (_ORCHARD_PATTERNS, "ORCHARD", "orchard"),
        (_PLOT_PATTERNS, "PLOT", "plot"),
    ):
        if slot in result:
            continue
        for pattern in patterns:
            m = pattern.search(value)
            if m:
                result[slot] = f"{prefix}_{int(m.group(1))}"
                break
    return result
