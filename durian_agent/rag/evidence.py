"""Evidence Check（架构文档 §29，任务 #30）——五项判断。

"搜到了东西"≠"能支撑回答"。五项判断（§29）：
1. 是否覆盖关键实体：Semantic Schema 识别出的实体（标准名/别名）
   必须出现在证据中；
2. 是否覆盖用户核心条件：症状词必须被证据覆盖
   （双路径定案的核心教训：KB 虫害关键词再多，没提「卷叶」就答不了卷叶）；
3. 是否覆盖数值条件：查询带数字+单位（剂量/阈值类）时证据须有数字；
4. 是否存在来源冲突：多份证据对同一数值条件给出矛盾数字；
5. TopK 是否只有单点相关：只有一份证据且相关度低，不足以支撑结论。

判断结论 = sufficient + reasons（可审计，§52 Trace 需要）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from durian_agent.glossary import load_glossary
from durian_agent.normalize import normalize_input

#: 单点相关判定：证据少于该数量视为单点
MIN_EVIDENCE_DOCS = 2

#: 数值条件模式（语言无关：数字+农事单位，与 #19 剂量兜底一致）
_NUM_COND_RE = re.compile(
    r"\d+\s*(?:ml|kg|ppm|กรัม|มิลลิลิตร|ลิตร|克|毫升|升|g|l)(?![a-z0-9])"
    r"|浓度|剂量|稀释",
    re.IGNORECASE,
)


def _entity_surface_forms(canonical: str) -> List[str]:
    """实体的全部表面形式：标准名 + 术语表别名（大小写不敏感匹配用）。"""
    forms = [canonical]
    for alias, value in load_glossary().items():
        if value == canonical and alias != canonical:
            forms.append(alias)
    return forms


def _contains_any(text: str, forms: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(form.lower() in lowered for form in forms if form)


def _extract_numbers(text: str) -> List[float]:
    return [float(m.replace(",", "")) for m in re.findall(r"\d+(?:\.\d+)?", text)]


def check_evidence(
    query: str,
    semantic: Optional[Dict[str, Any]],
    docs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """五项判断。docs 为 RRF+Rerank 后的证据列表（含 text 字段）。"""
    reasons: List[str] = []
    semantic = semantic or {}

    if not docs:
        return {"sufficient": False, "reasons": ["无任何证据"]}

    top_texts = [str(d.get("text", "")) for d in docs]
    top_joined = "\n".join(top_texts)

    # ── 1. 关键实体覆盖 ──
    entities = semantic.get("entities") or {}
    for slot, canonical in entities.items():
        if not canonical:
            continue
        if not _contains_any(top_joined, _entity_surface_forms(canonical)):
            reasons.append(f"关键实体未覆盖: {slot}={canonical}")

    # ── 2. 核心症状/条件覆盖 ──
    for symptom in semantic.get("symptoms") or []:
        if not _contains_any(top_joined, [symptom]):
            reasons.append(f"核心症状词未被证据覆盖: {symptom}")

    # ── 3. 数值条件覆盖 ──
    query_numbers = _extract_numbers(query) if _NUM_COND_RE.search(query) else []
    if query_numbers:
        evidence_numbers = _extract_numbers(top_joined)
        if not evidence_numbers:
            reasons.append("查询含数值条件但证据无任何数字")

    # ── 4. 来源冲突：同一证据集内对查询数值给出矛盾值 ──
    conflict = False
    if query_numbers:
        per_doc = [_extract_numbers(t) for t in top_texts]
        # 至少两份证据各自含数字、且对查询的关键数字无交集（简单保守判据）
        with_nums = [nums for nums in per_doc if nums]
        if len(with_nums) >= 2:
            key = query_numbers[0]
            hits = [any(abs(n - key) <= max(0.5, abs(key) * 0.05) for n in nums)
                    for nums in with_nums]
            if not any(hits):
                conflict = True
                reasons.append(f"证据数值与查询条件 {key} 不一致，疑来源冲突")

    # ── 5. 单点相关 ──
    if len(docs) < MIN_EVIDENCE_DOCS and not conflict:
        reasons.append(f"证据单点（{len(docs)} 份），不足以支撑专业结论")

    return {
        "sufficient": not reasons,
        "reasons": reasons,
        "evidence_count": len(docs),
        "entity_conflict": conflict,
    }
