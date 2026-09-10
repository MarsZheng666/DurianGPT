"""多语言检索评估集（架构文档 §55，任务 #60）。

底座：rag_eval/gold.jsonl——此前 RAG 工作沉淀的 1247 条四语金标
（zh 847 / en 122 / th 139 / ms 139，含金标答案与参考文档）。

本模块把它规范化为 §55 结构：
    {"query", "language", "intent", "difficulty",
     "gold_document_ids", "gold_chunk_ids"}

difficulty 维度（§55 六档）由启发式补全：
    Cross-language（cross_lingual 标记）> 原有 difficulty >
    Alias（查询命中术语表别名）> Long query（长查询）> Easy。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

GOLD_PATH = Path(__file__).resolve().parent.parent.parent / \
    "rag_eval" / "gold.jsonl"

#: §55 六档难度（封闭集）
DIFFICULTIES = ("Easy", "Alias", "Cross-language", "Professional term",
                "Long query", "Colloquial")

_LANG_KEYS = ("q_lang", "language")


def _pick_language(raw: Dict[str, Any]) -> str:
    for key in _LANG_KEYS:
        value = raw.get(key)
        if value in ("zh", "en", "th", "ms"):
            return value
    return "zh"


def _derive_difficulty(raw: Dict[str, Any], query: str,
                       alias_lookup=None) -> str:
    if raw.get("cross_lingual"):
        return "Cross-language"
    existing = raw.get("difficulty")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    if alias_lookup is not None and alias_lookup(query):
        return "Alias"
    # zh/th 无空格长查询；en/ms 按词数
    if len(query) > (30 if _pick_language(raw) in ("zh", "th") else 60):
        return "Long query"
    return "Easy"


def _alias_hit_factory():
    """查询是否含术语表别名（难度判据）。"""
    from durian_agent.glossary import load_glossary

    aliases = sorted((a.casefold() for a in load_glossary()),
                     key=len, reverse=True)

    def hit(query: str) -> bool:
        lowered = query.casefold()
        return any(alias in lowered for alias in aliases if len(alias) >= 3)

    return hit


def load_gold_set(path: Union[str, Path] = GOLD_PATH,
                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载并规范化为 §55 结构。"""
    alias_hit = _alias_hit_factory()
    entries: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            query = str(raw.get("question") or "").strip()
            if not query:
                continue
            ref_doc = raw.get("ref_doc_id")
            entries.append({
                "query": query,
                "language": _pick_language(raw),
                "intent": str(raw.get("category") or "unknown"),
                "difficulty": _derive_difficulty(raw, query, alias_hit),
                "gold_document_ids": [str(ref_doc)] if ref_doc else [],
                "gold_chunk_ids": [str(a) for a in (raw.get("anchors") or [])],
                "gold_answer": str(raw.get("gold_answer") or ""),
            })
            if limit is not None and len(entries) >= limit:
                break
    return entries


def coverage_report(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """§55 维度覆盖：语言 × intent × difficulty。"""
    languages: Dict[str, int] = {}
    intents: Dict[str, int] = {}
    difficulties: Dict[str, int] = {}
    for entry in entries:
        languages[entry["language"]] = languages.get(entry["language"], 0) + 1
        intents[entry["intent"]] = intents.get(entry["intent"], 0) + 1
        difficulties[entry["difficulty"]] = difficulties.get(
            entry["difficulty"], 0) + 1
    return {"total": len(entries), "languages": languages,
            "intents": intents, "difficulties": difficulties}
