"""多语言术语表（rag_build/glossary.json + glossary_v2.json）。

- v1：glossary.json，「分类 → {别名: 中文标准名}」扁平映射（321 别名 / 201 概念）；
- v2：glossary_v2.json，§10 规范结构——canonical_id / type / canonical_name /
  四语分组 aliases / related_entities。由 v1 确定性生成（build_glossary_v2），
  别名按文字自动分组（th/zh 按字符集；拉丁文按源类目区分 en/ms——
  malay_common 类目进 ms，其余进 en，是可用的最佳启发式）。

数据现状（诚实记账，2026-09-10）：
    321 别名 / 201 canonical 概念。§10 建议「300～500 concept +
    1000～2000 alias」——概念数已达建议区间，别名数低于区间；
    文档中「1330+」与实测数据不符，以本记账为准。
    扩充到 1000+ 别名需批量四语翻译（LLM 通道确认后进行）或人工补充。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "rag_build" / "glossary.json"
GLOSSARY_V2_PATH = Path(__file__).resolve().parent.parent / "rag_build" / "glossary_v2.json"

# 数据文件里以 _ 开头的是说明键（_note / _meta / _purpose），不是术语
_META_KEYS = {"_note", "_meta", "_purpose"}

#: v1 类目 → (canonical_id 前缀, type)
CATEGORY_TO_TYPE = {
    "variety_thai": ("CULTIVAR", "cultivar"),
    "variety_malaysia": ("CULTIVAR", "cultivar"),
    "variety_indonesia": ("CULTIVAR", "cultivar"),
    "disease": ("DISEASE", "disease"),
    "pest": ("PEST", "pest"),
    "nutrient": ("NUTRIENT", "nutrient"),
    "phenology": ("PHENOLOGY", "phenology"),
    "cultivation": ("CULTIVATION", "cultivation"),
    "postharvest": ("POSTHARVEST", "postharvest"),
    "plant_organ": ("PLANT_ORGAN", "plant_organ"),
    "symptom": ("SYMPTOM", "symptom"),
    "malay_common": ("MALAY_COMMON", "malay_common"),
    "fruit_identity": ("FRUIT", "fruit_identity"),
}

_THAI_RE = re.compile(r"[\u0e00-\u0e7f]")
_ZH_RE = re.compile(r"[\u4e00-\u9fff]")


def _read_raw(path: Union[str, Path]) -> Dict[str, Dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    categorized: Dict[str, Dict[str, str]] = {}
    for category, entries in data.items():
        if not isinstance(entries, dict):
            continue
        clean = {
            alias: canonical
            for alias, canonical in entries.items()
            if alias not in _META_KEYS and isinstance(canonical, str)
        }
        if clean:
            categorized[category] = clean
    return categorized


def load_glossary(path: Union[str, Path] = GLOSSARY_PATH) -> Dict[str, str]:
    """读取术语表，返回 {别名: 中文标准名}。

    别名覆盖四种文字（zh/en/th/ms），同一标准名可能有多个别名。
    键冲突时后写的分类覆盖先写的——当前数据无跨分类同名别名。
    """
    mapping: Dict[str, str] = {}
    for entries in _read_raw(path).values():
        mapping.update(entries)
    return mapping


def load_glossary_categorized(
    path: Union[str, Path] = GLOSSARY_PATH,
) -> Dict[str, Dict[str, str]]:
    """读取术语表，返回 {分类: {别名: 标准名}}（保留分类，供实体槽位映射）。"""
    return _read_raw(path)


# ══════════════════ v2：§10 canonical_id 体系 ══════════════════


def _lang_of(text: str, source_category: str) -> str:
    """别名的语言分组：th/zh 按字符集；拉丁文按源类目（malay_common→ms，余→en）。"""
    if _THAI_RE.search(text):
        return "th"
    if _ZH_RE.search(text):
        return "zh"
    return "ms" if source_category == "malay_common" else "en"


def build_glossary_v2(path: Union[str, Path] = GLOSSARY_PATH) -> Dict:
    """从 v1 确定性生成 §10 结构的 v2 术语库。

    - canonical_name 跨类目合并（同一标准名多处出现时归并来源）；
    - canonical_id = 前缀_三位序号，按 (type, canonical_name) 排序保证稳定；
    - 别名按语言分组；canonical_name 自身也进对应语言组；
    - related_entities 留空（当前数据无可推导的关系，扩充时人工/LLM 补）。
    """
    raw = _read_raw(path)
    # canonical_name → {source_categories, aliases: {alias: lang}}
    merged: Dict[str, Dict] = {}
    for category, entries in raw.items():
        prefix, _etype = CATEGORY_TO_TYPE.get(category, ("TERM", category))
        for alias, canonical in entries.items():
            record = merged.setdefault(canonical, {
                "prefix": prefix, "categories": [], "aliases": {},
            })
            record["categories"].append(category)
            if alias != canonical:
                record["aliases"][alias] = _lang_of(alias, category)

    ordered = sorted(merged.items(), key=lambda kv: (kv[1]["prefix"], kv[0]))
    entries_out = []
    counters: Dict[str, int] = {}
    for canonical, record in ordered:
        prefix = record["prefix"]
        counters[prefix] = counters.get(prefix, 0) + 1
        canonical_id = f"{prefix}_{counters[prefix]:03d}"
        aliases: Dict[str, List[str]] = {"zh": [], "en": [], "th": [], "ms": []}
        # 标准名自身归入其语言组
        aliases[_lang_of(canonical, record["categories"][0])].append(canonical)
        for alias, lang in sorted(record["aliases"].items()):
            if alias not in aliases[lang]:
                aliases[lang].append(alias)
        entries_out.append({
            "canonical_id": canonical_id,
            "type": CATEGORY_TO_TYPE.get(record["categories"][0],
                                         (prefix, "unknown"))[1],
            "canonical_name": canonical,
            "aliases": aliases,
            "related_entities": [],
            "source_categories": sorted(set(record["categories"])),
        })
    return {
        "_meta": {
            "purpose": "§10 canonical_id 术语体系（由 glossary.json 确定性生成）",
            "generated_by": "durian_agent.glossary.build_glossary_v2",
            "counts": {
                "concepts": len(entries_out),
                "aliases": sum(len(v) for e in entries_out
                               for v in e["aliases"].values()),
            },
            "note": "别名按文字分组：th/zh 按字符集，拉丁文按源类目 "
                    "（malay_common→ms，其余→en）。related_entities 待扩充。",
        },
        "entries": entries_out,
    }


def load_glossary_v2(
    path: Union[str, Path] = GLOSSARY_V2_PATH,
    *,
    build_if_missing: bool = True,
) -> Optional[Dict]:
    """加载 v2 术语库；文件缺失时可从 v1 现场生成（不落盘）。"""
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    if build_if_missing:
        return build_glossary_v2()
    return None


def save_glossary_v2(
    data: Optional[Dict] = None,
    path: Union[str, Path] = GLOSSARY_V2_PATH,
) -> int:
    """生成并落盘 v2 术语库，返回概念数。"""
    p = Path(path)
    data = data or build_glossary_v2()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(data["entries"])
