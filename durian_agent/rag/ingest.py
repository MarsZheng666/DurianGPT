"""Chunk 实体识别与 Metadata（架构文档 §22，任务 #21）。

把建库产物（rag_build/chunks.jsonl 或 chunk_document 的新分块）规范化为
§22 Chunk Metadata：

    chunk_id / document_id / language / title / section
    entities / domain / source_type / role_scope / orchard_scope

- entities：复用 #12 实体归一化（术语表确定性识别）；
- domain：由实体槽位 + 高置信关键词派生（无信号 → general）；
- source_type：provenance/block_type 映射（sop/document/literature/faq）；
- role_scope：默认全员可见（worker+manager）；数据权限过滤（#47）就绪前
  不做限制，防止误伤召回；
- orchard_scope：空列表 = 不限定园区（全园区适用）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from durian_agent.normalize import normalize_input
from durian_agent.semantic.entities import detect_entities

#: §22 Chunk Metadata 必备键
CHUNK_METADATA_KEYS = (
    "chunk_id", "document_id", "language", "title", "section",
    "entities", "domain", "source_type", "role_scope", "orchard_scope",
)

#: domain 词汇表（派生用，§22 示例 fertilization）
_DOMAINS = (
    "fertilization", "irrigation", "plant_protection", "flowering_management",
    "fruit_management", "pruning", "soil_management", "harvest", "postharvest",
    "variety", "general",
)

#: 实体槽位 → domain
_SLOT_DOMAIN = {
    "disease": "plant_protection",
    "pest": "plant_protection",
    "fertilizer": "fertilization",
    "cultivar": "variety",
}

#: 高置信 domain 关键词（保守：只在无实体信号时兜底）
_DOMAIN_KEYWORDS = [
    ("fertilization", ("施肥", "肥料", "fertiliz", "ใส่ปุ๋ย", "baja")),
    ("irrigation", ("灌溉", "浇水", "irrigat", "watering", "ให้น้ำ", "siram")),
    ("plant_protection", ("病害", "虫害", "农药", "disease", "pest", "โรค", "penyakit")),
    ("flowering_management", ("开花", "花期", "flowering", "ออกดอก", "berbunga")),
    ("fruit_management", ("果实", "坐果", "fruit set", "ผล", "buah")),
    ("pruning", ("修剪", "整枝", "pruning", "ตัดแต่ง", "pangkas")),
    ("soil_management", ("土壤", " acidity", "soil", "ดิน", "tanah")),
    ("harvest", ("采收", "收获", "harvest", "เก็บเกี่ยว", "tuai")),
    ("postharvest", ("采后", "后熟", "postharvest", "ripening", "การบ่ม", "pascapanen")),
]

#: provenance / block_type → source_type
_SOURCE_TYPE_MAP = {
    "legacy_faq": "faq",
    "legacy_thai": "document",
    "legacy_lit_salvage": "literature",
    "pdf_parse": "document",
}


def infer_domain(text: str, entities: Optional[Dict[str, str]] = None) -> str:
    """实体槽位优先、关键词兜底的 domain 派生。"""
    for slot, domain in _SLOT_DOMAIN.items():
        if entities and entities.get(slot):
            return domain
    value = normalize_input(text)
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(kw in value for kw in keywords):
            return domain
    return "general"


def build_chunk_record(
    text: str,
    *,
    chunk_id: str,
    document_id: str,
    language: str = "zh",
    title: str = "",
    section: str = "",
    source_type: str = "document",
    role_scope: Optional[List[str]] = None,
    orchard_scope: Optional[List[str]] = None,
    provenance: str = "",
    block_type: str = "text",
) -> Dict[str, Any]:
    """规范化为 §22 ChunkRecord（含实体识别与 domain 派生）。"""
    entities = detect_entities(text)
    source = _SOURCE_TYPE_MAP.get(provenance, source_type)
    if block_type == "qa_pair":
        source = "faq"
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "language": language or "zh",
        "title": title or document_id,
        "section": section or "",
        "entities": entities,
        "domain": infer_domain(text, entities),
        "source_type": source,
        "role_scope": role_scope if role_scope is not None else ["worker", "manager"],
        "orchard_scope": orchard_scope if orchard_scope is not None else [],
        # 索引用正文（display 与溯源信息由建库侧管理）
        "text": text,
    }


def ingest_corpus(
    path: Union[str, Path],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """接入建库产物 chunks.jsonl → ChunkRecord 列表。

    字段映射：id→chunk_id，source_file→document_id，
    metadata.src_lang→language（缺省 zh，索引文本是中文），
    provenance/block_type→source_type。
    """
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            text = str(raw.get("index_text") or raw.get("text") or "")
            if not text:
                continue
            metadata = raw.get("metadata") or {}
            records.append(build_chunk_record(
                text,
                chunk_id=str(raw.get("id") or ""),
                document_id=str(raw.get("source_file") or raw.get("doc") or ""),
                language=str(metadata.get("src_lang") or "zh"),
                section="",
                provenance=str(raw.get("provenance") or ""),
                block_type=str(raw.get("block_type") or "text"),
            ))
            if limit is not None and len(records) >= limit:
                break
    return records
