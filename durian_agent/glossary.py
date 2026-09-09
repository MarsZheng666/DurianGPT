"""多语言术语表加载（rag_build/glossary.json）。

glossary.json 是「分类 → {别名: 中文标准名}」的扁平映射（约 330 条别名）。
架构文档 §10 的 canonical_id 体系（任务 #15，阶段二）将以它为底座演进，
本模块只提供读取与基础筛选，不重复造数据。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "rag_build" / "glossary.json"

# 数据文件里以 _ 开头的是说明键（_note / _meta），不是术语
_META_KEYS = {"_note", "_meta", "_purpose"}


def load_glossary(path: Union[str, Path] = GLOSSARY_PATH) -> Dict[str, str]:
    """读取术语表，返回 {别名: 中文标准名}。

    别名覆盖四种文字（zh/en/th/ms），同一标准名可能有多个别名。
    键冲突时后写的分类覆盖先写的——当前数据无跨分类同名别名。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}
    for category, entries in data.items():
        if not isinstance(entries, dict):
            continue
        for alias, canonical in entries.items():
            if alias in _META_KEYS or not isinstance(canonical, str):
                continue
            mapping[alias] = canonical
    return mapping
