"""生育阶段识别（架构文档 §8，任务 #14）。

§8 的 10 个阶段（含 unknown 兜底）。识别两层：
- 规则层（本模块）：封闭小词表、最长优先——只收各语种**高精度**的
  阶段术语。这不是路由关键词表：阶段识别用于 RAG 元数据过滤与
  Schema 填充，精确率优先，宁可漏（回落 unknown）不可错；
- LLM 层（任务 #3 SemanticParse）：从上下文推断（如"谢花后两周"）。

中文术语优先对齐项目语料高频写法（开花期/采收期/抽梢），泰文/英文/
马来文次之。词表按长度降序匹配，避免「花芽分化」被「开花」截胡、
「post-harvest」被「harvest」截胡。
"""

from __future__ import annotations

from typing import List, Tuple

from durian_agent.normalize import normalize_input

#: §8 的 10 个生育阶段（顺序即文档顺序）
GROWTH_STAGES: Tuple[str, ...] = (
    "seedling",
    "vegetative",
    "pre_flowering",
    "flowering",
    "fruit_set",
    "fruit_development",
    "pre_harvest",
    "harvest",
    "post_harvest",
    "unknown",
)

UNKNOWN_STAGE = "unknown"
_VALID_STAGES = frozenset(GROWTH_STAGES)

# 高精度阶段术语（术语, 阶段）。多语种；保持小而准。
_STAGE_TERMS: List[Tuple[str, str]] = [
    # pre_flowering：花芽类（长词在前，避免被「开花」截胡）
    ("花芽分化", "pre_flowering"),
    ("花芽诱导", "pre_flowering"),
    ("花芽", "pre_flowering"),
    ("pre-flowering", "pre_flowering"),
    ("pre flowering", "pre_flowering"),
    ("诱导开花", "pre_flowering"),
    # flowering
    ("开花期", "flowering"),
    ("盛花期", "flowering"),
    ("花期", "flowering"),
    ("开花", "flowering"),
    ("anthesis", "flowering"),
    ("flowering", "flowering"),
    ("ออกดอก", "flowering"),
    ("berbunga", "flowering"),
    # fruit_set
    ("坐果期", "fruit_set"),
    ("坐果", "fruit_set"),
    ("幼果期", "fruit_set"),
    ("谢花后", "fruit_set"),
    ("fruit set", "fruit_set"),
    ("fruit-set", "fruit_set"),
    # fruit_development
    ("果实发育", "fruit_development"),
    ("膨果期", "fruit_development"),
    ("果实膨大", "fruit_development"),
    ("fruit development", "fruit_development"),
    # seedling
    ("苗期", "seedling"),
    ("幼苗期", "seedling"),
    ("幼苗", "seedling"),
    ("seedling", "seedling"),
    ("ต้นกล้า", "seedling"),
    # vegetative
    ("营养生长期", "vegetative"),
    ("营养生长", "vegetative"),
    ("抽梢", "vegetative"),
    ("vegetative", "vegetative"),
    ("flushing", "vegetative"),
    # pre_harvest
    ("采前期", "pre_harvest"),
    ("采前", "pre_harvest"),
    ("pre-harvest", "pre_harvest"),
    ("pre harvest", "pre_harvest"),
    # harvest
    ("采收期", "harvest"),
    ("收获期", "harvest"),
    ("采收", "harvest"),
    ("harvesting", "harvest"),
    ("harvest", "harvest"),
    ("เก็บเกี่ยว", "harvest"),
    ("menuai", "harvest"),
    # post_harvest
    ("采后处理", "post_harvest"),
    ("采后", "post_harvest"),
    ("后熟", "post_harvest"),
    ("post-harvest", "post_harvest"),
    ("post harvest", "post_harvest"),
    ("ripening", "post_harvest"),
    ("การบ่ม", "post_harvest"),
    ("pascapanen", "post_harvest"),
]

_STAGE_TERMS_SORTED = sorted(_STAGE_TERMS, key=lambda pair: len(pair[0]), reverse=True)


def is_valid_growth_stage(stage: str) -> bool:
    return stage in _VALID_STAGES


def normalize_growth_stage(stage: str) -> str:
    """LLM 输出清洗：非法值回落 unknown。"""
    cleaned = (stage or "").strip()
    return cleaned if cleaned in _VALID_STAGES else UNKNOWN_STAGE


def detect_growth_stage(text: str) -> str:
    """规则层生育阶段识别：最长术语优先；无命中回落 unknown。"""
    value = normalize_input(text)
    for term, stage in _STAGE_TERMS_SORTED:
        if term in value:
            return stage
    return UNKNOWN_STAGE
