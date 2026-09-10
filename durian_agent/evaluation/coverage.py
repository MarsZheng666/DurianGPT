"""有效语料解析覆盖率（架构文档 §56，任务 #63）。

    覆盖率 = 成功解析并进入可检索索引的有效知识单元
             / 人工标注的有效知识单元总数

按四类型分层：正文 text / 表格 table / 图片 image / 扫描页 scan_page。

- 人工标注基准（annotations）：每类型的有效知识单元数——需人工盘点
  源文档得出，本模块提供基准文件格式与计算；
- 索引侧（chunks）：语料按 block_type 统计（text/qa_pair 计入 text，
  table_broken 视为未成功解析、不计入有效索引）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

#: §56 四类型（封闭集）
COVERAGE_TYPES = ("text", "table", "image", "scan_page")

#: 语料 block_type → 覆盖率类型（table_broken 不算成功解析）
_BLOCK_TYPE_MAP = {
    "text": "text",
    "qa_pair": "text",
    "table": "table",
    "image": "image",
    "scan_page": "scan_page",
}


def indexed_counts(chunks: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """语料的可检索知识单元计数（按四类型）。"""
    counts = {t: 0 for t in COVERAGE_TYPES}
    for chunk in chunks:
        coverage_type = _BLOCK_TYPE_MAP.get(chunk.get("block_type", "text"))
        if coverage_type:
            counts[coverage_type] += 1
    return counts


def parse_coverage(
    chunks: Sequence[Dict[str, Any]],
    annotations: Dict[str, int],
) -> Dict[str, Any]:
    """分层覆盖率报告。

    annotations: {"text": 人工标注数, "table": ..., "image": ...,
    "scan_page": ...}——缺失类型视为无标注基准（该层不评估）。
    """
    indexed = indexed_counts(chunks)
    layers: Dict[str, Any] = {}
    total_indexed = total_annotated = 0
    for coverage_type in COVERAGE_TYPES:
        annotated = int(annotations.get(coverage_type, 0))
        got = indexed[coverage_type]
        if annotated > 0:
            ratio = round(got / annotated, 4)
            total_indexed += got
            total_annotated += annotated
        else:
            ratio = None   # 无基准：不评估该层
        layers[coverage_type] = {
            "indexed": got, "annotated": annotated, "coverage": ratio,
        }
    overall = (round(total_indexed / total_annotated, 4)
               if total_annotated else None)
    return {"layers": layers, "overall": overall,
            "total_indexed": total_indexed,
            "total_annotated": total_annotated}


def load_annotations(raw: Union[str, Dict[str, Any]]) -> Dict[str, int]:
    """基准文件/字典加载：{"text": N, ...}（字符串按 JSON 解析）。"""
    import json

    data = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    return {t: int(data.get(t, 0)) for t in COVERAGE_TYPES}
