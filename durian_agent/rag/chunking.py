"""文档解析与结构识别（架构文档 §18，任务 #17）。

输入以 MinerU content_list 风格的块序列为准（现有语料的产出格式），
把每块归类为：正文 text / 标题 title / 表格 table / 图片 image / 扫描页 scan_page，
后续按类型分别解析（§19 正文分块 / §20 表格 / §21 图片，扫描页走 OCR 链路）。

设计：
- 显式 type 字段优先（MinerU/解析器自带类型）；
- 缺失 type 时用高精度启发式（markdown 管道符=表格、图片路径/base64=图片、
  短行无句末标点=标题）；
- scan_page 独立于 image：整页图像且带页码上下文的按扫描页处理
  （OCR + 版面恢复是 §21/任务 #22 的职责，本模块只负责识别与分流）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

#: 块类型的封闭集（§18：正文/标题/表格/图片/扫描页）
BLOCK_TYPES = ("text", "title", "table", "image", "scan_page")

# MinerU 常见 type 值 → 规范类型
_KNOWN_TYPE_MAP = {
    "text": "text", "paragraph": "text", "body": "text",
    "title": "title", "header": "title", "section_header": "title",
    "table": "table", "tbl": "table",
    "image": "image", "img": "image", "figure": "image",
    "scan_page": "scan_page", "page_image": "scan_page",
}

_MD_TABLE_RE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_IMAGE_PATH_RE = re.compile(
    r"\.(?:jpg|jpeg|png|webp|tif|tiff|bmp)\b|^data:image/|^!\[", re.IGNORECASE
)
_SENTENCE_END = ("。", ".", "！", "!", "？", "?", "；", ";", "：", ":")


def _looks_like_title(text: str) -> bool:
    """标题启发式：短行、无句末标点、不是表格/图片。"""
    stripped = text.strip()
    if not (2 <= len(stripped) <= 60):
        return False
    if stripped.endswith(_SENTENCE_END):
        return False
    # 常见标题模式：编号开头 / 数字编号章节
    if re.match(r"^(?:第?[一二三四五六七八九十百\d]+[章节部分篇\.、\s]|[IVX]+\.|\d+\.\d*\s)", stripped):
        return True
    return False


def classify_block(block: Dict[str, Any]) -> str:
    """单个内容块 → BLOCK_TYPES 之一。"""
    btype = str(block.get("type", "") or "").strip().lower()
    if btype in _KNOWN_TYPE_MAP:
        mapped = _KNOWN_TYPE_MAP[btype]
        # 整页图像 + 页码上下文 → 扫描页
        if mapped == "image" and (block.get("page_idx") is not None
                                  and block.get("page_scan", False)):
            return "scan_page"
        return mapped

    text = str(block.get("text") or block.get("content") or "").strip()
    if not text:
        # 无文本：有图片特征则按图片（可能是扫描页，由上游 page_scan 标注细分）
        if _IMAGE_PATH_RE.search(str(block.get("image_path", "")) or ""):
            return "image"
        return "text"

    if _IMAGE_PATH_RE.search(text):
        return "image"
    if _MD_TABLE_RE.search(text):
        return "table"
    if _looks_like_title(text):
        return "title"
    return "text"


def iter_document_blocks(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """MinerU content_list → 规范块序列：每块附 block_type。

    保留原字段的其余内容（bbox/page_idx 等溯源信息原样透传），
    供 §22 Chunk Metadata 与 §56 语料覆盖率分层统计使用。
    """
    blocks: List[Dict[str, Any]] = []
    for raw in content:
        if not isinstance(raw, dict):
            continue
        block = dict(raw)
        block["block_type"] = classify_block(raw)
        blocks.append(block)
    return blocks
