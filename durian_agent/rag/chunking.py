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


# ══════════════════ 正文递归分块（§19，任务 #20）══════════════════

#: §19 建议区间：chunk 300～800 tokens（中文语料 ≈ 字符），overlap 10%～20%
CHUNK_MIN_CHARS = 300
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP_RATIO = 0.15
#: 建库噪声下限（对齐 rag_rechunk.MIN_CHUNK_CHARS=60 的实测值）
_MIN_PIECE_CHARS = 60


def chunk_text(
    text: str,
    *,
    size: int = CHUNK_MAX_CHARS,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
) -> List[str]:
    """单段正文按句边界递归分块（移植 rag_rechunk.split_text 的实测逻辑）。

    - 优先句号/分号/换行切点，其次逗号/空格（不把句子劈开）；
    - 相邻块重叠 overlap_ratio*size（§19：10%～20%）；
    - 低于 60 字的碎屑不进索引（对齐既有建库管线的噪声下限）；
      过短的碎尾并入前一块，长文的分块落在 300～800 区间。
    """
    t = str(text or "").strip()
    if len(t) <= size:
        return [t] if len(t) >= _MIN_PIECE_CHARS else []

    overlap = max(1, int(size * overlap_ratio))
    pieces: List[str] = []
    start, n = 0, len(t)
    while start < n:
        end = min(n, start + size)
        if end < n:
            window = t[start:end]
            cut = max(window.rfind("。"), window.rfind("."),
                      window.rfind("；"), window.rfind(";"), window.rfind("\n"))
            if cut < size * 0.5:
                cut = max(window.rfind("，"), window.rfind(","), window.rfind(" "))
            if cut > size * 0.4:
                end = start + cut + 1
        piece = t[start:end].strip()
        if len(piece) >= _MIN_PIECE_CHARS:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)

    # 碎尾合并：短于下限的最后一块并回前块（§19 区间约束）
    if len(pieces) >= 2 and len(pieces[-1]) < CHUNK_MIN_CHARS * 0.5:
        tail = pieces.pop()
        pieces[-1] = (pieces[-1] + tail).strip()
    return pieces


def chunk_document(
    blocks: List[Dict[str, Any]],
    *,
    size: int = CHUNK_MAX_CHARS,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
) -> List[Dict[str, Any]]:
    """标题→子标题→段落 的层级感知分块（§19）。

    - title 块开启新 section（标题作为 chunk 的 section 元数据，正文不吞标题）；
    - section 内 text 块合并后走 chunk_text 递归分块；
    - table/image/scan_page 块**整块保留**（表格按 §20 单独处理，不参与切分）；
    - 返回 [{"text", "section", "block_type", "source_index"}]。
    """
    chunks: List[Dict[str, Any]] = []
    section = ""
    buffer: List[str] = []
    buffer_start: Optional[int] = None

    def flush(end_index: int) -> None:
        nonlocal buffer, buffer_start
        if not buffer:
            return
        merged = "\n".join(buffer).strip()
        for piece in chunk_text(merged, size=size, overlap_ratio=overlap_ratio):
            chunks.append({
                "text": piece,
                "section": section,
                "block_type": "text",
                "source_index": buffer_start,
            })
        buffer, buffer_start = [], None

    for index, block in enumerate(blocks):
        btype = block.get("block_type", "text")
        text = str(block.get("text") or block.get("content") or "").strip()
        if btype == "title":
            flush(index)
            section = text
        elif btype in ("table", "image", "scan_page"):
            flush(index)
            if text or block.get("image_path"):
                chunks.append({
                    "text": text,
                    "section": section,
                    "block_type": btype,
                    "source_index": index,
                    "image_path": block.get("image_path"),
                })
        else:
            if not text:
                continue
            if buffer_start is None:
                buffer_start = index
            buffer.append(text)
    flush(len(blocks))
    return chunks


# ══════════════════ 表格处理（§20，任务 #16）══════════════════


def table_to_markdown(
    headers: List[str],
    rows: List[List[str]],
    *,
    caption: Optional[str] = None,
    units: Optional[Dict[str, str]] = None,
) -> str:
    """结构化表格 → Markdown 整块文本（§20）。

    保留四要素：
    - 表头：units 里的单位并入表头单元格（如 "剂量 (ml)"）；
    - 行列关系：单元格一一对应，列数以 headers 为准，缺位补空串；
    - 单位：不丢（并入表头而非单独一行，检索时与列名同现）；
    - 说明文字：caption 作为独立行放在表后（语料实测格式为【表格】前缀
      + 表体，此处沿用 caption 后置，前缀由调用方决定）。

    单元格内的换行转 <br>（与既有语料的表格分块格式一致，
    避免 Markdown 表格被单元格换行破坏）。
    """
    units = units or {}

    def cell(value: Any) -> str:
        text = str(value if value is not None else "").strip()
        return text.replace("|", "\\|").replace("\n", "<br>")

    header_cells = []
    for header in headers:
        h = cell(header)
        unit = units.get(str(header))
        if unit and f"({unit})" not in h:
            h = f"{h} ({cell(unit)})"
        header_cells.append(h)

    width = len(header_cells)
    lines = [
        "|" + "|".join(header_cells) + "|",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for row in rows:
        cells = [cell(v) for v in list(row)[:width]]
        cells += [""] * (width - len(cells))
        lines.append("|" + "|".join(cells) + "|")

    table = "\n".join(lines)
    if caption:
        table = f"{table}\n{cell(caption)}"
    return table
