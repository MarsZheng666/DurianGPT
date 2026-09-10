"""扫描件处理（架构文档 §21，任务 #22）：Scan → OCR → 版面恢复 → Chunk。

链路：
    扫描页图像 → OcrFn（行级识别：文本 + 近似行框）→ restore_layout
    （版面恢复：按垂直间距聚段落）→ chunk_document（#20 递归分块）

通道（OcrFn 依赖注入）：
- tesseract_ocr()：默认通道。本机未装 tesseract 二进制时抛 OcrConfigError
  （不静默降级——OCR 质量直接影响索引质量）；
- vlm_ocr(vlm_fn)：VLM 备选通道（复用 #18 的外部 API 约定）。
  无坐标输出，按线性阅读顺序退化处理（版面信息有限，已知局限）。

版面恢复是**单栏启发式**（行距阈值分段）——多栏扫描件需更强的版面
分析（未来接入 MinerU 等版面解析器时替换 restore_layout）。
"""

from __future__ import annotations

import shutil
import statistics
from typing import Any, Callable, Dict, List, Optional, Sequence

#: OcrFn 契约：image_bytes → [{"text", "top", "left", "width", "height"}]
OcrFn = Callable[[bytes], List[Dict[str, Any]]]

#: 段间距阈值 = 中位行高 × 该系数
PARAGRAPH_GAP_FACTOR = 1.5


class OcrConfigError(RuntimeError):
    """OCR 通道不可用。"""


VLM_OCR_PROMPT = """Extract all visible text lines from this scanned document image.
Return plain text only, one output line per visible text line, preserving reading order.
Do not add any explanation."""


def tesseract_ocr(*, lang: str = "chi_sim+eng") -> OcrFn:
    """tesseract 通道：word-level 输出按 (block, par, line) 聚成行。"""
    if shutil.which("tesseract") is None:
        raise OcrConfigError(
            "tesseract 未安装（brew install tesseract 及 chi_sim 语言包）。"
            "或者使用 vlm_ocr(vlm_fn) 作为 OCR 通道。"
        )

    def ocr(image_bytes: bytes) -> List[Dict[str, Any]]:
        import io

        import pytesseract
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        data = pytesseract.image_to_data(image, lang=lang,
                                         output_type=pytesseract.Output.DICT)
        rows: Dict[tuple, Dict[str, Any]] = {}
        for i, text in enumerate(data["text"]):
            if not str(text).strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            row = rows.setdefault(key, {"parts": [], "top": data["top"][i],
                                        "left": data["left"][i],
                                        "width": 0, "height": data["height"][i]})
            row["parts"].append(str(text))
            row["width"] += data["width"][i]
        lines = []
        for key in sorted(rows):
            row = rows[key]
            lines.append({
                "text": " ".join(row["parts"]),
                "top": row["top"], "left": row["left"],
                "width": row["width"], "height": row["height"],
            })
        return lines

    return ocr


def vlm_ocr(vlm_fn: Callable[[bytes, str], str]) -> OcrFn:
    """VLM 备选通道：无坐标 → 线性顺序（版面信息有限，已知局限）。"""

    def ocr(image_bytes: bytes) -> List[Dict[str, Any]]:
        raw = vlm_fn(image_bytes, VLM_OCR_PROMPT)
        lines = []
        for offset, text in enumerate((raw or "").splitlines()):
            text = text.strip()
            if not text:
                continue
            lines.append({
                "text": text,
                "top": offset * 20, "left": 0, "width": 0, "height": 20,
            })
        return lines

    return ocr


def _join_line(parts: Sequence[str]) -> str:
    """行内词拼接：仅拉丁-拉丁边界留空格；CJK 与数字/拉丁之间不加空格
    （语料惯例：「表层土深度超过50厘米」「每株50克」均无空格）。"""
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out and not (_ends_cjk(out) or _starts_cjk(part)):
            out += " "
        out += part
    return out


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _ends_cjk(text: str) -> bool:
    return bool(text) and _is_cjk(text[-1])


def _starts_cjk(text: str) -> bool:
    return bool(text) and _is_cjk(text[0])


def restore_layout(lines: List[Dict[str, Any]]) -> List[str]:
    """版面恢复（单栏启发式）：按 y 排序，行距超过阈值则分段。"""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda l: (l["top"], l["left"]))
    heights = [max(l.get("height", 0), 1) for l in ordered]
    median_h = statistics.median(heights) if heights else 1

    paragraphs: List[List[str]] = [[ordered[0]["text"]]]
    for prev, cur in zip(ordered, ordered[1:]):
        gap = cur["top"] - (prev["top"] + max(prev.get("height", 1), 1))
        if gap > median_h * PARAGRAPH_GAP_FACTOR:
            paragraphs.append([cur["text"]])
        else:
            paragraphs[-1].append(cur["text"])
    return [_join_line(p) for p in paragraphs if _join_line(p)]


def ocr_scan(image_bytes: bytes, ocr_fn: OcrFn) -> List[str]:
    """单页扫描件 → OCR + 版面恢复 → 段落列表。"""
    return restore_layout(ocr_fn(image_bytes))


def ocr_pages_to_chunks(
    pages: Sequence[bytes],
    ocr_fn: OcrFn,
    *,
    size: int = 800,
    overlap_ratio: float = 0.15,
) -> List[Dict[str, Any]]:
    """多页扫描件 → OCR → 版面恢复 → chunk_document 递归分块（§21 完整链路）。"""
    from durian_agent.rag.chunking import chunk_document

    blocks = []
    for page_index, page in enumerate(pages):
        for para_index, paragraph in enumerate(ocr_scan(page, ocr_fn)):
            blocks.append({"type": "text", "text": paragraph,
                           "page_idx": page_index, "par_idx": para_index})
    return chunk_document(blocks, size=size, overlap_ratio=overlap_ratio)
