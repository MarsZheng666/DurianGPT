#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重新分片管道：产出干净的 rag_build/chunks.jsonl。

背景约束（决定了本脚本的设计）：
- 22 篇文献的 MinerU 原始 JSON 与源 PDF 大部分已丢失，现存 clean_chunks.jsonl 里的
  文献 chunk 是 MinerU 输出被 `str()` 打平后**硬截断在 500 字符**的产物，不可逆。
  因此对这部分只能做「抢救」：剥离残渣、救回自然语言，无法恢复完整表格。
- 已找回 7 份权威 PDF（含 IPGRI 196页、马来西亚农业部 62页），这部分走正常解析管道。
- FAQ(1000) 与泰文 DOCX(651) chunk 完全干净，原样保留。

核心原则：
1. **index_text 与 display_text 分离**。前者供向量化，后者供展示/喂 LLM。
2. **宁可不召回，不可召回残表**。丢了表头的裸 <td> 数值不进 index_text。
3. 抢救而非丢弃。原 is_noise_text() 整块丢掉 22.8% 的 chunk，其中含真实正文。

用法：
    python rag_rechunk.py build      # 产出 rag_build/chunks.jsonl
    python rag_rechunk.py stats      # 只看统计不写文件
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
BUILD_DIR = BASE_DIR / "rag_build"
OUT_PATH = BUILD_DIR / "chunks.jsonl"
LEGACY_CHUNKS = BASE_DIR / "clean_chunks.jsonl"
PDF_DIRS = [BASE_DIR / "raw" / "pdfs", BASE_DIR / "rag_pdfs"]

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 60

# 按文档粒度覆盖 CHUNK_SIZE（其它文档默认 900 不变，chunk 边界完全不动——
# 泰文/中文 known-item 评测集依赖边界稳定）。键用 PDF 文件名前缀匹配：
# parse_pdfs 的 doc 名就是文件名（含 .pdf），前缀匹配对带不带后缀都成立。
# MY_DOA 2024 新版要切细（300）：信息密度比 2012 版高，900 会把多个管理主题
# 合进一个 chunk，检索粒度过粗。2c 重建时若 gp-1 落地的实际文件名与此前缀
# 不符，构建日志会打印「chunk 尺寸覆盖生效」标记——看不到该标记就要查文件名。
DOC_CHUNK_SIZE = {"MY_DOA_Pakej_Teknologi_Durian_2024": 300}

# 按文档粒度覆盖 CHUNK_OVERLAP（默认 150 不变，其它文档零影响）。
# OVERLAP=150 是绝对值：900 粒度下 17% 是设计意图，但 300 粒度下等于 50% 重叠
# ——相邻 chunk 共享一半内容，把「近重复候选挤占 top-5」烤进索引（马来文 FAQ
# 58.7% 近重复密度已让我们付过学费的失效模式）。让 overlap 与 size 等比：
# 300/50 与 900/150 同为 ~17%。
DOC_CHUNK_OVERLAP = {"MY_DOA_Pakej_Teknologi_Durian_2024": 50}


# ══════════════════════════ 文本清洗 ══════════════════════════

# MinerU 打平残渣的实测键名频次（clean_chunks.jsonl 文献部分 7751 条）：
#   bbox 33418 / type 24534 / content 14024 / index 9468 / angle 9094
#   lines 8869 / spans 8768 / blocks 908 / para_blocks 835 / page_idx 832
#   page_size 816 / discarded_blocks 716 / image_path 628 / sub_type 231 / html 168
_RESIDUE_KEYS = (
    "bbox|type|content|index|angle|lines|spans|blocks|para_blocks|page_idx"
    "|page_size|discarded_blocks|image_path|sub_type|html|text_level|score"
)

_RE_BBOX = re.compile(r"\{?\s*['\"]bbox['\"]\s*:\s*\[[^\]]*\]\s*,?")
# 键: 标量值（数字/布尔/None），整对删除
_RE_KV_SCALAR = re.compile(
    rf"['\"](?:{_RESIDUE_KEYS})['\"]\s*:\s*(?:-?\d+(?:\.\d+)?|true|false|null|None|True|False)\s*,?"
)
# 键: 后面接字符串值 —— 只删键名，保留值（这是正文所在）
_RE_KV_KEYONLY = re.compile(rf"['\"](?:{_RESIDUE_KEYS})['\"]\s*:\s*")

# MinerU 的 type 字段**值**（而非键名）。打平后这些值变成孤立的带引号字符串留在正文里，
# 实测残留 12025 处 'text'、2682 处 'inline_equation'、941 处 'title'。
# 它们不是内容，是结构标签，必须删。只在被引号包裹且前后不是正常词的情况下删，
# 避免误删正文里合法出现的同名单词。
_MINERU_TYPE_VALUES = (
    "text|title|list|table|image|figure|inline_equation|interline_equation"
    "|table_caption|image_caption|table_footnote|image_body|table_body"
    "|index|algorithm|code|equation|header|footer|page_number|discarded"
    "|ref_text|caption|footnote|title_level|content_list|abstract|formula"
)
_RE_TYPE_VALUE = re.compile(rf"['\"](?:{_MINERU_TYPE_VALUES})['\"]")
# 去掉引号后可能残留的 image_body / table_body 裸词（无引号形式）
_RE_BODY_WORD = re.compile(r"\b(?:image_body|table_body|para_blocks|discarded_blocks)\b")

_RE_HTML_TABLE = re.compile(r"</?(?:table|tr|td|th|thead|tbody)\b[^>]*>", re.I)
_RE_HTML_ANY = re.compile(r"</?[a-zA-Z][^>]{0,60}>")
_RE_ENTITY = re.compile(r"&(?:#x?[0-9a-fA-F]+|amp|quot|lt|gt|nbsp|apos);")
_RE_URL = re.compile(r"https?://\S+")
_RE_STRUCT_PUNCT = re.compile(r"[\[\]{}]+|(?<![\u4e00-\u9fff\w])['\"](?![\u4e00-\u9fff\w])")
_RE_WS = re.compile(r"\s+")
_RE_ORPHAN_PUNCT = re.compile(r"(?:\s*,\s*){2,}|(?:\s*'\s*){2,}")

# ── 打平残渣的两类顽固尾巴（独立判据抽检发现，第一轮清洗未覆盖）──
# 1) 孤立引号：MinerU 把每个 span 的值单独加引号，打平后变成 '3 kg' 这种包裹形式。
#    语义可读但引号是噪声，且会干扰精确原文匹配。
_RE_QUOTED_VALUE = re.compile(
    r"'\s*([^']{1,40}?)\s*'"
)
# 2) 裸 bbox 坐标残留：形如 " 13, 488, 409, 547 " 或 " 3, 671 " —— 由 bbox 数组被
#    去掉方括号后留下。必须要求是 2 个以上逗号分隔的整数，且不与单位相连，
#    才能安全删除而不误伤正文里的数字列举（如"3, 5, 7 年生"）。
_RE_BARE_BBOX = re.compile(
    r"(?<![\d.%℃×~-])\s\d{1,4}\s*,\s*\d{1,4}(?:\s*,\s*\d{1,4}){1,3}\s*(?![\d.]*\s*"
    r"(?:kg|g|mm|cm|m|km|%|℃|天|日|月|年|次|倍|亩|株|个|米|厘米|毫米))"
)


def strip_quote_noise(text: str) -> str:
    """去掉 MinerU 打平留下的孤立引号与裸 bbox 坐标。

    放在 strip_residue 之后单独执行：此时结构标记已清除，剩下的引号
    基本都是 span 值的包裹符，可以安全剥离。
    """
    s = str(text or "")
    s = _RE_BARE_BBOX.sub(" ", s)
    # 反复剥离引号包裹，处理嵌套/连续情况
    for _ in range(3):
        new = _RE_QUOTED_VALUE.sub(r"\1", s)
        if new == s:
            break
        s = new
    # 剩下的孤立单引号（配对失败的）直接删
    s = re.sub(r"(?<![\u4e00-\u9fffA-Za-z])'|'(?![\u4e00-\u9fffA-Za-z])", " ", s)
    s = _RE_WS.sub(" ", s)
    return s.strip(" ,;'\"")

_WORD = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0e00-\u0e7fA-Za-z]")


def normalize_units(text: str) -> str:
    """归一化 MinerU/LaTeX 单位残留。沿用 rag_llamaindex.normalize_ocr_units 的规则。

    实测 LaTeX 命令频次：mathrm 1673 / circ 316 / sim 205 / times 85 / cdot 83 / pm 70
    """
    v = str(text or "")
    v = re.sub(r"(\d+(?:\.\d+)?)\s*\\+mathrm\{(mm|cm|m|km|kg|g|mg|L|mL|ha)\}", r"\1 \2", v, flags=re.I)
    v = re.sub(r"(\d+(?:\.\d+)?)\s*\\+sim\s*(\d+(?:\.\d+)?)\s*(?:\^\{\\+circ\})?\s*\\+mathrm\{C\}",
               r"\1–\2℃", v, flags=re.I)
    v = re.sub(r"(\d+(?:\.\d+)?)\s*(?:\^\{\\+circ\})?\s*\\+mathrm\{C\}", r"\1℃", v, flags=re.I)
    v = re.sub(r"\\+sim", "~", v)
    v = re.sub(r"\\+times", "×", v)
    v = re.sub(r"\\+cdot", "·", v)
    v = re.sub(r"\\+pm", "±", v)
    v = re.sub(r"\\+mu\b", "μ", v)
    v = v.replace("\\geqslant", "≥").replace("\\leqslant", "≤")
    v = re.sub(r"\\+(?:mathrm|mathbf|mathbb|text|tag|rightharpoon|circ)\b", " ", v)
    v = re.sub(r"\\+([%&#_$])", r"\1", v)

    # MinerU 偶尔把温度的度数符号输出成 \%，仅在上下文明确是温度时才改
    def _deg(m: re.Match) -> str:
        s, e = m.span()
        ctx = v[max(0, s - 28): min(len(v), e + 16)].lower()
        if any(k in ctx for k in ("温度", "低温", "高温", "积温", "摄氏", "temperature")):
            return f"{m.group(1)}℃"
        return m.group(0)

    return re.sub(r"(\d+(?:\.\d+)?)\s*\\+%", _deg, v)


def strip_residue(text: str, drop_table_html: bool = True) -> str:
    """剥离 MinerU 结构残渣，尽量保留自然语言。

    与原 rag_llamaindex.clean_text() 的关键差别：
    - 区分「键:标量」（整对删）与「键:字符串」（只删键名，保值）—— 原版一律只删键名，
      导致 bbox 坐标数字大量留在正文里，把向量拖成噪声。
    - 显式清理 HTML 实体与孤立标点，原版不处理。
    """
    s = str(text or "")
    s = _RE_BBOX.sub(" ", s)
    s = _RE_KV_SCALAR.sub(" ", s)
    s = _RE_KV_KEYONLY.sub(" ", s)
    s = _RE_TYPE_VALUE.sub(" ", s)
    s = _RE_BODY_WORD.sub(" ", s)
    if drop_table_html:
        s = _RE_HTML_TABLE.sub(" ", s)
    s = _RE_HTML_ANY.sub(" ", s)
    s = _RE_ENTITY.sub(" ", s)
    s = _RE_URL.sub(" ", s)
    s = normalize_units(s)
    s = _RE_STRUCT_PUNCT.sub(" ", s)
    s = _RE_ORPHAN_PUNCT.sub(" ", s)
    s = _RE_WS.sub(" ", s)
    s = s.strip(" ,;'\"")
    return strip_quote_noise(s)


def word_ratio(text: str) -> float:
    t = str(text or "")
    return len(_WORD.findall(t)) / max(len(t), 1)


# ══════════════════════════ 质量分级 ══════════════════════════

_RE_TABLE_MARK = re.compile(r"</?t[dhr]\b|</?table\b", re.I)
_RE_RESIDUE_MARK = re.compile(rf"['\"](?:{_RESIDUE_KEYS})['\"]\s*:")


def grade(raw: str, cleaned: str) -> str:
    """分级：usable / table_fragment / junk。

    table_fragment 单独成类而非归入 junk —— 它们仍可能带 caption 等可用信息，
    但绝不能让裸数值进 index_text（幻觉高风险）。

    注：不再保留 salvaged_short 等级。实测清洗后 <40 字符的碎片全是
    `"cks': 'image', 'image_body'"` 这类残渣尾巴，没有检索价值，直接丢。
    """
    if not cleaned or len(cleaned) < MIN_CHUNK_CHARS:
        return "junk"
    if _RE_TABLE_MARK.search(raw):
        return "table_fragment"
    if word_ratio(cleaned) < 0.55:
        return "junk"
    if len(_RE_RESIDUE_MARK.findall(cleaned)) >= 1:
        return "junk"
    # 清洗后仍以数字为主 → 是被截断的数据行，不是正文
    digits = sum(1 for c in cleaned if c.isdigit())
    if digits / len(cleaned) > 0.35:
        return "junk"
    return "usable"


_RE_CAPTION = re.compile(
    r"(?:表|图|附表|Table|Fig(?:ure)?)\s*[\dIVX]{1,3}(?:[-–—.]\d{1,3})?"
    r"[\s.:：]{0,3}([^。；\n|]{4,70})"
)


def extract_caption(text: str) -> Optional[str]:
    """从表格残片里抢救 caption —— 这是残表唯一还有检索价值的部分。

    必须产出**纯 caption**：残表的裸数值若混进 index_text，就等于把
    "丢了表头的数字"重新放回检索路径，与本次改造的目的直接冲突。
    """
    s = strip_residue(text, drop_table_html=True)
    m = _RE_CAPTION.search(s)
    if not m:
        return None

    label = m.group(0)[: m.start(1) - m.start(0)].strip(" .:：")
    body = m.group(1)
    # 截到第一个明显的数据边界：连续数字、括号引用、多个空格
    body = re.split(r"\s{2,}|\d+\.\d+|\)\s*[.。]|\s\d{2,}", body)[0].strip(" ,.;:'\"()")
    cap = f"{label} {body}".strip()

    if len(cap) < 8 or len(cap) > 90:
        return None
    if word_ratio(cap) < 0.55:      # 必须以文字为主，不是数字堆
        return None
    if "'" in cap or '"' in cap:    # 仍带残渣引号，说明没切干净
        return None
    if "<" in cap or ">" in cap:    # 残留 HTML 尖括号
        return None
    # 必须是完整的 caption 起头，避免截出 "Table 14" 这种无信息量的空壳
    if not re.match(r"^(?:表|图|附表|Table|Fig(?:ure)?)\s*[\dIVX]", cap):
        return None
    tail = re.sub(r"^(?:表|图|附表|Table|Fig(?:ure)?)\s*[\dIVX]+(?:[-–—.]\d+)?[\s.:：]*", "", cap)
    if len(tail) < 8:               # 只有编号没有描述，无检索价值
        return None
    digits = sum(1 for c in cap if c.isdigit())
    if digits / len(cap) > 0.3:     # 数字占比过高，实为数据行
        return None
    return cap


# ══════════════════════════ 切分 ══════════════════════════

_SENT_END = re.compile(r"(?<=[。！？；\.;!?])\s*")


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """按句边界切分，避免把句子劈开。"""
    t = str(text or "").strip()
    if len(t) <= size:
        return [t] if len(t) >= MIN_CHUNK_CHARS else []

    out: List[str] = []
    start = 0
    n = len(t)
    while start < n:
        end = min(n, start + size)
        if end < n:
            window = t[start:end]
            # 优先句号，其次逗号/空格
            cut = max(window.rfind("。"), window.rfind("."), window.rfind("；"),
                      window.rfind(";"), window.rfind("\n"))
            if cut < size * 0.5:
                cut = max(window.rfind("，"), window.rfind(","), window.rfind(" "))
            if cut > size * 0.4:
                end = start + cut + 1
        piece = t[start:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS:
            out.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out


def sha1(s: str) -> str:
    return hashlib.sha1(str(s).encode("utf-8", "ignore")).hexdigest()


def make_chunk(
    *,
    doc: str,
    index_text: str,
    display_text: Optional[str] = None,
    block_type: str = "text",
    page: Optional[int] = None,
    source: str = "",
    extra_meta: Optional[Dict[str, Any]] = None,
    seq: int = 0,
) -> Dict[str, Any]:
    disp = display_text if display_text is not None else index_text
    meta: Dict[str, Any] = {"text_sha1": sha1(index_text)}
    if extra_meta:
        meta.update({k: v for k, v in extra_meta.items() if v is not None})
    return {
        "id": sha1(f"{doc}:{page}:{seq}:{meta['text_sha1']}"),
        "doc": doc,
        "source_file": doc,
        "page": page,
        "block_type": block_type,
        "provenance": source,
        "index_text": index_text,
        "display_text": disp,
        "text": index_text,  # 向后兼容旧读取路径
        "metadata": meta,
    }


# ══════════════════════════ 来源一：抢救旧 chunk ══════════════════════════

def group_of(doc: str) -> str:
    d = str(doc or "")
    if re.match(r"^FAQ_(en|zh|th|ms)_\d+$", d):
        return "faq"
    if d.endswith(".docx"):
        return "thai_docx"
    return "literature"


# 排除名单：pt_durian_2012_MinerU__*.json 是 MY_DOA_Pakej_Teknologi_Durian_2012.pdf
# （同一本书）的劣质重复 MinerU 解析 —— 已实测含系统性中文幻觉注入（38 处）与
# tanah→tidak 反义改词，OCR 质量差（hasil kacukan / telah kacukan 之别）。
# 该书已由 2024 新版 PDF 替代（parse_pdfs 直接解析），此重复解析不再入库。
# 注意：匹配的是 clean_chunks.jsonl 里的 doc 值（741 行输入，含 420 行幸存者）。
_SALVAGE_EXCLUDE_DOCS = ("pt_durian_2012_MinerU",)


def salvage_legacy(stats: Counter) -> List[Dict[str, Any]]:
    """处理 clean_chunks.jsonl：FAQ/泰文原样保留，文献做抢救清洗。"""
    if not LEGACY_CHUNKS.exists():
        print(f"[WARN] 缺少 {LEGACY_CHUNKS}")
        return []

    out: List[Dict[str, Any]] = []
    seq = defaultdict(int)

    for line in LEGACY_CHUNKS.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            stats["legacy_bad_json"] += 1
            continue

        doc = str(r.get("doc") or r.get("source_file") or "unknown")
        if any(pat in doc for pat in _SALVAGE_EXCLUDE_DOCS):
            stats["salvage_excluded"] += 1
            continue
        raw = str(r.get("text") or "")
        g = group_of(doc)
        stats[f"legacy_in_{g}"] += 1

        # FAQ：结构化问答，本就干净，原样保留
        if g == "faq":
            cleaned = _RE_WS.sub(" ", raw).strip()
            if len(cleaned) < 20:
                stats["faq_dropped"] += 1
                continue
            seq[doc] += 1
            out.append(make_chunk(
                doc=doc, index_text=cleaned, block_type="qa_pair",
                source="legacy_faq", seq=seq[doc],
                extra_meta={"category": r.get("category"), "question": r.get("question"),
                            "src_lang": "zh" if doc.startswith("FAQ_zh") else doc.split("_")[1]},
            ))
            stats["out_faq"] += 1
            continue

        # 泰文 DOCX：干净但语种为泰文，标记待翻译（阶段 4.5 处理）
        if g == "thai_docx":
            cleaned = _RE_WS.sub(" ", raw).strip()
            if len(cleaned) < 20:
                stats["thai_dropped"] += 1
                continue
            seq[doc] += 1
            out.append(make_chunk(
                doc=doc, index_text=cleaned, block_type="text",
                source="legacy_thai", seq=seq[doc],
                extra_meta={"src_lang": "th", "needs_translation": True,
                            "original_text": cleaned},
            ))
            stats["out_thai"] += 1
            continue

        # 文献：抢救清洗
        cleaned = strip_residue(raw, drop_table_html=True)
        gr = grade(raw, cleaned)
        stats[f"lit_grade_{gr}"] += 1

        if gr == "junk":
            continue

        if gr == "table_fragment":
            # 残表：只有 caption 可用，裸数值绝不进 index_text
            cap = extract_caption(raw)
            if not cap:
                stats["table_frag_no_caption_dropped"] += 1
                continue
            seq[doc] += 1
            out.append(make_chunk(
                doc=doc, index_text=cap,
                display_text=f"{cap}\n（注：该表格在历史数据中已损坏，表头丢失，数值不可引用）",
                block_type="table_broken", page=r.get("page"),
                source="legacy_lit_salvage", seq=seq[doc],
                extra_meta={"caption": cap, "damaged": True,
                            "raw_fragment": raw[:1200]},
            ))
            stats["out_table_broken"] += 1
            continue

        # usable
        for piece in split_text(cleaned):
            seq[doc] += 1
            out.append(make_chunk(
                doc=doc, index_text=piece, block_type="text", page=r.get("page"),
                source="legacy_lit_salvage", seq=seq[doc],
                extra_meta={"salvaged": True},
            ))
            stats["out_literature"] += 1

    return out


# ══════════════════════════ 来源二：PDF 正常解析 ══════════════════════════

_RE_TBL_JUNK_HDR = re.compile(r"^\s*\|?\s*Col\d", re.I)
# PyMuPDF 对合并单元格/无表头的表会生成 Col5|Col6|... 占位列名。
# 少量占位可容忍（真表格偶有空表头），过半则说明表结构未被正确识别。
_RE_COL_PLACEHOLDER = re.compile(r"\|Col\d+", re.I)


def table_to_markdown(tbl) -> Optional[str]:
    """把 PyMuPDF 表格转 Markdown，并过滤误检。

    find_tables() 会把页眉、摘要块误判为表格（实测 Frontiers p1 的作者行、
    JAD p1 的 ABSTRACT、马来西亚手册 p3 的目录都被误检），必须过滤。
    """
    try:
        md = tbl.to_markdown()
    except Exception:
        return None
    if not md:
        return None
    lines = [l for l in md.splitlines() if l.strip()]
    if len(lines) < 3:  # 至少 表头 + 分隔 + 1 数据行
        return None
    if _RE_TBL_JUNK_HDR.search(lines[0]):
        return None
    ncol = lines[0].count("|")
    if ncol < 3:  # 单列不算表
        return None

    # 占位列头过多 → 表结构未被正确识别，宁可不要
    n_ph = len(_RE_COL_PLACEHOLDER.findall(lines[0]))
    if ncol and n_ph / ncol > 0.4:
        return None

    # 各行列数一致性：误检的文本块列数往往不齐
    consistent = sum(1 for l in lines if abs(l.count("|") - ncol) <= 1)
    if consistent / len(lines) < 0.8:
        return None
    # 整表内容不能是同一个 cell 重复（目录误检的典型特征）
    cells = [c.strip() for l in lines[2:] for c in l.split("|") if c.strip()]
    if cells and len(set(cells)) <= 2:
        return None
    # 表格必须有实质文字，不能全是数字和符号（否则等同裸数值表）
    joined = " ".join(cells)
    if joined and word_ratio(joined) < 0.25:
        return None
    return md.strip()


def parse_pdfs(stats: Counter) -> List[Dict[str, Any]]:
    try:
        import pymupdf
    except ImportError:
        print("[WARN] 缺少 pymupdf，跳过 PDF 解析")
        return []

    out: List[Dict[str, Any]] = []
    seen_doc_hash: Dict[str, str] = {}

    files: List[Path] = []
    for d in PDF_DIRS:
        if d.exists():
            files.extend(sorted(d.glob("*.pdf")))

    for path in files:
        name = path.name
        chunk_size = next(
            (s for prefix, s in DOC_CHUNK_SIZE.items() if name.startswith(prefix)),
            CHUNK_SIZE,
        )
        chunk_overlap = next(
            (o for prefix, o in DOC_CHUNK_OVERLAP.items() if name.startswith(prefix)),
            CHUNK_OVERLAP,
        )
        if chunk_size != CHUNK_SIZE or chunk_overlap != CHUNK_OVERLAP:
            print(f"  ⚙️  chunk 尺寸覆盖生效: {name} -> size={chunk_size} overlap={chunk_overlap}")
        try:
            doc = pymupdf.open(str(path))
        except Exception as exc:
            print(f"  [WARN] 打不开 {name}: {exc}")
            stats["pdf_open_failed"] += 1
            continue

        if len(doc) == 0:
            print(f"  [WARN] 0 页（文件损坏）: {name}")
            stats["pdf_corrupt"] += 1
            doc.close()
            continue

        # 全文去重：LKE_Standard / SOP_V0 存在多份内容相同的副本
        full_text = "".join(p.get_text() for p in doc)
        h = sha1(full_text)
        if h in seen_doc_hash:
            print(f"  ⏭  内容与 {seen_doc_hash[h]} 重复，跳过: {name}")
            stats["pdf_duplicate"] += 1
            doc.close()
            continue
        seen_doc_hash[h] = name

        if len(full_text.strip()) < 200 * len(doc) / 10:
            # 文本层过稀疏（扫描件），标记待 OCR，不入库
            print(f"  ⚠️  文本层稀疏，疑似扫描件，标记待OCR: {name}")
            stats["pdf_needs_ocr"] += 1
            doc.close()
            continue

        seq = 0
        n_tbl = 0
        for pno in range(len(doc)):
            page = doc[pno]

            # ── 表格：整表一个 chunk，不参与切分 ──
            table_bboxes = []
            try:
                for tbl in page.find_tables().tables:
                    md = table_to_markdown(tbl)
                    if not md:
                        continue
                    table_bboxes.append(tbl.bbox)
                    seq += 1
                    n_tbl += 1
                    head = " ".join(md.splitlines()[:1])[:120]
                    out.append(make_chunk(
                        doc=name, index_text=f"【表格】{head}\n{md}"[:2400],
                        display_text=md, block_type="table", page=pno + 1,
                        source="pdf_parse", seq=seq,
                        extra_meta={"table_markdown": md},
                    ))
                    stats["out_table"] += 1
            except Exception:
                pass

            # ── 正文：排除表格区域，避免与表格 chunk 重复 ──
            try:
                if table_bboxes:
                    blocks = page.get_text("blocks")
                    keep = []
                    for b in blocks:
                        bx0, by0, bx1, by1 = b[:4]
                        inside = any(
                            bx0 >= t[0] - 4 and by0 >= t[1] - 4
                            and bx1 <= t[2] + 4 and by1 <= t[3] + 4
                            for t in table_bboxes
                        )
                        if not inside:
                            keep.append(b[4])
                    body = "\n".join(keep)
                else:
                    body = page.get_text("text")
            except Exception:
                body = ""

            body = strip_residue(body, drop_table_html=True)
            if len(body) < MIN_CHUNK_CHARS or word_ratio(body) < 0.5:
                continue
            for piece in split_text(body, size=chunk_size, overlap=chunk_overlap):
                seq += 1
                out.append(make_chunk(
                    doc=name, index_text=piece, block_type="text", page=pno + 1,
                    source="pdf_parse", seq=seq,
                ))
                stats["out_pdf_text"] += 1

            # ── 图片：占位符方案 I5，只记 metadata，不做 VLM（下一轮） ──
            try:
                imgs = page.get_images(full=True)
            except Exception:
                imgs = []
            if imgs:
                stats["images_seen"] += len(imgs)

        print(f"  ✅ {name[:52]:<54} 页{len(doc):>3}  表{n_tbl:>3}  chunk{seq:>4}")
        doc.close()

    return out


# ══════════════════════════ 主流程 ══════════════════════════

def build(write: bool = True) -> None:
    stats: Counter = Counter()

    print("═" * 78)
    print("来源一：抢救 clean_chunks.jsonl")
    print("═" * 78)
    legacy = salvage_legacy(stats)
    print(f"  产出 {len(legacy)} chunk")

    print()
    print("═" * 78)
    print("来源二：解析 PDF")
    print("═" * 78)
    pdfs = parse_pdfs(stats)
    print(f"  产出 {len(pdfs)} chunk")

    allc = legacy + pdfs

    # 全局去重
    seen = set()
    final = []
    for c in allc:
        k = re.sub(r"[\W_]+", "", c["index_text"].lower())[:400]
        if k and k not in seen:
            seen.add(k)
            final.append(c)
        else:
            stats["dedup_removed"] += 1

    print()
    print("═" * 78)
    print("统计")
    print("═" * 78)
    for k in sorted(stats):
        print(f"  {k:<38}{stats[k]:>7}")
    print(f"\n  去重后总 chunk: {len(final)}")
    bt = Counter(c["block_type"] for c in final)
    pv = Counter(c["provenance"] for c in final)
    print(f"  block_type: {dict(bt)}")
    print(f"  provenance: {dict(pv)}")

    if write:
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        with OUT_PATH.open("w", encoding="utf-8") as f:
            for c in final:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\n  已写入 {OUT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 重新分片管道")
    ap.add_argument("cmd", choices=["build", "stats"])
    args = ap.parse_args()
    build(write=(args.cmd == "build"))


if __name__ == "__main__":
    main()
