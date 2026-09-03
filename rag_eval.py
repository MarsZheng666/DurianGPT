#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RAG 检索质量评估框架。

用途：在重新分片前后跑同一套金标，量化对比检索效果。

设计要点：
1. **锚点匹配而非 ID 匹配**。金标里的 `doc_id`（doc_409）是 clean_chunks.jsonl 的行号
   指针，重新分片后必然失效。因此判定"是否召回正确证据"靠的是把金标 source_text 归一化后
   的文本锚点是否出现在召回内容里 —— 这个判据跨分片方案稳定，是 before/after 可比的前提。
2. **同时度量正向指标和负向指标**。只看召回率会掩盖"召回了但是垃圾"的情况，而这恰好是
   当前语料的主要问题（表格 HTML 碎片在数值类问题上被顶到 top-1）。
3. **分层报告**。按问题类型和难度分桶，因为整体均值会掩盖"数值类问题崩掉"这种局部失效。

用法：
    python rag_eval.py baseline --tag before_rechunk
    python rag_eval.py compare --before before_rechunk --after after_rechunk
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "rag_eval"
GOLD_PATH = EVAL_DIR / "gold.jsonl"


# ────────────────────────────── 文本归一化 ──────────────────────────────

_MINERU_KEYS = r"(?:content|bbox|spans|lines|para_blocks|discarded_blocks|type|image_path|page_idx|page_size|angle|index|html)"
_RESIDUE_KV = re.compile(rf"['\"]{_MINERU_KEYS}['\"]\s*:\s*")
_BBOX = re.compile(r"\{?\s*['\"]bbox['\"]\s*:\s*\[[^\]]*\]\s*,?")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]{0,80}>")
_TABLE_TAG = re.compile(r"</?t[dhr]\b|</?table\b", re.I)
_WORD = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0e00-\u0e7fA-Za-z]")


def strip_parser_residue(text: str) -> str:
    """剥离 MinerU 打平后的结构噪声，只留自然语言。"""
    s = str(text or "")
    s = _BBOX.sub(" ", s)
    s = _RESIDUE_KV.sub(" ", s)
    s = _HTML_TAG.sub(" ", s)
    s = re.sub(r"&#x27;|&quot;|&amp;|&gt;|&lt;", " ", s)
    s = re.sub(r"[\[\]{}'\"]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_for_match(text: str) -> str:
    """匹配用归一化：去掉所有非字母数字字符，规避标点/空白/全半角差异。"""
    s = strip_parser_residue(text).lower()
    return re.sub(r"[^\w\u4e00-\u9fff\u0e00-\u0e7f]+", "", s, flags=re.UNICODE)


# 锚点垃圾判据：这些是 MinerU 残渣被 strip 后残留的结构词根 / 纯坐标数字串。
# 它们不是自然语言，拿来当"正确证据"的判据只会制造假阴性。
_JUNK_ROOTS = (
    "blocks", "bbox", "spans", "pageidx", "imagepath", "htmlbody", "paperowner",
    "parablocks", "discarded", "pagesize", "figure", "tablebody", "textlevel",
)


def anchor_is_junk(anchor: str) -> bool:
    """判定锚点本身是否不可用（残渣派生 / 纯数字坐标 / 字母占比过低）。"""
    a = str(anchor or "")
    if len(a) < 14:
        return True
    head = a[:44]
    if any(root in head for root in _JUNK_ROOTS):
        return True
    # 纯数字或数字占比过高（多为 bbox 坐标、DOI 片段、页码串）
    digits = sum(1 for c in a if c.isdigit())
    if digits / len(a) > 0.45:
        return True
    # 必须含足够的字母/汉字/泰文
    if len(_WORD.findall(a)) / len(a) < 0.6:
        return True
    return False


def extract_anchors(
    source_text: str,
    window: int = 24,
    max_anchors: int = 8,
) -> List[str]:
    """从金标 source_text 提取用于判定召回正确性的文本锚点。

    锚点是归一化后的**定长短窗口**（默认 24 字符），而非整段长文本。
    这样设计的原因：索引 chunk 中位长度仅 322 字符，而金标 source_text 里的
    自然语言片段常达 100-180 字符，重新分片后必然跨 chunk 边界，
    用长锚点做包含匹配会产生大量假阴性（实测把 Hit@5 压到 3%）。
    短窗口 + 多锚点取或，才能稳定判定"这段内容有没有被召回到"。
    """
    cleaned = strip_parser_residue(source_text)
    _CH = r"\u4e00-\u9fff\u0e00-\u0e7fA-Za-z0-9"
    _TAIL = _CH + r"，。、；：%℃°\.\-\s"
    pattern = "[" + _CH + "][" + _TAIL + "]{20,}"

    candidates: List[str] = []
    for span in re.findall(pattern, cleaned):
        n = norm_for_match(span)
        if len(n) < window:
            continue
        # 在片段内均匀取窗口，步长为窗口的 1.5 倍以覆盖不同位置
        step = max(window, int(window * 1.5))
        for start in range(0, len(n) - window + 1, step):
            candidates.append(n[start:start + window])

    anchors: List[str] = []
    for c in candidates:
        if anchor_is_junk(c):
            continue
        if any(c in a or a in c for a in anchors):
            continue
        anchors.append(c)
        if len(anchors) >= max_anchors:
            break
    return anchors


# ────────────────────────────── 证据质量判定 ──────────────────────────────

def evidence_defect(text: str) -> Optional[str]:
    """判定单条证据是否为不可用垃圾。返回缺陷类型，None 表示正常。

    这是负向指标的基础。判据依据实测：当前语料的垃圾主要是
    裸表格 HTML（丢了表头的数值，幻觉高风险）和 MinerU 结构残渣。

    ⚠️ 循环论证风险：本函数的正则与 rag_rechunk.py 的清洗正则同源，
    因此清洗后本函数必然报 0% —— 实测确认过这一点（评估报 0.0%，
    独立判据报 27%）。**必须同时看 independent_defect()**，
    它用完全不同的判据（是否像人写的句子）交叉验证。
    """
    t = str(text or "").strip()
    if not t:
        return "empty"
    if _TABLE_TAG.search(t):
        return "table_html"
    if len(_RESIDUE_KV.findall(t)) >= 2 or "'para_blocks'" in t:
        return "parser_residue"
    if len(_WORD.findall(t)) / max(len(t), 1) < 0.55:
        return "noise"
    if len(t) < 30:
        return "too_short"
    return None


# ── 独立判据：故意不复用清洗管道的任何正则 ──
# 目的是交叉验证，防止「用清洗规则去检验清洗结果」的循环论证。
# 判断标准换成"这段文字像不像人写的句子"。
_IND_STRUCT = re.compile(r"[{}]|\[\s*\d+\s*,\s*\d+")
_IND_NUMSEQ = re.compile(r"(?:\b\d+(?:\.\d+)?\b[\s,|]+){4,}")
_IND_TEXTCHAR = re.compile(r"[\u4e00-\u9fff\u0e00-\u0e7fA-Za-z]")


def independent_defect(text: str) -> Optional[str]:
    """独立判据版的垃圾检测。与 evidence_defect 无共享正则。"""
    t = str(text or "").strip()
    if not t:
        return "empty"
    if _IND_STRUCT.search(t):
        return "struct_symbol"
    if t.count("'") >= 4 or t.count('"') >= 4:
        return "stray_quotes"
    if _IND_NUMSEQ.search(t):
        return "number_run"
    if len(_IND_TEXTCHAR.findall(t)) / max(len(t), 1) < 0.5:
        return "low_text_ratio"
    return None


# ────────────────────────────── 指标计算 ──────────────────────────────

def hit_rank(anchors: List[str], evidences: List[str]) -> Optional[int]:
    """返回首个命中任一锚点的证据排名（1-based），未命中返回 None。"""
    for rank, ev in enumerate(evidences, start=1):
        n = norm_for_match(ev)
        if not n:
            continue
        if any(a in n for a in anchors):
            return rank
    return None


def dcg_at_k(rank: Optional[int], k: int) -> float:
    import math
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


# ── 答案要点覆盖率：跨语言可用的补充指标 ──
# 现有金标 94% 是「中文提问 + 英文/泰文原文」，原文锚点匹配在这种情况下
# 会把"召回了语义正确的中文证据"误判为失败。但金标 answer 是中文，
# 召回证据也主要是中文，因此改为度量：答案里的实义要点有多少能在证据中找到支撑。
# 这是弱于锚点法的判据（不能证明来源同一），但跨语言稳定，且对
# before/after 对比有效——同一套判据下的相对变化仍然可信。

_STOP_TERMS = {
    "榴莲", "果实", "种植", "可以", "需要", "进行", "应该", "建议", "注意", "如果",
    "以及", "并且", "同时", "通过", "使用", "具有", "存在", "包括", "等等", "方面",
    "情况", "过程", "条件", "影响", "作用", "结论", "摘要", "步骤", "参数", "参考",
    "风险", "提示", "操作", "要点", "durian", "the", "and", "for", "with", "that",
}


def extract_answer_terms(answer: str, max_terms: int = 40) -> List[str]:
    """从金标答案提取实义要点：数值+单位、专有名词、2-4 字中文词。"""
    s = strip_parser_residue(answer)
    terms: List[str] = []

    # 数值 + 单位（最有判别力，也是幻觉高发区）
    for m in re.findall(r"\d+(?:\.\d+)?\s*(?:%|℃|°C|kg|g|mm|cm|m|天|日|月|年|次|倍|亩|株|度)", s):
        terms.append(norm_for_match(m))

    # 品种代号 / 拉丁名 / 英文专名
    for m in re.findall(r"\bD\s?\d{1,3}\b|[A-Z][a-z]{4,}", s):
        terms.append(norm_for_match(m))

    # 中文实义词：2-4 字
    for span in re.findall(r"[\u4e00-\u9fff]{2,}", s):
        for size in (4, 3, 2):
            for i in range(0, len(span) - size + 1):
                g = span[i:i + size]
                if g not in _STOP_TERMS:
                    terms.append(g)

    seen, out = set(), []
    for t in terms:
        if t and len(t) >= 2 and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_terms:
            break
    return out


def answer_coverage(answer_terms: List[str], evidences: List[str]) -> float:
    """答案要点在召回证据中的覆盖比例，0..1。"""
    if not answer_terms:
        return 0.0
    blob = norm_for_match(" ".join(evidences))
    if not blob:
        return 0.0
    hit = sum(1 for t in answer_terms if t in blob)
    return hit / len(answer_terms)


def _is_self_hit(question: str, evidence_text: str) -> bool:
    """判断这条证据是不是金标问题自身所在的 chunk。

    FAQ 金标的问答对本身就在索引里，用它的问题去检索必然命中它自己 ——
    实测英文 FAQ 的 top1 有 100% 是自命中，Hit@3 因此虚高到 100%。
    这不是检索能力，是数据泄漏，必须单独扣除。
    """
    qn = norm_for_match(question)[:40]
    if len(qn) < 12:
        return False
    return qn in norm_for_match(evidence_text)


def _is_qa_chunk(ev: Dict[str, Any]) -> bool:
    """判断证据是否为 FAQ 问答对 chunk。"""
    if str(ev.get("block_type") or "") == "qa_pair":
        return True
    meta = ev.get("metadata") or {}
    if str(meta.get("block_type") or "") == "qa_pair":
        return True
    doc = str(meta.get("doc") or ev.get("doc") or "")
    return bool(re.match(r"^FAQ_(en|zh|th|ms)_\d+$", doc))


def evaluate_one(
    item: Dict[str, Any],
    evidences: List[Dict[str, Any]],
    k_list: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, Any]:
    anchors = item["anchors"]
    texts = [str(e.get("text") or e.get("display_text") or "") for e in evidences]
    # 知识库口径：排除所有 FAQ 问答对证据。
    # 原因：FAQ 有 en/zh/th/ms 四个平行版本且本身在索引里，用 FAQ 问题检索
    # 会命中自己或其他语种的同一条问答（实测英文 FAQ 召回中 88% 是其他 FAQ）。
    # 这既不是自命中也不是真正的知识检索，必须单独扣除才能衡量
    # "能不能从真实文档语料里找到答案"。
    texts_kb = [t for t, e in zip(texts, evidences) if not _is_qa_chunk(e)]

    # 排除自身后的证据集 —— 这才是"从其他内容里检索"的真实能力
    q = item.get("question") or ""
    texts_ns = [t for t in texts if not _is_self_hit(q, t)]

    rank = hit_rank(anchors, texts)
    rank_ns = hit_rank(anchors, texts_ns)
    rank_kb = hit_rank(anchors, texts_kb)
    defects = [evidence_defect(t) for t in texts]
    n_defect = sum(1 for d in defects if d)
    ind_defects = [independent_defect(t) for t in texts]
    n_ind = sum(1 for d in ind_defects if d)

    out: Dict[str, Any] = {
        "qid": item["qid"],
        "category": item["category"],
        "difficulty": item.get("difficulty"),
        "n_returned": len(evidences),
        "hit_rank": rank,
        "n_defect": n_defect,
        "defect_types": [d for d in defects if d],
        "defect_ratio": n_defect / max(len(texts), 1),
        "top1_defect": bool(defects and defects[0]),
        "ind_defect_ratio": n_ind / max(len(texts), 1),
        "ind_top1_defect": bool(ind_defects and ind_defects[0]),
        "ind_defect_types": [d for d in ind_defects if d],
        # 无结果本身是一种失败模式，要单独记
        "empty_result": len(evidences) == 0,
    }
    for k in k_list:
        out[f"hit@{k}"] = bool(rank is not None and rank <= k)
        out[f"ndcg@{k}"] = dcg_at_k(rank, k)
    out["mrr"] = 1.0 / rank if rank else 0.0
    # 无泄漏口径
    out["n_self_hit"] = len(texts) - len(texts_ns)
    out["hit@5_ns"] = bool(rank_ns is not None and rank_ns <= 5)
    out["hit@3_ns"] = bool(rank_ns is not None and rank_ns <= 3)
    out["hit@1_ns"] = bool(rank_ns is not None and rank_ns <= 1)
    out["mrr_ns"] = 1.0 / rank_ns if rank_ns else 0.0
    out["answer_coverage_ns"] = answer_coverage(
        item.get("answer_terms") or [], texts_ns)
    # 知识库口径 —— 只看非 FAQ 证据
    out["n_qa_evidence"] = len(texts) - len(texts_kb)
    out["hit@1_kb"] = bool(rank_kb is not None and rank_kb <= 1)
    out["hit@3_kb"] = bool(rank_kb is not None and rank_kb <= 3)
    out["hit@5_kb"] = bool(rank_kb is not None and rank_kb <= 5)
    out["answer_coverage_kb"] = answer_coverage(
        item.get("answer_terms") or [], texts_kb)
    out["kb_evidence_rate"] = len(texts_kb) / max(len(texts), 1)
    # kb 口径仅在还剩足够非 FAQ 证据时才有意义。
    # 实测 FAQ 金标的召回中 91% 是 FAQ 证据，扣除后平均只剩 0.4 条，
    # 此时 Hit@3†/覆盖† 度量的是"残渣里有没有答案"，不能解读为检索能力。
    # 对照：legacy 金标（问题由文献生成）的 FAQ 证据占比仅 8-11%，
    # 说明"FAQ 挤占知识语料"并不存在，是 FAQ 问题匹配 FAQ 内容的合理行为。
    out["kb_valid"] = len(texts_kb) >= 2

    # 跨语言可用的补充指标
    cov = answer_coverage(item.get("answer_terms") or [], texts)
    out["answer_coverage"] = cov
    # 覆盖率 ≥0.35 视为"证据实质支撑了答案"，阈值由实测标定
    out["supported"] = cov >= 0.35
    return out


def aggregate(rows: List[Dict[str, Any]], k_list: Tuple[int, ...] = (1, 3, 5)) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {}
    agg: Dict[str, Any] = {"n": n}
    for k in k_list:
        agg[f"hit@{k}"] = sum(r[f"hit@{k}"] for r in rows) / n
        agg[f"ndcg@{k}"] = sum(r[f"ndcg@{k}"] for r in rows) / n
    agg["mrr"] = sum(r["mrr"] for r in rows) / n
    agg["empty_rate"] = sum(r["empty_result"] for r in rows) / n
    agg["defect_ratio"] = sum(r["defect_ratio"] for r in rows) / n
    agg["top1_defect_rate"] = sum(r["top1_defect"] for r in rows) / n
    agg["ind_defect_ratio"] = sum(r.get("ind_defect_ratio", 0) for r in rows) / n
    agg["ind_top1_defect_rate"] = sum(r.get("ind_top1_defect", False) for r in rows) / n
    agg["avg_returned"] = sum(r["n_returned"] for r in rows) / n
    agg["answer_coverage"] = sum(r["answer_coverage"] for r in rows) / n
    for k in ("hit@1_ns", "hit@3_ns", "hit@5_ns", "mrr_ns", "answer_coverage_ns",
              "kb_evidence_rate"):
        if k in rows[0]:
            agg[k] = sum(r.get(k, 0) for r in rows) / n
    # kb 指标只在"还剩 ≥2 条非 FAQ 证据"的样本上统计
    kb_rows = [r for r in rows if r.get("kb_valid")]
    agg["n_kb_valid"] = len(kb_rows)
    agg["kb_valid_rate"] = len(kb_rows) / n
    for k in ("hit@1_kb", "hit@3_kb", "hit@5_kb", "answer_coverage_kb"):
        agg[k] = (sum(r.get(k, 0) for r in kb_rows) / len(kb_rows)) if kb_rows else None
    agg["self_hit_rate"] = sum(r.get("n_self_hit", 0) for r in rows) / max(
        sum(r["n_returned"] for r in rows), 1)
    agg["supported_rate"] = sum(r["supported"] for r in rows) / n
    return agg


# ────────────────────────────── 问题分类 ──────────────────────────────

_CATEGORY_RULES = [
    ("numeric", ["多少", "几个", "含量", "浓度", "用量", "温度", "湿度", "株行距",
                 "亩产", "产量", "百分", "比例", "kg", "cm", "mm", "℃", "%",
                 "how much", "how many"]),
    ("fill_blank", ["____", "＿＿", "填空", "（　）"]),
    ("disease", ["病", "虫", "防治", "症状", "病斑", "药剂", "菌", "蠹", "螨", "蚧"]),
    ("variety", ["品种", "品系", "猫山王", "金枕", "苏丹王", "黑刺", "编号", "代号", "D1", "D2"]),
    ("cultivation", ["种植", "栽培", "定植", "施肥", "修剪", "灌溉", "管理", "嫁接", "授粉", "开花"]),
    ("postharvest", ["采收", "成熟", "保鲜", "贮藏", "储运", "包装", "运输", "催熟"]),
]


def classify_question(q: str) -> str:
    ql = str(q or "").lower()
    for name, kws in _CATEGORY_RULES:
        if any(kw.lower() in ql for kw in kws):
            return name
    return "general"


def detect_lang(text: str) -> str:
    """粗粒度语种判定，覆盖本项目的四个语种：zh / en / ms / th。

    ⚠️ 马来文用拉丁字母书写，只按字符集判断会把它误判成英文。
    实测项目里 440 个马来文 chunk（pt_durian_2012 的 309 条 +
    马来西亚农业部手册 127 条）一直被当作英文处理。
    因此拉丁字母的文本必须再用功能词做二次判别。
    """
    s = str(text or "")
    zh = len(re.findall(r"[\u4e00-\u9fff]", s))
    th = len(re.findall(r"[\u0e00-\u0e7f]", s))
    lat = len(re.findall(r"[A-Za-z]", s))

    if zh and zh >= th and zh * 4 >= lat:
        return "zh"
    if th and th >= zh:
        return "th"
    if not lat:
        return "unknown"

    ms_hits = len(_RE_MALAY_STOP.findall(s))
    en_hits = len(_RE_ENGLISH_STOP.findall(s))
    if ms_hits >= 3 and ms_hits > en_hits:
        return "ms"
    return "en"


# 马来语高频功能词与农业术语。选词标准：在马来语中极常见，
# 且不与英语拼写冲突（例如排除 "ini/itu" 之外的短词以免误命中）。
_RE_MALAY_STOP = re.compile(
    r"\b(?:yang|dan|untuk|dengan|pada|dari|adalah|akan|dapat|tidak|atau"
    r"|dalam|oleh|pokok|baja|buah|daun|tanah|penyakit|perosak|tanaman"
    r"|hasil|boleh|perlu|setiap|kawasan|jarak|cantasan|penuaian|rawatan"
    r"|semburan|racun|kulat|biji|akar|batang|bunga|ranting|sepokok|setahun"
    r"|disyorkan|kandungan|ialah|ke|di|itu|ini)\b",
    re.I,
)
_RE_ENGLISH_STOP = re.compile(
    r"\b(?:the|of|and|to|in|is|are|was|were|for|with|that|this|which"
    r"|from|be|been|have|has|were|their|these|those|been|also|such)\b",
    re.I,
)


# ────────────────────────────── 金标集构建 ──────────────────────────────

_RE_FAQ_DOC = re.compile(r"^FAQ_(en|zh|th|ms)_\d+$")


def build_gold_from_faq() -> List[Dict[str, Any]]:
    """从 FAQ 四语种问答对构建金标。

    为什么改用 FAQ 作主源：原金标取自 qa_dataset*.jsonl，其中 94% 是
    "中文提问 + 英文原文" —— 那是 QA 生成方式的产物，不代表真实使用。
    实际用户主要用英文提问，其次是中文/马来文/泰文。
    FAQ 恰好四语种齐全（en 347 / zh 357 / ms 155 / th 141），
    问题与答案同语种，是唯一能反映真实使用场景的金标源。

    锚点取自 answer（而非 source_text），因为 FAQ 的 answer 就是标准答案，
    且与 question 同语种，不存在跨语言失配。
    """
    src = BASE_DIR / "clean_chunks.jsonl"
    if not src.exists():
        print(f"[WARN] 缺少 {src}")
        return []

    gold: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    seen_q = set()

    for line in src.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue

        m = _RE_FAQ_DOC.match(str(r.get("doc") or ""))
        if not m:
            continue
        lang = m.group(1)

        q = str(r.get("question") or "").strip()
        a = str(r.get("answer") or "").strip()
        if not q or not a:
            stats[f"{lang}_missing_field"] += 1
            continue

        qk = norm_for_match(q)[:80]
        if qk in seen_q:
            stats[f"{lang}_dup"] += 1
            continue

        anchors = extract_anchors(a)
        if not anchors:
            stats[f"{lang}_no_anchor"] += 1
            continue

        seen_q.add(qk)
        gold.append({
            "qid": f"faq_{lang}_{len(gold):04d}",
            "question": q,
            "category": r.get("category") or classify_question(q),
            "difficulty": None,
            "origin": "faq",
            "anchors": anchors,
            # q_lang 必须由问题文本推导，不能用 FAQ 源文档的语种标记——
            # 实测 346 条 q_lang='en' 里有 237 条(68.5%)问题正文是纯中文
            # （金标构建早期按 doc 正则取语种，把整个桶标错了，导致
            # "英文 Hit@3 100%" 实为大半中文题的数字，真实英文从未被
            # 单独度量过）。q_lang_orig 保留原错误标注供审计与历史报告对账。
            "q_lang": detect_lang(q),
            "q_lang_orig": lang,
            "src_lang": lang,
            # 跨语言 = 问题与答案文本语种不同。不能用 q_lang != lang（源文档
            # 标记）——那 237 条"en"桶实为中文问答，若按源标记判跨语言会把
            # 同语种题错误踢出主口径。
            "cross_lingual": detect_lang(q) != detect_lang(a),
            "gold_answer": a[:2000],
            "answer_terms": extract_answer_terms(a),
            "ref_doc_id": r.get("doc"),
            "ref_source_file": r.get("source_file"),
        })
        stats[f"{lang}_kept"] += 1

    print(f"[gold] FAQ 金标 {len(gold)} 条")
    print(f"[gold] 明细: {dict(stats)}")
    return gold


def build_gold(limit: Optional[int] = None, include_legacy: bool = True) -> List[Dict[str, Any]]:
    """构建金标集。

    主源：FAQ 四语种（问答同语种，反映真实使用）。
    辅源：qa_dataset*.jsonl（跨语言，标记 cross_lingual=True，仅作参考口径）。
    """
    gold = build_gold_from_faq()
    seen_q = {norm_for_match(g["question"])[:80] for g in gold}

    if include_legacy:
        sources = [
            (BASE_DIR / "qa_dataset_v2.cleaned.jsonl", "v2_cleaned"),
            (BASE_DIR / "qa_dataset.jsonl", "v1"),
        ]
        stats: Counter = Counter()
        for path, origin in sources:
            if not path.exists():
                continue
            for line in path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                q = str(r.get("question") or "").strip()
                st = str(r.get("source_text") or "")
                if not q or not st:
                    continue
                qk = norm_for_match(q)[:80]
                if qk in seen_q:
                    stats["dup"] += 1
                    continue
                anchors = extract_anchors(st)
                if not anchors:
                    stats["no_anchor"] += 1
                    continue
                seen_q.add(qk)
                q_lang = detect_lang(q)
                src_lang = detect_lang(strip_parser_residue(st))
                gold.append({
                    "qid": f"{origin}_{len(gold):04d}",
                    "question": q,
                    "category": classify_question(q),
                    "difficulty": r.get("difficulty"),
                    "origin": origin,
                    "anchors": anchors,
                    "q_lang": q_lang,
                    "src_lang": src_lang,
                    "cross_lingual": q_lang != src_lang,
                    "gold_answer": str(r.get("answer") or "")[:2000],
                    "answer_terms": extract_answer_terms(str(r.get("answer") or "")),
                    "ref_doc_id": r.get("doc_id"),
                    "ref_source_file": r.get("source_file"),
                })
                stats["kept"] += 1
                if limit and len(gold) >= limit:
                    break
        print(f"[gold] legacy 补充: {dict(stats)}")

    print(f"\n[gold] 合计 {len(gold)} 条")
    print(f"[gold] 按提问语种: {dict(Counter(g['q_lang'] for g in gold))}")
    print(f"[gold] 按来源: {dict(Counter(g['origin'] for g in gold))}")
    print(f"[gold] 类型分布: {dict(Counter(g['category'] for g in gold))}")
    mono = [g for g in gold if not g["cross_lingual"]]
    print(f"[gold] 同语种（主口径）: {len(mono)}；跨语种（仅参考）: {len(gold) - len(mono)}")
    return gold


def load_gold() -> List[Dict[str, Any]]:
    if not GOLD_PATH.exists():
        raise SystemExit(f"金标集不存在: {GOLD_PATH}\n先运行: python rag_eval.py build-gold")
    return [json.loads(l) for l in GOLD_PATH.open(encoding="utf-8") if l.strip()]


def _index_corpus_blob() -> str:
    """把索引 docstore 里所有 node 文本拼成一个归一化大串，用于可达性预检。"""
    p = BASE_DIR / "rag_llamaindex_storage" / "docstore.json"
    if not p.exists():
        raise SystemExit(f"索引不存在: {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    parts = []
    for v in (d.get("docstore/data") or {}).values():
        dd = v.get("__data__") or v
        parts.append(norm_for_match(dd.get("text") or ""))
    return " ||| ".join(parts)


def audit_gold() -> Dict[str, Any]:
    """可达性预检：金标锚点是否真的存在于当前索引中。

    这一步是 before/after 可比的前提。若某条金标的内容压根不在索引里，
    检索再好也不可能召回，把它算作"检索失败"会掩盖真正的检索质量问题，
    也会让重新分片后的提升无法归因（究竟是分片变好了，还是内容补回来了）。
    """
    gold = load_gold()
    blob = _index_corpus_blob()
    reachable, unreachable = [], []
    for g in gold:
        if any(a in blob for a in g["anchors"]):
            reachable.append(g["qid"])
        else:
            unreachable.append(g["qid"])

    result = {
        "n_gold": len(gold),
        "n_reachable": len(reachable),
        "n_unreachable": len(unreachable),
        "reachable_rate": len(reachable) / max(len(gold), 1),
        "reachable_qids": reachable,
        "unreachable_qids": unreachable,
    }
    out = EVAL_DIR / "gold_audit.json"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    by_cat = Counter(g["category"] for g in gold if g["qid"] in set(unreachable))
    print(f"\n{'=' * 70}")
    print("金标可达性预检（内容是否真的在索引里）")
    print(f"{'=' * 70}")
    print(f"  金标总数      {len(gold)}")
    print(f"  可达（在索引） {len(reachable)}  ({result['reachable_rate'] * 100:.1f}%)")
    print(f"  不可达         {len(unreachable)}  ({(1 - result['reachable_rate']) * 100:.1f}%)")
    print(f"\n  不可达的按类型分布: {dict(by_cat)}")
    print(f"\n  → 语料覆盖率上限 = {result['reachable_rate'] * 100:.1f}%")
    print("    重新分片后应重跑此预检：可达率上升说明内容被挽回，")
    print("    这与检索排序改善是两件不同的事，必须分开归因。")
    print(f"\n  已写入 {out}")
    return result


def save_gold(gold: List[Dict[str, Any]]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with GOLD_PATH.open("w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"[gold] 已写入 {GOLD_PATH}")


# ────────────────────────────── 跑评估 ──────────────────────────────

def run_baseline(tag: str, top_k: int = 5, limit: Optional[int] = None) -> None:
    import rag_llamaindex as R

    gold = load_gold()
    if limit:
        gold = gold[:limit]

    # 可达性标记：分离"内容缺失"与"检索没排上来"两种失败
    audit_path = EVAL_DIR / "gold_audit.json"
    reachable_set: Optional[set] = None
    if audit_path.exists():
        au = json.loads(audit_path.read_text(encoding="utf-8"))
        reachable_set = set(au.get("reachable_qids") or [])
        print(f"[eval] 已加载可达性预检: {len(reachable_set)}/{au.get('n_gold')} 条可达")
    else:
        print("[eval] 未找到 gold_audit.json，将只报全量口径（建议先跑 audit-gold）")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    latencies: List[float] = []

    for i, item in enumerate(gold, 1):
        t1 = time.time()
        try:
            ev = R.retrieve(item["question"], top_k=top_k)
        except Exception as exc:
            print(f"  [WARN] qid={item['qid']} 检索异常: {exc}")
            ev = []
        latencies.append(time.time() - t1)
        row = evaluate_one(item, ev)
        row["reachable"] = (reachable_set is None) or (item["qid"] in reachable_set)
        row["cross_lingual"] = bool(item.get("cross_lingual"))
        row["q_lang"] = item.get("q_lang") or "unknown"
        # 核心口径：内容在库里 且 问题与原文同语种。
        # 只有这个子集能干净归因到"检索排序好不好"。
        row["core"] = row["reachable"] and not row["cross_lingual"]
        rows.append(row)
        if i % 50 == 0:
            print(f"  ...{i}/{len(gold)}")

    def _with_latency(rs: List[Dict], lats: List[float]) -> Dict[str, Any]:
        a = aggregate(rs)
        if a and lats:
            a["latency_p50"] = statistics.median(lats)
            a["latency_p95"] = sorted(lats)[int(len(lats) * 0.95)] if len(lats) > 1 else lats[0]
        return a

    overall = _with_latency(rows, latencies)
    reach_rows = [r for r in rows if r["reachable"]]
    core_rows = [r for r in rows if r["core"]]
    overall_reachable = aggregate(reach_rows)
    overall_core = aggregate(core_rows)

    def _bucket(rs: List[Dict], key: str) -> Dict[str, Any]:
        return {
            str(v): aggregate([r for r in rs if r[key] == v])
            for v in sorted({r[key] for r in rs}, key=lambda x: str(x))
        }

    defect_kinds = Counter()
    ind_kinds = Counter()
    for r in rows:
        defect_kinds.update(r["defect_types"])
        ind_kinds.update(r.get("ind_defect_types") or [])

    report = {
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "top_k": top_k,
        "n_gold": len(gold),
        "n_reachable": len(reach_rows),
        "n_core": len(core_rows),
        "reachable_rate": len(reach_rows) / max(len(rows), 1),
        "core_rate": len(core_rows) / max(len(rows), 1),
        "elapsed_sec": round(time.time() - t0, 1),
        "overall": overall,
        "overall_reachable": overall_reachable,
        "overall_core": overall_core,
        # by_lang 只用同语种金标 —— legacy 的 243 条跨语言金标（中文提问/英文原文）
        # 混进来会把中文那一行从 53.4% 拉低到 36.5%，是纯粹的口径污染。
        "by_lang": _bucket([r for r in rows if not r["cross_lingual"]], "q_lang"),
        "by_lang_all": _bucket(rows, "q_lang"),
        "by_lang_core": _bucket(core_rows, "q_lang"),
        "by_category": _bucket(rows, "category"),
        "by_category_core": _bucket(core_rows, "category"),
        "by_difficulty": _bucket(rows, "difficulty"),
        "defect_kinds": dict(defect_kinds),
        "ind_defect_kinds": dict(ind_kinds),
        "rows": rows,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"report_{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    print(f"\n[eval] 报告已写入 {out}")


def _pct(x: Any) -> str:
    return f"{float(x) * 100:5.1f}%" if isinstance(x, (int, float)) else "    -"


def print_report(rep: Dict[str, Any]) -> None:
    o = rep["overall"]
    orr = rep.get("overall_reachable") or {}
    oc = rep.get("overall_core") or {}
    print(f"\n{'=' * 82}")
    print(f"RAG 检索评估报告   tag={rep['tag']}   金标 {rep['n_gold']} 条   top_k={rep['top_k']}")
    print(f"{'=' * 82}")
    print(f"口径说明：")
    print(f"  全量     {rep['n_gold']:>4} 条 —— 端到端体验，含语料缺失与跨语言干扰")
    if rep.get("n_reachable") is not None:
        print(f"  可达     {rep['n_reachable']:>4} 条 —— 内容确实在索引里（"
              f"{rep.get('reachable_rate', 0) * 100:.1f}%）")
    if rep.get("n_core") is not None:
        print(f"  核心     {rep['n_core']:>4} 条 —— 可达 ∩ 同语种（"
              f"{rep.get('core_rate', 0) * 100:.1f}%）★主指标，唯一可干净归因")

    print("\n【正向指标 —— 该找到的有没有找到】")
    print(f"  {'口径':<14}{'n':>6}{'Hit@1':>9}{'Hit@3':>9}{'Hit@5':>9}{'MRR':>9}{'nDCG@5':>9}")
    print(f"  {'─' * 65}")
    for label, a in (("全量", o), ("可达", orr), ("核心 ★", oc)):
        if not a:
            continue
        print(f"  {label:<14}{a.get('n', 0):>6}{_pct(a['hit@1']):>9}{_pct(a['hit@3']):>9}"
              f"{_pct(a['hit@5']):>9}{a['mrr']:>9.4f}{a['ndcg@5']:>9.4f}")

    print("\n【答案支撑度 —— 跨语言可用，本项目主力指标】")
    print(f"  {'口径':<14}{'n':>6}{'要点覆盖率':>12}{'支撑率':>10}")
    print(f"  {'─' * 44}")
    for label, a in (("全量", o), ("可达", orr), ("核心 ★", oc)):
        if not a:
            continue
        print(f"  {label:<14}{a.get('n', 0):>6}{_pct(a['answer_coverage']):>12}{_pct(a['supported_rate']):>10}")
    print("  覆盖率 = 金标答案实义要点在召回证据中的命中比例；支撑率 = 覆盖率≥0.35 的占比")

    print("\n【负向指标 —— 找回来的是不是垃圾】（全量口径，垃圾与语种无关）")
    print(f"  证据垃圾率      {_pct(o['defect_ratio'])}   （召回证据中不可用的占比）")
    print(f"  Top1 垃圾率     {_pct(o['top1_defect_rate'])}   ★（排第一的就是垃圾，危害最大）")
    print(f"  空结果率        {_pct(o['empty_rate'])}")
    print(f"  ── 独立判据交叉验证（不复用清洗正则，防循环论证）──")
    print(f"  独立证据垃圾率  {_pct(o.get('ind_defect_ratio', 0))}")
    print(f"  独立Top1垃圾率  {_pct(o.get('ind_top1_defect_rate', 0))} ★★ 最可信的负向指标")
    print(f"  平均返回条数    {o['avg_returned']:.2f}")
    print(f"\n【性能】 P50 {o.get('latency_p50', 0) * 1000:.0f}ms   P95 {o.get('latency_p95', 0) * 1000:.0f}ms")

    if rep.get("defect_kinds"):
        print(f"\n【垃圾类型分布】 管道判据 {rep['defect_kinds']}")
    if rep.get("ind_defect_kinds"):
        print(f"【垃圾类型分布】 独立判据 {rep['ind_defect_kinds']}")

    bl = rep.get("by_lang") or {}
    if bl:
        _LN = {"en": "英文 ★主要用户", "zh": "中文", "ms": "马来文", "th": "泰文",
               "unknown": "未知"}
        print(f"\n{'─' * 82}")
        print(f"{'按提问语种（仅同语种金标）':<24}{'n':>5}{'Hit@3':>8}{'Hit@3†':>9}"
              f"{'覆盖':>8}{'覆盖†':>9}{'FAQ证据':>9}{'空结果':>8}")
        print(f"{'─' * 82}")
        for lg, a in sorted(bl.items(), key=lambda kv: -(kv[1].get("n") or 0)):
            if not a:
                continue
            qa_rate = 1.0 - a.get("kb_evidence_rate", 1.0)
            nkb = a.get("n_kb_valid", 0)
            h3kb = _pct(a["hit@3_kb"]) if a.get("hit@3_kb") is not None else "  n/a"
            ckb = _pct(a["answer_coverage_kb"]) if a.get("answer_coverage_kb") is not None else "  n/a"
            if nkb < max(20, a["n"] * 0.15):
                h3kb = ckb = " 样本不足"
            print(f"{_LN.get(lg, lg):<24}{a['n']:>5}{_pct(a['hit@3']):>8}"
                  f"{h3kb:>9}"
                  f"{_pct(a.get('answer_coverage', 0)):>8}"
                  f"{ckb:>9}"
                  f"{_pct(qa_rate):>9}"
                  f"{_pct(a['empty_rate']):>8}")
        print("  † = **知识库口径**：只算非 FAQ 证据。FAQ 有四语种平行版本且本身在索引里，")
        print("    用 FAQ 问题检索会命中自己或其他语种的同一条问答（实测占召回 88%），")
        print("    扣除后才反映\"能否从真实文档语料中找到答案\"。")
        print("  「FAQ证据」列 = 召回结果中 FAQ 问答对所占比例。")
        print("  FAQ 金标的 FAQ证据占比达 91%，扣除后剩余样本不足以支撑 † 指标，故标为")
        print("  「样本不足」。对照 legacy 金标（问题由文献生成）该占比仅 8-11%，说明")
        print("  这是 FAQ 问题匹配 FAQ 内容的合理行为，不是 FAQ 挤占知识语料。")

    for title, key in (("按问题类型（核心口径 ★）", "by_category_core"),
                       ("按问题类型（全量）", "by_category")):
        buckets = rep.get(key) or {}
        if not buckets:
            continue
        print(f"\n{'─' * 82}")
        print(f"{title:<24}{'n':>5}{'Hit@1':>9}{'Hit@3':>9}{'Hit@5':>9}{'垃圾率':>9}{'Top1垃圾':>10}")
        print(f"{'─' * 82}")
        for cat, a in sorted(buckets.items(), key=lambda kv: -(kv[1].get("n") or 0)):
            if not a:
                continue
            print(f"{cat:<24}{a['n']:>5}{_pct(a['hit@1']):>9}{_pct(a['hit@3']):>9}"
                  f"{_pct(a['hit@5']):>9}{_pct(a['defect_ratio']):>9}{_pct(a['top1_defect_rate']):>10}")

    bd = rep.get("by_difficulty") or {}
    if bd:
        print(f"\n{'─' * 78}")
        print(f"{'按难度':<20}{'n':>5}{'Hit@1':>9}{'Hit@3':>9}{'Hit@5':>9}{'垃圾率':>9}{'Top1垃圾':>10}")
        print(f"{'─' * 78}")
        for d, a in bd.items():
            if not a:
                continue
            print(f"{d:<20}{a['n']:>5}{_pct(a['hit@1']):>9}{_pct(a['hit@3']):>9}"
                  f"{_pct(a['hit@5']):>9}{_pct(a['defect_ratio']):>9}{_pct(a['top1_defect_rate']):>10}")


# ────────────────────────────── 对比 ──────────────────────────────

_HIGHER_BETTER = {"hit@1", "hit@3", "hit@5", "mrr", "ndcg@1", "ndcg@3", "ndcg@5",
                  "answer_coverage", "supported_rate",
                  "hit@1_ns", "hit@3_ns", "hit@5_ns", "answer_coverage_ns",
                  "hit@1_kb", "hit@3_kb", "hit@5_kb", "answer_coverage_kb"}
_LOWER_BETTER = {"defect_ratio", "top1_defect_rate", "empty_rate", "latency_p50",
                 "latency_p95", "ind_defect_ratio", "ind_top1_defect_rate"}


def compare(before_tag: str, after_tag: str) -> None:
    pb = EVAL_DIR / f"report_{before_tag}.json"
    pa = EVAL_DIR / f"report_{after_tag}.json"
    for p in (pb, pa):
        if not p.exists():
            raise SystemExit(f"报告不存在: {p}")
    b = json.loads(pb.read_text(encoding="utf-8"))
    a = json.loads(pa.read_text(encoding="utf-8"))

    if b["n_gold"] != a["n_gold"]:
        print(f"[WARN] 两次金标条数不同 ({b['n_gold']} vs {a['n_gold']})，对比可能不公平")

    print(f"\n{'=' * 84}")
    print(f"对比: {before_tag}  →  {after_tag}")
    print(f"{'=' * 84}")

    def show(title: str, ob: Dict, oa: Dict, keys: List[str]) -> None:
        print(f"\n{title}")
        print(f"  {'指标':<20}{'before':>11}{'after':>11}{'变化':>12}  判定")
        print(f"  {'─' * 68}")
        for k in keys:
            if k not in ob or k not in oa:
                continue
            vb, va = float(ob[k]), float(oa[k])
            d = va - vb
            if k in _HIGHER_BETTER:
                verdict = "✅ 改善" if d > 0.005 else ("❌ 退化" if d < -0.005 else "≈ 持平")
            elif k in _LOWER_BETTER:
                verdict = "✅ 改善" if d < -0.005 else ("❌ 退化" if d > 0.005 else "≈ 持平")
            else:
                verdict = ""
            is_pct = k not in {"mrr", "ndcg@1", "ndcg@3", "ndcg@5", "latency_p50",
                               "latency_p95", "avg_returned"}
            if is_pct:
                print(f"  {k:<20}{vb * 100:>10.1f}%{va * 100:>10.1f}%{d * 100:>+11.1f}pp  {verdict}")
            else:
                print(f"  {k:<20}{vb:>11.4f}{va:>11.4f}{d:>+11.4f}  {verdict}")

    show("【总体（全量金标）】", b["overall"], a["overall"],
         ["answer_coverage", "supported_rate", "hit@1", "hit@3", "hit@5", "mrr", "ndcg@5",
          "defect_ratio", "top1_defect_rate", "ind_defect_ratio",
          "ind_top1_defect_rate", "empty_rate"])

    if b.get("overall_core") and a.get("overall_core"):
        show("【核心口径 ★ 可达∩同语种 —— 纯检索排序能力】",
             b["overall_core"], a["overall_core"],
             ["hit@1", "hit@3", "hit@5", "mrr", "ndcg@5"])

    rb_rate = b.get("reachable_rate")
    ra_rate = a.get("reachable_rate")
    if rb_rate is not None and ra_rate is not None:
        print(f"\n【语料覆盖率】 金标内容可达率 "
              f"{rb_rate * 100:.1f}% → {ra_rate * 100:.1f}% "
              f"({(ra_rate - rb_rate) * 100:+.1f}pp)")
        print("  这一项代表内容是否被挽回，与检索排序改善是两件事，不要混为一谈。")
        print(f"  核心可比子集: {b.get('n_core')} → {a.get('n_core')} 条")

    if b.get("by_lang") and a.get("by_lang"):
        _LN = {"en": "英文★", "zh": "中文", "ms": "马来文", "th": "泰文"}
        print(f"\n【分语种 Hit@3† / 覆盖率†（知识库口径，已排除 FAQ 证据）】")
        print(f"  {'语种':<10}{'n':>5}{'Hit@3 before':>14}{'after':>9}{'变化':>10}"
              f"{'覆盖 before':>13}{'after':>9}{'变化':>10}")
        print(f"  {'─' * 80}")
        for lg in sorted(set(b["by_lang"]) | set(a["by_lang"])):
            ab, aa = b["by_lang"].get(lg), a["by_lang"].get(lg)
            if not ab or not aa:
                continue
            hd = aa.get("hit@3_kb", 0) - ab.get("hit@3_kb", 0)
            cd = aa.get("answer_coverage_kb", 0) - ab.get("answer_coverage_kb", 0)
            print(f"  {_LN.get(lg, lg):<10}{aa['n']:>5}{ab.get('hit@3_kb',0)*100:>13.1f}%"
                  f"{aa.get('hit@3_kb',0)*100:>8.1f}%{hd*100:>+9.1f}pp"
                  f"{ab.get('answer_coverage_kb',0)*100:>12.1f}%"
                  f"{aa.get('answer_coverage_kb',0)*100:>8.1f}%{cd*100:>+9.1f}pp")

    print(f"\n【分类型 Hit@3 / 垃圾率】")
    print(f"  {'类型':<14}{'n':>5}{'Hit@3 before':>14}{'after':>9}{'变化':>10}"
          f"{'垃圾率 before':>15}{'after':>9}{'变化':>10}")
    print(f"  {'─' * 84}")
    cats = sorted(set(b["by_category"]) | set(a["by_category"]))
    for c in cats:
        ab, aa = b["by_category"].get(c), a["by_category"].get(c)
        if not ab or not aa:
            continue
        h_d = aa["hit@3"] - ab["hit@3"]
        g_d = aa["defect_ratio"] - ab["defect_ratio"]
        print(f"  {c:<14}{aa['n']:>5}{ab['hit@3'] * 100:>13.1f}%{aa['hit@3'] * 100:>8.1f}%"
              f"{h_d * 100:>+9.1f}pp{ab['defect_ratio'] * 100:>14.1f}%"
              f"{aa['defect_ratio'] * 100:>8.1f}%{g_d * 100:>+9.1f}pp")

    # 逐题回归/进步清单 —— 定位具体是哪些问题变好变坏
    rb = {r["qid"]: r for r in b["rows"]}
    ra = {r["qid"]: r for r in a["rows"]}
    common = set(rb) & set(ra)
    fixed = [q for q in common if not rb[q]["hit@3"] and ra[q]["hit@3"]]
    broken = [q for q in common if rb[q]["hit@3"] and not ra[q]["hit@3"]]
    print(f"\n【逐题变化】共同题 {len(common)} 条")
    print(f"  新召回成功（before 失败 → after 成功）: {len(fixed)} 条")
    print(f"  召回退化（before 成功 → after 失败）  : {len(broken)} 条")
    if broken:
        print(f"  ⚠️  退化题目需人工复核，前 10 条 qid: {broken[:10]}")

    gold = {g["qid"]: g for g in load_gold()}
    if broken:
        print(f"\n  退化样例:")
        for q in broken[:3]:
            print(f"    - [{gold.get(q, {}).get('category')}] {gold.get(q, {}).get('question', '')[:60]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 检索质量评估")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("build-gold", help="从 QA 数据集构建金标集")
    g.add_argument("--limit", type=int, default=None)

    sub.add_parser("audit-gold", help="可达性预检：金标内容是否真的在索引里")

    r = sub.add_parser("baseline", help="跑一次评估并存报告")
    r.add_argument("--tag", required=True, help="报告标签，如 before_rechunk")
    r.add_argument("--top-k", type=int, default=5)
    r.add_argument("--limit", type=int, default=None)

    c = sub.add_parser("compare", help="对比两份报告")
    c.add_argument("--before", required=True)
    c.add_argument("--after", required=True)

    s = sub.add_parser("show", help="打印已有报告")
    s.add_argument("--tag", required=True)

    args = ap.parse_args()

    if args.cmd == "build-gold":
        save_gold(build_gold(limit=args.limit))
    elif args.cmd == "audit-gold":
        audit_gold()
    elif args.cmd == "baseline":
        run_baseline(args.tag, top_k=args.top_k, limit=args.limit)
    elif args.cmd == "compare":
        compare(args.before, args.after)
    elif args.cmd == "show":
        p = EVAL_DIR / f"report_{args.tag}.json"
        print_report(json.loads(p.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
