#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线翻译管道：把 rag_build/chunks.jsonl 里的非中文内容译成中文。

为什么要做（实测依据，见 docs/RAG_跨语言检索方案.md）：
  embedding 是中文单语 bge-small-zh-v1.5，实测中文查询下——
    中文同义句相似度 0.873，中文无关内容 0.501
    英文同义句 0.339，泰文同义句 0.295
  英文/泰文同义句的分数**低于中文无关内容**，意味着任意一条不相关的中文 chunk
  都会排在真正相关的外文 chunk 前面。外文内容对中文查询是死的。

为什么不换多语言 embedding：
  实测 bge-reranker-v2-m3(XLM-R) 跨语言对齐分数 英文 5.640 / 马来文 2.038 /
  泰文 -1.961，对齐质量随语言资源量呈数量级差距。换模型是把对齐负担交给模型
  实时承担；离线翻译在入库阶段就消灭跨语言问题，检索时纯中文对纯中文。

设计要点：
  1. **幂等缓存**：以 text_sha1 为 key 存 translations.jsonl，重跑不重复调用。
  2. **原文永不覆盖**：写入 metadata.original_text。泰文源 docx 已丢失，
     现存 chunk 是仅存副本。
  3. **术语表按需注入**：只把当前 chunk 实际命中的术语放进 prompt，
     而不是塞整张表（省 token，也避免无关术语误导模型按位置硬凑——
     首次测试时因列举 Musang King 导致泰国品种 Kan Yao 被错译为猫山王）。
  4. **数值校验**：译文与原文的数字集合必须一致，不一致则重试；仍不一致
     则标记 number_mismatch 供人工复核。这是防幻觉的硬闸门。
  5. FAQ 的 en/th/ms 条目**不翻译** —— 它们是刻意做的多语种问答，
     用于服务对应语种的用户，译成中文会破坏产品设计。

用法：
    python rag_translate.py plan              # 只统计工作量，不调用模型
    python rag_translate.py run               # 执行翻译（可中断，断点续跑）
    python rag_translate.py run --limit 50    # 小批量试跑
    python rag_translate.py apply             # 把译文合并回 chunks.jsonl
    python rag_translate.py verify            # 抽检译文质量
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
BUILD_DIR = BASE_DIR / "rag_build"
CHUNKS_PATH = BUILD_DIR / "chunks.jsonl"
CACHE_PATH = BUILD_DIR / "translations.jsonl"
GLOSSARY_PATH = BUILD_DIR / "glossary.json"
OUT_PATH = BUILD_DIR / "chunks_translated.jsonl"

LLM_URL = os.getenv("DURIAN_TRANSLATE_URL", "http://127.0.0.1:8010/v1/chat/completions")
LLM_MODEL = os.getenv("DURIAN_TRANSLATE_MODEL", "durian-base")
MAX_WORKERS = int(os.getenv("DURIAN_TRANSLATE_WORKERS", "4"))
MAX_RETRY = 3


# ══════════════════════════ 语种判定 ══════════════════════════

_RE_ZH = re.compile(r"[\u4e00-\u9fff]")
_RE_TH = re.compile(r"[\u0e00-\u0e7f]")
_RE_LAT = re.compile(r"[A-Za-z]")


def detect_lang(text: str) -> str:
    s = str(text or "")
    counts = {"zh": len(_RE_ZH.findall(s)), "th": len(_RE_TH.findall(s)),
              "en": len(_RE_LAT.findall(s))}
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else "unknown"


def needs_translation(chunk: Dict[str, Any]) -> bool:
    """判断是否需要翻译。

    FAQ 的多语种条目刻意保留 —— 它们服务对应语种用户，译成中文会破坏产品设计。
    """
    if chunk.get("block_type") == "qa_pair":
        return False
    if chunk.get("provenance") == "legacy_faq":
        return False
    lang = detect_lang(chunk.get("index_text") or "")
    if lang not in ("en", "th"):
        return False
    # 已经译过的（metadata 里有标记）跳过
    if (chunk.get("metadata") or {}).get("translated"):
        return False
    return True


# ══════════════════════════ 术语表 ══════════════════════════

# fruit_identity 段无条件注入（不依赖是否命中）——实测模型会把榴莲误译为芒果/荔枝
ALWAYS_TERMS: List[Tuple[str, str]] = []


def load_glossary() -> List[Tuple[str, str]]:
    """展平术语表为 (源词, 译名) 列表，按源词长度降序（长词优先匹配）。

    副作用：填充 ALWAYS_TERMS（fruit_identity 段），该段无条件进 prompt。
    """
    global ALWAYS_TERMS
    if not GLOSSARY_PATH.exists():
        print(f"[WARN] 术语表不存在: {GLOSSARY_PATH}")
        return []
    data = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    pairs: List[Tuple[str, str]] = []
    always: List[Tuple[str, str]] = []
    for section, items in data.items():
        if section.startswith("_") or not isinstance(items, dict):
            continue
        for src, dst in items.items():
            if src.startswith("_") or not isinstance(dst, str):
                continue
            if section == "fruit_identity":
                always.append((src, dst))
            else:
                pairs.append((src, dst))
    pairs.sort(key=lambda kv: -len(kv[0]))
    ALWAYS_TERMS = always
    return pairs


def relevant_terms(text: str, glossary: List[Tuple[str, str]], limit: int = 18) -> List[Tuple[str, str]]:
    """只挑当前文本实际出现的术语。

    这一步很关键：把整张术语表塞进 prompt 会让模型"按位置硬凑"——
    首次测试时 prompt 里列了 Musang King=猫山王，泰文里的 Kan Yao(长柄)
    就被错译成了猫山王。只给实际命中的术语能避免这类干扰。
    """
    low = text.lower()
    hits: List[Tuple[str, str]] = []
    for src, dst in glossary:
        if len(src) < 3:
            continue
        # 拉丁词用词边界匹配，泰文/中文直接子串匹配
        if _RE_LAT.search(src):
            if re.search(r"\b" + re.escape(src.lower()) + r"\b", low):
                hits.append((src, dst))
        elif src in text:
            hits.append((src, dst))
        if len(hits) >= limit:
            break
    return hits


# ══════════════════════════ Prompt ══════════════════════════

_LANG_NAME = {"en": "英文", "th": "泰文", "unknown": "外文"}


def build_prompt(text: str, lang: str, terms: List[Tuple[str, str]]) -> str:
    all_terms = ALWAYS_TERMS + [t for t in terms if t not in ALWAYS_TERMS]
    term_block = ""
    if all_terms:
        lines = "\n".join(f"  {s} → {d}" for s, d in all_terms)
        term_block = f"\n必须使用以下固定译名（本项目标准术语）：\n{lines}\n"

    return f"""你是榴莲种植领域的专业翻译。把下面的{_LANG_NAME.get(lang, '外文')}资料翻译成简体中文。

⚠️ 最重要：**这份资料讲的是榴莲**（durian / ทุเรียน / Durio zibethinus）。
绝对不要译成芒果、荔枝、龙眼、菠萝蜜或任何其他水果。凡指代该作物之处一律译作"榴莲"。

硬性要求：
1. 只输出译文正文，不要任何解释、前言、标题或"以下是译文"之类的话
2. **所有数字、单位、化学配比、品种编号必须原样保留**，绝不换算、绝不改动
   例如 45.2% 保持 45.2%，8 m x 8 m 保持 8 m × 8 m，NPK 15:15:15 保持原样，D24 保持 D24
3. 拉丁学名（如 Phytophthora palmivora）保留原文，可在后面加中文名
4. 保持原文的段落和列表结构
5. 若原文是不完整的片段，照原样译出片段，不要自行补全或扩写
{term_block}
原文：
{text}"""


# ══════════════════════════ LLM 调用 ══════════════════════════

def call_llm(prompt: str, max_tokens: int = 2048, timeout: int = 240) -> str:
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 0.9,
        # Qwen3 需要显式关闭思考模式，否则输出里会混入 <think> 段
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(LLM_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        data = json.loads(f.read())
    return data["choices"][0]["message"]["content"]


# ── 输出健康度检测（实测发现的三类失败）──
# A) Prompt 泄漏：模型把指令当输出复述。实测两条译文原样吐回了 prompt，
#    连 prompt 里的示例数字 45.2 / 8 / 15:15:15 也一并输出。
_PROMPT_MARKERS = (
    "最重要", "这份资料讲的是榴莲", "硬性要求", "绝对不要译成",
    "只输出译文", "必须使用以下固定译名", "原样保留", "本项目标准术语",
)
# B) 退化循环：实测出现 "3.3" 重复 211 次、"29 次" 重复 148 次
_RE_DEGENERATE = re.compile(r"(.{2,14}?)\1{6,}", re.S)


def output_is_unhealthy(out: str, src: str) -> Optional[str]:
    """检测输出是否不可用。返回问题类型，None 表示健康。"""
    o = str(out or "")
    hits = [m for m in _PROMPT_MARKERS if m in o]
    if len(hits) >= 2:
        return f"prompt_leak: {hits[:3]}"
    # 单个强特征也算泄漏 —— 实测有译文只复述了"严禁误译为芒果荔枝"这一句，
    # 命中数 1 而漏检。这些短语只可能来自 prompt，不可能是原文内容。
    for strong in ("误译为芒果", "绝对不要译成", "这份资料讲的是榴莲",
                   "凡指代该作物", "Durio zibethinus）"):
        if strong in o:
            return f"prompt_leak_strong: {strong}"
    m = _RE_DEGENERATE.search(o)
    if m:
        return f"degenerate_loop: {m.group(1)[:12]!r} 重复"
    # 译文长度异常膨胀（正常中译不会超过原文 2.5 倍字符）
    if len(o) > max(300, len(src) * 2.5):
        return f"length_blowup: {len(o)} vs src {len(src)}"
    return None


_RE_THINK = re.compile(r"<think>.*?</think>", re.S)
_RE_PREAMBLE = re.compile(
    r"^\s*(?:以下是|这是|译文[:：]|翻译[:：]|中文翻译[:：]|Here is|Translation:)[^\n]{0,20}\n+",
    re.I,
)


def clean_output(text: str) -> str:
    s = _RE_THINK.sub(" ", str(text or "")).strip()
    s = _RE_PREAMBLE.sub("", s)
    # 去掉整体包裹的代码块
    s = re.sub(r"^```[a-z]*\n(.*)\n```$", r"\1", s, flags=re.S)
    return s.strip()


# ══════════════════════════ 数值校验（防幻觉硬闸门）══════════════════════════

_RE_NUM = re.compile(r"\d+(?:\.\d+)?")

# 实质数值：小数、千分位数、带单位的数、百分比、3 位以上整数。
# 排除孤立的 1-2 位小整数 —— 泰文原文是 PPT 讲义式的，充满 (1) (2) 23 24
# 这类页码与小节编号，译文合理地省略或合并它们，不应判为数值错误。
# 首次试跑 14/30 报"数值不一致"，逐条核对后确认全部是这类误报。
_RE_SUBSTANTIVE_NUM = re.compile(
    r"\d+\.\d+"                                    # 小数 5.5 / 45.2
    r"|\d{1,3}(?:,\d{3})+"                         # 千分位 1,600
    r"|\d+\s*(?:%|℃|°C)"                           # 百分比、温度
    r"|\d+(?:\.\d+)?\s*(?:kg|g|mg|mm|cm|m|km|ha|L|mL|t|ppm|"
    r"公斤|克|毫米|厘米|米|公里|公顷|亩|吨|升|毫升|天|日|月|年|次|倍|株|棵|个)"
    r"|\d{3,}"                                     # 3 位以上整数
    r"|\d+\s*[:：]\s*\d+(?:\s*[:：]\s*\d+)*"        # 配比 15:15:15
)


def substantive_numbers(text: str) -> Counter:
    """只提取实质数值，忽略页码/序号类小整数。"""
    s = str(text or "")
    found = []
    for m in _RE_SUBSTANTIVE_NUM.finditer(s):
        # 归一化：去空白、去千分位逗号，只保留数字与小数点和冒号
        tok = re.sub(r"[^\d.:]", "", m.group(0))
        if tok:
            found.append(tok)
    return Counter(found)


def number_multiset(text: str) -> Counter:
    """提取文本里所有数字。保留给需要全量比较的场景。"""
    return Counter(_RE_NUM.findall(str(text or "")))


def check_numbers(src: str, dst: str) -> Optional[str]:
    """校验译文的**实质数值**是否与原文一致。返回问题描述，None 表示通过。

    这是防幻觉的硬闸门：翻译最危险的失败模式不是措辞生硬，
    而是把 45.2% 写成 45%、把 5.5-6.5 写成 5-6。这类错误在
    农业指导场景会造成实际损失。
    """
    # 跨语言比较必须用"该数字在对侧是否出现过"，而不是直接比实质数值集合。
    # 原因：实质数值的判定依赖单位词表，而单位词表只覆盖中英文。
    # 泰文 "50 เซนติเมตร" 里的 50 会被当成页码忽略，中文 "50厘米" 却被计入，
    # 于是误报"译文多出 50"。实测 34/37 告警都源于此。
    a_sub, b_sub = substantive_numbers(src), substantive_numbers(dst)
    a_all, b_all = number_multiset(src), number_multiset(dst)

    def _norm(c: Counter) -> set:
        out = set()
        for k in c:
            out.add(k)
            out.add(k.rstrip("0").rstrip("."))   # 20.00 ≡ 20
        return out

    # 复合配比（1:4:100、15:15:15）在 number_multiset 里会被拆成原子数，
    # 因此复合形式永远不会出现在对侧的 all 集合中，必然误报"缺失+多出"。
    # 解决：把两侧的复合形式各自并入自己的 all 集合，并额外登记其原子分解。
    def _augment(all_c: Counter, sub_c: Counter) -> set:
        s = _norm(all_c)
        for k in sub_c:
            s.add(k)
            if ":" in k:
                s.update(k.split(":"))
        return s

    a_all_n, b_all_n = _augment(a_all, a_sub), _augment(b_all, b_sub)
    missing = Counter({k: v for k, v in a_sub.items()
                       if k not in b_all_n and k.rstrip("0").rstrip(".") not in b_all_n})
    added = Counter({k: v for k, v in b_sub.items()
                     if k not in a_all_n and k.rstrip("0").rstrip(".") not in a_all_n})
    # 佛历→公历转换是正确行为（佛历 2546 = 公元 2003），不算错
    for bs in list(missing):
        try:
            ce = str(int(bs) - 543)
        except ValueError:
            continue
        if 1900 <= int(bs) - 543 <= 2100 and ce in added:
            del missing[bs]
            del added[ce]
    parts = []
    if missing:
        parts.append(f"译文缺失 {dict(missing)}")
    if added:
        parts.append(f"译文多出 {dict(added)}")
    return "; ".join(parts) if parts else None


# ── 主体身份校验（防"榴莲译成芒果"类幻觉）──
# 实测 durian-base 对泰文 ทุเรียน 识别不稳，曾把榴莲译成芒果、荔枝。
# 整库主题都是榴莲，译错主体等于污染语料，必须硬拦。
# 只保留"最可能被误认成榴莲主体"的几个。香蕉/木瓜/椰子等在榴莲园语境里
# 常作为防风林或间作作物被合理提及，实测确认是误报，已剔除。
_WRONG_FRUITS = ("芒果", "荔枝", "龙眼", "菠萝蜜", "木菠萝", "山竹", "红毛丹")
_RE_DURIAN_SRC = re.compile(r"durian|ทุเรียน|Durio", re.I)


def check_subject(src: str, dst: str) -> Optional[str]:
    """原文提到榴莲时，译文必须译成榴莲，且不得出现其他水果主体。"""
    if not _RE_DURIAN_SRC.search(str(src or "")):
        return None
    d = str(dst or "")
    wrong = [f for f in _WRONG_FRUITS if f in d]
    if wrong and "榴莲" not in d:
        return f"主体误译为 {wrong}（原文是榴莲）"
    # 榴莲与他果同现多为合理对比/间作描述，不重试，仅在 metadata 留痕
    # "译文无榴莲字样"也不重试 —— 片段式原文（如纯参数列表）本就不含主体词
    return None


# ══════════════════════════ 缓存 ══════════════════════════

_cache_lock = Lock()


def load_cache() -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not CACHE_PATH.exists():
        return cache
    for line in CACHE_PATH.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        k = r.get("key")
        if k:
            cache[k] = r
    return cache


def append_cache(record: Dict[str, Any]) -> None:
    with _cache_lock:
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ══════════════════════════ 单条翻译 ══════════════════════════

def translate_one(
    key: str,
    text: str,
    lang: str,
    glossary: List[Tuple[str, str]],
) -> Dict[str, Any]:
    terms = relevant_terms(text, glossary)
    prompt = build_prompt(text, lang, terms)
    # 中文比英文紧凑，但留足余量；泰文译中文膨胀较多
    max_tok = min(3000, max(400, int(len(text) * 1.4) + 200))

    last_err = ""
    last_out = ""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            raw = call_llm(prompt, max_tokens=max_tok)
        except Exception as exc:
            last_err = f"call_failed: {exc}"
            time.sleep(min(2 ** attempt, 8))
            continue

        out = clean_output(raw)
        last_out = out

        if not out or len(out) < 4:
            last_err = "empty_output"
            continue

        unhealthy = output_is_unhealthy(out, text)
        if unhealthy:
            last_err = unhealthy
            continue
        # 译文里不应还剩大量原语种字符
        if lang == "th" and len(_RE_TH.findall(out)) > len(_RE_TH.findall(text)) * 0.3:
            last_err = "thai_not_translated"
            continue
        if not _RE_ZH.search(out):
            last_err = "no_chinese_in_output"
            continue

        # 主体身份校验优先 —— 译错水果等于污染语料，比数值偏差更严重
        subj_issue = check_subject(text, out)
        if subj_issue and attempt < MAX_RETRY:
            last_err = f"subject_mismatch: {subj_issue}"
            continue

        num_issue = check_numbers(text, out)
        if num_issue and attempt < MAX_RETRY:
            last_err = f"number_mismatch: {num_issue}"
            continue

        return {
            "key": key, "src_lang": lang, "ok": True,
            "translation": out,
            "terms_used": [f"{s}={d}" for s, d in terms],
            "number_issue": num_issue,     # 末次仍不一致则保留，供人工复核
            "subject_issue": subj_issue,   # 同上
            "attempts": attempt,
        }

    return {
        "key": key, "src_lang": lang, "ok": False,
        "translation": last_out or "",
        "error": last_err, "attempts": MAX_RETRY,
    }


# ══════════════════════════ 主流程 ══════════════════════════

def load_chunks() -> List[Dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"缺少 {CHUNKS_PATH}，先运行 python rag_rechunk.py build")
    return [json.loads(l) for l in CHUNKS_PATH.open(encoding="utf-8") if l.strip()]


def plan() -> None:
    chunks = load_chunks()
    todo = [c for c in chunks if needs_translation(c)]
    cache = load_cache()
    done = sum(1 for c in todo if (c.get("metadata") or {}).get("text_sha1") in cache)

    by_lang = Counter(detect_lang(c["index_text"]) for c in todo)
    by_prov = Counter(c.get("provenance") for c in todo)
    chars = sum(len(c["index_text"]) for c in todo)

    print(f"总 chunk           {len(chunks)}")
    print(f"需翻译             {len(todo)}  ({chars:,} 字符)")
    print(f"已在缓存           {done}")
    print(f"待处理             {len(todo) - done}")
    print(f"\n按语种: {dict(by_lang)}")
    print(f"按来源: {dict(by_prov)}")

    skipped_faq = [c for c in chunks
                   if c.get("provenance") == "legacy_faq"
                   and detect_lang(c["index_text"]) in ("en", "th")]
    print(f"\n刻意跳过的 FAQ 多语种条目: {len(skipped_faq)} 条"
          f"（服务对应语种用户，译成中文会破坏产品设计）")


def run(limit: Optional[int] = None) -> None:
    chunks = load_chunks()
    glossary = load_glossary()
    print(f"[glossary] 载入 {len(glossary)} 条术语")

    cache = load_cache()
    todo = []
    for c in chunks:
        if not needs_translation(c):
            continue
        key = (c.get("metadata") or {}).get("text_sha1")
        if not key or key in cache:
            continue
        todo.append((key, c["index_text"], detect_lang(c["index_text"])))

    # 同一 text_sha1 只译一次
    seen = set()
    uniq = []
    for k, t, lg in todo:
        if k not in seen:
            seen.add(k)
            uniq.append((k, t, lg))
    todo = uniq

    if limit:
        todo = todo[:limit]

    if not todo:
        print("没有待翻译内容（可能已全部命中缓存）")
        return

    print(f"[run] 待翻译 {len(todo)} 条，并发 {MAX_WORKERS}")
    t0 = time.time()
    stats = Counter()
    n_done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(translate_one, k, t, lg, glossary): k
                for k, t, lg in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as exc:
                stats["crash"] += 1
                print(f"  [ERR] {exc}")
                continue
            append_cache(rec)
            stats["ok" if rec["ok"] else "failed"] += 1
            if rec.get("number_issue"):
                stats["number_issue"] += 1
            if rec.get("subject_issue"):
                stats["subject_issue"] += 1
            n_done += 1
            if n_done % 25 == 0 or n_done == len(todo):
                el = time.time() - t0
                rate = n_done / max(el, 0.01)
                eta = (len(todo) - n_done) / max(rate, 0.001)
                print(f"  {n_done}/{len(todo)}  "
                      f"{rate:.1f} 条/秒  已用 {el/60:.1f}min  剩余 ~{eta/60:.1f}min  "
                      f"{dict(stats)}")

    print(f"\n[run] 完成，用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"[run] 统计: {dict(stats)}")
    print(f"[run] 缓存: {CACHE_PATH}")


def apply_translations() -> None:
    """把缓存里的译文合并回 chunks，产出 chunks_translated.jsonl。"""
    chunks = load_chunks()
    cache = load_cache()
    stats = Counter()
    out: List[Dict[str, Any]] = []

    for c in chunks:
        meta = dict(c.get("metadata") or {})
        key = meta.get("text_sha1")

        if not needs_translation(c) or not key or key not in cache:
            if needs_translation(c):
                stats["missing_translation"] += 1
                meta["translation_missing"] = True
                c = {**c, "metadata": meta}
            out.append(c)
            stats["passthrough"] += 1
            continue

        rec = cache[key]
        if not rec.get("ok") or not rec.get("translation"):
            # 翻译失败：保留原文，标记出来，不阻塞流程
            meta["translation_failed"] = True
            meta["translation_error"] = rec.get("error", "")
            out.append({**c, "metadata": meta})
            stats["failed_keep_original"] += 1
            continue

        original = c["index_text"]
        translated = rec["translation"]

        # 原文永不覆盖 —— 泰文源 docx 已丢失，现存 chunk 是仅存副本
        meta["original_text"] = original
        meta["src_lang"] = rec.get("src_lang")
        meta["translated"] = True
        meta["translated_by"] = LLM_MODEL
        if rec.get("terms_used"):
            meta["terms_used"] = rec["terms_used"]
        if rec.get("number_issue"):
            meta["number_issue"] = rec["number_issue"]
            stats["number_issue"] += 1
        if rec.get("subject_issue"):
            meta["subject_issue"] = rec["subject_issue"]
            stats["subject_issue"] += 1
        meta.pop("needs_translation", None)

        out.append({
            **c,
            "index_text": translated,       # 中文，供向量化
            "display_text": translated,     # 展示也用中文；原文在 metadata
            "text": translated,
            "metadata": meta,
        })
        stats["translated"] += 1

    OUT_PATH.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out),
        encoding="utf-8",
    )
    print(f"[apply] {dict(stats)}")
    print(f"[apply] 已写入 {OUT_PATH}（{len(out)} chunk）")
    print(f"[apply] 下一步：把 rag_llamaindex.py 的 RECHUNK_FILE 指向该文件，或直接覆盖 chunks.jsonl")


def verify(n: int = 12) -> None:
    """抽检译文质量。"""
    import random
    cache = load_cache()
    oks = [r for r in cache.values() if r.get("ok")]
    bad = [r for r in cache.values() if not r.get("ok")]
    numiss = [r for r in cache.values() if r.get("number_issue")]

    print(f"缓存 {len(cache)} 条：成功 {len(oks)}，失败 {len(bad)}，数值不一致 {len(numiss)}")
    if bad:
        print(f"\n失败原因分布: {dict(Counter(r.get('error','?').split(':')[0] for r in bad))}")

    chunks = {(_c.get('metadata') or {}).get('text_sha1'): _c for _c in load_chunks()}
    random.seed(7)
    print(f"\n=== 随机抽检 {n} 条 ===")
    for r in random.sample(oks, min(n, len(oks))):
        src = (chunks.get(r["key"]) or {}).get("index_text", "")
        print(f"\n─── [{r['src_lang']}] 术语: {r.get('terms_used') or '无'}")
        print(f"  原文: {src[:150]}")
        print(f"  译文: {r['translation'][:150]}")

    if numiss:
        print(f"\n=== 数值不一致的 {len(numiss)} 条（需人工复核）===")
        for r in numiss[:5]:
            src = (chunks.get(r["key"]) or {}).get("index_text", "")
            print(f"\n  ⚠️ {r['number_issue']}")
            print(f"     原文: {src[:120]}")
            print(f"     译文: {r['translation'][:120]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="离线翻译管道")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="统计工作量")
    r = sub.add_parser("run", help="执行翻译")
    r.add_argument("--limit", type=int, default=None)
    sub.add_parser("apply", help="合并译文回 chunks")
    v = sub.add_parser("verify", help="抽检质量")
    v.add_argument("-n", type=int, default=12)
    args = ap.parse_args()

    if args.cmd == "plan":
        plan()
    elif args.cmd == "run":
        run(limit=args.limit)
    elif args.cmd == "apply":
        apply_translations()
    elif args.cmd == "verify":
        verify(n=args.n)


if __name__ == "__main__":
    main()
