from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "rag_pdfs"
STORAGE_DIR = BASE_DIR / "rag_llamaindex_storage"

EXTRA_CHUNK_FILES = [
    BASE_DIR / "rag_auto" / "chunks_cache.json",
    BASE_DIR / "rag_auto" / "auto_chunks.jsonl",
    BASE_DIR / "clean_chunks.jsonl",
    BASE_DIR / "chunks.jsonl",
]

# 重新分片产物（rag_rechunk.py 生成）。存在时**取代** EXTRA_CHUNK_FILES 与 PDF 目录，
# 因为它已经把 legacy chunk、PDF 解析、表格提取全部合并处理过了，
# 再叠加旧来源会重复入库。
RECHUNK_FILE = BASE_DIR / "rag_build" / "chunks.jsonl"

# 先用 CPU，避免和 vLLM 抢 5090 显存。
# 支持环境变量覆盖（本地部署时指向 models/bge-small-zh-v1.5）
EMBED_MODEL_NAME = os.getenv(
    "DURIAN_LI_EMBED_MODEL_PATH",
    str(BASE_DIR / "models" / "bge-small-zh-v1.5"),
)

# LlamaIndex 检索分数一般是相似度分数，不同模型范围会有差异。
# 先保守一点，太低就不返回 evidence。
DEFAULT_MIN_SCORE = 0.42

_INDEX_CACHE = None
_LEXICAL_NODE_CACHE: List[Dict[str, Any]] | None = None
_NEIGHBOR_CACHE: Dict[tuple, List[Dict[str, Any]]] | None = None



GENERIC_STOPWORDS = {
    "榴莲", "榴莲树", "请问", "哪些", "什么", "怎么", "如何", "是否", "有没有",
    "介绍", "说明", "一下", "相关", "资料", "内容", "的是", "有哪", "可以",
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "about", "durian",
}


INTENT_RULES = {
    "code": {
        "triggers": ["编号", "代号", "代码", "品种号", "品种编号"],
        "required": [
            "编号", "代号", "代码", "品种号", "品种编号",
            "D24", "D197", "D200", "D101",
            "猫山王", "苏丹王", "黑刺", "金枕", "青尼", "干尧", "托曼尼",
        ],
    },
    "disease": {
        "triggers": ["病害", "病虫害", "炭疽病", "疫病", "根腐病", "叶斑", "防治", "症状", "病原"],
        "required": [
            "病害", "病虫害", "炭疽病", "疫病", "根腐病", "叶斑",
            "防治", "症状", "病原", "病斑", "菌", "药剂",
            "Colletotrichum", "Phytophthora",
        ],
    },
    "variety": {
        "triggers": ["品种", "品系", "种类", "猫山王", "金枕", "黑刺", "苏丹王"],
        "required": [
            "品种", "品系", "种类", "猫山王", "金枕", "黑刺", "苏丹王",
            "D24", "D197", "D200",
        ],
    },
    "planting": {
        "triggers": ["种植", "栽培", "定植", "施肥", "修剪", "灌溉", "管理"],
        "required": ["种植", "栽培", "定植", "施肥", "修剪", "灌溉", "管理", "土壤", "温度", "水分"],
    },
}


def setup_llamaindex() -> None:
    """Configure local embedding model."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device="cpu",
    )

    # 这里只用 LlamaIndex 做检索，不让它调用默认 LLM。
    Settings.llm = None


def normalize_ocr_units(text: str) -> str:
    """Normalize common MinerU/LaTeX unit artifacts before retrieval."""
    value = str(text or "")
    value = re.sub(
        r"(\d+(?:\.\d+)?)\s*\\+mathrm\{mm\}",
        r"\1 mm",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"(\d+(?:\.\d+)?)\s*\\+mathrm\{cm\}",
        r"\1 cm",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"(\d+(?:\.\d+)?)\s*\\+sim\s*(\d+(?:\.\d+)?)"
        r"\s*(?:\^\{\\+circ\})?\s*\\+mathrm\{C\}",
        r"\1–\2℃",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"(\d+(?:\.\d+)?)\s*(?:\^\{\\+circ\})?\s*\\+mathrm\{C\}",
        r"\1℃",
        value,
        flags=re.I,
    )
    value = value.replace("\\geqslant", "≥").replace("\\leqslant", "≤")

    # MinerU occasionally emits temperature degree symbols as ``\%``.
    # Convert only when the surrounding text clearly describes temperature;
    # keep real humidity/percentage values unchanged.
    bad_degree = re.compile(r"(\d+(?:\.\d+)?)\s*\\+%")

    def replace_bad_degree(match: re.Match) -> str:
        start, end = match.span()
        context = value[max(0, start - 28) : min(len(value), end + 16)]
        temperature_markers = (
            "温度",
            "低温",
            "高温",
            "积温",
            "摄氏",
            "temperature",
        )
        if any(marker in context.lower() for marker in temperature_markers):
            return f"{match.group(1)}℃"
        return match.group(0)

    return bad_degree.sub(replace_bad_degree, value)


def clean_text(text: str) -> str:
    text = str(text or "")

    # 清理常见 OCR / parser 泄漏
    text = re.sub(r"\{?\s*['\"]bbox['\"]\s*:\s*\[[^\]]*\]\s*,?", " ", text)
    text = re.sub(r"['\"](?:page_idx|page_size|image_path|angle|index|type)['\"]\s*:\s*[^,}\]]+", " ", text)
    text = re.sub(r"['\"](?:content|text|type)['\"]\s*:\s*", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = normalize_ocr_units(text)

    # 统一空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── legacy 泰文声调符号修复 ───────────────────────────────────────────────
# 那批 legacy 泰文 .docx 是用非 Unicode 泰文字体排版的，抽取时按字形位置错映射。
# 实测存在**两套互不相容的损坏方案**，逐文档分布，不能用同一组替换处理：
#
#   方案 C（คู่มือ...ชุมพร.docx，251 chunk）：
#     U+0E4D NIKHAHIT       → ่ MAI EK   （2653 处）
#     U+0E4E YAMAKKAN       → ้ MAI THO  （2525 处）
#     U+02DA RING ABOVE + า → ำ SARA AM  （1247 处，U+02DA 是非泰文字符）
#   方案 AB（การผลิตทุเรียน.docx / ...ภาคใต้ตอนล่างn.docx，400 chunk）：
#     U+0E4D + า            → ำ SARA AM  （413 处，即预组合 ำ 被分解）
#     无 U+0E4E、无 U+02DA
#
# 两套方案对 `ํา` 的判读完全相反（C 判 `่า`、AB 判 `ำ`），这正是把两套混在一起
# 统计时会看到「233 支持 ำ / 96 支持 ่า」那种自相矛盾结果的原因。
# 用干净参照语料（泰文 FAQ index_text + 泰文金标，31.7 万字符）逐处裁决，
# **分方案后各自零反例**：AB 的 `ํา` 231:0 判 ำ，C 的 `ํา` 1:91 判 `่า`、
# C 的 `๎า` 38:0 判 `้า`。
#
# 判据（**这里必须小心**）：早先只用「文本内出现 U+0E4E 或 U+02DA/U+02D9」判方案 C。
# 那个判据在整条 chunk（500 字符）上 0 例误判，但它是**长程依赖** —— 判据字符可能
# 离待判的 `ํา` 很远，于是同一个子串换个切分窗口就判出不同结果。实测：
#   '...ของโรครากเนําโคนเนําในต'   → เน่า（窗口内有 ๎ 和 ˚，判 C，对）
#   '...ทนทานตํอโรครากเนําโคนเนํา' → เนำ （窗口内无判据字符，判 AB，错）
# 这类失配是静默的。补一条**局部**证据把长程依赖压下去：方案 AB 里 U+0E4D 几乎
# 只以 `ํา` 形态出现（343/348），而方案 C 里 81% 的 U+0E4D 后面不是 า。所以
# 「出现裸 ํ」是方案 C 的正面证据。但 AB 也有 7 个裸 ํ，无条件采信会在整条 chunk
# 上引入 2-3% 误判、进而让索引失同步，所以再加一个比例闸：裸 ํ 的个数要不少于
# 干净声调符号（่ ้）的个数 —— AB 里 `่`/`้` 用量正常（35-41‰）而裸 ํ 只有 7 个，
# 比值极小；C 里两者数量相当。实测误判率（含 `ํา` 的窗口，步长 7 全扫）：
#   窗口 500/250（建索引用的粒度）：C 0.0% / AB 0.0%   ← 与旧判据相同，索引不失同步
#   窗口 60（查询粒度）：C 17.5%→10.7%，AB 0.0%→0.0%
#   窗口 36：C 35.0%→23.5%，AB 0.0%→0.2%
# 注：`ํา` 的歧义**无法完全局部化**。实测同一词形 `ทํา` 在 AB 是 ทำ、在 C 是 ท่า
# （`อ˚าเภอทําแซะ` = 春蓬府 ท่าแซะ 县），只看前 1 个字符时 919 处里有 392 处歧义
# （所以「前置辅音表」这个思路是不成立的）。这里只能把依赖范围从「整段任意位置」
# 缩到「同一小窗口内的统计比例」，做不到 0。
_THAI_STRONG_C = re.compile(
    r"\u0e4e"
    r"|[\u02da\u02d9]\u0e32"
    r"|(?<=[\u0e00-\u0e7f])[\u02da\u02d9]"
    r"|[\u02da\u02d9](?=[\u0e00-\u0e7f])"
)
_THAI_BARE_NIKHAHIT = re.compile(r"\u0e4d(?!\u0e32)")
_THAI_CLEAN_TONE = re.compile(r"[\u0e48\u0e49]")
# 组合符号顺序颠倒：上元音应在声调符号之前。legacy 语料里颠倒 141 处
# （`้ื`81 `่ิ`36 `่ื`18 `้ึ`2 `้ี`2 `้ิ`1 `่ี`1），最典型的是
# เช้ือรา → เชื้อรา（真菌，56 处）。干净 FAQ 语料里这 7 种颠倒形态**一个都没有**，
# 而正序形态有数百上千处（`ี่` 1255、`ื้` 815、`ื่` 727…），零反例。
# 与上面几条规则无交互：实测 ํ/๎ 后面从不跟上元音（各 0 处），所以顺序无关。
_THAI_SUSPECT = re.compile(
    r"[\u0e4d\u0e4e\u02da\u02d9]|[\u0e48-\u0e4b][\u0e34-\u0e37]|([\u0e48-\u0e4b])\1"
)
# 双写声调符号 + า（两个方向）：源文档把 ำ 的分解叠写进一步错成了「声调翻倍」。
#   น้้า / ให้น้้า / ซา้้ / นา้้  →  น้ำ / ให้น้ำ / ซ้ำ / น้ำ
# 泰文一个音节最多一个声调符号，相邻双声调（无论相同与否）在干净参照语料
# （31.7 万字符）里出现 0 次，在 legacy 里只以这 15 处「相同双写」形态出现
# （相邻但不同的双声调也是 0 处），所以按「相邻相同 + 紧邻 า」收敛成 声调+ำ。
# 两个方向都要：`้้า`（声调在前）x12、`า้้`（声调在后）x3。
# 用独立 sub 而不是并入 _THAI_SCAN：正则反向引用放进命名分组会破坏 lastgroup
# 语义；且这两个模式与主扫描的任何规则都无交集（主扫描不消费孤立的声调符号
# 和 า），先后顺序不影响结果。
_THAI_DOUBLE_TONE_AM = re.compile(r"([\u0e48-\u0e4b])\1(\u0e32)|(\u0e32)([\u0e48-\u0e4b])\4")
# U+02DA/U+02D9 在非泰文里是正常字符（"25˚C" 的度数符号），只有和泰文同处一段
# 才是损坏证据。不加这道闸的话 "25˚C" 会被改成 "25ำC" —— 当前语料和金标里
# 各 0 处，但查询侧同样走这个函数，用户打一个度数符号就会被改坏。
_THAI_CHAR = re.compile(r"[\u0e00-\u0e7f]")
# U+02DA / U+02D9（RING ABOVE / DOT ABOVE）是 SARA AM 的上半部，后面的 า 是下半部。
# 三个分支按可靠性排序，缺一不可：
#   1. 后跟 า —— 最可靠，度数符号不可能后跟 SARA AA。必须放第一位，
#      这样 น้˚า 会连 า 一起吃掉替换成 ำ。
#   2/3. 裸符号但紧邻泰文 —— 语料里有 3 处 า 在换行时丢了、5 处 ˚ 与前面的
#      辅音被排版空格隔开（เพือก่ ˚าจัดเ = เพื่อกำจัด），这两支兜住它们。
# 只用「同段落有泰文」当闸是不够的：泰文查询里写 "อุณหภูมิ 25˚C" 会被
# 改成 "25ำC"。加上邻接条件后 1264+5=1269 处语料判读一个不少，
# 而 "25˚C" 无论有没有泰文陪同都不再被误改。
_THAI_SARA_AM = re.compile(
    r"[\u02da\u02d9]\u0e32"
    r"|(?<=[\u0e00-\u0e7f])[\u02da\u02d9]"
    r"|[\u02da\u02d9](?=[\u0e00-\u0e7f])"
)
# 方案 C 的判据：U+0E4E，或出现「确实是 SARA AM 的 ˚/˙」。
# 不能直接用 [\u0e4e\u02da\u02d9]，否则一个度数符号就会把方案判成 C。
_THAI_SCHEME_C = re.compile(r"\u0e4e")

# 通用判据判不对的个别词，用显式词表兜住（先于方案规则应用）。
# ไฟทอปธอรํา：Phytophthora（疫霉）。方案 C 会把它判成 ไฟทอปธอร่า，
# 但正确写法是 ไฟทอปธอรา —— 这里的 ํ 是源文档凭空多出来的，应删除而非替换。
# 语料里 17 处全在方案 C 那份文档，而方案 AB 文档里写的正是正确的 ไฟทอปธอรา。
_THAI_FIX_WORDS = {
    "\u0e44\u0e1f\u0e17\u0e2d\u0e1b\u0e18\u0e2d\u0e23\u0e4d\u0e32":
        "\u0e44\u0e1f\u0e17\u0e2d\u0e1b\u0e18\u0e2d\u0e23\u0e32",
    # 源文档里还有 1 处把 ป 打成  บ 的 ไฟทอบธอรํา。不兜住的话它会走方案 C 变成
    # ไฟทอบธอร่า，凭空多出第四种拼法，一并归到同一个写法。
    "\u0e44\u0e1f\u0e17\u0e2d\u0e1a\u0e18\u0e2d\u0e23\u0e4d\u0e32":
        "\u0e44\u0e1f\u0e17\u0e2d\u0e1b\u0e18\u0e2d\u0e23\u0e32",
}

# 单次扫描：所有规则合成一条正则，一次遍历、最长/最高优先分支胜出、
# 匹配后从消费点之后继续（re.sub 语义），不会像链式 str.replace 那样
# 让前一条规则的产物影响后一条规则的匹配位置。
# 分支顺序即优先级：词表 > 顺序颠倒 > SARA AM > `ํา` > 裸 ํ > ๎。
_THAI_SCAN = re.compile(
    "(?P<word>"
    + "|".join(
        re.escape(k)
        for k in sorted(_THAI_FIX_WORDS, key=len, reverse=True)
    )
    + ")"
    r"|(?P<swap>[\u0e48-\u0e4b][\u0e34-\u0e37])"
    # 叠写 SARA AM + 声调：น + ํ + ้ + า。声调夹在 nikhahit 和 า 中间。
    # 正确结果是 声调 + ำ（น้ำ）。不加这条的话裸 ํ 会先被映射成 ่，
    # 产出 น่้า 这种双声调符号的废词。语料 7 处（นํ้า x5、ตํ่า x2，全在方案 AB）。
    # 注意与 101 处 ้ํา（声调在前）区分：那种形态 ํ 直接跟 า，走 sara_am 分支已正确。
    r"|(?P<stacked_am>\u0e4d[\u0e48-\u0e4b]\u0e32)"
    r"|(?P<am>[\u02da\u02d9]\u0e32"
    r"|(?<=[\u0e00-\u0e7f])[\u02da\u02d9]"
    r"|[\u02da\u02d9](?=[\u0e00-\u0e7f]))"
    r"|(?P<sara_am_or_ek>\u0e4d\u0e32)"
    r"|(?P<ek>\u0e4d)"
    r"|(?P<tho>\u0e4e)"
)


def _thai_scheme_is_c(text: str) -> bool:
    """判断文本属于哪套损坏方案。见上方 _THAI_STRONG_C 处的完整论证。"""
    if _THAI_STRONG_C.search(text):
        return True
    n_bare = len(_THAI_BARE_NIKHAHIT.findall(text))
    if not n_bare:
        return False
    return n_bare >= max(len(_THAI_CLEAN_TONE.findall(text)), 1)


def normalize_thai(text: str) -> str:
    """修 legacy 泰文 .docx 的声调符号错映射。见上方两套损坏方案的说明。

    幂等：处理后文本里不再有 U+0E4D/0E4E/02DA/02D9、也不再有颠倒的
    「声调符号+上元音」，二次调用在 _THAI_SUSPECT 处直接短路返回。
    对干净泰文、中文、英文、马来文完全无影响。

    **只对 BM25 的字符级 n-gram 匹配有意义，对 dense 向量无影响**：泰文字符
    在 bge-small-zh 的 tokenizer 下无论对错都塌缩成 [UNK]，实测规范化前后
    token 数完全相同。不要指望它改善向量质量。

    **`ํา` 的判读依赖输入窗口**（方案判据是统计性的，见上）。索引侧总是拿整条
    chunk 调用、实测 0 例误判；查询侧的干净泰文根本不触发本函数。但如果把
    语料切成几十字符的碎片再调用，`ํา` 有可能判反 —— 这是已知且无法完全消除
    的限制，不要在碎片上调用本函数。
    """
    if not isinstance(text, str) or not text:
        return text
    if not _THAI_SUSPECT.search(text) or not _THAI_CHAR.search(text):
        return text

    # 双写声调 + า → 声调 + ำ（先做；见 _THAI_DOUBLE_TONE_AM 处的顺序论证）
    if _THAI_SUSPECT.search(text):
        text = _THAI_DOUBLE_TONE_AM.sub(
            lambda m: (m.group(1) or m.group(4)) + "\u0e33", text
        )

    is_c = _thai_scheme_is_c(text)

    def replace(match: re.Match) -> str:
        kind = match.lastgroup
        if kind == "word":
            return _THAI_FIX_WORDS[match.group()]
        if kind == "swap":
            return match.group()[1] + match.group()[0]
        if kind == "stacked_am":
            return match.group()[1] + "\u0e33"   # ํ้า -> ้ำ（声调+ำ）
        if kind == "am":
            return "\u0e33"
        if kind == "sara_am_or_ek":
            # 唯一一条随方案分叉的规则
            return "\u0e48\u0e32" if is_c else "\u0e33"
        if kind == "ek":
            return "\u0e48"
        return "\u0e49"

    return _THAI_SCAN.sub(replace, text)


FILL_BLANK_RE = re.compile(
    r"(?:_{2,}|＿{2,}|（\s*）|\(\s*\)|\[\s*\]|【\s*】|<\s*空\s*>)"
)


def is_exact_answer_query(query: str) -> bool:
    value = str(query or "")
    return bool(
        FILL_BLANK_RE.search(value)
        or ("填空" in value and any(mark in value for mark in ("___", "（", "(")))
    )


def is_noise_text(text: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    lower = compact.lower()

    if len(compact) < 20:
        return True

    parser_markers = [
        "'bbox'", '"bbox"', "bbox:",
        "'spans'", '"spans"',
        "'lines'", '"lines"',
        "'para_blocks'", '"para_blocks"',
        "'discarded_blocks'", '"discarded_blocks"',
        "'page_idx'", '"page_idx"',
        "'image_path'", '"image_path"',
        "'content'", '"content"',
    ]
    if sum(1 for m in parser_markers if m in lower) >= 3:
        return True

    # 版权页/前言特征词。原实现只要命中一个就整块丢弃，会误杀正文里
    # 正常讨论"出版""主编"的段落 —— 实测这是索引丢失 22.8% 内容的原因之一。
    # 改为要求同时满足：命中 ≥2 个特征词，且文本较短（真正的版权页都很短）。
    front_matter = [
        "本书由", "主编", "副研究员", "助理研究员", "负责图书框架",
        "主要负责", "资料整理", "责任编辑", "出版",
    ]
    hits = sum(1 for m in front_matter if m in compact)
    if hits >= 2 and len(compact) < 220:
        return True

    return False


def load_pdf_documents() -> List[Document]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        try:
            pdf = fitz.open(str(pdf_path))
        except Exception as exc:
            print(f"[WARN] cannot open PDF: {pdf_path.name}: {exc}")
            continue

        for page_idx, page in enumerate(pdf, start=1):
            try:
                text = page.get_text("text")
            except Exception as exc:
                print(f"[WARN] cannot read page: {pdf_path.name} p{page_idx}: {exc}")
                continue

            text = clean_text(text)

            if is_noise_text(text):
                continue

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source_file": pdf_path.name,
                        "doc": pdf_path.name,
                        "page": page_idx,
                    },
                )
            )

        pdf.close()

    return docs



def load_extra_chunk_documents() -> List[Document]:
    """Load existing RAG JSON / JSONL chunks into LlamaIndex."""
    docs: List[Document] = []

    def iter_objects(path: Path):
        suffix = path.suffix.lower()

        if suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[WARN] cannot read JSON chunks: {path}: {exc}")
                return

            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
                return

            if isinstance(data, dict):
                for key in ["chunks", "data", "items", "documents"]:
                    val = data.get(key)
                    if isinstance(val, list):
                        for obj in val:
                            if isinstance(obj, dict):
                                yield obj
                        return
            return

        # JSONL
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except Exception as exc:
            print(f"[WARN] cannot read JSONL chunks: {path}: {exc}")
            return

    for path in EXTRA_CHUNK_FILES:
        if not path.exists():
            continue

        print(f"[INFO] loading extra chunks: {path}")
        before = len(docs)

        for obj in iter_objects(path):
            text = (
                obj.get("display_text")
                or obj.get("text")
                or obj.get("index_text")
                or obj.get("original_text")
                or obj.get("content")
                or obj.get("evidence")
                or ""
            )

            text = clean_text(text)

            if is_noise_text(text):
                continue

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source_file": obj.get("source_file") or path.name,
                        "doc": obj.get("doc") or obj.get("source_file") or path.name,
                        "page": obj.get("page") or obj.get("page_idx"),
                        "item_index": obj.get("item_index"),
                        "chunk_index": obj.get("chunk_index"),
                        "text_sha1": obj.get("text_sha1"),
                        "source_path": str(path),
                    },
                )
            )

        print(f"[INFO] loaded from {path.name}: {len(docs) - before}")

    return docs



def load_rechunk_nodes() -> List[Any]:
    """加载 rag_build/chunks.jsonl 并直接构造 TextNode。

    关键设计：**表格与图片类 node 绕过 SentenceSplitter**。
    原 rebuild_index() 走 from_documents(transformations=[splitter])，会把所有
    Document 无差别过一遍 splitter，chunk_size=512 必然把表格从中间切断，
    导致表头与数据行分家 —— 这是表格检索失效的根本原因。

    另外这里区分 index_text（向量化）与 display_text（展示/喂 LLM），
    不再像旧版那样把两者赋成同一个值。
    """
    from llama_index.core.schema import TextNode

    if not RECHUNK_FILE.exists():
        return []

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)
    nodes: List[Any] = []
    n_split = 0
    n_whole = 0
    n_bilingual_th = 0

    for line in RECHUNK_FILE.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
        except Exception:
            continue

        index_text = str(c.get("index_text") or c.get("text") or "").strip()
        if not index_text:
            continue

        block_type = str(c.get("block_type") or "text")
        raw_meta = dict(c.get("metadata") or {})
        display_text = str(c.get("display_text") or index_text)

        # ── legacy 泰文正文：index_text 拼上泰文原文，让同一 node 双通路可达 ──
        # 这 651 条此前只有中文译文进索引，泰文原文躺在 metadata.original_text，
        # 而 metadata 不参与 BM25（_build_lexical_node_cache 调的 node.get_content()
        # 只返回 text），所以泰文查询在词法上完全触达不到这批知识库正文 ——
        # 这是「泰文 Hit@3 94.2% 但知识库覆盖率只有 7.4%」的结构性成因。
        # 拼接后 dense 走中文译文、BM25 走泰文 n-gram，命中同一个 node。
        #
        # 三个已实测的约束：
        #   1. 必须绕过 splitter。chunk_size=512 会把拼接结果切成"纯中文块 +
        #      纯泰文块"两条，等于原地踏步。所以下面改道进整块分支。
        #   2. 绕过不会截断 embedding。651 条拼接后 bge token 数 median 207 /
        #      max 343，0 条超过 512（500 个泰文字符只压成约 60 个 token，
        #      连续泰文塌缩成单个 [UNK]）。
        #   3. 中文放前面。即使将来有截断也先保住中文，且 embedding 由中文主导。
        # display_text 保持纯中文译文 —— 展示层与喂 LLM 的内容一字不变。
        bilingual_th = (
            block_type == "text"
            and raw_meta.get("src_lang") == "th"
            and bool(raw_meta.get("original_text"))
        )
        if bilingual_th:
            display_text = index_text
            index_text = (
                index_text
                + "\n"
                + normalize_thai(str(raw_meta["original_text"])).strip()
            )

        base_meta = {
            "doc": c.get("doc"),
            "source_file": c.get("source_file") or c.get("doc"),
            "page": c.get("page"),
            "block_type": block_type,
            "provenance": c.get("provenance"),
        }

        if block_type in ("table", "table_broken", "image", "qa_pair") or bilingual_th:
            # 整块成 node，不切分。长字段（display_text / table_markdown /
            # original_text）只挂在这里是安全的，因为不走 splitter。
            meta = dict(base_meta)
            meta["display_text"] = display_text
            for k in ("table_markdown", "caption", "damaged", "image_path",
                      "original_text", "src_lang", "category", "question",
                      "needs_translation", "text_sha1"):
                if raw_meta.get(k) is not None:
                    meta[k] = raw_meta[k]
            excluded = [k for k in meta if k not in ("doc", "source_file", "page")]
            nodes.append(TextNode(
                text=index_text, metadata=meta,
                excluded_embed_metadata_keys=excluded,
                excluded_llm_metadata_keys=excluded,
            ))
            n_whole += 1
            if bilingual_th:
                n_bilingual_th += 1
        else:
            # 正文过 splitter。**metadata 必须保持短小** ——
            # SentenceSplitter 会把 metadata 序列化长度算进 chunk_size，
            # 把 display_text（可达千字）放进来会直接报
            # "Metadata length is longer than chunk size"。
            # 正文的 display_text 与 index_text 相同，无需单独存。
            meta = dict(base_meta)
            for k in ("src_lang", "needs_translation"):
                if raw_meta.get(k) is not None:
                    meta[k] = raw_meta[k]
            excluded = [k for k in meta if k not in ("doc", "source_file", "page")]
            doc = Document(text=index_text, metadata=meta)
            doc.excluded_embed_metadata_keys = excluded
            doc.excluded_llm_metadata_keys = excluded
            for nd in splitter.get_nodes_from_documents([doc]):
                nodes.append(nd)
                n_split += 1

    print(f"[INFO] 重新分片产物: 整块 node {n_whole}（表格/图片/QA + legacy泰文双语，"
          f"未切分），其中 legacy 泰文双语拼接 node {n_bilingual_th}，"
          f"正文切分后 node {n_split}")
    return nodes


def rebuild_index() -> None:
    setup_llamaindex()

    # 优先走重新分片产物
    nodes = load_rechunk_nodes()
    if nodes:
        print(f"[INFO] 使用 {RECHUNK_FILE}，共 {len(nodes)} node")
        index = VectorStoreIndex(nodes=nodes, show_progress=True)
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(STORAGE_DIR))
        print(f"[INFO] LlamaIndex RAG index persisted to: {STORAGE_DIR}")
        return

    # 回退：旧路径（rag_build/chunks.jsonl 不存在时）
    print(f"[WARN] 未找到 {RECHUNK_FILE}，回退到旧的 PDF + EXTRA_CHUNK_FILES 路径")
    pdf_docs = load_pdf_documents()
    extra_docs = load_extra_chunk_documents()
    docs = pdf_docs + extra_docs

    if not docs:
        raise RuntimeError(f"No valid documents found in {PDF_DIR} or EXTRA_CHUNK_FILES")

    print(f"[INFO] loaded PDF pages: {len(pdf_docs)}")
    print(f"[INFO] loaded extra chunks: {len(extra_docs)}")
    print(f"[INFO] loaded total documents: {len(docs)}")

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=80,
    )

    index = VectorStoreIndex.from_documents(
        docs,
        transformations=[splitter],
        show_progress=True,
    )

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))

    print(f"[INFO] LlamaIndex RAG index persisted to: {STORAGE_DIR}")


def load_index() -> VectorStoreIndex:
    """Load LlamaIndex index once and reuse it in backend."""
    global _INDEX_CACHE

    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    setup_llamaindex()

    if not STORAGE_DIR.exists():
        raise RuntimeError(f"Index storage not found: {STORAGE_DIR}. Run rebuild first.")

    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    _INDEX_CACHE = load_index_from_storage(storage_context)
    return _INDEX_CACHE


def extract_query_terms(query: str) -> List[str]:
    q = str(query or "").strip()
    if not q:
        return []

    terms = re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_-]{1,}|D\s*-?\s*\d{1,4}", q)

    result = []
    seen = set()

    for term in terms:
        t = re.sub(r"\s+", "", term).strip()
        if not t:
            continue
        if t in GENERIC_STOPWORDS or t.lower() in GENERIC_STOPWORDS:
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            result.append(t)

    return result


def detect_intents(query: str) -> List[str]:
    q = str(query or "")
    intents = []

    for name, rule in INTENT_RULES.items():
        if any(t in q for t in rule["triggers"]):
            intents.append(name)

    return intents


def evidence_matches_query(query: str, text: str, score: float, min_score: float = DEFAULT_MIN_SCORE) -> bool:
    text = str(text or "")
    query = str(query or "")

    if not text or is_noise_text(text):
        return False

    # 对精确原文/长句查询：只要相似度够高，直接保留。
    # 例如：“叶片病斑多从叶尖或叶缘开始，个别从叶内发生”
    if len(query) >= 18 and score >= 0.55:
        return True

    intents = detect_intents(query)
    terms = extract_query_terms(query)

    # 编号/代号类问题必须严格，避免“3～50朵”这种数字误命中。
    if "code" in intents:
        required = INTENT_RULES["code"]["required"]
        return score >= 0.55 and any(word in text for word in required)

    # 病害类问题：只要证据里有病害/症状/防治相关词，并且分数不是太低，就保留。
    if "disease" in intents:
        disease_required = INTENT_RULES["disease"]["required"]
        if score >= 0.45 and any(word in text for word in disease_required):
            return True
        return False

    # 品种类问题：要求有品种相关词。
    if "variety" in intents:
        variety_required = INTENT_RULES["variety"]["required"]
        if score >= 0.45 and any(word in text for word in variety_required):
            return True
        return False

    # 种植/栽培类问题：要求有管理相关词。
    if "planting" in intents:
        planting_required = INTENT_RULES["planting"]["required"]
        if score >= 0.45 and any(word in text for word in planting_required):
            return True
        return False

    # 没有明确意图的泛问题，不强行返回 RAG evidence。
    if not terms:
        return False

    matched = sum(1 for t in terms if t in text)
    coverage = matched / max(len(terms), 1)

    # 多关键词问题，至少覆盖一部分，并且分数过线。
    if len(terms) >= 2:
        return score >= 0.45 and coverage >= 0.34

    # 单关键词太宽，默认不返回，避免“榴莲”到处乱命中。
    return False


def _normalized_dedupe_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower(), flags=re.UNICODE)[:900]


def _exact_anchor_coverage(query: str, text: str) -> float:
    if not is_exact_answer_query(query):
        return 0.0
    body = str(query or "")
    if "：" in body:
        prefix, suffix = body.split("：", 1)
        if any(mark in prefix for mark in ("填空", "原文", "知识库", "只填")):
            body = suffix
    elif ":" in body:
        prefix, suffix = body.split(":", 1)
        if any(mark in prefix for mark in ("填空", "原文", "知识库", "只填")):
            body = suffix
    anchors = [
        _normalized_dedupe_key(segment)
        for segment in FILL_BLANK_RE.split(body)
    ]
    anchors = [anchor for anchor in anchors if len(anchor) >= 2]
    if not anchors:
        return 0.0
    haystack = _normalized_dedupe_key(text)
    matched = sum(1 for anchor in anchors if anchor in haystack)
    return matched / len(anchors)


GLOSSARY_PATH = BASE_DIR / "rag_build" / "glossary.json"

# 别名索引：小写词 → 该词所属同义组的全部说法。惰性构建，只读一次文件。
_ALIAS_INDEX: Optional[Dict[str, set]] = None


def _load_alias_index() -> Dict[str, set]:
    """从术语表构建双向别名索引。

    术语表原本是「别名 → 标准名」的单向映射（供翻译统一译名用），
    这里把同一标准名下的所有说法互相打通，使任一说法都能扩展出全组。
    例如 Monthong / Mon Thong / หมอนทอง / Golden Pillow / 金枕 互为别名。
    """
    global _ALIAS_INDEX
    if _ALIAS_INDEX is not None:
        return _ALIAS_INDEX

    index: Dict[str, set] = {}
    if not GLOSSARY_PATH.exists():
        _ALIAS_INDEX = index
        return index

    try:
        data = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] cannot read glossary: {exc}")
        _ALIAS_INDEX = index
        return index

    groups: Dict[str, set] = {}
    for section, items in data.items():
        if section.startswith("_") or not isinstance(items, dict):
            continue
        for src, dst in items.items():
            if src.startswith("_") or not isinstance(dst, str):
                continue
            groups.setdefault(dst, set()).update({src, dst})

    for words in groups.values():
        for w in words:
            index.setdefault(w.lower(), set()).update(words)

    _ALIAS_INDEX = index
    return index


_RE_NON_LATIN_WORD = re.compile(r"[\u4e00-\u9fff\u0e00-\u0e7f]")


def _expand_aliases(text: str) -> List[str]:
    """把文本里命中的实体扩展为其全部同义说法。

    匹配策略按字符类型区分：
    - 含中文/泰文的词条用子串匹配（这些语言无词边界）
    - 纯拉丁词条用词边界匹配，避免 "D24" 命中 "D240" 之类
    """
    index = _load_alias_index()
    if not index:
        return []

    text = normalize_thai(text)   # 与索引侧保持一致；规则幂等，非泰文不受影响
    lowered = text.lower()
    out: set = set()
    for word, group in index.items():
        if len(word) < 3:
            continue          # 过短的词条噪声太大（如 "ke"、"di"）
        if _RE_NON_LATIN_WORD.search(word):
            if word in text or word in lowered:
                out |= group
        elif re.search(r"\b" + re.escape(word) + r"\b", lowered):
            out |= group
    return sorted(out)


def _lexical_terms(text: str) -> List[str]:
    """Build compact multilingual terms suitable for an in-memory BM25 scan.

    覆盖四个语种：中文、英文/马来文（拉丁）、泰文。

    泰文必须单独处理：它和中文一样不用空格分词，但字符范围
    \\u0e00-\\u0e7f 此前不在任何一条正则里，导致泰文查询的分词结果为 **0**，
    BM25 通路完全空转。加上 dense 侧的向量塌缩（4 个语义不同的泰文问题
    余弦相似度精确等于 1.0），泰文原本是两条通路同时失效。
    """
    value = clean_text(text)
    value = normalize_thai(value)   # 与索引侧保持一致；规则幂等，非泰文不受影响
    value = FILL_BLANK_RE.sub(" ", value)
    value = re.sub(
        r"^(?:请|请严格)?(?:根据|依据)?(?:知识库|参考资料|原文)?"
        r"(?:进行)?(?:填空|回答)?(?:，|,|：|:|\s)*(?:只填写答案[：:]?)?",
        " ",
        value,
    )

    terms: List[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,}|\d+(?:\.\d+)?", value):
        lowered = token.lower()
        if lowered not in GENERIC_STOPWORDS:
            terms.append(lowered)

    for span in re.findall(r"[\u4e00-\u9fff]+", value):
        if 2 <= len(span) <= 12 and span not in GENERIC_STOPWORDS:
            terms.append(span)
        for size in (2, 3):
            if len(span) < size:
                continue
            for index in range(0, len(span) - size + 1):
                gram = span[index : index + size]
                if gram not in GENERIC_STOPWORDS:
                    terms.append(gram)

    # 泰文：用 3-4 gram。泰文一个音节常占 3-4 个字符
    # （辅音 + 元音符号 + 声调符号叠写），用中文的 2-3 gram 会把单音节切碎。
    for span in re.findall(r"[\u0e00-\u0e7f]+", value):
        if 2 <= len(span) <= 14:
            terms.append(span)
        for size in (3, 4):
            if len(span) < size:
                continue
            for index in range(0, len(span) - size + 1):
                terms.append(span[index : index + size])

    # 别名扩展：把查询里命中的实体，按术语表补全其全部同义说法。
    # 解决"用户说法与文档写法不一致"导致的漏召回 —— 实测 6 组别名对里
    # 3 组漏召回（Monthong↛金枕、Phytophthora↛疫霉、soil moisture sensor↛土壤湿度传感器），
    # 加扩展后修回 2 组（第 3 组是设备名，术语表未收录）。
    terms.extend(_expand_aliases(value))

    seen = set()
    result = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            result.append(term)
    return result[:80]


def _build_lexical_node_cache(index: VectorStoreIndex) -> List[Dict[str, Any]]:
    global _LEXICAL_NODE_CACHE, _NEIGHBOR_CACHE
    if _LEXICAL_NODE_CACHE is not None:
        return _LEXICAL_NODE_CACHE

    nodes: List[Dict[str, Any]] = []
    neighbors: Dict[tuple, List[Dict[str, Any]]] = {}
    for node_id, node in index.docstore.docs.items():
        try:
            text = clean_text(node.get_content())
        except Exception:
            continue
        if not text or is_noise_text(text):
            continue
        # 泰文声调符号规范化放在噪声判定之后：保证过滤行为与改动前逐字节一致。
        # 规范化后的文本同时用于 BM25 匹配与 _expand_adjacent_text 拼接上下文，
        # 展示层拿到的也是修复后的正确泰文。
        text = normalize_thai(text)
        metadata = dict(getattr(node, "metadata", {}) or {})
        item = {
            "node_id": str(node_id),
            "text": text,
            "metadata": metadata,
            "length": max(1, len(re.sub(r"\s+", "", text))),
        }
        nodes.append(item)

        doc = str(
            metadata.get("doc")
            or metadata.get("source_file")
            or "unknown"
        )
        if metadata.get("item_index") is not None:
            try:
                sequence = ("item", doc, int(metadata["item_index"]))
                neighbors.setdefault(sequence, []).append(item)
            except (TypeError, ValueError):
                pass
        elif metadata.get("page") is not None:
            try:
                sequence = ("page", doc, int(metadata["page"]))
                neighbors.setdefault(sequence, []).append(item)
            except (TypeError, ValueError):
                pass

    _LEXICAL_NODE_CACHE = nodes
    _NEIGHBOR_CACHE = neighbors
    return nodes


def _bm25_retrieve(
    index: VectorStoreIndex,
    query: str,
    limit: int,
) -> List[Dict[str, Any]]:
    nodes = _build_lexical_node_cache(index)
    terms = _lexical_terms(query)
    if not nodes or not terms:
        return []

    document_frequency: Counter = Counter()
    matched_rows = []
    for node in nodes:
        text_lower = node["text"].lower()
        frequencies = {}
        for term in terms:
            count = text_lower.count(term)
            if count:
                frequencies[term] = count
                document_frequency[term] += 1
        if frequencies:
            matched_rows.append((node, frequencies))

    if not matched_rows:
        return []

    total_docs = len(nodes)
    average_length = sum(node["length"] for node in nodes) / max(total_docs, 1)
    k1 = 1.5
    b = 0.75
    scored = []
    for node, frequencies in matched_rows:
        score = 0.0
        for term, tf in frequencies.items():
            df = document_frequency[term]
            inverse_frequency = math.log(
                1.0 + (total_docs - df + 0.5) / (df + 0.5)
            )
            denominator = tf + k1 * (
                1.0 - b + b * node["length"] / max(average_length, 1.0)
            )
            score += inverse_frequency * (tf * (k1 + 1.0)) / denominator
        if score > 0:
            scored.append(
                {
                    **node,
                    "bm25_score": float(score),
                }
            )

    scored.sort(key=lambda item: item["bm25_score"], reverse=True)
    return scored[: max(1, int(limit))]


def _dense_retrieve(
    index: VectorStoreIndex,
    query: str,
    limit: int,
) -> List[Dict[str, Any]]:
    retriever = index.as_retriever(similarity_top_k=max(12, int(limit)))
    output = []
    for node in retriever.retrieve(query):
        text = clean_text(node.node.get_content())
        if not text or is_noise_text(text):
            continue
        # 必须与 _build_lexical_node_cache 做同样的泰文规范化：RRF 融合的
        # merge key 是 _normalized_dedupe_key(text)（按文本而非 node_id 去重），
        # 两侧文本不一致会让同一个 node 裂成两条候选、RRF 分数被劈开而掉出 Top-K。
        text = normalize_thai(text)
        output.append(
            {
                "node_id": str(getattr(node.node, "node_id", "") or ""),
                "text": text,
                "metadata": dict(node.node.metadata or {}),
                "dense_score": float(node.score or 0.0),
            }
        )
    return output


def _expand_adjacent_text(item: Dict[str, Any], max_chars: int = 2600) -> str:
    if not _NEIGHBOR_CACHE:
        return str(item.get("text") or "")
    metadata = dict(item.get("metadata") or {})
    doc = str(
        metadata.get("doc")
        or metadata.get("source_file")
        or "unknown"
    )
    sequence_type = ""
    sequence_value = None
    if metadata.get("item_index") is not None:
        sequence_type = "item"
        sequence_value = metadata.get("item_index")
    elif metadata.get("page") is not None:
        sequence_type = "page"
        sequence_value = metadata.get("page")
    try:
        sequence_value = int(sequence_value)
    except (TypeError, ValueError):
        return str(item.get("text") or "")

    pieces = []
    seen = set()
    for offset in (-1, 0, 1):
        key = (sequence_type, doc, sequence_value + offset)
        for neighbor in _NEIGHBOR_CACHE.get(key, []):
            text = str(neighbor.get("text") or "").strip()
            dedupe = _normalized_dedupe_key(text)
            if text and dedupe and dedupe not in seen:
                seen.add(dedupe)
                pieces.append(text)
    combined = " ".join(pieces).strip()
    return combined[:max_chars] if combined else str(item.get("text") or "")


def _is_thai_query(query: str) -> bool:
    """泰文查询判定。字符集判据，零歧义。

    判据是「泰文字符数 >= 中文字符数 × 3」。那个 ×3 是必要的：泰文一个
    音节占 3-4 个字符，裸字符数比较会系统性偏向泰文，把「榴莲品种
    หมอนทอง 的特点」这类中文查询（中文 7 字 / 泰文 8 字）误判成泰文。

    风险是不对称的，所以宁严不宽：误判成泰文会让一条中文查询白白丢掉
    好用的 dense（净损失）；误判成非泰文只是回到改动前的现状（无新增损失）。

    实测 1247 条金标：泰文 139/139 全对，zh/en/ms 共 1108 条无一被判成
    泰文（假阳性 0），与 rag_eval.detect_lang() 的泰文分支在金标上等价
    （金标里的泰文题都是纯泰文，两种判据不产生分歧）。

    为什么泰文要跳过 dense：bge-small-zh-v1.5 是中文单语模型，对泰文查询
    的 tokenizer 输出中位 66.7% 是 UNK，139 条泰文查询只剩 13 个不同的
    token 序列（最大簇 68 条共享同一个长度 3 的序列）。dense 通路对泰文
    是纯噪声，却在 RRF 里占 1.0 权重，把 BM25 的正确结果挤出 top-k。
    实测跳过后 ns@3 22.3%→30.9%（McNemar p=0.0018）、
    ns@1 5.0%→21.6%（p<0.0001），且省一次 embedding 推理（81-87ms）。

    ⚠️ 验收时别按 +8.6pp 要求：收益集中在近重复富集的子集里。那 13 条
    新增命中的题 4-gram 最近邻 Jaccard 中位 0.719，而全体是 0.375；在
    低近重复子集（Jaccard<0.5，98 条）上只有 +3.1pp。**合理预期是 ~+3pp。**
    另注：泰文金标的近重复密度（3-gram Jaccard 中位 0.421）与英文 0.404
    同量级，早期"泰文近重复只有 1.4%"是按空格分词算的假数——泰文连写会
    被切成一个巨型 token，Jaccard 必然趋零。跨语种比较必须用字符 n-gram。

    ⚠️ 前提条件：本分支成立的前提是「泰文查询的证据在泰文向量簇里、而
    bge 对泰文塌缩」。如果将来把泰文内容搬进中文簇（或换多语言
    embedding），泰文 dense 不再是噪声，**这个分支必须撤掉或重新评估**。
    """
    s = str(query or "")
    th = len(re.findall(r"[\u0e00-\u0e7f]", s))
    if not th:
        return False
    zh = len(re.findall(r"[\u4e00-\u9fff]", s))
    return th >= zh * 3


def _hybrid_retrieve(
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    index = load_index()
    exact_mode = is_exact_answer_query(query)
    candidate_limit = max(36, int(top_k) * 8)
    # 泰文跳过 dense，理由见 _is_thai_query 的 docstring。
    # 直接不调用而非事后降权 —— 省掉一次 embedding 推理。
    # dense 为空列表时 merge_candidates 自然不贡献候选，无需加权重分支。
    dense = [] if _is_thai_query(query) else _dense_retrieve(index, query, candidate_limit)
    lexical = _bm25_retrieve(index, query, candidate_limit)

    merged: Dict[str, Dict[str, Any]] = {}
    rrf_constant = 60.0

    def merge_candidates(items, source, weight):
        for rank, item in enumerate(items, start=1):
            key = _normalized_dedupe_key(item.get("text") or "")
            if not key:
                continue
            target = merged.setdefault(
                key,
                {
                    "text": item.get("text") or "",
                    "metadata": dict(item.get("metadata") or {}),
                    "dense_score": 0.0,
                    "bm25_score": 0.0,
                    "hybrid_score": 0.0,
                },
            )
            target["hybrid_score"] += weight / (rrf_constant + rank)
            if source == "dense":
                target["dense_score"] = max(
                    target["dense_score"],
                    float(item.get("dense_score") or 0.0),
                )
            else:
                target["bm25_score"] = max(
                    target["bm25_score"],
                    float(item.get("bm25_score") or 0.0),
                )

    merge_candidates(dense, "dense", 1.0)
    merge_candidates(lexical, "bm25", 1.35 if exact_mode else 1.0)

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("hybrid_score") or 0.0),
            float(item.get("bm25_score") or 0.0),
            float(item.get("dense_score") or 0.0),
        ),
        reverse=True,
    )

    results = []
    seen_locations = set()
    seen_expanded = set()
    for item in ranked:
        dense_score = float(item.get("dense_score") or 0.0)
        bm25_score = float(item.get("bm25_score") or 0.0)
        text = str(item.get("text") or "")
        if not exact_mode and not (
            evidence_matches_query(query, text, dense_score)
            or bm25_score > 1.2
        ):
            continue
        metadata = dict(item.get("metadata") or {})
        if exact_mode:
            text = _expand_adjacent_text(item)
            if _exact_anchor_coverage(query, text) < 0.6:
                continue

            doc = str(
                metadata.get("doc")
                or metadata.get("source_file")
                or "unknown"
            )
            sequence_value = metadata.get("item_index")
            sequence_type = "item"
            if sequence_value is None:
                sequence_value = metadata.get("page")
                sequence_type = "page"
            try:
                numeric_value = int(sequence_value)
            except (TypeError, ValueError):
                numeric_value = None
            if numeric_value is not None:
                nearby_locations = {
                    (sequence_type, doc, numeric_value + offset)
                    for offset in (-1, 0, 1)
                }
                if seen_locations.intersection(nearby_locations):
                    continue
                seen_locations.add((sequence_type, doc, numeric_value))

            expanded_key = _normalized_dedupe_key(text)
            if expanded_key in seen_expanded:
                continue
            seen_expanded.add(expanded_key)
        results.append(
            {
                "source_file": (
                    metadata.get("source_file")
                    or metadata.get("doc")
                    or "unknown"
                ),
                "doc": (
                    metadata.get("doc")
                    or metadata.get("source_file")
                    or "unknown"
                ),
                "page": metadata.get("page"),
                "score": round(
                    dense_score
                    if dense_score
                    else float(item.get("hybrid_score") or 0.0),
                    4,
                ),
                "dense_score": round(dense_score, 4),
                "bm25_score": round(bm25_score, 4),
                "hybrid_score": round(
                    float(item.get("hybrid_score") or 0.0),
                    6,
                ),
                "text": text,
                "display_text": text,
                "metadata": metadata,
                "neighbor_expanded": bool(exact_mode),
            }
        )
        if len(results) >= top_k:
            break
    return results



def retrieve(query: str, top_k: int = 5, min_score: float = DEFAULT_MIN_SCORE) -> List[Dict[str, Any]]:
    return _hybrid_retrieve(query, top_k=max(1, int(top_k)))



def raw_retrieve_debug(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Return raw LlamaIndex retrieval results without conservative filtering."""
    index = load_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    raw_nodes = retriever.retrieve(query)

    results = []
    for i, node in enumerate(raw_nodes, start=1):
        text = clean_text(node.node.get_content())
        meta = dict(node.node.metadata or {})
        results.append({
            "rank": i,
            "score": round(float(node.score or 0.0), 4),
            "doc": meta.get("doc") or meta.get("source_file") or "unknown",
            "page": meta.get("page"),
            "text_preview": text[:500],
        })
    return results

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rag_llamaindex.py rebuild")
        print("  python rag_llamaindex.py query '问题'")
        raise SystemExit(1)

    cmd = sys.argv[1]

    if cmd == "rebuild":
        rebuild_index()
        return

    if cmd == "query":
        if len(sys.argv) < 3:
            raise SystemExit("missing query")
        query = sys.argv[2]
        results = retrieve(query)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if cmd == "raw":
        if len(sys.argv) < 3:
            raise SystemExit("missing query")
        query = sys.argv[2]
        results = raw_retrieve_debug(query)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()


def retrieve_for_backend(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return evidence objects compatible with DurianGPT frontend/backend."""
    items = retrieve(query, top_k=top_k)

    evidence = []
    for i, item in enumerate(items):
        text = item.get("text") or item.get("display_text") or ""
        doc = item.get("doc") or item.get("source_file") or "unknown"
        meta = item.get("metadata", {}) or {}

        # index_text 与 display_text 分离：检索命中的是 index_text（可能是表格摘要、
        # 图片描述、译文），但喂给 LLM 和展示给用户的应该是 display_text
        # （完整 Markdown 表、原始格式）。旧实现把四个字段全赋同值，
        # 导致"为检索优化"和"为展示保留原格式"两个目标无法并存。
        display = str(meta.get("display_text") or text)
        block_type = str(meta.get("block_type") or "text")

        row = {
            "id": f"llamaindex_{i}_{abs(hash((doc, text[:80])))}",
            "source_file": item.get("source_file") or doc,
            "doc": doc,
            "page": item.get("page"),
            "item_index": meta.get("item_index"),
            "chunk_index": meta.get("chunk_index"),
            "score": item.get("score", 0.0),
            "dense_score": item.get("dense_score", item.get("score", 0.0)),
            "lexical_score": item.get("bm25_score", 0.0),
            "bm25_score": item.get("bm25_score", 0.0),
            "hybrid_score": item.get("hybrid_score", 0.0),
            "neighbor_expanded": bool(item.get("neighbor_expanded")),
            "block_type": block_type,
            "text": display,
            "index_text": text,
            "original_text": str(meta.get("original_text") or display),
            "display_text": display,
            "canonical_language": str(meta.get("src_lang") or "zh"),
            "text_language": str(meta.get("src_lang") or "zh"),
            "display_language": "zh",
        }

        # 表格：透传 Markdown 供前端渲染
        if meta.get("table_markdown"):
            row["table_markdown"] = meta["table_markdown"]
        # 损坏的表格：明确标记，前端与 LLM 都不应引用其数值
        if meta.get("damaged"):
            row["damaged_table"] = True
        # 图片：透传路径供回显
        if meta.get("image_path"):
            row["image_path"] = meta["image_path"]

        evidence.append(row)

    return evidence
