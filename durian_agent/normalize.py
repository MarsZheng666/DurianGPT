"""输入文本归一化（架构文档 §11）——InputNormalize 节点的核心逻辑。

设计边界：
- 只做「形式归一」，不做「内容理解」：不改词、不删词、不翻译；
- 默认保留大小写：normalized_query 会进入 SemanticParse 与最终展示，
  Musang King 这类实体名不应被改小写；检索分词侧需要统一小写时
  传 fold_case=True（与 rag_llamaindex._lexical_terms 的小写行为对齐）；
- 句末 ?/！保留（疑问语气是 SemanticParse 的信号），只压缩重复；
- 数字/单位归一只覆盖高置信模式：全角字符（NFKC）、泰文数字、
  温度符号变体、数字与常用单位间的空格压缩。
  千分位逗号（1,000）等有歧义的形式**不做**——它同时可能是列表分隔符。

NFKC 的坑（实测，见 tests/test_agent_normalize.py）：
- ℃ (U+2103) 会被分解成 "°C"，˚ (U+02DA) 会被拆成 "空格+U+030A"，
  … 会被拆成 "..."，º (U+00BA) 会变成 "o"；
- 其中 ˚ 在泰文语料里是 legacy 损坏修复（rag_llamaindex.normalize_thai）
  的判据字符，绝不能被 NFKC 破坏 —— 所以用占位符保护后再做 NFKC；
- 泰文数字 ๐-๙ 没有 NFKC 映射，需要显式转换。

泰语词典优先分词在 durian_agent/thai.py（任务 #2）。
"""

from __future__ import annotations

import re
import unicodedata

# ── 占位符（NFKC 保护）────────────────────────────────────────
# 用不会被 NFKC 触及的控制字符做往返占位
_PLACE_ELLIPSIS = "\x02"   # …  保护中文省略号 "……" 不被拆成 "......"
_PLACE_RING = "\x03"       # ˚  泰文 SARA AM 损坏判据字符（U+02DA）
_PLACE_DOT = "\x04"        # ˙  同上（U+02D9）

_THAI_DIGIT_TABLE = str.maketrans(
    "๐๑๒๓๔๕๖๗๘๙",          # U+0E50-0E59，NFKC 不处理
    "0123456789",
)

# 零宽字符：不可见的排版残留，会静默破坏 BM25 精确匹配
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")

# ── 温度归一 ─────────────────────────────────────────────────
# 顺序：(a) 数字前缀式 "25 ° C"→"25℃"；(b) 无数字式 "室温°C"→"室温℃"；
# (c) 收掉残留的 "数字 ℃" 中间空格。
# 字符类含 °(U+00B0, NFKC 不变) ∘(U+2218, 不变) 和恢复回来的 ˚(U+02DA)。
# º 在 NFKC 之前就换成 °（否则会被拆成 "o"）。
_DEGREE_DIGIT_RE = re.compile(r"(\d)\s*[°∘˚]\s*[cC]\b")
_DEGREE_BARE_RE = re.compile(r"\s*[°∘˚]\s*[cC]\b")
_DEGREE_SPACE_RE = re.compile(r"\s+℃")

# ── 单位空格压缩 ─────────────────────────────────────────────
_UNITS = r"(?:mm|cm|km|kg|mg|ml|ppm|ha|℃|%|m|g|l)"
_UNIT_SPACE_RE = re.compile(rf"(\d)\s+({_UNITS})\b", re.IGNORECASE)

# ── 标点 ─────────────────────────────────────────────────────
# 3 个以上连续 ?/! 压成 1 个；中文句读连打压成 1 个。
# 不能压 "……"（中文省略号规范写法，已由占位符保护）。
_REPEAT_BANG_RE = re.compile(r"([!?！？])\1{2,}")
_REPEAT_CJK_PUNCT_RE = re.compile(r"([。，、；：])\1+")

# 首尾剥离：空白与句读，不含 ?/！（句末语气保留）
_STRIP_CHARS = " \t\r\n\u3000。，、；：,;:~～"


def normalize_input(text: str, *, fold_case: bool = False) -> str:
    """四语种（zh/en/th/ms）通用的输入归一化。

    幂等：normalize_input(normalize_input(x)) == normalize_input(x)。
    """
    if not isinstance(text, str):
        text = str(text or "")

    # 1) NFKC 前的保护：省略号、泰文修复判据字符（˚/˙）、º
    text = text.replace("…", _PLACE_ELLIPSIS)
    text = text.replace("˚", _PLACE_RING)
    text = text.replace("˙", _PLACE_DOT)
    text = text.replace("º", "°")

    # 2) 泰文数字显式转换（NFKC 无映射）
    text = text.translate(_THAI_DIGIT_TABLE)

    # 3) Unicode NFKC：全角→半角、兼容数字、㎡→m2 等
    text = unicodedata.normalize("NFKC", text)

    # 4) 还原被保护的字符
    text = (text
            .replace(_PLACE_ELLIPSIS, "…")
            .replace(_PLACE_RING, "˚")
            .replace(_PLACE_DOT, "˙"))

    # 5) 零宽字符
    text = _ZERO_WIDTH_RE.sub("", text)

    # 6) 温度符号统一为 ℃（含 NFKC 把 ℃ 拆回 °C 的往返）
    text = _DEGREE_DIGIT_RE.sub(r"\1℃", text)
    text = _DEGREE_BARE_RE.sub("℃", text)
    text = _DEGREE_SPACE_RE.sub("℃", text)

    # 7) 空白折叠
    text = re.sub(r"\s+", " ", text).strip()

    # 8) 数字+单位空格压缩
    text = _UNIT_SPACE_RE.sub(r"\1\2", text)

    # 9) 标点处理
    text = _REPEAT_BANG_RE.sub(r"\1", text)
    text = _REPEAT_CJK_PUNCT_RE.sub(r"\1", text)
    text = text.strip(_STRIP_CHARS)

    # 10) 大小写（可选；检索分词侧用）
    if fold_case:
        text = text.lower()

    return text
