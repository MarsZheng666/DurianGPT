"""泰语归一化与词典优先分词（架构文档 §11 泰语管线，任务 #2）。

管线：Thai text → 基础 normalization（normalize.normalize_input）
              → legacy 声调符号修复（下方 verbatim 迁移段）
              → 领域词典优先匹配 + 剩余片段 n-gram
              → domain-aware tokens

════════════════════════════════════════════════════════════════════
【verbatim 迁移段】以下 legacy 泰文声调符号修复代码原样复制自
rag_llamaindex.py:178-357（2026-09-10 快照）。该文件有未提交的本地
修改不能直接 import，且运行时不希望拖上 llamaindex 全家桶，故整段
迁移。**与原处的同步由 tests/test_agent_thai.py 的 parity 测试强制
保证**：任何一侧改动导致行为分叉，测试即失败。修改请两侧同步。
════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import re
from typing import Dict, List

from durian_agent.glossary import load_glossary
from durian_agent.normalize import normalize_input

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

# ════════════════════ verbatim 迁移段结束 ════════════════════


# ════════════════════ 输入侧泰语归一化（§11）════════════════════

def normalize_thai_input(text: str) -> str:
    """§11 泰语管线前两步：基础归一化 + legacy 声调修复。

    顺序上 normalize_input 在前：温度 "25˚C" 先统一成 "25℃"，
    ˚ 被消费掉，不会再与 SARA AM 修复规则纠缠（两个规则对 ˚ 的
    消费条件互斥：°度符号后跟 C，SARA AM 后跟 า）。
    """
    return normalize_thai(normalize_input(text))


# ════════════════════ 领域词典优先分词（§11）════════════════════

_THAI_RUN_RE = re.compile(r"[\u0e00-\u0e7f]+")


def thai_domain_terms(mapping: Dict[str, str] | None = None) -> List[str]:
    """术语表中以泰文书写的别名（词典优先匹配只对泰文文本有意义）。"""
    if mapping is None:
        mapping = load_glossary()
    return [alias for alias in mapping if _THAI_CHAR.search(alias)]


def tokenize_thai(text: str, terms: List[str] | None = None) -> List[str]:
    """词典优先最长匹配 + 剩余片段 n-gram → domain-aware tokens。

    规则（§11）：
    1. 先过 normalize_thai_input（基础归一化 + 声调修复）；
    2. 术语表别名按「最长优先」贪心匹配，命中的整词作为一个完整 token
       （专业术语绝不切碎，如 หมอนทอง 不能拆成 n-gram）；
    3. 未被词典占用的泰文连续段，沿用 rag_llamaindex._lexical_terms
       实测调优的 3-4 gram 策略（泰文一个音节常占 3-4 字符，
       中文式 2-3 gram 会把单音节切碎）；
    4. 非泰文部分不在此处理（拉丁/中文分词是检索侧 _lexical_terms 的职责）。
    """
    if terms is None:
        terms = thai_domain_terms()
    value = normalize_thai_input(text)
    if not value:
        return []

    # 词典优先：长词先占位，占过的区间不再被短词/词典重叠使用
    occupied = [False] * len(value)
    matched: List[tuple] = []   # (start, term)
    for term in sorted(terms, key=len, reverse=True):
        if not term:
            continue
        start = value.find(term)
        while start != -1:
            end = start + len(term)
            if not any(occupied[start:end]):
                for i in range(start, end):
                    occupied[i] = True
                matched.append((start, term))
            start = value.find(term, start + 1)
    matched.sort()

    tokens: List[str] = [term for _, term in matched]

    # 剩余泰文连续段（跳过被词典占用的字符）→ 整段 + 3-4 gram
    for run in _THAI_RUN_RE.finditer(value):
        span_start, span_text = run.start(), run.group()
        free = "".join(
            ch for offset, ch in enumerate(span_text)
            if not occupied[span_start + offset]
        )
        if 2 <= len(free) <= 14:
            tokens.append(free)
        for size in (3, 4):
            if len(free) < size:
                continue
            for i in range(len(free) - size + 1):
                tokens.append(free[i:i + size])

    # 去重保序
    seen = set()
    result = []
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result
