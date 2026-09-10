"""Token Budget（架构文档 §34，任务 #57）。

§34 分配比例：
    System 10% / Summary 15% / Recent History 25% / RAG 35% /
    Current Query 5% / Output Reserve 10%

- estimate_tokens：CJK ≈ 1 字符/token，拉丁 ≈ 4 字符/token 的保守估计
  （与 bge 系 tokenizer 的经验行为对齐；精确计数交给模型侧报错兜底）；
- TokenBudget.cap(section)：该节预算（token）；
- TokenBudget.fit(section, text)：超预算截断（各节独立，互不挤占）；
- RAG 拿最大份额（35%）——证据是专业结论的依据；
  Query 最小（5%）——用户问题本身不长。
"""

from __future__ import annotations

import re
from typing import Dict, Optional

DEFAULT_TOTAL_TOKENS = 8000

#: §34 比例（冻结清单，改动须经评估）
DEFAULT_BUDGET_RATIOS: Dict[str, float] = {
    "system": 0.10,
    "summary": 0.15,
    "history": 0.25,
    "rag": 0.35,
    "query": 0.05,
    "output": 0.10,
}

_CJK_RE = re.compile(r"[\u2e80-\u9fff\u0e00-\u0e7f\uf900-\ufaff]")  # 中日韩+泰文


def estimate_tokens(text: str) -> int:
    """保守 token 估计：CJK 1 字符=1 token；其余 4 字符=1 token。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def _token_budget_to_chars(tokens: int) -> int:
    """截断用的字符数：按最坏情况（全 CJK）折算，保证不超预算。"""
    return max(0, tokens)


class TokenBudget:
    """§34 六节预算。"""

    def __init__(self, total_tokens: int = DEFAULT_TOTAL_TOKENS,
                 ratios: Optional[Dict[str, float]] = None):
        self.total = total_tokens
        self.ratios = ratios or DEFAULT_BUDGET_RATIOS

    def cap(self, section: str) -> int:
        ratio = self.ratios.get(section)
        if ratio is None:
            raise KeyError(f"未知预算节: {section}，可选: {sorted(self.ratios)}")
        return int(self.total * ratio)

    def fit(self, section: str, text: str) -> str:
        """截断到该节预算内（保尾部——对话内容关键信息常在近处）。"""
        cap = self.cap(section)
        chars = _token_budget_to_chars(cap)
        if len(text) <= chars:
            return text
        return text[-chars:]

    def report(self, used: Dict[str, str]) -> Dict[str, Dict[str, int]]:
        """各节实际用量 vs 预算（§52 可观测）。"""
        out = {}
        for section, text in used.items():
            used_tokens = estimate_tokens(text)
            out[section] = {"used": used_tokens, "cap": self.cap(section),
                            "over": used_tokens > self.cap(section)}
        return out
