"""Intent 分类（架构文档 §5/§7，任务 #5）。

两层设计：
- LLM 层（主力）：任意语言 → 封闭标签集的结构化输出，一次调用同时
  拿 language / intent / secondary_intents；
- 规则层（兜底）：只覆盖**确定性高**的少数意图（任务操作类 + 告警/资产），
  LLM 不可用或输出非法时使用；其余意图交给 LLM——不为覆盖面堆关键词表。

多语言测试样例集见 testdata/intent_samples.json（20 意图 × 3 条，
zh/en/th/ms 混合），同时是任务 #61 评估集的种子。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from durian_agent.llm import LLMProvider
from durian_agent.normalize import normalize_input
from durian_agent.semantic.labels import (
    INTENTS,
    UNKNOWN_INTENT,
    normalize_intent,
    normalize_secondary_intents,
)

# ══════════════════ LLM prompt（§5：只回答"用户在问什么"）══════════════════

INTENT_SYSTEM_PROMPT = """You are a multilingual intent classifier for a durian plantation system.

The user query may be in Chinese, English, Thai, or Malay.

Classify the user's question into exactly one primary intent from this closed set:
{intents}

You may also return up to 2 secondary intents from the same set if the question clearly involves them.

Return JSON only:
{{"language": "zh|en|th|ms", "intent": "<label>", "secondary_intents": ["<label>", ...]}}

Rules:
- Do NOT answer the question.
- Do NOT diagnose diseases or pests.
- Do NOT invent labels outside the closed set.
- If the question is ambiguous, choose the closest intent and keep secondary_intents empty.
""".format(intents=", ".join(INTENTS))

# ══════════════════ 规则兜底（高置信子集）══════════════════

# 语言混合但模式稳定：任务操作动词 + 工单词；告警/资产名词
_RULE_PATTERNS: List[tuple] = [
    ("task_create", [
        re.compile(r"(创建|新建|帮我建|建一个).{0,8}(工单|任务)"),
        re.compile(r"create.{0,15}task", re.IGNORECASE),
        re.compile(r"buat.{0,12}tugasan", re.IGNORECASE),
        re.compile(r"(สร้าง|เปิด).{0,12}(งาน|ใบสั่งงาน)"),
    ]),
    ("task_update", [
        re.compile(r"(工单|任务)\s*\d*\s*(号)?\s*(改|更新|标记|完成|取消|关闭|设为)"),
        re.compile(r"(把|将).{0,10}(工单|任务).{0,10}(改|更新|完成|取消|关闭)"),
        re.compile(r"(mark|update|set).{0,12}task.{0,15}(completed|done|status)", re.IGNORECASE),
        re.compile(r"kemaskini.{0,12}tugasan", re.IGNORECASE),
    ]),
    ("task_query", [
        re.compile(r"(查询|查一下|查看|看看).{0,6}(工单|任务)"),
        re.compile(r"(工单|任务).{0,6}(状态|进度|列表)"),
        re.compile(r"(check|show|list).{0,20}tasks?", re.IGNORECASE),
        re.compile(r"semak.{0,10}tugasan", re.IGNORECASE),
    ]),
    ("alarm_query", [
        re.compile(r"(告警|报警|警报)"),
        re.compile(r"alarms?", re.IGNORECASE),
        re.compile(r"การแจ้งเตือน"),
        re.compile(r"amaran", re.IGNORECASE),
    ]),
    ("asset_query", [
        re.compile(r"(传感器|设备|资产).{0,6}(有|多少|列表|清单|几)"),
        re.compile(r"(有多少|几台|几个|多少台|多少个).{0,4}(传感器|设备|资产)"),
        re.compile(r"(list|show).{0,12}(devices?|sensors?|assets?)", re.IGNORECASE),
        re.compile(r"senarai.{0,12}(peralatan|aset)", re.IGNORECASE),
    ]),
]


def _rule_classify(query: str) -> Optional[str]:
    value = normalize_input(query)
    for intent, patterns in _RULE_PATTERNS:
        if any(p.search(value) for p in patterns):
            return intent
    return None


# ══════════════════ 分类器 ═══════════════════

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出中提取 JSON 对象（容忍 ```json 围栏与前后杂文）。"""
    if not isinstance(text, str):
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if not brace:
            return None
        text = brace.group(0)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class IntentClassifier:
    """intent 分类器：LLM 主力 + 规则兜底。"""

    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm

    def classify(self, query: str) -> Dict[str, Any]:
        """返回 {"language", "intent", "secondary_intents"}。

        LLM 输出非法或未配置 LLM 时落到规则兜底；规则也无命中则 unknown
        （路由层对 unknown 按保守策略处理，不猜 general_knowledge）。
        """
        if self.llm is not None:
            try:
                raw = self.llm.complete(INTENT_SYSTEM_PROMPT, query)
                parsed = _extract_json(raw)
                if parsed is not None:
                    intent = normalize_intent(str(parsed.get("intent", "")))
                    if intent != UNKNOWN_INTENT:
                        return {
                            "language": str(parsed.get("language", "")).strip() or "zh",
                            "intent": intent,
                            "secondary_intents": normalize_secondary_intents(
                                parsed.get("secondary_intents") or []
                            ),
                        }
            except Exception:
                pass  # LLM 失败 → 规则兜底（错误处理策略属任务 #56，这里先降级）
        return self._fallback(query)

    def _fallback(self, query: str) -> Dict[str, Any]:
        intent = _rule_classify(query) or UNKNOWN_INTENT
        return {"language": "", "intent": intent, "secondary_intents": []}
