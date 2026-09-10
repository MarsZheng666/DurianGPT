"""LangGraph conversation runtime for Durian GPT.

This module owns orchestration and thread memory only. Domain-specific model,
RAG, prompt, and image-context behavior is supplied by the FastAPI adapter so
the existing Durian GPT implementation remains the single source of truth.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, Iterator, List, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph, add_messages


logger = logging.getLogger("durian--langgraph")


class DurianGraphState(TypedDict, total=False):
    # 【流程核心】messages 挂在 add_messages reducer 上：节点返回新 id 消息=追加，
    # 同 id 消息=原地更新，RemoveMessage=删除（见 _trim_memory_node）。
    messages: Annotated[List[AnyMessage], add_messages]
    turn_messages: List[Dict[str, Any]]
    requested_language: str
    response_language: str
    use_rag: bool
    generation: Dict[str, Any]
    user_query: str
    route_decision: Dict[str, Any]
    context_resolution: Dict[str, Any]
    context_messages: List[Dict[str, Any]]
    rag_query: str
    evidence: List[Dict[str, Any]]
    evidence_quality: str
    evidence_text: str
    final_messages: List[Dict[str, str]]
    answer: str
    active_context_card: Optional[Dict[str, Any]]
    working_memory: Dict[str, Any]
    topic_memories: List[Dict[str, Any]]
    rolling_summary: str
    long_term_memories: List[Dict[str, Any]]
    memory_event: Dict[str, Any]
    user_id: str
    resolved_query: str
    rag_allowed: bool
    clarification_needed: bool
    turn_count: int
    exact_mode: bool
    error: str


_GENERIC_FOLLOWUP_PATTERNS = (
    r"^(再)?详细(一)?点[吧。！!？?]*$",
    r"^信息(再)?详细(一)?点[吧。！!？?]*$",
    r"^(再)?具体(一)?点[吧。！!？?]*$",
    r"^(请)?展开(说说|说明|讲讲)?[吧。！!？?]*$",
    r"^(请)?继续(说|讲|分析|回答)?[吧。！!？?]*$",
    r"^(然后|接着|还有)(呢)?[。！!？?]*$",
    r"^(为什么|怎么说|什么意思|有多严重)[。！!？?]*$",
    r"^(说详细点|讲详细点|多说一点|多讲一点)[。！!？?]*$",
    r"^(再)?细说(一下|一点|一地)?[吧。！!？?]*$",
    r"^(再)?讲(一下|一点|一地)?[吧。！!？?]*$",
    r"^(再)?说(一下|一点|一地)?[吧。！!？?]*$",
    r"^(展开|展开一下|展开一点)[吧。！!？?]*$",
    r"^(more details|explain more|go on|continue|why)[.!?]*$",
)

_MEMORY_SENSITIVE_MARKERS = (
    "密码",
    "口令",
    "验证码",
    "身份证",
    "银行卡",
    "api key",
    "apikey",
    "access token",
    "secret",
    "password",
)

_MEMORY_STOP_TERMS = {
    "这个",
    "那个",
    "然后",
    "还是",
    "可以",
    "怎么",
    "什么",
    "一下",
    "一个",
    "用户",
    "问题",
    "回答",
    "榴莲",
}


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def is_generic_followup(query: str, has_prior_context: bool) -> bool:
    """Return True only for short utterances that need a previous turn."""
    if not has_prior_context:
        return False
    compact = _compact_text(query)
    if not compact:
        return False
    return any(re.fullmatch(pattern, compact, flags=re.IGNORECASE) for pattern in _GENERIC_FOLLOWUP_PATTERNS)


def _message_role(message: AnyMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    role = getattr(message, "type", "") or ""
    return {"human": "user", "ai": "assistant"}.get(role, role or "user")


def message_to_payload(message: AnyMessage) -> Dict[str, Any]:
    extra = dict(getattr(message, "additional_kwargs", {}) or {})
    return {
        "id": getattr(message, "id", None),
        "role": _message_role(message),
        "content": str(getattr(message, "content", "") or ""),
        "image_url": extra.get("image_url"),
        "active_context_card": extra.get("active_context_card"),
        "evidence": extra.get("evidence"),
        "evidence_quality": extra.get("evidence_quality", "none"),
    }


def _stable_message_id(payload: Dict[str, Any]) -> str:
    material = {
        "role": payload.get("role") or "user",
        "content": payload.get("content") or "",
        "image_url": payload.get("image_url"),
        "active_context_card": payload.get("active_context_card"),
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return f"durian-{hashlib.sha256(encoded).hexdigest()[:32]}"


def payload_to_message(payload: Dict[str, Any]) -> AnyMessage:
    role = str(payload.get("role") or "user").lower()
    content = str(payload.get("content") or "")
    extra = {
        key: payload.get(key)
        for key in ("image_url", "active_context_card", "evidence", "evidence_quality")
        if payload.get(key) is not None
    }
    message_id = str(payload.get("id") or _stable_message_id(payload))
    if role == "assistant":
        return AIMessage(content=content, additional_kwargs=extra, id=message_id)
    if role == "system":
        return SystemMessage(content=content, additional_kwargs=extra, id=message_id)
    return HumanMessage(content=content, additional_kwargs=extra, id=message_id)


def _last_content(messages: List[AnyMessage], role: str) -> str:
    for message in reversed(messages):
        if _message_role(message) == role:
            return str(getattr(message, "content", "") or "").strip()
    return ""


def build_followup_retrieval_query(messages: List[AnyMessage], query: str) -> str:
    """Create a clean standalone query for a vague follow-up.

    Past assistant prose is deliberately excluded: an unsupported answer must
    never become retrieval input and reinforce its own hallucinations.
    """
    previous = messages[:-1] if messages else []
    previous_user = ""
    for message in reversed(previous):
        if _message_role(message) != "user":
            continue
        candidate = str(getattr(message, "content", "") or "").strip()
        if candidate and not is_generic_followup(candidate, True):
            previous_user = candidate
            break
    # Retrieval needs the stable subject, not the conversational instruction
    # ("详细一点", "为什么", ...). The latter belongs to generation intent.
    return previous_user[:320]


def _context_card_anchor_terms(card: Dict[str, Any]) -> str:
    haystack = " ".join(
        str(card.get(key) or "")
        for key in ("main_subject", "last_user_query", "last_answer_summary")
    ).lower()
    anchors = []
    anchor_rules = [
        (("炭疽", "anthracnose", "colletotrichum"), "炭疽病 Anthracnose 叶斑病 真菌病害"),
        (("叶斑", "leaf spot"), "叶斑病 叶片病斑 防治"),
        (("蒂腐", "茎端腐", "stem end rot"), "榴莲蒂腐病 Stem End Rot 防治"),
        (("赤衣", "pink disease"), "榴莲赤衣病 Pink Disease 防治"),
        (("白根", "root disease", "root rot"), "榴莲白根病 根腐病 防治"),
        (("蛀果虫", "fruit borer"), "榴莲蛀果虫 Fruit Borer 防治"),
        (("变质", "果肉", "腐烂", "霉变", "果腐", "软腐", "spoilage", "fruit rot"), "榴莲果实腐烂 霉变 果肉变质 果腐病 食用安全"),
    ]
    for triggers, label in anchor_rules:
        if any(trigger in haystack for trigger in triggers):
            anchors.append(label)
    return " ".join(anchors[:3])


def build_context_card_retrieval_query(card: Dict[str, Any], query: str) -> str:
    """Turn a vague follow-up into a query anchored to the selected topic card."""
    parts = [
        _context_card_anchor_terms(card),
        str(card.get("main_subject") or "").strip(),
        str(card.get("last_user_query") or "").strip()[:220],
        str(card.get("last_answer_summary") or "").strip()[:360],
        str(query or "").strip(),
    ]
    return "；".join(part for part in parts if part)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_terms(value: str) -> set[str]:
    text = re.sub(r"\s+", " ", str(value or "").lower()).strip()
    terms = {
        word
        for word in re.findall(r"[a-z0-9][a-z0-9._-]{1,}", text)
        if word not in _MEMORY_STOP_TERMS
    }
    for block in re.findall(r"[\u3400-\u9fff]{2,}", text):
        if block not in _MEMORY_STOP_TERMS and len(block) <= 8:
            terms.add(block)
        terms.update(
            block[index : index + 2]
            for index in range(max(0, len(block) - 1))
            if block[index : index + 2] not in _MEMORY_STOP_TERMS
        )
    return terms


def _normalize_memory_content(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n，。；;！!?？")
    text = re.sub(r"^(?:请|麻烦你|帮我)?(?:记住|记一下|记得)[：:\s]*", "", text)
    return text[:400].strip()


def _classify_explicit_memory(sentence: str) -> Optional[str]:
    value = str(sentence or "").strip()
    lower = value.lower()
    if not value or any(marker in lower for marker in _MEMORY_SENSITIVE_MARKERS):
        return None
    if re.search(r"(?:我|本人).{0,5}(?:喜欢|偏好|更喜欢|不喜欢|希望|习惯|常用|倾向)", value):
        return "preference"
    if re.search(r"(?:以后|今后).{0,12}(?:回答|回复|说明|称呼|使用)", value):
        return "preference"
    if re.search(
        r"(?:我的|我家|本人).{0,20}(?:果园|农场|榴莲|地区|位置|土壤|品种|树龄|面积|目标|设备|语言)",
        value,
    ):
        return "profile"
    if re.search(
        r"(?:我|本人)(?:在|来自|位于|住在|种植|栽培|管理|经营|使用|有|没有|主要种)",
        value,
    ):
        return "profile"
    return None


def extract_explicit_user_memories(query: str) -> List[Dict[str, str]]:
    """Extract only user-authored facts/preferences; never infer model facts."""
    raw = re.sub(r"\s+", " ", str(query or "")).strip()
    if not raw:
        return []

    explicit_command = bool(
        re.match(
            r"^(?:请|麻烦你|帮我)?(?:记住|记一下|记得)(?!吗|没|了没)",
            raw,
        )
    )
    if ("?" in raw or "？" in raw) and not explicit_command:
        return []

    candidates = [
        item.strip()
        for item in re.split(r"[。；;\n]+", raw)
        if item.strip()
    ]
    result: List[Dict[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates[:6]:
        content = _normalize_memory_content(candidate)
        if not content:
            continue
        if any(
            marker in content.lower()
            for marker in _MEMORY_SENSITIVE_MARKERS
        ):
            continue
        kind = _classify_explicit_memory(content)
        if explicit_command and kind is None:
            kind = "fact"
        if kind is None:
            continue
        normalized = _compact_text(content)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append({"kind": kind, "content": content})
    return result


class UserMemoryStore:
    """Small, user-scoped SQLite store for explicit long-term memories."""

    def __init__(self, path: str):
        self.path = str(Path(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, normalized)
                );
                CREATE INDEX IF NOT EXISTS idx_user_memories_user_updated
                    ON user_memories(user_id, updated_at DESC);
                """
            )
            self._conn.commit()

    def upsert(self, user_id: str, kind: str, content: str) -> None:
        normalized = _compact_text(content)
        if not user_id or not normalized:
            return
        memory_id = "memory-" + hashlib.sha256(
            f"{user_id}|{normalized}".encode("utf-8")
        ).hexdigest()[:32]
        now = _utc_now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO user_memories(
                    id, user_id, kind, content, normalized, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, normalized) DO UPDATE SET
                    kind=excluded.kind,
                    content=excluded.content,
                    updated_at=excluded.updated_at
                """,
                (
                    memory_id,
                    str(user_id),
                    str(kind or "fact"),
                    str(content)[:400],
                    normalized,
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def clear(self, user_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM user_memories WHERE user_id = ?",
                (str(user_id),),
            )
            self._conn.commit()
            return int(cursor.rowcount or 0)

    def forget_matching(self, user_id: str, query: str) -> int:
        target = re.sub(
            r"^(?:请|帮我)?(?:忘记|忘掉|删除)(?:关于|我告诉你的|这条)?",
            "",
            str(query or "").strip(),
        )
        target = re.sub(r"(?:的)?记忆$", "", target).strip()
        target_terms = _memory_terms(target)
        if not target_terms:
            return 0
        with self._lock:
            rows = list(
                self._conn.execute(
                    "SELECT id, content FROM user_memories WHERE user_id = ?",
                    (str(user_id),),
                )
            )
            ids = [
                str(row["id"])
                for row in rows
                if _memory_terms(str(row["content"])) & target_terms
            ]
            for memory_id in ids:
                self._conn.execute(
                    "DELETE FROM user_memories WHERE id = ? AND user_id = ?",
                    (memory_id, str(user_id)),
                )
            self._conn.commit()
        return len(ids)

    def apply_user_message(self, user_id: str, query: str) -> Dict[str, Any]:
        text = str(query or "").strip()
        if not user_id or not text:
            return {}
        if re.search(
            r"(?:清空|删除|忘记|忘掉).{0,8}(?:全部|所有).{0,4}(?:记忆|资料|信息)",
            text,
        ):
            return {"action": "cleared", "count": self.clear(user_id)}
        if re.search(r"^(?:请|帮我)?(?:忘记|忘掉|删除)(?:关于|我告诉你的)", text):
            return {
                "action": "forgot",
                "count": self.forget_matching(user_id, text),
            }

        memories = extract_explicit_user_memories(text)
        for memory in memories:
            self.upsert(user_id, memory["kind"], memory["content"])
        if not memories:
            return {}
        return {
            "action": "stored",
            "count": len(memories),
            "items": memories,
        }

    def retrieve(
        self,
        user_id: str,
        query: str = "",
        max_items: int = 6,
    ) -> List[Dict[str, Any]]:
        if not user_id:
            return []
        with self._lock:
            rows = list(
                self._conn.execute(
                    """
                    SELECT id, kind, content, created_at, updated_at
                    FROM user_memories
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 80
                    """,
                    (str(user_id),),
                )
            )
        query_terms = _memory_terms(query)
        ranked: List[tuple[float, int, sqlite3.Row]] = []
        for index, row in enumerate(rows):
            memory_terms = _memory_terms(str(row["content"]))
            overlap = len(query_terms & memory_terms)
            score = float(overlap * 4)
            if query_terms:
                score += overlap / max(1, len(query_terms))
            if str(row["kind"]) in {"preference", "profile"}:
                score += 0.75
            score += max(0.0, 0.5 - index * 0.01)
            ranked.append((score, -index, row))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = ranked[: max(1, int(max_items))]
        return [
            {
                "id": str(row["id"]),
                "kind": str(row["kind"]),
                "content": str(row["content"]),
                "updated_at": str(row["updated_at"]),
                "score": round(float(score), 4),
            }
            for score, _index, row in selected
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class DurianLangGraphRuntime:
    """Persistent, streaming LangGraph runtime for one Durian GPT process."""

    def __init__(
        self,
        *,
        adapter: Any,
        checkpoint_path: str,
        user_memory_path: Optional[str] = None,
        max_memory_messages: int = 12,
        max_prompt_messages: int = 8,
        max_topic_memories: int = 10,
    ):
        self.adapter = adapter
        self.checkpoint_path = str(Path(checkpoint_path))
        self.max_memory_messages = max(4, int(max_memory_messages))
        self.max_prompt_messages = max(2, int(max_prompt_messages))
        self.max_topic_memories = max(3, int(max_topic_memories))
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        memory_path = user_memory_path or str(
            Path(self.checkpoint_path).with_name("durian_user_memories.sqlite")
        )
        self.user_memory_store = UserMemoryStore(memory_path)

        self._saver_context = SqliteSaver.from_conn_string(self.checkpoint_path)
        self._saver = self._saver_context.__enter__()
        self._saver.setup()
        self.graph = self._build_graph()
        logger.info(
            "LangGraph runtime ready: checkpoint=%s user_memory=%s",
            self.checkpoint_path,
            memory_path,
        )

    def _build_graph(self):
        builder = StateGraph(DurianGraphState)
        builder.add_node("route", self._route_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("compose", self._compose_node)
        builder.add_node("generate", self._generate_node)
        builder.add_node("trim_memory", self._trim_memory_node)
        builder.add_edge(START, "route")
        builder.add_edge("route", "retrieve")
        builder.add_edge("retrieve", "compose")
        builder.add_edge("compose", "generate")
        builder.add_edge("generate", "trim_memory")
        builder.add_edge("trim_memory", END)
        # 【流程核心】compile() 把"节点+边"编译成可执行图（Pregel 实例），
        # 并绑定 SqliteSaver：此后每个 superstep 结束都会按 thread_id 把整个 state 落盘。
        return builder.compile(checkpointer=self._saver, name="durian_conversation")

    @staticmethod
    def _user_id_from_thread(thread_id: str) -> str:
        return str(thread_id or "").split(":", 1)[0].strip()

    def _memory_card_payloads(
        self,
        cards: List[Dict[str, Any]],
        payloads: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        existing_ids = {
            str((payload.get("active_context_card") or {}).get("context_id") or "")
            for payload in payloads
            if isinstance(payload.get("active_context_card"), dict)
        }
        result: List[Dict[str, Any]] = []
        for card in cards[-self.max_topic_memories :]:
            context_id = str(card.get("context_id") or "")
            if context_id and context_id in existing_ids:
                continue
            result.append(
                {
                    "role": "assistant",
                    "content": str(
                        card.get("last_answer_summary")
                        or card.get("summary")
                        or ""
                    )[:800],
                    "active_context_card": dict(card),
                }
            )
        return result

    @staticmethod
    def _build_memory_instruction(
        *,
        route_intent: str,
        resolved_query: str,
        original_query: str,
        working_memory: Dict[str, Any],
        rolling_summary: str,
        long_term_memories: List[Dict[str, Any]],
        memory_event: Dict[str, Any],
    ) -> str:
        sections: List[str] = []
        if resolved_query and resolved_query.strip() != original_query.strip():
            sections.append(
                "已结合上下文补全的当前问题："
                + resolved_query.strip()[:800]
            )
        if route_intent == "follow_up" and working_memory:
            sections.append(
                "当前工作记忆："
                + json.dumps(working_memory, ensure_ascii=False, default=str)[:1800]
            )
        if route_intent == "follow_up" and rolling_summary:
            sections.append("对话滚动摘要：" + str(rolling_summary)[:1800])
        if long_term_memories:
            memory_lines = [
                f"- [{item.get('kind', 'fact')}] {item.get('content', '')}"
                for item in long_term_memories[:6]
                if str(item.get("content") or "").strip()
            ]
            if memory_lines:
                sections.append(
                    "用户明确确认的长期记忆（仅在与当前问题相关时使用）：\n"
                    + "\n".join(memory_lines)
                )
        if memory_event.get("action") in {"cleared", "forgot"}:
            sections.append(
                "本轮用户要求删除记忆，已执行；不要继续引用已删除内容。"
            )
        if not sections:
            return ""
        return (
            "[Conversation Memory V2]\n"
            + "\n\n".join(sections)
            + "\n\n优先级：当前用户问题 > 当前工作记忆 > 用户明确确认的长期记忆 > 知识库资料。"
            "长期记忆只用于补充用户已明确陈述的信息，不得把助手过去的推测当成用户事实。"
        )

    @staticmethod
    def _working_memory_from_card(
        card: Optional[Dict[str, Any]],
        query: str,
        answer: str = "",
    ) -> Dict[str, Any]:
        if not isinstance(card, dict) or not card:
            return {
                "active_subject": str(query or "")[:240],
                "last_user_goal": str(query or "")[:500],
                "last_answer_summary": str(answer or "")[:800],
            }
        return {
            "context_id": str(card.get("context_id") or ""),
            "active_subject": str(
                card.get("main_subject")
                or card.get("active_subject")
                or query
                or ""
            )[:300],
            "scenario": str(card.get("scenario") or ""),
            "category_type": str(card.get("category_type") or ""),
            "source": str(card.get("source") or ""),
            "last_user_goal": str(
                card.get("last_user_query") or query or ""
            )[:500],
            "key_facts": [
                str(item)[:260]
                for item in list(card.get("key_facts") or [])[-8:]
                if str(item).strip()
            ],
            "last_answer_summary": str(
                card.get("last_answer_summary")
                or card.get("summary")
                or answer
                or ""
            )[:900],
            "image_url": str(card.get("image_url") or ""),
        }

    def _merge_topic_memories(
        self,
        existing: List[Dict[str, Any]],
        card: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        cards = [dict(item) for item in existing if isinstance(item, dict)]
        if not isinstance(card, dict) or not card:
            return cards[-self.max_topic_memories :]
        context_id = str(card.get("context_id") or "")
        if context_id:
            cards = [
                item
                for item in cards
                if str(item.get("context_id") or "") != context_id
            ]
        cards.append(dict(card))
        return cards[-self.max_topic_memories :]

    @staticmethod
    def _build_rolling_summary(cards: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for card in cards[-4:]:
            subject = str(card.get("main_subject") or "未命名主题").strip()
            user_goal = str(card.get("last_user_query") or "").strip()
            answer = str(
                card.get("last_answer_summary")
                or card.get("summary")
                or ""
            ).strip()
            facts = "；".join(
                str(item).strip()
                for item in list(card.get("key_facts") or [])[-3:]
                if str(item).strip()
            )
            parts = [f"主题={subject}"]
            if user_goal:
                parts.append(f"用户目标={user_goal[:260]}")
            if facts:
                parts.append(f"关键信息={facts[:420]}")
            if answer:
                parts.append(f"最近结论={answer[:520]}")
            lines.append("；".join(parts))
        return "\n".join(lines)[-2400:]

    @staticmethod
    def _clarification_reply(
        response_language: str,
        working_memory: Dict[str, Any],
    ) -> str:
        subject = str(working_memory.get("active_subject") or "").strip()
        replies = {
            "zh": (
                f"我还不能确定你指的是“{subject}”还是更早的话题。请补充一下具体对象或问题。"
                if subject
                else "我还不能确定你指的是哪一项内容。请补充一下具体对象或问题。"
            ),
            "en": (
                f"I cannot yet tell whether you mean “{subject}” or an earlier topic. Please specify the subject or question."
                if subject
                else "I cannot yet tell which item you mean. Please specify the subject or question."
            ),
            "ms": (
                f"Saya belum pasti sama ada anda merujuk kepada “{subject}” atau topik terdahulu. Sila nyatakan objek atau soalan dengan lebih khusus."
                if subject
                else "Saya belum pasti perkara yang anda maksudkan. Sila nyatakan objek atau soalan dengan lebih khusus."
            ),
            "th": (
                f"ฉันยังไม่แน่ใจว่าคุณหมายถึง “{subject}” หรือหัวข้อก่อนหน้า โปรดระบุสิ่งหรือคำถามให้ชัดเจนขึ้น"
                if subject
                else "ฉันยังไม่แน่ใจว่าคุณหมายถึงเรื่องใด โปรดระบุสิ่งหรือคำถามให้ชัดเจนขึ้น"
            ),
        }
        return replies.get(response_language, replies["zh"])

    def _route_node(self, state: DurianGraphState) -> Dict[str, Any]:
        """【流程 1】理解问题：意图分类 → 上下文消解 → 产出 resolved_query / rag_query / 工作记忆。"""
        route_started = time.perf_counter()
        try:
            get_stream_writer()({"type": "status", "phase": "route_start", "message": "正在理解问题..."})
        except Exception:
            pass
        messages = list(state.get("messages") or [])
        checkpoint_payloads = [message_to_payload(message) for message in messages]
        incoming_payloads = [
            dict(payload)
            for payload in list(state.get("turn_messages") or [])
            if str(payload.get("content") or "").strip()
        ]

        # A browser replay carries the exact visible order of the recent
        # conversation. Prefer it whenever it includes history. LangGraph's
        # add_messages reducer updates an existing ID in place, which is useful
        # for deduplication but must not be allowed to make an older answer look
        # newer than a freshly completed image diagnosis.
        payloads = (
            incoming_payloads
            if len(incoming_payloads) > 1
            else checkpoint_payloads
        )
        topic_memories = [
            dict(item)
            for item in list(state.get("topic_memories") or [])
            if isinstance(item, dict)
        ][-self.max_topic_memories :]
        rolling_summary = str(state.get("rolling_summary") or "")
        prior_working_memory = dict(state.get("working_memory") or {})
        long_term_memories = [
            dict(item)
            for item in list(state.get("long_term_memories") or [])
            if isinstance(item, dict)
        ]
        memory_event = dict(state.get("memory_event") or {})
        resolver_payloads = (
            self._memory_card_payloads(topic_memories, payloads) + payloads
        )
        ordered_messages = [payload_to_message(payload) for payload in payloads]
        query = _last_content(ordered_messages, "user")
        response_language = self.adapter.resolve_language(
            query,
            str(state.get("requested_language") or "zh"),
        )
        has_prior_context = any(
            _message_role(message) in {"user", "assistant"}
            for message in ordered_messages[:-1]
        ) or bool(topic_memories or rolling_summary or prior_working_memory)
        exact_detector = getattr(self.adapter, "is_exact_query", None)
        exact_mode = bool(exact_detector(query)) if callable(exact_detector) else False
        contextual_exact_detector = getattr(
            self.adapter,
            "is_contextual_exact_query",
            None,
        )
        contextual_exact_mode = bool(
            exact_mode
            and callable(contextual_exact_detector)
            and contextual_exact_detector(query, resolver_payloads)
        )
        casual_detector = getattr(self.adapter, "is_casual_query", None)
        casual_mode = bool(casual_detector(query)) if callable(casual_detector) else False
        generic_followup = is_generic_followup(query, has_prior_context)
        clean_followup_query = (
            build_followup_retrieval_query(ordered_messages, query)
            if generic_followup
            else ""
        )
        clean_followup_subject = (
            clean_followup_query.strip()
        )

        # 【流程 1.1】意图分类，优先级从上到下：casual 短路；exact 精确事实题不许继承
        # 历史；generic_followup 走正则识别；其余交给 adapter.route_context 语义路由。
        if casual_mode:
            route = {
                "intent": "casual",
                "use_history": False,
                "use_image_context": False,
                "reason": "short conversational message bypasses domain generation",
                "source": "langgraph_rule",
            }
        elif exact_mode and not contextual_exact_mode:
            route = {
                "intent": "new_topic",
                "use_history": False,
                "use_image_context": False,
                "reason": "exact fact/source question must not inherit stale context",
                "source": "langgraph_rule",
            }
        elif generic_followup:
            route = {
                "intent": "follow_up",
                "use_history": True,
                "use_image_context": self.adapter.has_image_context(resolver_payloads),
                "reason": "generic follow-up requires previous turn",
                "source": "langgraph_rule",
            }
        else:
            route = dict(self.adapter.route_context(query, payloads, response_language) or {})

        intent = str(route.get("intent") or "new_topic")
        if intent not in {"greeting", "casual", "new_topic", "follow_up"}:
            intent = "new_topic"
        if intent == "follow_up" and not has_prior_context:
            logger.info(
                "LangGraph route corrected: follow_up without prior context -> new_topic query=%s source=%s",
                query[:120].replace("\n", " "),
                route.get("source"),
            )
            intent = "new_topic"
            route["reason"] = "no prior context; corrected to new_topic"
        route["intent"] = intent
        route["use_history"] = intent == "follow_up"

        context_resolution: Dict[str, Any] = {}
        # A generic follow-up may skip the extra semantic resolver for latency,
        # but it must never skip retrieval. Its subject is reconstructed from
        # the latest non-generic user turn below.
        generic_followup_fast_path = str(os.getenv("DURIAN_GENERIC_FOLLOWUP_FAST_PATH", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if (
            intent not in {"greeting", "casual"}
            and not (generic_followup and generic_followup_fast_path)
            and (not exact_mode or contextual_exact_mode)
        ):
            # 【流程 1.2】语义上下文消解：从主题卡片判断是否在继续上一话题；
            # 选出 active card 会把 intent 升级为 follow_up（见下方 selected 分支）。
            context_resolution = dict(
                self.adapter.resolve_context(
                    query,
                    resolver_payloads,
                    response_language,
                )
                or {}
            )
        elif generic_followup and generic_followup_fast_path:
            logger.info("LangGraph generic follow-up skipped semantic context resolver: query=%s", query[:120].replace("\n", " "))

        if context_resolution.get("selected"):
            route.update(
                {
                    "intent": "follow_up",
                    "use_history": True,
                    "use_image_context": str(
                        (context_resolution.get("card") or {}).get("source") or ""
                    ).startswith("image"),
                    "reason": (
                        (context_resolution.get("selection") or {}).get("reason")
                        or route.get("reason")
                        or "active context selected"
                    ),
                }
            )

        selected_card = (
            dict(context_resolution.get("card") or {})
            if context_resolution.get("selected")
            else {}
        )
        latest_visible_card: Dict[str, Any] = {}
        for payload in reversed(payloads[:-1]):
            card = payload.get("active_context_card")
            if payload.get("role") == "assistant" and isinstance(card, dict) and card:
                latest_visible_card = dict(card)
                if not latest_visible_card.get("last_answer_summary"):
                    summary = re.sub(
                        r"<!--[\s\S]*?-->",
                        " ",
                        str(payload.get("content") or ""),
                    )
                    summary = re.sub(r"\s+", " ", summary).strip()
                    if summary:
                        latest_visible_card["last_answer_summary"] = summary[:520]
                break
        # 【流程 1.3】补全 resolved_query：把"详细一点"重写成"上一主题：详细一点"，防歧义。
        if generic_followup and clean_followup_subject:
            resolved_query = f"{clean_followup_subject}：{query}".strip()
        elif context_resolution.get("selected"):
            resolved_query = str(
                context_resolution.get("standalone_query") or query
            ).strip()
        elif (
            route["intent"] == "follow_up"
            and latest_visible_card.get("main_subject")
        ):
            resolved_query = (
                f"{latest_visible_card.get('main_subject')}：{query}"
            ).strip()
        elif route["intent"] == "follow_up" and prior_working_memory.get(
            "active_subject"
        ) and len(incoming_payloads) <= 1:
            resolved_query = (
                f"{prior_working_memory.get('active_subject')}：{query}"
            ).strip()
        else:
            resolved_query = str(query or "").strip()

        if generic_followup and clean_followup_subject and not route.get("use_image_context"):
            # Do not put an earlier assistant answer into working memory here.
            # If it had no evidence, doing so turns hallucination into context.
            working_memory = {
                "active_subject": clean_followup_subject[:300],
                "last_user_goal": resolved_query[:500],
                "last_answer_summary": "",
                "source": "grounded_text_followup",
            }
        elif selected_card:
            working_memory = self._working_memory_from_card(
                selected_card,
                resolved_query,
            )
        elif route["intent"] == "follow_up" and latest_visible_card:
            working_memory = self._working_memory_from_card(
                latest_visible_card,
                resolved_query,
            )
        elif route["intent"] == "follow_up" and prior_working_memory:
            working_memory = dict(prior_working_memory)
            working_memory["last_user_goal"] = resolved_query[:500]
        elif route["intent"] in {"greeting", "casual"}:
            working_memory = dict(prior_working_memory)
        else:
            working_memory = self._working_memory_from_card(None, query)

        clarification_needed = bool(
            context_resolution.get("transition") == "clarify"
            and not context_resolution.get("selected")
            and has_prior_context
            and not generic_followup
        )

        # 【流程 1.4】检索查询 ≠ 用户原话：模糊追问会被重构成上一轮稳定的用户问题，
        # 绝不能拿上一轮 AI 回答当检索输入（否则自我强化幻觉，见 build_followup_retrieval_query）。
        if generic_followup and clean_followup_query and not route.get("use_image_context"):
            rag_query = clean_followup_query
        elif context_resolution.get("selected"):
            if generic_followup:
                rag_query = build_context_card_retrieval_query(
                    selected_card,
                    resolved_query,
                )
            else:
                rag_query = resolved_query
        elif route["intent"] == "follow_up":
            if generic_followup and latest_visible_card:
                rag_query = build_context_card_retrieval_query(
                    latest_visible_card,
                    query,
                )
            elif generic_followup:
                rag_query = build_followup_retrieval_query(
                    ordered_messages,
                    query,
                )
            elif latest_visible_card.get("main_subject"):
                rag_query = build_context_card_retrieval_query(
                    latest_visible_card,
                    query,
                )
            elif prior_working_memory.get("active_subject"):
                rag_query = resolved_query
            else:
                rag_query = build_followup_retrieval_query(
                    ordered_messages,
                    query,
                )
        else:
            rag_query = self.adapter.enhance_query(query, response_language)

        memory_instruction = self._build_memory_instruction(
            route_intent=str(route.get("intent") or "new_topic"),
            resolved_query=resolved_query,
            original_query=query,
            working_memory=working_memory,
            rolling_summary="" if generic_followup and not route.get("use_image_context") else rolling_summary,
            long_term_memories=long_term_memories,
            memory_event=memory_event,
        )
        system_messages: List[Dict[str, Any]] = []
        if (
            context_resolution.get("selected")
            and context_resolution.get("system_instruction")
        ):
            system_messages.append(
                {
                    "role": "system",
                    "content": str(
                        context_resolution.get("system_instruction") or ""
                    ),
                }
            )
        if memory_instruction:
            system_messages.append(
                {"role": "system", "content": memory_instruction}
            )

        if route["intent"] == "follow_up":
            if generic_followup and not route.get("use_image_context"):
                # Keep user intent, but omit ungrounded assistant prose. Valid
                # assistant turns can be retained because their references are
                # now carried in the history payload.
                followup_payloads: List[Dict[str, Any]] = []
                for payload in payloads[-self.max_prompt_messages :]:
                    if payload.get("role") != "assistant":
                        followup_payloads.append(payload)
                        continue
                    prior_evidence = payload.get("evidence")
                    prior_quality = str(payload.get("evidence_quality") or "none")
                    if isinstance(prior_evidence, list) and prior_evidence and prior_quality != "none":
                        followup_payloads.append(payload)
                context_messages = system_messages + followup_payloads
            else:
                context_messages = system_messages + payloads[-self.max_prompt_messages :]
        elif route["intent"] in {"greeting", "casual"}:
            context_messages = [payloads[-1]] if payloads else []
        else:
            context_messages = system_messages + (
                [payloads[-1]] if payloads else []
            )

        # One factual path: every substantive Durian turn goes through
        # RAGFlow. Context routing may rewrite the query, but it may not bypass
        # retrieval. Only conversational fast paths and clarification skip it.
        # 【流程 1.5】只有 greeting/casual/待澄清可跳过检索——一切实质问题必须走 RAG。
        rag_allowed = bool(
            route.get("intent") not in {"greeting", "casual"}
            and not clarification_needed
        )
        if generic_followup and generic_followup_fast_path:
            logger.info("LangGraph generic follow-up uses clean retrieval query: query=%s rag_query=%s", query[:120].replace("\n", " "), rag_query[:180].replace("\n", " "))

        adjusted_generation = dict(state.get("generation") or {})

        self.adapter.log_route(
            query=query,
            language=response_language,
            route=route,
            rag_query=rag_query,
            resolved_query=resolved_query,
            history_count=max(0, len(payloads) - 1),
        )
        logger.info(
            "LangGraph route timing: total=%.3fs intent=%s history=%s query=%s",
            time.perf_counter() - route_started,
            route.get("intent"),
            route.get("use_history"),
            query[:120].replace("\n", " "),
        )
        return {
            "user_query": query,
            "response_language": response_language,
            "route_decision": route,
            "generation": adjusted_generation,
            "context_resolution": context_resolution,
            "context_messages": context_messages,
            "rag_query": rag_query,
            "resolved_query": resolved_query,
            "rag_allowed": rag_allowed,
            "clarification_needed": clarification_needed,
            "evidence": [],
            "evidence_quality": "none",
            "evidence_text": "",
            "final_messages": [],
            "answer": "",
            "active_context_card": None,
            "working_memory": working_memory,
            "topic_memories": topic_memories,
            "rolling_summary": rolling_summary,
            "long_term_memories": long_term_memories,
            "memory_event": memory_event,
            "user_id": str(state.get("user_id") or ""),
            "turn_count": int(state.get("turn_count") or 0),
            "exact_mode": exact_mode,
            "error": "",
        }

    def _retrieve_node(self, state: DurianGraphState) -> Dict[str, Any]:
        """【流程 2】检索：很薄的一层，领域逻辑全在 adapter；产出 evidence / evidence_quality。"""
        started = time.perf_counter()
        route = state.get("route_decision") or {}
        query = str(state.get("rag_query") or state.get("user_query") or "")
        try:
            get_stream_writer()({"type": "status", "phase": "retrieval_start", "message": "正在检索参考资料..."})
        except Exception:
            pass
        skip_reason = ""
        if route.get("intent") in {"greeting", "casual"}:
            skip_reason = f"intent={route.get('intent')}"
        elif not bool(state.get("rag_allowed", state.get("use_rag", True))):
            skip_reason = "rag_not_allowed"
        elif not self.adapter.rag_available():
            skip_reason = "rag_unavailable"
        if skip_reason:
            logger.info(
                "LangGraph retrieve skipped: reason=%s elapsed=%.3fs query=%s",
                skip_reason,
                time.perf_counter() - started,
                query[:120].replace("\n", " "),
            )
            return {"evidence": [], "evidence_quality": "none", "evidence_text": ""}

        evidence, quality, evidence_text = self.adapter.retrieve(
            query,
            str(state.get("response_language") or "zh"),
        )
        logger.info(
            "LangGraph retrieve timing: total=%.3fs evidence=%d quality=%s query=%s",
            time.perf_counter() - started,
            len(evidence or []),
            str(quality or "none"),
            query[:120].replace("\n", " "),
        )
        return {
            "evidence": list(evidence or []),
            "evidence_quality": str(quality or "none"),
            "evidence_text": str(evidence_text or ""),
        }

    def _compose_node(self, state: DurianGraphState) -> Dict[str, Any]:
        """【流程 3】组装 prompt：裁剪后的历史 + 检索证据 → 最终 LLM 消息列表（≤ max_prompt_messages 条）。"""
        if (state.get("route_decision") or {}).get("intent") in {"greeting", "casual"}:
            return {"final_messages": []}
        if bool(state.get("clarification_needed")):
            return {"final_messages": []}
        final_messages = self.adapter.build_final_messages(
            list(state.get("context_messages") or []),
            str(state.get("evidence_text") or ""),
            self.max_prompt_messages,
            str(state.get("response_language") or "zh"),
        )
        return {"final_messages": list(final_messages or [])}

    def _generate_node(self, state: DurianGraphState) -> Dict[str, Any]:
        """【流程 4】生成：一条优先级递降的"逃生梯"，多数分支不调 LLM 直接给出答案。"""
        writer = get_stream_writer()
        generation_started = time.perf_counter()
        first_content_logged = False
        route = state.get("route_decision") or {}
        response_language = str(state.get("response_language") or "zh")
        evidence = list(state.get("evidence") or [])
        evidence_quality = str(state.get("evidence_quality") or "none")
        client_evidence = self.adapter.prepare_evidence(evidence, response_language)
        source_missing = bool(
            route.get("intent") not in {"greeting", "casual"}
            and not bool(state.get("clarification_needed"))
            and not evidence
        )
        has_image_context = bool(
            route.get("use_image_context")
            or str((state.get("working_memory") or {}).get("source") or "").startswith("image")
        )
        grounded_followup_answer: Optional[str] = None
        grounded_diagnosis_answer: Optional[str] = None
        soft_no_source_answer: Optional[str] = None
        if source_missing and not has_image_context:
            soft_answer_builder = getattr(
                self.adapter,
                "soft_no_source_answer",
                None,
            )
            if callable(soft_answer_builder):
                soft_no_source_answer = soft_answer_builder(
                    str(state.get("user_query") or ""),
                    response_language,
                )
        diagnosis_extractor = getattr(
            self.adapter,
            "extract_grounded_diagnosis_answer",
            None,
        )
        if evidence and callable(diagnosis_extractor):
            grounded_diagnosis_answer = diagnosis_extractor(
                str(state.get("user_query") or ""),
                evidence,
                response_language,
            )
        if (
            route.get("intent") == "follow_up"
            and is_generic_followup(str(state.get("user_query") or ""), True)
            and evidence
        ):
            followup_extractor = getattr(
                self.adapter,
                "extract_grounded_followup_answer",
                None,
            )
            if callable(followup_extractor):
                grounded_followup_answer = followup_extractor(
                    evidence,
                    response_language,
                )

        writer(
            {
                "type": "metadata",
                "phase": "context_ready",
                "route": route.get("intent"),
                "history_used": bool(route.get("use_history")),
                "answer_mode": (
                    "clarification"
                    if bool(state.get("clarification_needed"))
                    else (
                        "exact_extraction"
                        if bool(state.get("exact_mode"))
                        else (
                            "soft_no_source_generation"
                            if source_missing
                            else "grounded_generation"
                        )
                    )
                ),
                "evidence_quality": evidence_quality,
                "evidence": client_evidence,
                "resolved_query": str(
                    state.get("resolved_query")
                    or state.get("user_query")
                    or ""
                ),
                "memory_used": len(
                    list(state.get("long_term_memories") or [])
                ),
                "working_memory_subject": str(
                    (state.get("working_memory") or {}).get(
                        "active_subject"
                    )
                    or ""
                ),
            }
        )
        writer({"type": "status", "phase": "generation_start", "message": "正在生成回答..."})

        try:
            # 逃生梯从上到下：greeting/casual 固定话术 → 澄清模板 → exact 从证据抄答案
            # （绝不靠模型记忆）→ 证据抽取 → 免责声明+通用生成 → 兜底才是真正的流式 LLM。
            if route.get("intent") == "greeting":
                full_response = self.adapter.greeting_reply(response_language)
                writer({"type": "content", "content": full_response})
            elif route.get("intent") == "casual":
                casual_responder = getattr(self.adapter, "casual_reply", None)
                full_response = (
                    casual_responder(
                        str(state.get("user_query") or ""),
                        response_language,
                    )
                    if callable(casual_responder)
                    else self.adapter.greeting_reply(response_language)
                )
                writer({"type": "content", "content": full_response})
            elif bool(state.get("clarification_needed")):
                full_response = self._clarification_reply(
                    response_language,
                    dict(state.get("working_memory") or {}),
                )
                writer({"type": "content", "content": full_response})
            elif bool(state.get("exact_mode")):
                # Exact facts must be copied from the retrieved source window,
                # never regenerated from model memory. This covers both
                # name -> identifier (``X 官方编号``) and identifier -> name
                # (``Dxxx 是什么品种``), including entities not present in a
                # hard-coded alias table.
                exact_extractor = getattr(
                    self.adapter,
                    "extract_exact_answer",
                    None,
                )
                exact_query = str(
                    state.get("rag_query")
                    or state.get("resolved_query")
                    or state.get("user_query")
                    or ""
                )
                full_response = (
                    exact_extractor(
                        exact_query,
                        evidence,
                        response_language,
                    )
                    if callable(exact_extractor)
                    else None
                )
                if not full_response:
                    exact_not_found = getattr(
                        self.adapter,
                        "exact_not_found",
                        None,
                    )
                    full_response = (
                        exact_not_found(response_language)
                        if callable(exact_not_found)
                        else "No reliable source passage was found for this exact question."
                    )
                logger.info(
                    "LangGraph exact extraction: found=%s evidence=%d query=%s",
                    bool(evidence and full_response),
                    len(evidence),
                    exact_query[:120].replace("\n", " "),
                )
                writer({"type": "content", "content": str(full_response)})
            elif grounded_diagnosis_answer:
                full_response = grounded_diagnosis_answer
                writer({"type": "content", "content": full_response})
            elif grounded_followup_answer:
                full_response = grounded_followup_answer
                writer({"type": "content", "content": full_response})
            elif soft_no_source_answer:
                caution = {
                    "zh": "参考资料不足，以下内容仅作一般性判断：\n\n",
                    "en": "Reference material is insufficient; the following is only a general assessment:\n\n",
                    "ms": "Bahan rujukan tidak mencukupi; berikut hanyalah penilaian umum:\n\n",
                    "th": "ข้อมูลอ้างอิงไม่เพียงพอ ข้อความต่อไปนี้เป็นเพียงการประเมินทั่วไป:\n\n",
                }.get(response_language, "参考资料不足，以下内容仅作一般性判断：\n\n")
                full_response = caution + soft_no_source_answer
                writer({"type": "content", "content": full_response})
            elif (
                source_missing
                and not has_image_context
                and route.get("intent") == "follow_up"
                and is_generic_followup(str(state.get("user_query") or ""), True)
            ):
                full_response = {
                    "zh": "没有检索到能够支持上一轮内容的可靠资料，因此我不能继续扩写，避免把未经证实的信息当成事实。你可以补充更具体的品种名称、编号或资料来源，我再重新检索。",
                    "en": "I could not retrieve reliable sources supporting the previous answer, so I will not expand it as fact. Please provide a more specific variety name, code, or source and I will search again.",
                    "ms": "Saya tidak menemui sumber yang boleh dipercayai untuk menyokong jawapan sebelumnya, jadi saya tidak akan menghuraikannya sebagai fakta. Sila berikan nama varieti, kod atau sumber yang lebih khusus.",
                    "th": "ฉันไม่พบแหล่งข้อมูลที่น่าเชื่อถือซึ่งรองรับคำตอบก่อนหน้า จึงจะไม่ขยายความเป็นข้อเท็จจริง โปรดระบุชื่อพันธุ์ รหัส หรือแหล่งข้อมูลให้ชัดเจนขึ้น",
                }.get(response_language, "没有检索到能够支持上一轮内容的可靠资料，因此我不能继续扩写。")
                writer({"type": "content", "content": full_response})
            else:
                parts: List[str] = []
                if source_missing and not has_image_context:
                    caution = {
                        "zh": "参考资料不足，以下内容仅作一般性判断：\n\n",
                        "en": "Reference material is insufficient; the following is only a general assessment:\n\n",
                        "ms": "Bahan rujukan tidak mencukupi; berikut hanyalah penilaian umum:\n\n",
                        "th": "ข้อมูลอ้างอิงไม่เพียงพอ ข้อความต่อไปนี้เป็นเพียงการประเมินทั่วไป:\n\n",
                    }.get(response_language, "参考资料不足，以下内容仅作一般性判断：\n\n")
                    parts.append(caution)
                    writer({"type": "content", "content": caution})
                chunks = self.adapter.stream_generate(
                    list(state.get("final_messages") or []),
                    dict(state.get("generation") or {}),
                    response_language,
                )
                # 每个 token chunk 经 get_stream_writer 发出，穿透图直达 stream_turn 再推给前端 SSE。
                for chunk in chunks:
                    if chunk:
                        text = str(chunk)
                        if not first_content_logged:
                            first_content_logged = True
                            logger.info(
                                "LangGraph first content latency: %.3fs route=%s evidence=%d quality=%s query=%s",
                                time.perf_counter() - generation_started,
                                route.get("intent"),
                                len(evidence),
                                evidence_quality,
                                str(state.get("user_query") or "")[:120].replace("\n", " "),
                            )
                        parts.append(text)
                        writer({"type": "content", "content": text})
                full_response = "".join(parts)

            clean_response = self.adapter.clean_output(full_response)
            context_resolution = state.get("context_resolution") or {}
            if (
                route.get("intent") in {"greeting", "casual"}
                or bool(state.get("clarification_needed"))
            ):
                active_context_card = None
            else:
                # 生成主题卡片（答案摘要）：下一轮 follow_up 的上下文消解就靠它。
                active_context_card = self.adapter.build_context_card(
                    str(
                        state.get("resolved_query")
                        or state.get("user_query")
                        or ""
                    ),
                    clean_response,
                    response_language,
                    base_card=(
                        context_resolution.get("card")
                        if context_resolution.get("selected")
                        else None
                    ),
                    source=(
                        "text_followup"
                        if route.get("intent") == "follow_up"
                        else "text"
                    ),
                )
            topic_memories = self._merge_topic_memories(
                list(state.get("topic_memories") or []),
                active_context_card,
            )
            rolling_summary = self._build_rolling_summary(topic_memories)
            if active_context_card:
                working_memory = self._working_memory_from_card(
                    active_context_card,
                    str(
                        state.get("resolved_query")
                        or state.get("user_query")
                        or ""
                    ),
                    clean_response,
                )
            else:
                working_memory = dict(state.get("working_memory") or {})
            assistant_payload = {
                "role": "assistant",
                "content": clean_response,
                "active_context_card": active_context_card,
                "evidence": client_evidence,
                "evidence_quality": evidence_quality,
            }
            assistant_message = payload_to_message(assistant_payload)
            tokens_generated = self.adapter.approx_tokens(clean_response)
            logger.info(
                "LangGraph generation timing: total=%.3fs chars=%d tokens=%d route=%s evidence=%d quality=%s query=%s",
                time.perf_counter() - generation_started,
                len(clean_response),
                tokens_generated,
                route.get("intent"),
                len(evidence),
                evidence_quality,
                str(state.get("user_query") or "")[:120].replace("\n", " "),
            )
            writer(
                {
                    "type": "metadata",
                    "phase": "complete",
                    "tokens_generated": tokens_generated,
                    "active_context_card": active_context_card,
                    "memory_event": dict(state.get("memory_event") or {}),
                    "memory_used": len(
                        list(state.get("long_term_memories") or [])
                    ),
                }
            )
            # 返回 assistant 消息 → add_messages reducer 把它追加进 checkpoint（state 的核心合并规则）。
            return {
                "messages": [assistant_message],
                "answer": clean_response,
                "active_context_card": active_context_card,
                "working_memory": working_memory,
                "topic_memories": topic_memories,
                "rolling_summary": rolling_summary,
                "turn_count": int(state.get("turn_count") or 0) + 1,
                "error": "",
            }
        except Exception as exc:
            logger.exception("LangGraph generation failed after %.3fs", time.perf_counter() - generation_started)
            writer({"type": "error", "error": str(exc)})
            return {"answer": "", "active_context_card": None, "error": str(exc)}

    def _trim_memory_node(self, state: DurianGraphState) -> Dict[str, Any]:
        """【流程 5】裁剪：消息超过上限时用 RemoveMessage 删最旧的（add_messages reducer 的删除能力）。"""
        messages = list(state.get("messages") or [])
        if len(messages) <= self.max_memory_messages:
            return {}
        old_messages = messages[: -self.max_memory_messages]
        removals = [
            RemoveMessage(id=message.id)
            for message in old_messages
            if getattr(message, "id", None)
        ]
        if removals:
            logger.info(
                "LangGraph thread memory trimmed: %d -> %d",
                len(messages),
                self.max_memory_messages,
            )
        return {"messages": removals}

    def stream_turn(
        self,
        *,
        thread_id: str,
        messages: List[Dict[str, Any]],
        requested_language: str,
        use_rag: bool,
        generation: Dict[str, Any],
    ) -> Iterator[Dict[str, Any]]:
        # 【流程 0】入口：FastAPI 每收到一轮对话调用一次。
        # thread_id 是记忆的钥匙——get_state 按 thread_id 从 checkpoint 读出上一轮的完整 state。
        config = {"configurable": {"thread_id": str(thread_id)}}
        snapshot = self.graph.get_state(config)
        snapshot_values = dict(snapshot.values or {})
        existing_messages = list(snapshot_values.get("messages") or [])
        existing_by_content: Dict[tuple[str, str], Dict[str, Any]] = {}
        for existing in existing_messages:
            payload = message_to_payload(existing)
            key = (str(payload.get("role") or ""), str(payload.get("content") or ""))
            existing_by_content[key] = payload

        # 【流程 0.2】消息对账：前端会重放最近几轮可见历史，用 (role, content) 对齐
        # checkpoint 里的旧消息并复用其 id，让 add_messages 做原地更新而非重复追加。
        normalized_payloads: List[Dict[str, Any]] = []
        for index, original in enumerate(messages):
            if not original.get("content"):
                continue
            payload = dict(original)
            key = (str(payload.get("role") or ""), str(payload.get("content") or ""))
            prior = existing_by_content.get(key)
            is_current_user = (
                index == len(messages) - 1
                and str(payload.get("role") or "user") == "user"
            )
            # Replayed frontend history should update, not duplicate, the
            # checkpointed message. The final user item is a new turn unless
            # the checkpoint already ends with the same pending user message.
            can_reuse_current = bool(
                existing_messages
                and _message_role(existing_messages[-1]) == "user"
                and str(getattr(existing_messages[-1], "content", "") or "")
                == str(payload.get("content") or "")
            )
            if prior and (not is_current_user or can_reuse_current):
                payload["id"] = prior.get("id")
                if payload.get("image_url") is None:
                    payload["image_url"] = prior.get("image_url")
                if payload.get("active_context_card") is None:
                    payload["active_context_card"] = prior.get("active_context_card")
            elif is_current_user and prior:
                seed = (
                    f"{thread_id}|{getattr(existing_messages[-1], 'id', '')}|"
                    f"{payload.get('content') or ''}"
                )
                payload["id"] = f"durian-turn-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:32]}"
            normalized_payloads.append(payload)

        incoming = [payload_to_message(payload) for payload in normalized_payloads]
        if not incoming:
            raise ValueError("messages must contain at least one non-empty message")
        latest_user_query = ""
        for payload in reversed(normalized_payloads):
            if str(payload.get("role") or "user") == "user":
                latest_user_query = str(payload.get("content") or "").strip()
                break
        user_id = self._user_id_from_thread(thread_id)
        # 【流程 0.3】写长期记忆：从用户话中抽显式陈述（"请记住…/我在…"）存入
        # 独立的 user_memories.sqlite；含问号的句子和密码类敏感词会被拒收。
        memory_event = self.user_memory_store.apply_user_message(
            user_id,
            latest_user_query,
        )
        # 【流程 0.4】读长期记忆：按词重叠打分取回最相关的 6 条，随 graph_input 进图。
        long_term_memories = self.user_memory_store.retrieve(
            user_id,
            latest_user_query,
            max_items=6,
        )
        graph_input: DurianGraphState = {
            "messages": incoming,
            "turn_messages": normalized_payloads,
            "requested_language": requested_language or "zh",
            "use_rag": bool(use_rag),
            "generation": dict(generation or {}),
            "user_id": user_id,
            "memory_event": memory_event,
            "long_term_memories": long_term_memories,
        }
        stream_started = time.perf_counter()
        event_count = 0
        completed = False
        try:
            # 【流程 0.5】驱动图执行：route→retrieve→compose→generate→trim_memory 线性跑完。
            # stream_mode="custom" 只透传节点内 get_stream_writer() 发出的事件（状态提示/流式 token）。
            for event in self.graph.stream(
                graph_input,
                config=config,
                stream_mode="custom",
            ):
                if isinstance(event, dict):
                    event_count += 1
                    yield event
            completed = True
            # 【流程收尾】图跑完后再次 get_state 取最终 state（已被 SqliteSaver 落盘），
            # 发 final 事件给前端；下一轮 get_state 读到的就是这份 state，记忆闭环。
            final_snapshot = self.graph.get_state(config)
            final_values = dict(final_snapshot.values or {})
            answer = str(final_values.get("answer") or "")
            active_context_card = final_values.get("active_context_card")
            yield {
                "type": "final",
                "response": answer,
                "active_context_card": active_context_card if isinstance(active_context_card, dict) else None,
                "evidence_quality": final_values.get("evidence_quality") or "none",
                "evidence": final_values.get("evidence") or [],
            }
            event_count += 1
        finally:
            logger.info(
                "LangGraph turn timing: total=%.3fs events=%d thread=%s query=%s",
                time.perf_counter() - stream_started,
                event_count,
                str(thread_id)[:80],
                latest_user_query[:120].replace("\n", " "),
            )

    def record_external_turn(
        self,
        *,
        thread_id: str,
        user_message: Dict[str, Any],
        assistant_message: Dict[str, Any],
    ) -> None:
        """Persist a completed non-chat turn, such as image analysis."""
        payloads = [dict(user_message), dict(assistant_message)]
        incoming = [payload_to_message(payload) for payload in payloads]
        config = {"configurable": {"thread_id": str(thread_id)}}
        snapshot = self.graph.get_state(config)
        values = dict(snapshot.values or {})
        card = assistant_message.get("active_context_card")
        topic_memories = self._merge_topic_memories(
            list(values.get("topic_memories") or []),
            card if isinstance(card, dict) else None,
        )
        working_memory = (
            self._working_memory_from_card(
                card,
                str(user_message.get("content") or ""),
                str(assistant_message.get("content") or ""),
            )
            if isinstance(card, dict) and card
            else dict(values.get("working_memory") or {})
        )
        user_id = self._user_id_from_thread(thread_id)
        user_query = str(user_message.get("content") or "")
        memory_event = self.user_memory_store.apply_user_message(
            user_id,
            user_query,
        )
        self.graph.update_state(
            config,
            {
                "messages": incoming,
                "turn_messages": payloads,
                "working_memory": working_memory,
                "topic_memories": topic_memories,
                "rolling_summary": self._build_rolling_summary(
                    topic_memories
                ),
                "user_id": user_id,
                "memory_event": memory_event,
                "long_term_memories": self.user_memory_store.retrieve(
                    user_id,
                    user_query,
                    max_items=6,
                ),
                "turn_count": int(values.get("turn_count") or 0) + 1,
            },
        )
        logger.info(
            "LangGraph external turn recorded: thread=%s source=%s",
            thread_id,
            str(
                (assistant_message.get("active_context_card") or {}).get("source")
                or "external"
            ),
        )

    def get_thread_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        snapshot = self.graph.get_state(
            {"configurable": {"thread_id": str(thread_id)}}
        )
        return [
            message_to_payload(message)
            for message in list((snapshot.values or {}).get("messages") or [])
        ]

    def get_thread_memory(self, thread_id: str) -> Dict[str, Any]:
        snapshot = self.graph.get_state(
            {"configurable": {"thread_id": str(thread_id)}}
        )
        values = dict(snapshot.values or {})
        return {
            "working_memory": dict(values.get("working_memory") or {}),
            "topic_memories": [
                dict(item)
                for item in list(values.get("topic_memories") or [])
                if isinstance(item, dict)
            ],
            "rolling_summary": str(values.get("rolling_summary") or ""),
            "turn_count": int(values.get("turn_count") or 0),
        }

    def get_user_memories(
        self,
        user_id: str,
        query: str = "",
    ) -> List[Dict[str, Any]]:
        return self.user_memory_store.retrieve(
            str(user_id),
            str(query or ""),
            max_items=20,
        )

    def delete_thread(self, thread_id: str) -> None:
        delete = getattr(self._saver, "delete_thread", None)
        if callable(delete):
            delete(str(thread_id))

    def close(self) -> None:
        if self.user_memory_store is not None:
            self.user_memory_store.close()
            self.user_memory_store = None
        if self._saver_context is not None:
            self._saver_context.__exit__(None, None, None)
            self._saver_context = None
