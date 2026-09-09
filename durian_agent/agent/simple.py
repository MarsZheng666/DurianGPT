"""SimpleAgent（架构文档 §14，任务 #35）：SIMPLE 路径的普通问答。

§14 定位：普通解释类问题（什么是坐果/为何修剪/滴灌喷灌区别/会话内总结）
直接 LLM 回答，不强制走 RAG 闭环；「模型认为需要知识时可调用 RAG」
体现为可选的 rag_fn 注入——检索服务自带证据充分性判断（#29），
证据充分才注入上下文，否则纯 LLM 直答。

不做的事（对齐 §1 设计原则「简单问题不强制进入复杂 Agent 流程」）：
- 不做多工具编排（那是 COMPLEX/ReAct 路径）；
- 不做 Evidence Retry 循环（高风险问题的强制闭环是 MUST_RAG 路径）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from durian_agent.llm import LLMProvider

SIMPLE_SYSTEM_PROMPT = """You are a helpful assistant for a durian plantation team.
Answer the user's question directly and concisely in the user's language
(Chinese, English, Thai, or Malay).

If reference evidence is provided below, use it when relevant; you may still
answer from general knowledge for common-sense questions.

Do NOT invent specific pesticide dosages, legal requirements, or agricultural
thresholds — if the question needs those and evidence is missing, say you are
not sure and suggest asking with more specifics.
"""

_RAG_PROMPT_HEADER = "Reference evidence (may or may not be relevant):\n"


class SimpleAgent:
    """SIMPLE 路径执行器。answer() 可直接作为 LangGraph 节点的逻辑内核。"""

    def __init__(self, llm: LLMProvider):
        if llm is None:
            raise ValueError("SimpleAgent 需要 LLM（SIMPLE 路径的定义就是 LLM 直答）")
        self.llm = llm

    def answer(
        self,
        query: str,
        rag_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """回答 SIMPLE 类问题。

        rag_fn: 可选的检索函数（query → {"documents": [...], "evidence_sufficient": bool}，
        即 §60 /api/rag/search 契约）。证据充分才注入；检索服务自身的
        Evidence Check 决定「需要知识」与否。
        """
        evidence_docs: List[Dict[str, Any]] = []
        if rag_fn is not None:
            try:
                result = rag_fn(query) or {}
                if result.get("evidence_sufficient"):
                    evidence_docs = list(result.get("documents") or [])[:3]
            except Exception:
                evidence_docs = []  # SIMPLE 路径检索失败不阻塞直答

        system = SIMPLE_SYSTEM_PROMPT
        if evidence_docs:
            evidence_text = "\n".join(
                f"- [{doc.get('document_id', '?')}] {doc.get('text', '')}"
                for doc in evidence_docs
            )
            system = f"{system}\n\n{_RAG_PROMPT_HEADER}{evidence_text}"

        answer = self.llm.complete(system, query)

        return {
            "answer": answer,
            "used_rag": bool(evidence_docs),
            "sources": [
                {
                    "document_id": doc.get("document_id", "?"),
                    "chunk_id": doc.get("chunk_id", "?"),
                    "title": doc.get("metadata", {}).get("title", ""),
                    "section": doc.get("metadata", {}).get("section", ""),
                }
                for doc in evidence_docs
            ],
        }
