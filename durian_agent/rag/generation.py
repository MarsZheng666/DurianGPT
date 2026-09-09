"""Final Generation（架构文档 §30/§48）——阶段一内核。

#32（阶段二）做完整验收（诱导补剂量被拒绝等）；
本模块提供图拓扑（#1）answer 节点所需的证据约束生成：
只根据提供的证据回答专业结论，证据不足时明确声明。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

GROUNDED_SYSTEM_PROMPT = """Answer agricultural professional claims only from the supplied evidence.

If evidence is insufficient:
state that the current knowledge base does not provide enough evidence.

Do not use model prior knowledge to invent:
- dosage
- pesticide recommendations
- thresholds
- legal requirements
- SOP requirements

Answer in the user's language (Chinese, English, Thai, or Malay).
Cite the chunk_id of the evidence you relied on, like [CHUNK_123].
"""


def build_evidence_block(docs: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for doc in docs:
        record = doc.get("record", {})
        lines.append(
            f"[{doc.get('chunk_id', '?')}] ({record.get('document_id', '?')}) "
            f"{doc.get('text', '')}"
        )
    return "\n".join(lines)


def generate_grounded_answer(llm, query: str, docs: List[Dict[str, Any]]) -> str:
    """证据约束生成：system 约束 + 证据块 + 用户问题 → 回答文本。"""
    system = f"{GROUNDED_SYSTEM_PROMPT}\n\nEvidence:\n{build_evidence_block(docs)}"
    return llm.complete(system, query)
