"""Final Generation（架构文档 §30/§31/§48，任务 #31/#33）。

- §48 约束：只根据证据回答专业结论；证据不足明确声明；
  不得用模型先验编造剂量/农药/阈值/法规/SOP 要求；
- §31 引用：回答中标注 [chunk_id]，cited_chunk_ids 与提供的证据
  一一对应——未被引用的证据不进 sources。
"""

from __future__ import annotations

import re
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

#: 证据不足时的固定声明（§48：明确声明，不硬答）
ABSTAIN_ANSWER = "当前知识库没有提供足够的证据来回答这个问题。请补充更多细节（品种、症状、园区环境），我会重新检索。"

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-]+)\]")


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
    """证据约束生成：system 约束 + 证据块 + 用户问题 → 回答文本。

    无证据时返回固定声明（不调用 LLM，不硬答）。
    """
    if not docs:
        return ABSTAIN_ANSWER
    system = f"{GROUNDED_SYSTEM_PROMPT}\n\nEvidence:\n{build_evidence_block(docs)}"
    return llm.complete(system, query)


def parse_citations(answer: str, docs: Sequence[Dict[str, Any]]) -> List[str]:
    """从回答中解析 [chunk_id] 引用，只保留确实提供的证据（§31 一一对应）。"""
    provided = {str(d.get("chunk_id")) for d in docs}
    cited = []
    for marker in _CITATION_RE.findall(answer or ""):
        if marker in provided and marker not in cited:
            cited.append(marker)
    return cited


def generate_with_citations(
    llm, query: str, docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """生成 + 引用解析：{"answer", "cited_chunk_ids"}（#33）。"""
    answer = generate_grounded_answer(llm, query, docs)
    return {"answer": answer, "cited_chunk_ids": parse_citations(answer, docs)}
