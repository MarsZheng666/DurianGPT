"""AgricultureRagTool（架构文档 §16/§17，任务 #36）。

把 RAG 闭环（四路召回 → RRF → 可选重排 → 证据判断）封装为 ReAct 工具：
- Observation = Top 证据文本（带 chunk_id 标注，供最终回答引用）；
- 证据不足时如实告知（ReAct 可换关键词再查，但不得无证据下专业结论——
  ToolPolicyGate #38 在 answer 前强制拦截）；
- 证据回写 ctx.state（reranked_docs/evidence_sufficient），
  供图 answer 节点生成带引用的最终回答。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from durian_agent.rag.evidence import check_evidence
from durian_agent.rag.rerank import rerank_documents
from durian_agent.rag.rrf import weighted_rrf
from durian_agent.tools.base import ToolContext, ToolRegistry, ToolSpec

SPEC = ToolSpec(
    name="agriculture_rag",
    description="检索农业知识库（品种/病害/施肥/灌溉/SOP 等专业资料）。"
                "回答专业农艺结论前必须先用它取证。",
    args_hint='{"query": "猫山王开花期施肥"}',
)


def register(registry: ToolRegistry, retriever, reranker=None,
             semantic: Optional[Dict[str, Any]] = None) -> None:
    """retriever: FourWayRetriever；reranker: 可选 RerankFn。"""
    registry.register(
        SPEC, lambda args, ctx: _execute(args, ctx, retriever, reranker, semantic))


def _execute(args, ctx: ToolContext, retriever, reranker, semantic) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "参数错误: query 必填"
    if retriever is None:
        return "知识库检索不可用（未配置检索器）"

    result = retriever.recall(query, top_k_each=10) or {"hits": {}, "rankings": {}}
    fused = weighted_rrf(result["rankings"])
    by_id = {d["chunk_id"]: d
             for hits in result["hits"].values() for d in hits}
    docs: List[Dict[str, Any]] = []
    for item in fused[:8]:
        doc = dict(by_id.get(item["chunk_id"], {}))
        doc["chunk_id"] = item["chunk_id"]
        doc["rrf_rank"] = item["rrf_rank"]
        docs.append(doc)

    if reranker is not None:
        docs = rerank_documents(query, docs, reranker, top_k=5)

    verdict = check_evidence(query, semantic, docs)

    # 证据回写图状态（answer 节点生成带引用回答用）
    if ctx.state is not None:
        ctx.state["reranked_docs"] = docs
        ctx.state["evidence_sufficient"] = verdict["sufficient"]

    if not docs:
        return "知识库未检索到相关资料。请换更具体的关键词（品种/症状/生育期）。"
    lines = [f"[{d['chunk_id']}] {str(d.get('text', ''))[:300]}" for d in docs[:3]]
    note = "" if verdict["sufficient"] else (
        "（证据可能不足：" + "；".join(verdict["reasons"][:2]) + "）")
    return "\n".join(lines) + note


def search_for_api(retriever, reranker, query: str,
                   semantic: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """供 §60 /api/rag/search（#11，阶段五）复用的独立检索入口。"""
    ctx = ToolContext(state={})
    observation = _execute({"query": query}, ctx, retriever, reranker, semantic)
    docs = ctx.state.get("reranked_docs", [])
    return {
        "documents": [
            {"chunk_id": d["chunk_id"], "document_id": d.get("record", {}).get(
                "document_id", d.get("document_id", "")),
             "text": d.get("text", ""), "metadata": d.get("record", {})}
            for d in docs
        ],
        "evidence_sufficient": ctx.state.get("evidence_sufficient", False),
        "observation": observation,
    }
