"""LangGraph 图拓扑（架构文档 §3/§44/§45，任务 #1）。

§44 的 **19 个节点**全量注册（拆分任务时误记为 18，以文档为准），
§45 的全部边分支接通：

    START → contextInit → normalize → semanticParse → route
    route: MUST_RAG → ragQueryBuild / SIMPLE → simpleAgent / COMPLEX → reactAgent
    ragQueryBuild → retrieve → rrf → rerank → evidenceCheck
    evidenceCheck: sufficient → answer
                   insufficient & retrievalCount < max → rewriteQuery → retrieve
                   insufficient & max reached → fallback（诚实拒答）
    reactAgent → toolPolicy →（有 tool call）permissionCheck → toolNode → reactAgent
                          →（无 tool call）answer
    simpleAgent / answer / fallback → memoryCompress → END

各节点的阶段成熟度（诚实标注，后续任务逐个替换内核）：
- 阶段一已完整：contextInit / normalize / semanticParse / route /
  ragQueryBuild / retrieve（四路）/ rrf / simpleAgent / answer / fallback
- 占位实现（拓扑先通、内核后补）：
  rerank        → 透传 RRF 序（#28 Phase2：Cross Encoder）
  evidenceCheck → 有证据即充分（#30 Phase2：五项判断）
  rewriteQuery  → 用扩展查询重试（#28 Phase2：规则+LLM 两层改写）
  reactAgent    → 降级提示（#35/#39 Phase3：ReAct 循环）
  toolPolicy / toolNode / permissionCheck / confirmation → no-op（Phase3/5）
  memoryCompress → no-op（#53 Phase4：滑动窗口与摘要）
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph import add_messages  # noqa: F401  (re-export convenience)

from durian_agent.agent.simple import SimpleAgent
from durian_agent.llm import LLMProvider
from durian_agent.normalize import normalize_input
from durian_agent.rag.recall import FourWayRetriever, build_queries
from durian_agent.rag.rrf import weighted_rrf
from durian_agent.router import route as route_decision
from durian_agent.semantic.parse import SemanticParser
from durian_agent.state import AgentState, DegradeLevel, init_state
from durian_agent.thai import normalize_thai_input

_THAI_RE = re.compile(r"[\u0e00-\u0e7f]")

#: Evidence Retry 上限（§29：2～3 次，取下限）
MAX_RETRIEVALS = 2


class GraphState(AgentState, total=False):
    """AgentState（§4 冻结清单）+ 图运行时键。

    LangGraph 只保留 schema 中声明的键——未声明的返回值会被静默丢弃，
    运行时键（路由理由/引用/待执行工具调用等）在这里显式声明。
    """

    route_reason: str
    rag_sources: list
    used_rag: bool
    pending_tool_calls: list


class DurianAgentGraph:
    """组装并编译完整 Agent 图。依赖全部注入（LLM/索引可缺省）。"""

    def __init__(
        self,
        *,
        llm: Optional[LLMProvider] = None,
        retriever: Optional[FourWayRetriever] = None,
        max_retrievals: int = MAX_RETRIEVALS,
        checkpointer=None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.max_retrievals = max_retrievals
        self._parser = SemanticParser(llm)
        self._simple = SimpleAgent(llm) if llm is not None else None
        self.graph = self._build(checkpointer)

    # ══════════════════ 节点实现 ══════════════════

    def _context_init(self, state: AgentState) -> Dict[str, Any]:
        """补齐缺省状态键（§4 init_state 安全默认值）+ 落一条 HumanMessage。

        messages 通道由 LangGraph 预初始化为空列表（"in state" 恒真），
        所以无条件追加——add_messages 语义下每个 invoke 恰好新增一条。
        """
        defaults = init_state(
            thread_id=state.get("thread_id", ""),
            user_id=state.get("user_id", ""),
            role=state.get("role", "worker"),
            language=state.get("language", "zh"),
        )
        patch = {k: v for k, v in defaults.items() if k not in state}
        patch["messages"] = [HumanMessage(content=state.get("original_query", ""))]
        return patch

    def _normalize(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("original_query", "")
        if _THAI_RE.search(query):
            normalized = normalize_thai_input(query)
        else:
            normalized = normalize_input(query)
        return {"normalized_query": normalized}

    def _semantic_parse(self, state: AgentState) -> Dict[str, Any]:
        semantic = self._parser.parse(state.get("normalized_query", ""))
        language = semantic.get("language") or state.get("language", "zh")
        semantic["language"] = language
        return {"semantic": semantic, "language": language}

    def _route(self, state: AgentState) -> Dict[str, Any]:
        decision = route_decision(state.get("semantic", {}))
        return {"route": decision["route"], "route_reason": decision["reason"]}

    def _rag_query_build(self, state: AgentState) -> Dict[str, Any]:
        queries = build_queries(state.get("normalized_query", ""))
        return {"search_queries": [queries["original"], queries["canonical"],
                                   queries["expanded"]]}

    def _retrieve(self, state: AgentState) -> Dict[str, Any]:
        """四路召回（§25）。重试轮次用最新改写查询（rewrite 更新 search_queries[0]）。"""
        if self.retriever is None:
            return {"retrieved_docs": [],
                    "retrieval_count": state.get("retrieval_count", 0) + 1}
        query = state["search_queries"][0] if state.get("search_queries") \
            else state.get("normalized_query", "")
        result = self.retriever.recall(query, top_k_each=10) or {"hits": {}}
        flattened = []
        for route, route_hits in result["hits"].items():
            for rank, hit in enumerate(route_hits, start=1):
                doc = dict(hit)
                doc["source_route"] = route
                doc["route_rank"] = rank
                flattened.append(doc)
        return {"retrieved_docs": flattened,
                "retrieval_count": state.get("retrieval_count", 0) + 1}

    def _rrf(self, state: AgentState) -> Dict[str, Any]:
        docs = state.get("retrieved_docs", [])
        by_id = {d["chunk_id"]: d for d in docs}
        rankings: Dict[str, list] = {r: [] for r in
                                     ("dense_original", "dense_canonical",
                                      "bm25_original", "bm25_expanded")}
        for doc in docs:   # hits 展开时保序，直接按路分组重建名次
            route = doc.get("source_route")
            if route in rankings and doc["chunk_id"] not in rankings[route]:
                rankings[route].append(doc["chunk_id"])
        fused = weighted_rrf(rankings)
        reranked = []
        for item in fused:
            doc = dict(by_id.get(item["chunk_id"], {}))
            doc["chunk_id"] = item["chunk_id"]
            doc["rrf_rank"] = item["rrf_rank"]
            doc["rrf_score"] = item["score"]
            reranked.append(doc)
        return {"reranked_docs": reranked}

    def _rerank(self, state: AgentState) -> Dict[str, Any]:
        """占位（#28 Phase2）：透传 RRF 序，Cross Encoder 重排后替换。"""
        return {}

    def _evidence_check(self, state: AgentState) -> Dict[str, Any]:
        """占位（#30 Phase2）：有融合证据即视为充分，五项判断后替换。"""
        sufficient = bool(state.get("reranked_docs"))
        return {"evidence_sufficient": sufficient}

    def _rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """占位（#28 Phase2）：无 LLM 改写，退化为扩展查询重试。"""
        queries = state.get("search_queries") or []
        if len(queries) >= 3:
            queries = [queries[2]] + queries[1:]   # 用 expanded 查询重试
        return {"search_queries": queries,
                "retry_count": state.get("retry_count", 0) + 1}

    def _rag_search_helper(self, query: str) -> Dict[str, Any]:
        """SimpleAgent 的 rag_fn：四路召回 + RRF，返回 §60 契约。"""
        if self.retriever is None:
            return {"documents": [], "evidence_sufficient": False}
        result = self.retriever.recall(query, top_k_each=10) or {"hits": {}, "rankings": {}}
        fused = weighted_rrf(result["rankings"])
        by_id = {d["chunk_id"]: d
                 for hits in result["hits"].values() for d in hits}
        documents = []
        for item in fused[:3]:
            doc = dict(by_id.get(item["chunk_id"], {}))
            doc["chunk_id"] = item["chunk_id"]
            doc.setdefault("metadata", {"title": doc.get("record", {}).get("document_id", ""),
                                        "section": ""})
            documents.append(doc)
        return {"documents": documents,
                "evidence_sufficient": bool(fused)}

    def _simple_agent(self, state: AgentState) -> Dict[str, Any]:
        if self._simple is None:
            answer = "（SIMPLE 路径需要 LLM，当前未配置。）"
            return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
        result = self._simple.answer(
            state.get("normalized_query", ""),
            rag_fn=self._rag_search_helper,
        )
        return {"final_answer": result["answer"],
                "rag_sources": result["sources"],
                "used_rag": result["used_rag"],
                "messages": [AIMessage(content=result["answer"])]}

    def _react_agent(self, state: AgentState) -> Dict[str, Any]:
        """占位（#39 Phase3）：ReAct 循环未实现，降级为提示性回答。"""
        answer = ("复杂任务编排（多工具 ReAct）将在第三阶段上线；"
                  "当前可先拆解为具体问题分别提问。")
        return {"final_answer": answer, "degrade_level": int(DegradeLevel.MINIMAL),
                "messages": [AIMessage(content=answer)]}

    def _tool_policy(self, state: AgentState) -> Dict[str, Any]:
        """占位（#38 Phase3）：ToolPolicyGate。"""
        return {"pending_tool_calls": []}

    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """占位（Phase3 工具层）。"""
        return {}

    def _permission_check(self, state: AgentState) -> Dict[str, Any]:
        """占位（#49 Phase5：Tool Permission）。"""
        return {}

    def _confirmation(self, state: AgentState) -> Dict[str, Any]:
        """占位（#54 Phase5：敏感操作二次确认）。"""
        return {}

    def _memory_compress(self, state: AgentState) -> Dict[str, Any]:
        """占位（#53 Phase4：滑动窗口与摘要）。"""
        return {}

    def _answer(self, state: AgentState) -> Dict[str, Any]:
        """MUST_RAG 终点：Final Generation（§30 证据约束生成）。"""
        docs = state.get("reranked_docs", [])
        if state.get("evidence_sufficient") and docs and self.llm is not None:
            from durian_agent.rag.generation import generate_grounded_answer
            answer = generate_grounded_answer(
                self.llm, state.get("normalized_query", ""), docs)
            sources = [
                {"document_id": d.get("record", {}).get("document_id", d.get("document_id", "?")),
                 "chunk_id": d["chunk_id"],
                 "title": d.get("record", {}).get("document_id", ""),
                 "section": ""}
                for d in docs[:3]
            ]
            return {"final_answer": answer, "rag_sources": sources,
                    "messages": [AIMessage(content=answer)]}
        return {}   # 证据不足由 fallback 处理（evidence_check 已分流，不会到此）

    def _fallback(self, state: AgentState) -> Dict[str, Any]:
        """§30/§48：证据不足时诚实拒答，禁止退化成模型自由回答。"""
        answer = (f"知识库中暂无足够证据回答「{state.get('original_query', '')[:50]}」。"
                  "请补充：具体品种、园区、症状细节或照片，我会重新检索。"
                  "（高风险问题检索未命中时不作无依据回答）")
        return {"final_answer": answer, "last_error": "RAG_NO_RESULT",
                "messages": [AIMessage(content=answer)]}

    # ══════════════════ 组图（§45）══════════════════

    def _build(self, checkpointer=None) -> StateGraph:
        builder = StateGraph(GraphState)
        builder.add_node("contextInit", self._context_init)
        builder.add_node("normalize", self._normalize)
        builder.add_node("semanticParse", self._semantic_parse)
        builder.add_node("route", self._route)
        builder.add_node("ragQueryBuild", self._rag_query_build)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("rrf", self._rrf)
        builder.add_node("rerank", self._rerank)
        builder.add_node("evidenceCheck", self._evidence_check)
        builder.add_node("rewriteQuery", self._rewrite_query)
        builder.add_node("simpleAgent", self._simple_agent)
        builder.add_node("reactAgent", self._react_agent)
        builder.add_node("toolPolicy", self._tool_policy)
        builder.add_node("toolNode", self._tool_node)
        builder.add_node("permissionCheck", self._permission_check)
        builder.add_node("confirmation", self._confirmation)
        builder.add_node("memoryCompress", self._memory_compress)
        builder.add_node("answer", self._answer)
        builder.add_node("fallback", self._fallback)

        builder.add_edge(START, "contextInit")
        builder.add_edge("contextInit", "normalize")
        builder.add_edge("normalize", "semanticParse")
        builder.add_edge("semanticParse", "route")

        def route_branch(state: AgentState):
            return {"MUST_RAG": "ragQueryBuild",
                    "SIMPLE": "simpleAgent",
                    "COMPLEX_TASK": "reactAgent"}[state["route"]]

        builder.add_conditional_edges("route", route_branch,
                                      ["ragQueryBuild", "simpleAgent", "reactAgent"])

        builder.add_edge("ragQueryBuild", "retrieve")
        builder.add_edge("retrieve", "rrf")
        builder.add_edge("rrf", "rerank")
        builder.add_edge("rerank", "evidenceCheck")

        def evidence_branch(state: AgentState):
            if state.get("evidence_sufficient"):
                return "answer"
            if state.get("retrieval_count", 0) < self.max_retrievals:
                return "rewriteQuery"
            return "fallback"

        builder.add_conditional_edges("evidenceCheck", evidence_branch,
                                      ["answer", "rewriteQuery", "fallback"])
        builder.add_edge("rewriteQuery", "retrieve")

        builder.add_edge("reactAgent", "toolPolicy")

        def tool_policy_branch(state: AgentState):
            return "permissionCheck" if state.get("pending_tool_calls") else "answer"

        builder.add_conditional_edges("toolPolicy", tool_policy_branch,
                                      ["permissionCheck", "answer"])
        builder.add_edge("permissionCheck", "toolNode")
        builder.add_edge("toolNode", "reactAgent")

        builder.add_edge("simpleAgent", "memoryCompress")
        builder.add_edge("answer", "memoryCompress")
        builder.add_edge("fallback", "memoryCompress")
        builder.add_edge("confirmation", "toolNode")
        builder.add_edge("memoryCompress", END)

        return builder.compile(checkpointer=checkpointer)

    # ══════════════════ 调用入口 ══════════════════

    def invoke(self, query: str, *, thread_id: str = "default",
               user_id: str = "", role: str = "worker", language: str = "zh",
               config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged_config = {"configurable": {"thread_id": thread_id}}
        if config:
            merged_config.update(config)
        result = self.graph.invoke(
            {"original_query": query, "thread_id": thread_id,
             "user_id": user_id, "role": role, "language": language},
            config=merged_config,
        )
        return result
