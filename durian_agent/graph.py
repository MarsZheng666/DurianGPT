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
    evidence_reasons: list
    react_done: bool
    react_steps: int
    pending_confirmation: dict
    rag_forced: bool
    tenant_id: str
    user_orchard_scope: list
    memory_compressions: int


class DurianAgentGraph:
    """组装并编译完整 Agent 图。依赖全部注入（LLM/索引可缺省）。"""

    def __init__(
        self,
        *,
        llm: Optional[LLMProvider] = None,
        retriever: Optional[FourWayRetriever] = None,
        max_retrievals: int = MAX_RETRIEVALS,
        checkpointer=None,
        reranker=None,
        tools=None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.max_retrievals = max_retrievals
        self._parser = SemanticParser(llm)
        self._simple = SimpleAgent(llm) if llm is not None else None
        # ReAct（#39）：显式传 tools 或（有 LLM 时）默认九工具装配
        from durian_agent.tools import build_default_registry
        from durian_agent.tools.react import ReActEngine
        if tools is None and llm is not None:
            tools = build_default_registry(retriever=retriever, reranker=reranker)
        self.tools = tools
        self._react = (ReActEngine(llm, tools)
                       if llm is not None and tools is not None else None)
        self.graph = self._build(checkpointer)

    # ══════════════════ 节点实现 ══════════════════

    def _context_init(self, state: GraphState) -> Dict[str, Any]:
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

    def _normalize(self, state: GraphState) -> Dict[str, Any]:
        query = state.get("original_query", "")
        if _THAI_RE.search(query):
            normalized = normalize_thai_input(query)
        else:
            normalized = normalize_input(query)
        return {"normalized_query": normalized}

    def _semantic_parse(self, state: GraphState) -> Dict[str, Any]:
        semantic = self._parser.parse(state.get("normalized_query", ""))
        language = semantic.get("language") or state.get("language", "zh")
        semantic["language"] = language
        return {"semantic": semantic, "language": language}

    def _route(self, state: GraphState) -> Dict[str, Any]:
        decision = route_decision(state.get("semantic", {}))
        return {"route": decision["route"], "route_reason": decision["reason"]}

    def _rag_query_build(self, state: GraphState) -> Dict[str, Any]:
        queries = build_queries(state.get("normalized_query", ""))
        return {"search_queries": [queries["original"], queries["canonical"],
                                   queries["expanded"]]}

    def _retrieve(self, state: GraphState) -> Dict[str, Any]:
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

    def _rrf(self, state: GraphState) -> Dict[str, Any]:
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

    def _rerank(self, state: GraphState) -> Dict[str, Any]:
        """Cross Encoder 重排（§27，#32）：RRF 序 → rerank 分 → Top5。"""
        if self.reranker is None:
            return {}   # 未配置重排器：透传 RRF 序（阶段一行为）
        from durian_agent.rag.rerank import rerank_documents

        docs = state.get("reranked_docs", [])
        reranked = rerank_documents(
            state.get("normalized_query", ""), docs, self.reranker, top_k=5)
        return {"reranked_docs": reranked}

    def _evidence_check(self, state: GraphState) -> Dict[str, Any]:
        """五项判断（§29，#30）：实体覆盖/核心条件/数值条件/来源冲突/单点。"""
        from durian_agent.rag.evidence import check_evidence

        result = check_evidence(
            state.get("normalized_query", ""),
            state.get("semantic"),
            state.get("reranked_docs", []),
        )
        return {"evidence_sufficient": result["sufficient"],
                "evidence_reasons": result["reasons"]}

    def _rewrite_query(self, state: GraphState) -> Dict[str, Any]:
        """两层改写（§24/§47，#28）：LLM 层（§47 禁猜清单+数字护栏）→ 规则层兜底。"""
        from durian_agent.rag.rewrite import rewrite_query

        query = state["search_queries"][0] if state.get("search_queries") \
            else state.get("normalized_query", "")
        rewritten = rewrite_query(
            query,
            semantic=state.get("semantic"),
            docs=state.get("reranked_docs"),
            llm=self.llm,
        )
        queries = list(state.get("search_queries") or [])
        if queries:
            queries[0] = rewritten
        else:
            queries = [rewritten]
        return {"search_queries": queries,
                "retry_count": state.get("retry_count", 0) + 1}

    def _rag_search_helper(self, query: str,
                       scope_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """SimpleAgent 的 rag_fn：四路召回 + RRF，返回 §60 契约。"""
        if self.retriever is None:
            return {"documents": [], "evidence_sufficient": False}
        from durian_agent.rag.permissions import (
            build_scope_expr, scope_from_graph_state,
        )
        expr = None
        if scope_state is not None and scope_state.get("tenant_id"):
            expr = build_scope_expr(scope_from_graph_state(scope_state))
        result = self.retriever.recall(query, top_k_each=10, expr=expr) \
            or {"hits": {}, "rankings": {}}
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

    def _simple_agent(self, state: GraphState) -> Dict[str, Any]:
        if self._simple is None:
            answer = "（SIMPLE 路径需要 LLM，当前未配置。）"
            return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
        from durian_agent.memory.context import business_state_from_graph_state

        result = self._simple.answer(
            state.get("normalized_query", ""),
            rag_fn=lambda q: self._rag_search_helper(q, scope_state=state),
            history_summary=state.get("history_summary", ""),
            recent_messages=state.get("recent_messages") or [],
            business_state=business_state_from_graph_state(state),
        )
        return {"final_answer": result["answer"],
                "rag_sources": result["sources"],
                "used_rag": result["used_rag"],
                "messages": [AIMessage(content=result["answer"])]}

    def _react_agent(self, state: GraphState) -> Dict[str, Any]:
        """ReAct 一步推理（#39）；未配置 LLM/工具时降级提示。"""
        if self._react is None:
            answer = ("复杂任务编排需要 LLM 与工具配置；"
                      "当前可拆解为具体问题分别提问。")
            return {"final_answer": answer, "react_done": True,
                    "degrade_level": int(DegradeLevel.MINIMAL),
                    "messages": [AIMessage(content=answer)]}
        return self._react.step(state)

    def _tool_policy(self, state: GraphState) -> Dict[str, Any]:
        """ToolPolicyGate（#38，§16/§64）：想下专业结论且无证据 → 强制 RAG。"""
        if state.get("pending_tool_calls"):
            return {}
        if state.get("react_done"):
            from durian_agent.tools.policy import gate_final_answer
            forced = gate_final_answer(
                state.get("semantic"), state,
                state.get("normalized_query", ""))
            if forced:
                return {"pending_tool_calls": [forced], "react_done": False,
                        "rag_forced": True}
        return {}

    def _tool_node(self, state: GraphState) -> Dict[str, Any]:
        """执行待决工具调用（#39）；敏感操作被 Registry 拦截为待确认草稿。"""
        import uuid

        from langchain_core.messages import ToolMessage

        from durian_agent.tools import ToolContext, parse_pending_confirmation

        calls = state.get("pending_tool_calls") or []
        if not calls or self.tools is None:
            return {}
        # 工具可回写的状态（RAG 证据）；LangGraph 节点 state 是快照，
        # 回写需通过返回 patch 传播
        tool_state = {"semantic": state.get("semantic")}
        ctx = ToolContext(
            user_id=state.get("user_id", ""),
            role=state.get("role", "worker"),
            thread_id=state.get("thread_id", ""),
            language=state.get("language", "zh"),
            state=tool_state,
        )
        patch: Dict[str, Any] = {
            "pending_tool_calls": [],
            "react_steps": state.get("react_steps", 0) + 1,
        }
        # §4 tool_calls 留痕（§52 Trace 消费）：LastValue 通道需整表返回
        tool_log = list(state.get("tool_calls") or [])
        for index, call in enumerate(calls):
            observation = self.tools.execute(
                call.get("tool", ""), call.get("args") or {}, ctx)
            tool_log.append({
                "tool_name": call.get("tool", ""),
                "arguments": call.get("args") or {},
                "status": "ok" if not str(observation).startswith(
                    ("工具执行失败", "权限不足", "工具不存在")) else "error",
            })
            patch.setdefault("messages", []).append(
                ToolMessage(content=observation,
                            tool_call_id=f"call-{patch['react_steps']}-{index}"))
            pending = parse_pending_confirmation(observation)
            if pending:
                patch["tool_calls"] = tool_log
                patch["pending_confirmation"] = {
                    "confirmation_id": f"cfm-{uuid.uuid4().hex[:10]}",
                    **pending,
                }
                # §43：确认回合的回答附带判断依据（证据引用）——
                # 综合判断已由 ReAct 基于这些证据作出
                evidence_ids = [str(d.get("chunk_id"))
                                for d in state.get("reranked_docs") or []
                                if d.get("chunk_id")][:3]
                evidence_note = (f"（判断依据：{' '.join(f'[{cid}]' for cid in evidence_ids)}）"
                                 if evidence_ids else "")
                patch["final_answer"] = (
                    f"以下操作需要您确认后才会执行：{pending['tool']} "
                    f"{pending['args'].get('title', '')}{evidence_note}".strip())
                patch["react_done"] = True
                return patch
        patch["tool_calls"] = tool_log
        if "reranked_docs" in tool_state:
            patch["reranked_docs"] = tool_state["reranked_docs"]
        if "evidence_sufficient" in tool_state:
            patch["evidence_sufficient"] = tool_state["evidence_sufficient"]
        return patch

    def _permission_check(self, state: GraphState) -> Dict[str, Any]:
        """占位（#49 Phase5：Tool Permission 角色过滤）。当前全放行。"""
        return {}


    def _confirmation(self, state: GraphState) -> Dict[str, Any]:
        """占位（#54 Phase5：敏感操作二次确认）。"""
        return {}

    def _memory_compress(self, state: GraphState) -> Dict[str, Any]:
        """§35（#55）：窗口切分 + 溢出摘要 + RemoveMessage 收缩。"""
        from langchain_core.messages import RemoveMessage

        from durian_agent.memory.window import compress

        result = compress(
            state.get("messages") or [],
            previous_summary=state.get("history_summary", ""),
            llm=self.llm,
        )
        patch: Dict[str, Any] = {
            "recent_messages": result["recent"],
            "history_summary": result["history_summary"],
        }
        if result["trimmed_ids"]:
            patch["messages"] = [RemoveMessage(id=mid)
                                 for mid in result["trimmed_ids"]]
            patch["memory_compressions"] = \
                state.get("memory_compressions", 0) + 1
        return patch

    def _answer(self, state: GraphState) -> Dict[str, Any]:
        """MUST_RAG 终点 / COMPLEX 收尾（ReAct 已产出回答时透传+抽引用）。"""
        # COMPLEX_TASK：ReAct 的 final_answer 已就绪 → 只补引用，不重新生成
        if state.get("route") == "COMPLEX_TASK" and state.get("final_answer"):
            docs = state.get("reranked_docs", [])
            from durian_agent.rag.generation import parse_citations
            cited = set(parse_citations(state["final_answer"], docs))
            sources = [
                {"document_id": d.get("record", {}).get("document_id",
                                                        d.get("document_id", "?")),
                 "chunk_id": d["chunk_id"],
                 "title": d.get("record", {}).get("document_id", ""),
                 "section": ""}
                for d in docs if d["chunk_id"] in cited
            ]
            return {"rag_sources": sources}

        docs = state.get("reranked_docs", [])
        if state.get("evidence_sufficient") and docs and self.llm is not None:
            from durian_agent.rag.generation import generate_with_citations

            result = generate_with_citations(
                self.llm, state.get("normalized_query", ""), docs)
            cited = set(result["cited_chunk_ids"])
            # §31：sources 只含被回答实际引用的证据（未引用的不进）
            sources = [
                {"document_id": d.get("record", {}).get("document_id",
                                                        d.get("document_id", "?")),
                 "chunk_id": d["chunk_id"],
                 "title": d.get("record", {}).get("document_id", ""),
                 "section": ""}
                for d in docs if d["chunk_id"] in cited
            ]
            return {"final_answer": result["answer"], "rag_sources": sources,
                    "messages": [AIMessage(content=result["answer"])]}
        return {}   # 证据不足由 fallback 处理（evidence_check 已分流，不会到此）

    def _fallback(self, state: GraphState) -> Dict[str, Any]:
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

        def route_branch(state: GraphState):
            return {"MUST_RAG": "ragQueryBuild",
                    "SIMPLE": "simpleAgent",
                    "COMPLEX_TASK": "reactAgent"}[state["route"]]

        builder.add_conditional_edges("route", route_branch,
                                      ["ragQueryBuild", "simpleAgent", "reactAgent"])

        builder.add_edge("ragQueryBuild", "retrieve")
        builder.add_edge("retrieve", "rrf")
        builder.add_edge("rrf", "rerank")
        builder.add_edge("rerank", "evidenceCheck")

        def evidence_branch(state: GraphState):
            if state.get("evidence_sufficient"):
                return "answer"
            if state.get("retrieval_count", 0) < self.max_retrievals:
                return "rewriteQuery"
            return "fallback"

        builder.add_conditional_edges("evidenceCheck", evidence_branch,
                                      ["answer", "rewriteQuery", "fallback"])
        builder.add_edge("rewriteQuery", "retrieve")

        builder.add_edge("reactAgent", "toolPolicy")

        def tool_policy_branch(state: GraphState):
            return "permissionCheck" if state.get("pending_tool_calls") else "answer"

        builder.add_conditional_edges("toolPolicy", tool_policy_branch,
                                      ["permissionCheck", "answer"])
        builder.add_edge("permissionCheck", "toolNode")

        def tool_node_branch(state: GraphState):
            # 敏感操作被拦截（react_done）→ 直接收尾提问确认；
            # 正常观察 → 回 reactAgent 继续推理
            return "answer" if state.get("react_done") else "reactAgent"

        builder.add_conditional_edges("toolNode", tool_node_branch,
                                      ["reactAgent", "answer"])

        builder.add_edge("simpleAgent", "memoryCompress")
        builder.add_edge("answer", "memoryCompress")
        builder.add_edge("fallback", "memoryCompress")
        builder.add_edge("confirmation", "toolNode")
        builder.add_edge("memoryCompress", END)

        return builder.compile(checkpointer=checkpointer)

    # ══════════════════ 调用入口 ══════════════════

    def invoke(self, query: str, *, thread_id: str = "default",
               user_id: str = "", role: str = "worker", language: str = "zh",
               tenant_id: str = "", config: Optional[Dict[str, Any]] = None,
               ) -> Dict[str, Any]:
        # §37：Checkpoint key = tenantId:userId:threadId（缺省维用占位）
        tenant = tenant_id or "default"
        user = user_id or "anonymous"
        checkpoint_key = f"{tenant}:{user}:{thread_id}"
        merged_config = {"configurable": {"thread_id": checkpoint_key}}
        if config:
            merged_config.update(config)
        result = self.graph.invoke(
            {"original_query": query, "thread_id": thread_id,
             "user_id": user_id, "role": role, "language": language,
             "tenant_id": tenant},
            config=merged_config,
        )
        return result
