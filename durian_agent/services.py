"""三服务拆分（架构文档 §59，任务 #66）。

    Agent Service        → durian_agent.api.app.create_app（已有：
                            LangGraph / NLU / Router / ReAct / Memory）
    RAG Service          → create_rag_service（Query Normalize/Expansion/
                            Dense/BM25/RRF/Rerank/Evidence）
    Business Tool Service→ create_tool_service（Weather/Sensor/Asset/
                            Task/Alarm）

单仓库内以独立 FastAPI 应用划分服务边界（§59「第一版不必微服务化过度」）；
独立部署时分别起 uvicorn 即可，接口契约不变。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException

from durian_agent.tools.base import ToolContext, ToolRegistry


def create_rag_service(retriever, reranker=None) -> FastAPI:
    """RAG Service：检索即服务（§60 /api/rag/search 契约）。"""
    app = FastAPI(title="Durian RAG Service", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "rag"}

    @app.post("/api/rag/search")
    def search(request: Dict[str, Any]) -> Dict[str, Any]:
        from durian_agent.tools.rag_tool import search_for_api

        query = str(request.get("query") or "").strip()
        if not query:
            raise HTTPException(status_code=422, detail="query 必填")
        if retriever is None:
            raise HTTPException(status_code=409, detail="检索器未配置")
        result = search_for_api(
            retriever, reranker, query,
            semantic=request.get("semantic_schema") or None)
        return {"documents": result["documents"][:int(request.get("top_k") or 5)],
                "evidence_sufficient": result["evidence_sufficient"]}

    return app


def create_tool_service(
    registry: ToolRegistry,
    *,
    service_role: str = "admin",
) -> FastAPI:
    """Business Tool Service：工具执行即服务。

    服务间内网调用（service_role=admin 全量），终端用户经 Agent Service
    走 §49 角色过滤——工具服务自身不做用户级鉴权（部署于内网）。
    """
    app = FastAPI(title="Durian Business Tool Service", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "tool",
                "tools": sorted(s.name for s in registry.specs())}

    @app.get("/tools")
    def tools() -> list:
        return [{"name": s.name, "description": s.description,
                 "args_hint": s.args_hint,
                 "confirmation_required": s.confirmation_required}
                for s in registry.specs()]

    @app.post("/tools/execute")
    def execute(request: Dict[str, Any]) -> Dict[str, Any]:
        tool = str(request.get("tool") or "")
        if not registry.has(tool):
            raise HTTPException(status_code=404, detail=f"工具不存在: {tool}")
        ctx = ToolContext(role=service_role,
                          thread_id=str(request.get("thread_id") or ""),
                          confirmed=bool(request.get("confirmed")))
        observation = registry.execute(tool, request.get("args") or {}, ctx)
        return {"tool": tool, "observation": observation}

    return app
