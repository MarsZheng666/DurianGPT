"""POST /api/chat 对话主接口（架构文档 §60，任务 #4）。

契约（§60）：
    Request:  {"thread_id": "...", "message": "...", "language": "auto"}
    Response: {"answer", "route", "sources", "pending_confirmation"}

扩展（超出 §60 但必要）：响应附带 thread_id——客户端未传 thread_id 时
网关自动分配，必须回传才能续接会话。

阶段一边界：
- pending_confirmation 恒为 null（敏感操作确认流在 #54，阶段五）；
- 图依赖注入：未注入时用离线模式（无 LLM/无检索）——SIMPLE 回答会
  提示需要配置 LLM，MUST_RAG 走 fallback 拒答，接口契约仍完整。
"""

from __future__ import annotations

from typing import Any, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from durian_agent.api.gateway import ConversationGateway, GatewayError
from durian_agent.graph import DurianAgentGraph


class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str = Field(min_length=1, max_length=8000)
    language: Optional[str] = None


class Source(BaseModel):
    document_id: str = ""
    chunk_id: str = ""
    title: str = ""
    section: str = ""


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: List[Source]
    pending_confirmation: Optional[dict] = None
    thread_id: str


def create_app(
    graph: Optional[DurianAgentGraph] = None,
    gateway: Optional[ConversationGateway] = None,
) -> FastAPI:
    """应用工厂：图与网关均可注入（测试替身/生产装配各取所需）。"""
    app = FastAPI(title="Durian Agent", version="0.1.0")
    _graph = graph if graph is not None else DurianAgentGraph()
    _gateway = gateway if gateway is not None else ConversationGateway()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(
        request: ChatRequest,
        x_user_id: Optional[str] = Header(default=None),
        x_role: Optional[str] = Header(default=None),
    ) -> ChatResponse:
        try:
            context = _gateway.resolve_context(
                user_id=x_user_id,
                role=x_role,
                thread_id=request.thread_id,
                language=request.language,
            )
        except GatewayError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # language=auto 交给 SemanticParse 判定，图入口用 zh 做缺省
        language = context["language"] if context["language"] != "auto" else "zh"
        state = _graph.invoke(
            request.message,
            thread_id=context["thread_id"],
            user_id=context["user_id"],
            role=context["role"],
            language=language,
        )
        return ChatResponse(
            answer=state.get("final_answer", ""),
            route=state.get("route", "SIMPLE"),
            sources=[
                Source(**{k: s.get(k, "") for k in
                         ("document_id", "chunk_id", "title", "section")})
                for s in state.get("rag_sources", [])
            ],
            pending_confirmation=None,
            thread_id=context["thread_id"],
        )

    return app
