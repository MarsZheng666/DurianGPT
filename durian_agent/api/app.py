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

from durian_agent.api.confirmations import ConfirmationStore
from durian_agent.api.gateway import ConversationGateway, GatewayError
from durian_agent.graph import DurianAgentGraph


class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str = Field(min_length=1, max_length=8000)
    language: Optional[str] = None


class ConfirmRequest(BaseModel):
    thread_id: str
    confirmation_id: str
    approved: bool


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
    confirmations: Optional[ConfirmationStore] = None,
) -> FastAPI:
    """应用工厂：图与网关均可注入（测试替身/生产装配各取所需）。"""
    app = FastAPI(title="Durian Agent", version="0.1.0")
    _graph = graph if graph is not None else DurianAgentGraph()
    _gateway = gateway if gateway is not None else ConversationGateway()
    _confirmations = confirmations if confirmations is not None else ConfirmationStore()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(
        request: ChatRequest,
        x_user_id: Optional[str] = Header(default=None),
        x_role: Optional[str] = Header(default=None),
        x_tenant_id: Optional[str] = Header(default=None),
    ) -> ChatResponse:
        try:
            context = _gateway.resolve_context(
                user_id=x_user_id,
                role=x_role,
                thread_id=request.thread_id,
                language=request.language,
                tenant_id=x_tenant_id,
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
            tenant_id=context["tenant_id"],
        )
        # §41：图侧拦截的敏感操作 → 注册确认存储并回传给客户端
        pending = state.get("pending_confirmation")
        if pending:
            cid = _confirmations.register(
                context["thread_id"], pending.get("tool", ""),
                pending.get("args", {}),
                confirmation_id=pending.get("confirmation_id"),
                role=context["role"], user_id=context["user_id"])
            pending = {**pending, "confirmation_id": cid}
        return ChatResponse(
            answer=state.get("final_answer", ""),
            route=state.get("route", "SIMPLE"),
            sources=[
                Source(**{k: s.get(k, "") for k in
                         ("document_id", "chunk_id", "title", "section")})
                for s in state.get("rag_sources", [])
            ],
            pending_confirmation=pending,
            thread_id=context["thread_id"],
        )

    @app.post("/api/chat/confirm")
    def confirm(request: ConfirmRequest) -> dict:
        """§41：approved=true → 以确认态执行；false → 取消。"""
        item = _confirmations.get(request.confirmation_id)
        if item is None:
            raise HTTPException(status_code=404, detail="确认请求不存在或已处理")
        if item["thread_id"] != request.thread_id:
            raise HTTPException(status_code=403, detail="确认请求不属于该会话")
        _confirmations.pop(request.confirmation_id)
        if not request.approved:
            return {"status": "cancelled",
                    "confirmation_id": request.confirmation_id}
        if _graph.tools is None:
            raise HTTPException(status_code=409, detail="工具层不可用")
        from durian_agent.tools import ToolContext
        ctx = ToolContext(
            thread_id=request.thread_id, confirmed=True,
            role=item.get("role", "manager"),
            user_id=item.get("user_id", ""))
        result = _graph.tools.execute(item["tool"], item["args"], ctx)
        return {"status": "executed",
                "confirmation_id": request.confirmation_id,
                "tool": item["tool"],
                "result": result}

    return app
