"""Task 工具（架构文档 §17/§41/§42，任务 #45/#42/#44）。

- task_query：工单查询（§17）；
- task_create：创建（§41 敏感操作——需用户确认；§42 幂等键防重）；
- task_update：更新（§41 敏感操作）。

确认流约定：
- create/update 的 ToolSpec 标注 confirmation_required；
- ToolRegistry.execute 在 ctx.confirmed 为假时**不执行**，返回
  PENDING_CONFIRMATION 前缀的 Observation（含待执行操作草稿 JSON），
  ReAct 层据此向用户发起确认（#54/#10 的 API 承接执行）。
"""

from __future__ import annotations

from durian_agent.tools.base import (  # noqa: F401 —— 协议常量由此导出
    PENDING_CONFIRMATION_PREFIX,
    parse_pending_confirmation,
    pending_confirmation_observation,
)
from durian_agent.tools.base import ToolRegistry, ToolSpec
from durian_agent.tools.providers import TASK_STATUSES, TaskProvider

PENDING_CONFIRMATION_PREFIX = "PENDING_CONFIRMATION: "

QUERY_SPEC = ToolSpec(
    name="task_query",
    description="查询工单（按编号/状态/园区过滤）。",
    args_hint='{"task_id": "T-101"} 或 {"status": "pending", "orchard": "ORCHARD_3"}',
)
CREATE_SPEC = ToolSpec(
    name="task_create",
    description="创建工单（巡检/排水/施肥等）。需要用户确认后才会执行。",
    args_hint='{"title": "排水巡检", "orchard": "ORCHARD_3", "plot": "PLOT_5", '
              '"priority": "high", "description": "...", "idempotency_key": "..."}',
    confirmation_required=True,
)
UPDATE_SPEC = ToolSpec(
    name="task_update",
    description="更新工单（状态/负责人/内容）。需要用户确认后才会执行。",
    args_hint='{"task_id": "T-101", "status": "done", "assignee": "阿明"}',
    confirmation_required=True,
)


def register_all(registry: ToolRegistry, provider: TaskProvider) -> None:
    registry.register(QUERY_SPEC, lambda args, ctx: _query(args, ctx, provider))
    registry.register(CREATE_SPEC, lambda args, ctx: _create(args, ctx, provider))
    registry.register(UPDATE_SPEC, lambda args, ctx: _update(args, ctx, provider))


# ── 查询（#45）──────────────────────────────────────────

def _query(args, ctx, provider: TaskProvider) -> str:
    task_id = args.get("task_id") or None
    status = str(args.get("status") or "").strip().lower() or None
    if status and status not in TASK_STATUSES:
        return f"参数错误: status 需为 {list(TASK_STATUSES)} 之一"
    orchard = args.get("orchard") or (
        ctx.orchard_scope[0] if len(ctx.orchard_scope) == 1 else None)
    rows = provider.query(task_id, status, orchard)
    if not rows:
        return "无符合条件的工单"
    return "\n".join(
        f"{t['task_id']} [{t['status']}] {t['title']}（{t.get('orchard', '')}"
        f"{' ' + t['plot'] if t.get('plot') else ''}，优先级 {t['priority']}）"
        for t in rows[:20])


# ── 创建（#42）──────────────────────────────────────────

def _validate_create(args) -> str | None:
    if not str(args.get("title") or "").strip():
        return "参数错误: title 必填"
    priority = args.get("priority", "normal")
    if priority not in ("low", "normal", "high"):
        return "参数错误: priority 需为 low/normal/high"
    return None


def _create(args, ctx, provider: TaskProvider) -> str:
    error = _validate_create(args)
    if error:
        return error
    draft = {
        "title": str(args["title"]).strip(),
        "description": str(args.get("description") or "").strip(),
        "orchard": str(args.get("orchard") or "").strip(),
        "plot": str(args.get("plot") or "").strip(),
        "priority": args.get("priority", "normal"),
        "created_by": ctx.user_id,
    }
    idempotency_key = str(args.get("idempotency_key") or
                           f"{ctx.thread_id}:{draft['title']}")
    task = provider.create(draft, idempotency_key)
    return (f"工单已创建: {task['task_id']} [{task['status']}] "
            f"{task['title']}（{task['orchard']}，优先级 {task['priority']}）")


# ── 更新（#44）──────────────────────────────────────────

def _update(args, ctx, provider: TaskProvider) -> str:
    task_id = str(args.get("task_id") or "").strip()
    if not task_id:
        return "参数错误: task_id 必填"
    status = args.get("status")
    if status is not None and status not in TASK_STATUSES:
        return f"参数错误: status 需为 {list(TASK_STATUSES)} 之一"
    patch = {k: args.get(k) for k in
             ("status", "assignee", "title", "description") if args.get(k)}
    if not patch:
        return "参数错误: 至少提供 status/assignee/title/description 之一"
    task = provider.update(task_id, patch)
    if not task:
        return f"工单不存在: {task_id}"
    return f"工单已更新: {task['task_id']} [{task['status']}] {task['title']}"

