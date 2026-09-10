"""AlarmTool（架构文档 §17，任务 #34）：告警查询。"""

from __future__ import annotations

from durian_agent.tools.base import ToolRegistry, ToolSpec
from durian_agent.tools.providers import AlarmProvider

SPEC = ToolSpec(
    name="alarm_query",
    description="查询园区告警（等级/内容/状态），可按园区与状态过滤。",
    args_hint='{"orchard": "ORCHARD_3", "status": "active"}（均可省略）',
)

_VALID_STATUS = {"active", "resolved"}


def register(registry: ToolRegistry, provider: AlarmProvider) -> None:
    registry.register(SPEC, lambda args, ctx: _execute(args, ctx, provider))


def _execute(args, ctx, provider: AlarmProvider) -> str:
    orchard = args.get("orchard") or (
        ctx.orchard_scope[0] if len(ctx.orchard_scope) == 1 else None)
    status = str(args.get("status") or "").strip().lower() or None
    if status and status not in _VALID_STATUS:
        return f"参数错误: status 需为 {sorted(_VALID_STATUS)} 之一"
    rows = provider.query(orchard, status)
    if not rows:
        where = f"{orchard} " if orchard else ""
        return f"无 {where}告警记录"
    return "\n".join(
        f"[{a['level']}] {a['id']} ({a['orchard']}, {a['status']}, "
        f"{a['raised']}): {a['message']}" for a in rows[:20])
