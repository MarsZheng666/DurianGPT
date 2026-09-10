"""UserContextTool（架构文档 §17，任务 #43）。

返回当前用户/角色/权限范围——Agent 需要知道「我是谁、能用什么」
才能正确编排（如 Worker 不能建工单时给出准确提示而非盲目尝试）。
"""

from __future__ import annotations

from durian_agent.tools.base import ToolContext, ToolRegistry, ToolSpec

SPEC = ToolSpec(
    name="user_context",
    description="查询当前用户身份、角色与可用工具范围",
    args_hint="{}（无需参数）",
)


def register(registry: ToolRegistry) -> None:
    registry.register(SPEC, _execute)


def _execute(args, ctx: ToolContext) -> str:
    from durian_agent.tools.base import ROLE_TOOLS

    allowed = sorted(ROLE_TOOLS.get(ctx.role, ROLE_TOOLS["worker"]))
    # user_context 自身可用；§39 未分配的工具在此角色下不可用
    return (
        f"当前用户: {ctx.user_id}（角色: {ctx.role}）\n"
        f"可用工具: {', '.join(['user_context'] + allowed)}\n"
        f"园区范围: {', '.join(ctx.orchard_scope) if ctx.orchard_scope else '全部园区'}"
    )
