"""工具层基础（架构文档 §17/§39）。

- ToolSpec：名字 + 描述 + 参数提示（进 ReAct prompt 的封闭工具清单）；
- ToolRegistry：注册/执行；Agent 只注册当前用户允许的工具（§39，
  角色过滤的强制执行在 #49，阶段五）；
- ToolContext：用户身份 + 图状态引用（RAG 工具回写证据用）。

工具执行契约：fn(args: dict, ctx: ToolContext) -> str（Observation 文本）。
异常不逃逸——捕获后返回错误文本作为 Observation，ReAct 可据此换路。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

#: §39 角色工具表（数据化，可审计）。admin = 全部注册工具；
#: alarm_query 未在 §39 清单分配 → 仅 admin（待业务确认后调整）。
ROLE_TOOLS: Dict[str, frozenset] = {
    "worker": frozenset({"agriculture_rag", "asset_query", "task_query"}),
    "manager": frozenset({
        "agriculture_rag", "asset_query", "task_query",
        "task_create", "task_update", "weather", "sensor",
    }),
}


def allowed_for_role(tool_name: str, role: str) -> bool:
    """§39：当前角色是否可用该工具。user_context 全员可用；
    admin 全量；未入表的角色按 worker（最小权限）。"""
    if role == "admin":
        return True
    if tool_name == "user_context":
        return True
    return tool_name in ROLE_TOOLS.get(role, ROLE_TOOLS["worker"])


@dataclass
class ToolSpec:
    name: str
    description: str
    args_hint: str = ""
    #: §41 敏感操作：需用户确认后才执行（Registry 强制拦截）
    confirmation_required: bool = False


@dataclass
class ToolContext:
    """工具执行上下文。state 为图状态的引用（工具可回写证据等）。"""
    user_id: str = "anonymous"
    role: str = "worker"
    thread_id: str = ""
    language: str = "zh"
    orchard_scope: List[str] = field(default_factory=list)
    state: Optional[Dict[str, Any]] = None
    #: §41：经 /api/chat/confirm 确认后的执行置 True
    confirmed: bool = False


ToolFn = Callable[[Dict[str, Any], ToolContext], str]


class ToolError(Exception):
    pass


#: §41 待确认 Observation 协议（ReAct 层与确认 API 由此衔接）
PENDING_CONFIRMATION_PREFIX = "PENDING_CONFIRMATION: "


def pending_confirmation_observation(tool: str, args: Dict[str, Any]) -> str:
    import json

    return (PENDING_CONFIRMATION_PREFIX
            + json.dumps({"tool": tool, "args": args}, ensure_ascii=False))


def parse_pending_confirmation(observation: str) -> Optional[Dict[str, Any]]:
    """识别待确认 Observation；非该协议返回 None。"""
    if isinstance(observation, str) and observation.startswith(PENDING_CONFIRMATION_PREFIX):
        import json

        try:
            parsed = json.loads(observation[len(PENDING_CONFIRMATION_PREFIX):])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


class ToolRegistry:
    """工具注册表。"""

    def __init__(self):
        self._tools: Dict[str, tuple] = {}

    def register(self, spec: ToolSpec, fn: ToolFn) -> None:
        self._tools[spec.name] = (spec, fn)

    def has(self, name: str) -> bool:
        return name in self._tools

    def specs(self) -> List[ToolSpec]:
        return [spec for spec, _ in self._tools.values()]

    def spec_text(self, role: str = "manager") -> str:
        """进 ReAct prompt 的工具清单文本（§39：只列该角色允许的工具）。"""
        from durian_agent.tools.base import allowed_for_role

        lines = []
        for spec, _fn in self._tools.values():
            if not allowed_for_role(spec.name, role):
                continue
            hint = f" args: {spec.args_hint}" if spec.args_hint else ""
            lines.append(f"- {spec.name}: {spec.description}{hint}")
        return "\n".join(lines)

    def execute(self, name: str, args: Dict[str, Any],
                ctx: ToolContext) -> str:
        """执行工具；未注册/执行异常都转成错误 Observation（不逃逸）。

        §41 强制拦截：confirmation_required 的工具在 ctx.confirmed
        为假时**不执行**，返回待确认 Observation（含操作草稿）。
        """
        entry = self._tools.get(name)
        if entry is None:
            return f"工具不存在: {name}"
        spec, fn = entry
        from durian_agent.tools.base import allowed_for_role
        if not allowed_for_role(name, ctx.role):
            return f"权限不足: 角色 {ctx.role} 不能使用 {name}"
        if spec.confirmation_required and not ctx.confirmed:
            return pending_confirmation_observation(name, args or {})
        try:
            return fn(args or {}, ctx)
        except ToolError as exc:
            return f"工具拒绝: {exc}"
        except Exception as exc:  # noqa: BLE001 —— Observation 语义需要兜底
            return f"工具执行失败: {type(exc).__name__}: {exc}"
