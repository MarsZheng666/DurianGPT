"""OrchardAssetTool（架构文档 §17，任务 #41）：园区/地块/设备/人员资产查询。"""

from __future__ import annotations

from durian_agent.tools.base import ToolRegistry, ToolSpec
from durian_agent.tools.providers import AssetProvider

SPEC = ToolSpec(
    name="asset_query",
    description="查询园区/地块/设备/人员资产信息。",
    args_hint='{"asset_type": "plots|devices|workers|orchards", "orchard": "ORCHARD_3"}',
)

_VALID_TYPES = {"orchards": "orchard", "plots": "plot",
                "devices": "device", "workers": "worker",
                # 单数别名
                "orchard": "orchard", "plot": "plot",
                "device": "device", "worker": "worker"}


def register(registry: ToolRegistry, provider: AssetProvider) -> None:
    registry.register(SPEC, lambda args, ctx: _execute(args, ctx, provider))


def _execute(args, ctx, provider: AssetProvider) -> str:
    raw_type = str(args.get("asset_type") or args.get("type") or "").strip().lower()
    asset_type = _VALID_TYPES.get(raw_type)
    if not asset_type:
        return f"参数错误: asset_type 需为 {sorted(set(_VALID_TYPES))} 之一"
    orchard = args.get("orchard") or (
        ctx.orchard_scope[0] if len(ctx.orchard_scope) == 1 else None)
    rows = provider.query(asset_type, orchard)
    if not rows:
        where = f"{orchard} " if orchard else ""
        return f"无 {where}{asset_type} 资产记录"
    lines = []
    for row in rows[:20]:
        detail = " ".join(f"{k}={v}" for k, v in row.items() if k != "type")
        lines.append(detail)
    return "\n".join(lines)
