"""SoilSensorTool（架构文档 §17，任务 #37）：土壤/传感器实时数据。"""

from __future__ import annotations

from durian_agent.tools.base import ToolRegistry, ToolSpec
from durian_agent.tools.providers import SensorProvider

SPEC = ToolSpec(
    name="sensor",
    description="查询土壤传感器实时数据（湿度/地温/pH）。判断是否需要灌水前先查。",
    args_hint='{"orchard": "ORCHARD_3", "plot": "PLOT_5"}（plot 可省略）',
)


def register(registry: ToolRegistry, provider: SensorProvider) -> None:
    registry.register(SPEC, lambda args, ctx: _execute(args, ctx, provider))


def _execute(args, ctx, provider: SensorProvider) -> str:
    orchard = str(args.get("orchard") or
                  (ctx.orchard_scope[0] if ctx.orchard_scope else ""))
    plot = args.get("plot") or None
    if not orchard:
        return "参数错误: 需要 orchard（或用户园区范围）"
    data = provider.soil_moisture(orchard, plot)
    if not data:
        where = f"{orchard}" + (f" {plot}" if plot else "")
        return f"无 {where} 的传感器数据"
    location = f"{data['orchard']}" + (f" 地块{data['plot']}" if data.get("plot") else "")
    return (f"{location} 实时数据（{data['updated']}）："
            f"土壤湿度 {data['moisture_pct']}%，"
            f"地温 {data['temp_c']}℃，pH {data['ph']}")
