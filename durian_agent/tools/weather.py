"""WeatherTool（架构文档 §17，任务 #40）：天气预报查询。"""

from __future__ import annotations

from durian_agent.tools.base import ToolRegistry, ToolSpec
from durian_agent.tools.providers import WeatherProvider

SPEC = ToolSpec(
    name="weather",
    description="查询未来 N 天天气预报（降雨/温度）。判断灌水、施药、采收时机时先查天气。",
    args_hint='{"location": "ORCHARD_3", "days": 3}',
)


def register(registry: ToolRegistry, provider: WeatherProvider) -> None:
    registry.register(SPEC, lambda args, ctx: _execute(args, ctx, provider))


def _execute(args, ctx, provider: WeatherProvider) -> str:
    location = str(args.get("location") or
                   (ctx.orchard_scope[0] if ctx.orchard_scope else "DEFAULT"))
    days = args.get("days", 3)
    try:
        days = int(days)
    except (TypeError, ValueError):
        return "参数错误: days 必须是整数"
    days = max(1, min(days, 7))   # 工具侧钳制（provider 侧另有兜底）
    forecast = provider.forecast(location, days)
    if not forecast:
        return f"无 {location} 的天气数据"
    lines = [f"{f['date']} {f['location']}: 降雨{f['rainfall_mm']}mm"
             f"（{'有雨' if f['rain'] else '无雨'}），"
             f"{f['temp_c_min']}~{f['temp_c_max']}℃"
             for f in forecast]
    return "\n".join(lines)
