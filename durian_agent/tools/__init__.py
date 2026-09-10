"""durian_agent.tools — ReAct 工具层（§17/§39）。"""

from durian_agent.tools.base import (
    ToolContext,
    ToolRegistry,
    ToolSpec,
    parse_pending_confirmation,
    pending_confirmation_observation,
)


def build_default_registry(*, retriever=None, reranker=None) -> ToolRegistry:
    """默认工具装配：九工具全注册 + InMemory 数据源（开发模式）。

    生产装配（阶段五服务拆分）替换 Provider 即可，工具逻辑不变。
    """
    from durian_agent.tools import (
        alarm,
        asset,
        rag_tool,
        sensor,
        task,
        user_context,
        weather,
    )
    from durian_agent.tools.providers import (
        InMemoryAlarms,
        InMemoryAssets,
        InMemorySensors,
        InMemoryTasks,
        InMemoryWeather,
    )

    registry = ToolRegistry()
    weather.register(registry, InMemoryWeather())
    sensor.register(registry, InMemorySensors())
    asset.register(registry, InMemoryAssets())
    alarm.register(registry, InMemoryAlarms())
    task.register_all(registry, InMemoryTasks())
    rag_tool.register(registry, retriever, reranker)
    user_context.register(registry)
    return registry


__all__ = [
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "parse_pending_confirmation",
    "pending_confirmation_observation",
    "build_default_registry",
]
