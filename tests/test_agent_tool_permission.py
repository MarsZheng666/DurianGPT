"""任务 #49 验证：Tool Permission（§39：Agent 只注册当前用户允许的工具）。

双层验证：
1. Prompt 层：ReAct system prompt 只列该角色允许的工具（Agent 看不见未授权工具）；
2. 执行层：Registry.execute 强制拦截（即使 LLM 硬调也不执行）。
"""

import unittest

from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM
from durian_agent.tools import ToolContext, ToolRegistry, build_default_registry
from durian_agent.tools.base import allowed_for_role
from durian_agent.tools.react import build_system_prompt


class TestAllowedForRole(unittest.TestCase):

    def test_worker_minimum_set(self):
        for tool in ("agriculture_rag", "asset_query", "task_query",
                     "user_context"):
            self.assertTrue(allowed_for_role(tool, "worker"), tool)
        for tool in ("task_create", "task_update", "weather", "sensor",
                     "alarm_query"):
            self.assertFalse(allowed_for_role(tool, "worker"), tool)

    def test_manager_extended(self):
        for tool in ("task_create", "task_update", "weather", "sensor"):
            self.assertTrue(allowed_for_role(tool, "manager"), tool)

    def test_admin_all(self):
        registry = build_default_registry()
        for spec in registry.specs():
            self.assertTrue(allowed_for_role(spec.name, "admin"), spec.name)

    def test_alarm_admin_only_per_doc(self):
        """alarm_query 未入 §39 角色表 → 仅 admin（待业务确认后调整）。"""
        self.assertFalse(allowed_for_role("alarm_query", "worker"))
        self.assertFalse(allowed_for_role("alarm_query", "manager"))
        self.assertTrue(allowed_for_role("alarm_query", "admin"))


class TestPromptLayer(unittest.TestCase):

    def test_worker_prompt_hides_unauthorized(self):
        registry = build_default_registry()
        prompt = build_system_prompt(registry, "worker")
        self.assertNotIn("- weather:", prompt)
        self.assertNotIn("- task_create:", prompt)
        self.assertIn("- agriculture_rag:", prompt)

    def test_manager_prompt_shows_extended(self):
        registry = build_default_registry()
        prompt = build_system_prompt(registry, "manager")
        self.assertIn("- weather:", prompt)
        self.assertIn("- task_create:", prompt)
        self.assertNotIn("- alarm_query:", prompt)


class TestEnforcementLayer(unittest.TestCase):

    def test_worker_cannot_execute_write_tool(self):
        registry = build_default_registry()
        out = registry.execute(
            "task_create", {"title": "x"},
            ToolContext(role="worker", confirmed=True))
        self.assertIn("权限不足", out)
        self.assertIn("worker", out)

    def test_manager_can(self):
        registry = build_default_registry()
        out = registry.execute(
            "task_create", {"title": "巡检", "idempotency_key": "k"},
            ToolContext(role="manager", confirmed=True))
        self.assertIn("工单已创建", out)

    def test_graph_worker_react_denied_observation(self):
        """图内：worker 的 ReAct 硬调未授权工具 → 拒绝 Observation 回给 LLM。"""

        class ScriptLLM(FakeLLM):
            def __init__(self):
                super().__init__("")
                self.react_calls = 0

            def complete(self, system, user, *, temperature=0.0):
                if "Respond with EXACTLY ONE JSON" in system:
                    self.react_calls += 1
                    if self.react_calls == 1:
                        return ('{"action": {"tool": "weather", '
                                '"args": {"days": 3}}}')
                    return '{"final_answer": "没有天气权限，无法判断。"}'
                return "不是JSON"

        graph = DurianAgentGraph(llm=ScriptLLM())
        result = graph.invoke("看下传感器数据和天气再决定浇水",
                              thread_id="t-perm", role="worker")
        observations = [m.content for m in result["messages"]
                        if m.__class__.__name__ == "ToolMessage"]
        self.assertTrue(any("权限不足" in o for o in observations))
        self.assertIn("没有天气权限", result["final_answer"])
        # worker 的 prompt 里根本没列 weather（看不见）
        self.assertNotIn("- weather:", graph._react._prompt_cache["worker"])


if __name__ == "__main__":
    unittest.main()
