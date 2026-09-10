"""任务 #54 验证：敏感操作二次确认完整流程（§41 五步）。

§41：Agent 生成待执行操作 → Validation → 展示给用户 → 用户确认 →
Tool Execute。拦截/展示/确认/执行已随 #42/#10 落地，本验收补齐
**Validation 时序**：无效参数在草稿产生前被拒（不进确认流）。
"""

import unittest

from fastapi.testclient import TestClient

from durian_agent.api.app import create_app
from durian_agent.api.confirmations import ConfirmationStore
from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM
from durian_agent.tools import ToolContext, ToolRegistry, build_default_registry
from durian_agent.tools.base import parse_pending_confirmation


class TestValidationStage(unittest.TestCase):

    def test_invalid_create_draft_rejected_before_confirmation(self):
        """§41 Validation：title 缺失的草稿不产生（错误 Observation）。"""
        registry = build_default_registry()
        out = registry.execute(
            "task_create", {"orchard": "ORCHARD_3"},
            ToolContext(role="manager"))
        self.assertIn("title 必填", out)
        self.assertIsNone(parse_pending_confirmation(out))   # 非草稿

    def test_invalid_update_draft_rejected(self):
        registry = build_default_registry()
        out = registry.execute(
            "task_update", {"status": "paused"},
            ToolContext(role="manager"))
        self.assertIn("参数错误", out)
        self.assertIsNone(parse_pending_confirmation(out))

    def test_valid_draft_proceeds_to_confirmation(self):
        registry = build_default_registry()
        out = registry.execute(
            "task_create", {"title": "排水巡检"},
            ToolContext(role="manager"))
        self.assertIsNotNone(parse_pending_confirmation(out))


class TestFullSection41Flow(unittest.TestCase):
    """§41 五步端到端（API 层）。"""

    def test_five_steps_end_to_end(self):
        from durian_agent.tools.providers import InMemoryTasks

        class ReactLLM(FakeLLM):
            def __init__(self):
                super().__init__("")
                self.calls = 0

            def complete(self, system, user, *, temperature=0.0):
                if "Respond with EXACTLY ONE JSON" in system:
                    self.calls += 1
                    return ('{"action": {"tool": "task_create", "args": '
                            '{"title": "排水巡检", "orchard": "ORCHARD_3"}}}')
                return "不是JSON"

        tasks = InMemoryTasks()
        registry = build_default_registry()
        # 共享工单存储
        from durian_agent.tools import task as task_tools
        registry._tools.pop("task_create")
        task_tools.register_all(registry, tasks)

        graph = DurianAgentGraph(llm=ReactLLM(), tools=registry)
        client = TestClient(create_app(graph=graph,
                                        confirmations=ConfirmationStore()))

        # 1-3. 生成操作 → Validation → 展示给用户（pending_confirmation）
        resp = client.post(
            "/api/chat",
            json={"thread_id": "t-41", "message": "创建排水巡检工单"},
            headers={"X-Role": "manager"})
        pending = resp.json()["pending_confirmation"]
        self.assertIsNotNone(pending)                       # Validation 过 → 展示
        self.assertEqual(pending["args"]["title"], "排水巡检")
        self.assertEqual(tasks.query(), [])                 # 未执行

        # 4-5. 用户确认 → Tool Execute
        confirm = client.post("/api/chat/confirm", json={
            "thread_id": "t-41",
            "confirmation_id": pending["confirmation_id"],
            "approved": True})
        self.assertEqual(confirm.json()["status"], "executed")
        self.assertEqual(len(tasks.query()), 1)             # 真执行
        self.assertEqual(tasks.query()[0]["title"], "排水巡检")


if __name__ == "__main__":
    unittest.main()
