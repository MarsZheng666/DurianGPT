"""任务 #10 验证：POST /api/chat/confirm（§41/§60）。

验证点：
1. §41 全链路：chat 产生待确认操作 → 响应携带 pending_confirmation
   （含 confirmation_id/tool/args）→ confirm approved=true 执行
   （ctx.confirmed=True）→ 工单真的创建；
2. approved=false → 取消，不执行；
3. 未知 confirmation_id → 404；会话不匹配 → 403；
4. 确认是一次性的：重复 confirm → 404。
"""

import unittest

from fastapi.testclient import TestClient

from durian_agent.api.app import create_app
from durian_agent.api.confirmations import ConfirmationStore
from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM
from durian_agent.tools import build_default_registry
from durian_agent.tools.providers import InMemoryTasks

CREATE_ACTION = ('{"action": {"tool": "task_create", "args": '
                 '{"title": "排水巡检", "orchard": "ORCHARD_3", '
                 '"priority": "high"}}}')


class ReactScriptLLM(FakeLLM):
    """语义解析回退规则层；ReAct 按脚本走。"""

    def __init__(self, react_responses):
        super().__init__("")
        self.react_responses = list(react_responses)
        self.react_calls = 0

    def complete(self, system, user, *, temperature=0.0):
        if "Respond with EXACTLY ONE JSON" in system:
            index = min(self.react_calls, len(self.react_responses) - 1)
            self.react_calls += 1
            return self.react_responses[index]
        return "不是JSON"


def make_app(tasks):
    """注入共享工单存储：图内拦截的草稿与确认执行落在同一存储。"""
    from durian_agent.tools import ToolRegistry
    from durian_agent.tools import task as task_tools
    from durian_agent.tools.providers import (
        InMemoryAlarms, InMemoryAssets, InMemorySensors, InMemoryWeather,
    )
    from durian_agent.tools import (
        alarm,
        asset,
        rag_tool,
        sensor,
        user_context,
        weather,
    )

    registry = ToolRegistry()
    weather.register(registry, InMemoryWeather())
    sensor.register(registry, InMemorySensors())
    asset.register(registry, InMemoryAssets())
    alarm.register(registry, InMemoryAlarms())
    task_tools.register_all(registry, tasks)
    rag_tool.register(registry, None)
    user_context.register(registry)

    graph = DurianAgentGraph(
        llm=ReactScriptLLM([CREATE_ACTION]), tools=registry,
        # confirm 端点以 ctx(role=manager) 执行；图内 ReAct 也用 manager
    )
    return TestClient(create_app(graph=graph,
                                 confirmations=ConfirmationStore()))


class TestConfirmFlow(unittest.TestCase):

    def setUp(self):
        self.tasks = InMemoryTasks()
        self.client = make_app(self.tasks)

    def _chat_creates_pending(self):
        resp = self.client.post(
            "/api/chat",
            json={"thread_id": "t-cfm",
                  "message": "帮我创建一个排水巡检工单"},
            headers={"X-Role": "manager", "X-User-Id": "u-mgr"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsNotNone(body["pending_confirmation"])
        self.assertEqual(body["pending_confirmation"]["tool"], "task_create")
        return body

    def test_full_flow_approved_executes(self):
        """§41 全链路：草稿 → 确认 → 执行。"""
        body = self._chat_creates_pending()
        cid = body["pending_confirmation"]["confirmation_id"]
        # 未确认前：工单未创建
        self.assertEqual(self.tasks.query(), [])
        resp = self.client.post("/api/chat/confirm", json={
            "thread_id": "t-cfm", "confirmation_id": cid, "approved": True})
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertEqual(result["status"], "executed")
        self.assertIn("工单已创建", result["result"])
        # 工单真的创建了（幂等键防重也生效）
        self.assertEqual(len(self.tasks.query()), 1)

    def test_declined_cancels(self):
        body = self._chat_creates_pending()
        cid = body["pending_confirmation"]["confirmation_id"]
        resp = self.client.post("/api/chat/confirm", json={
            "thread_id": "t-cfm", "confirmation_id": cid, "approved": False})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "cancelled")
        self.assertEqual(self.tasks.query(), [])   # 未执行

    def test_unknown_id_404(self):
        resp = self.client.post("/api/chat/confirm", json={
            "thread_id": "t-cfm", "confirmation_id": "cfm-nope",
            "approved": True})
        self.assertEqual(resp.status_code, 404)

    def test_thread_mismatch_403(self):
        body = self._chat_creates_pending()
        cid = body["pending_confirmation"]["confirmation_id"]
        resp = self.client.post("/api/chat/confirm", json={
            "thread_id": "别的会话", "confirmation_id": cid, "approved": True})
        self.assertEqual(resp.status_code, 403)
        # 未执行（403 分支先 pop 之前 return——确认项保留与否不强制，
        # 关键是没执行）
        self.assertEqual(self.tasks.query(), [])

    def test_confirmation_is_one_shot(self):
        body = self._chat_creates_pending()
        cid = body["pending_confirmation"]["confirmation_id"]
        first = self.client.post("/api/chat/confirm", json={
            "thread_id": "t-cfm", "confirmation_id": cid, "approved": True})
        self.assertEqual(first.status_code, 200)
        again = self.client.post("/api/chat/confirm", json={
            "thread_id": "t-cfm", "confirmation_id": cid, "approved": True})
        self.assertEqual(again.status_code, 404)   # 已消费
        # 幂等：确认只执行一次
        self.assertEqual(len(self.tasks.query()), 1)


if __name__ == "__main__":
    unittest.main()
