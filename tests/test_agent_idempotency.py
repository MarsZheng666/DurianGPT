"""任务 #47 验证：Task Tool 幂等（§42）。

§42 要求：TaskCreate 必须带 request_id/idempotency_key，
避免 **Agent retry** 与 **网络 retry** 导致重复工单。

幂等内核已随 #42 落地（InMemoryTasks.create 的 key 去重 +
Registry 确认闸），#10 的 API 测试覆盖了「确认重放不重复建单」。
本验收补齐 §42 点名的两类重放路径：
1. Agent retry：ReAct 循环内同参数重试（步数循环重放同一 action）；
2. 网络 retry：confirm 请求重放（已由 #10 test_confirmation_is_one_shot
   覆盖，此处直接复验）。
"""

import unittest

from durian_agent.tools import ToolContext, ToolRegistry
from durian_agent.tools.providers import InMemoryTasks
from durian_agent.tools.task import register_all

ARGS = {"title": "排水巡检", "orchard": "ORCHARD_3",
        "idempotency_key": "req-net-001"}
CONFIRMED = ToolContext(confirmed=True, role="manager")


class TestIdempotencyAcceptance(unittest.TestCase):

    def _registry(self):
        provider = InMemoryTasks()
        registry = ToolRegistry()
        register_all(registry, provider)
        return registry, provider

    def test_agent_retry_same_action_no_duplicate(self):
        """Agent retry：ReAct 循环重放同一 action（同 key 同参数）。"""
        registry, provider = self._registry()
        for _ in range(3):   # Agent 因观察超时等重试三次
            registry.execute("task_create", ARGS, CONFIRMED)
        self.assertEqual(len(provider.query()), 1)

    def test_network_retry_confirm_replay_no_duplicate(self):
        """网络 retry：confirm 请求重放（#10 场景复验）。"""
        registry, provider = self._registry()
        registry.execute("task_create", ARGS, CONFIRMED)
        registry.execute("task_create", ARGS, CONFIRMED)   # 网络重放
        self.assertEqual(len(provider.query()), 1)

    def test_different_keys_create_separate_tasks(self):
        """不同幂等键 = 不同业务请求，正常建多单。"""
        registry, provider = self._registry()
        registry.execute("task_create", {**ARGS, "idempotency_key": "k1"},
                         CONFIRMED)
        registry.execute("task_create", {**ARGS, "idempotency_key": "k2"},
                         CONFIRMED)
        self.assertEqual(len(provider.query()), 2)

    def test_default_key_derived_from_thread_and_title(self):
        """缺省 key 由 thread+标题派生（同会话同标题重试去重）。"""
        registry, provider = self._registry()
        ctx = ToolContext(confirmed=True, role="manager", thread_id="t-42")
        registry.execute("task_create", {"title": "施肥作业"}, ctx)
        registry.execute("task_create", {"title": "施肥作业"}, ctx)
        self.assertEqual(len(provider.query()), 1)


if __name__ == "__main__":
    unittest.main()
