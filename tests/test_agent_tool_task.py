"""任务 #45/#42/#44 验证：Task 工具三件套（§17/§41/§42）。

#45 task_query：过滤查询；
#42 task_create：§41 未确认不执行（草稿 Observation）；§42 幂等键防重；
#44 task_update：状态流转 + 未确认拦截 + 参数校验。
"""

import unittest

from durian_agent.tools import ToolContext, ToolRegistry
from durian_agent.tools.base import parse_pending_confirmation
from durian_agent.tools.providers import InMemoryTasks
from durian_agent.tools.task import register_all


def make(**ctx):
    provider = InMemoryTasks()
    registry = ToolRegistry()
    register_all(registry, provider)
    return registry, provider, ToolContext(role="manager", **ctx)


class TestTaskQuery(unittest.TestCase):
    """#45：工单查询。"""

    def test_query_by_id(self):
        registry, provider, ctx = make()
        provider.create({"title": "排水巡检", "orchard": "ORCHARD_3",
                         "priority": "high"}, "k1")
        out = registry.execute("task_query", {"task_id": "T-101"}, ctx)
        self.assertIn("T-101", out)
        self.assertIn("排水巡检", out)
        self.assertIn("pending", out)

    def test_query_filter_status_and_orchard(self):
        registry, provider, ctx = make()
        provider.create({"title": "任务A", "orchard": "ORCHARD_3"}, "k1")
        provider.create({"title": "任务B", "orchard": "ORCHARD_5"}, "k2")
        out = registry.execute(
            "task_query", {"status": "pending", "orchard": "ORCHARD_5"}, ctx)
        self.assertIn("任务B", out)
        self.assertNotIn("任务A", out)

    def test_invalid_status(self):
        registry, _, ctx = make()
        self.assertIn("参数错误", registry.execute(
            "task_query", {"status": "paused"}, ctx))

    def test_no_match(self):
        registry, _, ctx = make()
        self.assertIn("无符合条件的工单", registry.execute(
            "task_query", {"task_id": "T-999"}, ctx))


class TestTaskCreate(unittest.TestCase):
    """#42：创建 + §41 确认拦截 + §42 幂等。"""

    ARGS = {"title": "排水巡检", "orchard": "ORCHARD_3", "plot": "PLOT_5",
            "priority": "high"}

    def test_unconfirmed_returns_pending_draft(self):
        """§41：未确认时不执行，返回草稿 Observation。"""
        registry, provider, ctx = make()
        out = registry.execute("task_create", self.ARGS, ctx)
        pending = parse_pending_confirmation(out)
        self.assertIsNotNone(pending, msg=f"非待确认协议: {out!r}")
        self.assertEqual(pending["tool"], "task_create")
        self.assertEqual(pending["args"]["title"], "排水巡检")
        # 未执行：库里无工单
        self.assertEqual(provider.query(), [])

    def test_confirmed_executes(self):
        registry, provider, ctx = make(confirmed=True)
        out = registry.execute("task_create", self.ARGS, ctx)
        self.assertIn("工单已创建: T-101", out)
        self.assertEqual(len(provider.query()), 1)

    def test_idempotency_key_dedupes(self):
        """§42：同幂等键重复创建只产生一个工单（Agent retry/网络 retry）。"""
        registry, provider, ctx = make(confirmed=True)
        args = {**self.ARGS, "idempotency_key": "req-42"}
        registry.execute("task_create", args, ctx)
        registry.execute("task_create", args, ctx)   # 重试
        self.assertEqual(len(provider.query()), 1)

    def test_default_key_from_thread_and_title(self):
        registry, provider, ctx = make(confirmed=True, thread_id="t-1")
        registry.execute("task_create", self.ARGS, ctx)
        registry.execute("task_create", self.ARGS, ctx)   # 同线程同标题 → 同 key
        self.assertEqual(len(provider.query()), 1)

    def test_validation(self):
        registry, _, ctx = make(confirmed=True)
        self.assertIn("title 必填", registry.execute(
            "task_create", {"title": ""}, ctx))
        self.assertIn("priority", registry.execute(
            "task_create", {"title": "x", "priority": "urgent"}, ctx))


class TestTaskUpdate(unittest.TestCase):
    """#44：更新。"""

    def _with_task(self, **ctx):
        registry, provider, ctx = make(**ctx)
        provider.create({"title": "施肥作业", "orchard": "ORCHARD_3"}, "k")
        return registry, provider, ctx

    def test_unconfirmed_intercepted(self):
        registry, provider, ctx = self._with_task()
        out = registry.execute(
            "task_update", {"task_id": "T-101", "status": "done"}, ctx)
        self.assertIsNotNone(parse_pending_confirmation(out))
        # 未执行：状态未变
        self.assertEqual(provider.query(task_id="T-101")[0]["status"], "pending")

    def test_confirmed_status_transition(self):
        registry, provider, ctx = self._with_task(confirmed=True)
        out = registry.execute(
            "task_update", {"task_id": "T-101", "status": "done",
                            "assignee": "阿明"}, ctx)
        self.assertIn("工单已更新", out)
        task = provider.query(task_id="T-101")[0]
        self.assertEqual(task["status"], "done")
        self.assertEqual(task["assignee"], "阿明")

    def test_invalid_status_and_missing_task(self):
        registry, _, ctx = self._with_task(confirmed=True)
        self.assertIn("参数错误", registry.execute(
            "task_update", {"task_id": "T-101", "status": "paused"}, ctx))
        self.assertIn("工单不存在", registry.execute(
            "task_update", {"task_id": "T-999", "status": "done"}, ctx))
        self.assertIn("task_id 必填", registry.execute(
            "task_update", {"status": "done"}, ctx))


if __name__ == "__main__":
    unittest.main()
