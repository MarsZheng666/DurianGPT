"""任务 #43 验证：UserContextTool（§17）。"""

import unittest

from durian_agent.tools import ToolContext, ToolRegistry
from durian_agent.tools import user_context
from durian_agent.tools.base import ROLE_TOOLS


class TestUserContextTool(unittest.TestCase):

    def _run(self, **ctx_kwargs):
        registry = ToolRegistry()
        user_context.register(registry)
        ctx = ToolContext(**ctx_kwargs)
        return registry.execute("user_context", {}, ctx)

    def test_returns_identity_role_scope(self):
        out = self._run(user_id="u-1", role="manager",
                        orchard_scope=["ORCHARD_3"])
        self.assertIn("u-1", out)
        self.assertIn("manager", out)
        self.assertIn("ORCHARD_3", out)
        # §39：manager 可用工具清单进结果
        for tool in ["task_create", "task_update", "weather", "sensor"]:
            self.assertIn(tool, out)

    def test_worker_role_least_tools(self):
        out = self._run(user_id="u-2", role="worker")
        self.assertIn("agriculture_rag", out)
        self.assertIn("task_query", out)
        self.assertNotIn("task_create", out)   # Worker 无写操作

    def test_default_scope_all_orchards(self):
        out = self._run(user_id="u-3")
        self.assertIn("全部园区", out)

    def test_role_table_matches_doc_section39(self):
        self.assertEqual(ROLE_TOOLS["worker"],
                         {"agriculture_rag", "asset_query", "task_query"})
        self.assertEqual(ROLE_TOOLS["manager"],
                         {"agriculture_rag", "asset_query", "task_query",
                          "task_create", "task_update", "weather", "sensor"})

    def test_registry_contract(self):
        registry = ToolRegistry()
        user_context.register(registry)
        self.assertTrue(registry.has("user_context"))
        self.assertIn("user_context", registry.spec_text())


if __name__ == "__main__":
    unittest.main()
