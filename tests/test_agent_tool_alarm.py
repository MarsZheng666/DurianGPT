"""任务 #34 验证：AlarmTool（§17）。"""

import unittest

from durian_agent.tools import ToolContext, ToolRegistry
from durian_agent.tools.alarm import register
from durian_agent.tools.providers import InMemoryAlarms


def run(args, **ctx):
    registry = ToolRegistry()
    register(registry, InMemoryAlarms())
    return registry.execute("alarm_query", args, ToolContext(**ctx))


class TestAlarmTool(unittest.TestCase):

    def test_all_alarms(self):
        out = run({})
        self.assertIn("ALM-101", out)
        self.assertIn("ALM-103", out)

    def test_filter_by_orchard(self):
        out = run({"orchard": "ORCHARD_3"})
        self.assertIn("ALM-101", out)
        self.assertNotIn("ALM-103", out)      # 五号园的被过滤

    def test_filter_active_only(self):
        out = run({"status": "active"})
        self.assertIn("ALM-102", out)
        self.assertNotIn("ALM-103", out)      # resolved 被过滤

    def test_combined_filter(self):
        out = run({"orchard": "ORCHARD_3", "status": "active"})
        self.assertIn("ALM-101", out)
        self.assertNotIn("ALM-103", out)

    def test_invalid_status(self):
        self.assertIn("参数错误", run({"status": "firing"}))

    def test_no_records(self):
        self.assertIn("无 ORCHARD_9 告警记录", run({"orchard": "ORCHARD_9"}))


if __name__ == "__main__":
    unittest.main()
