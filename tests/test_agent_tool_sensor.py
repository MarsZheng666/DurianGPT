"""任务 #37 验证：SoilSensorTool（§17）。"""

import unittest

from durian_agent.tools import ToolContext, ToolRegistry
from durian_agent.tools import sensor
from durian_agent.tools.providers import InMemorySensors


def run(args, **ctx):
    registry = ToolRegistry()
    sensor.register(registry, InMemorySensors())
    return registry.execute("sensor", args, ToolContext(role="manager", **ctx))


class TestSensorTool(unittest.TestCase):

    def test_plot_level_reading(self):
        out = run({"orchard": "ORCHARD_3", "plot": "PLOT_5"})
        self.assertIn("ORCHARD_3 地块PLOT_5", out)
        self.assertIn("土壤湿度 42.0%", out)
        self.assertIn("pH 5.8", out)

    def test_orchard_level_without_plot(self):
        out = run({"orchard": "ORCHARD_3"})
        self.assertIn("土壤湿度 45.0%", out)
        self.assertNotIn("地块", out)

    def test_orchard_defaults_to_scope(self):
        out = run({}, orchard_scope=["ORCHARD_3"])
        self.assertIn("ORCHARD_3", out)

    def test_no_orchard_error_observation(self):
        self.assertIn("参数错误", run({}))

    def test_unknown_plot_no_data(self):
        out = run({"orchard": "ORCHARD_9", "plot": "PLOT_1"})
        self.assertIn("无 ORCHARD_9 PLOT_1 的传感器数据", out)


if __name__ == "__main__":
    unittest.main()
