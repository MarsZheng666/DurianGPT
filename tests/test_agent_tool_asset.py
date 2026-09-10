"""任务 #41 验证：OrchardAssetTool（§17）。"""

import unittest

from durian_agent.tools import ToolContext, ToolRegistry
from durian_agent.tools.asset import register
from durian_agent.tools.providers import InMemoryAssets


def run(args, **ctx):
    registry = ToolRegistry()
    register(registry, InMemoryAssets())
    return registry.execute("asset_query", args, ToolContext(**ctx))


class TestAssetTool(unittest.TestCase):

    def test_query_devices(self):
        out = run({"asset_type": "devices", "orchard": "ORCHARD_3"})
        self.assertIn("SENSOR-31", out)
        self.assertIn("土壤湿度传感器", out)
        self.assertIn("在线", out)

    def test_query_plots_with_cultivar(self):
        out = run({"asset_type": "plots", "orchard": "ORCHARD_3"})
        self.assertIn("PLOT_5", out)
        self.assertIn("cultivar=猫山王", out)

    def test_workers(self):
        out = run({"asset_type": "workers", "orchard": "ORCHARD_3"})
        self.assertIn("阿明", out)

    def test_singular_alias(self):
        out = run({"asset_type": "device", "orchard": "ORCHARD_3"})
        self.assertIn("SENSOR-31", out)

    def test_invalid_type(self):
        self.assertIn("参数错误", run({"asset_type": "buildings"}))

    def test_scope_single_orchard_default(self):
        out = run({"asset_type": "plots"}, orchard_scope=["ORCHARD_3"])
        self.assertIn("PLOT_5", out)

    def test_no_records(self):
        out = run({"asset_type": "plots", "orchard": "ORCHARD_99"})
        self.assertIn("无 ORCHARD_99 plot 资产记录", out)


if __name__ == "__main__":
    unittest.main()
