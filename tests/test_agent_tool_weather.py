"""任务 #40 验证：WeatherTool（§17）。"""

import unittest

from durian_agent.tools import ToolContext, ToolRegistry
from durian_agent.tools import weather
from durian_agent.tools.providers import InMemoryWeather


def make_registry(provider=None):
    registry = ToolRegistry()
    weather.register(registry, provider or InMemoryWeather())
    return registry


class TestWeatherTool(unittest.TestCase):

    def test_forecast_with_location(self):
        out = make_registry().execute(
            "weather", {"location": "ORCHARD_3", "days": 3}, ToolContext(role="manager"))
        self.assertIn("ORCHARD_3", out)
        self.assertIn("降雨28.0mm（有雨）", out)   # mock 第2天有雨
        self.assertIn("2026-09-11", out)

    def test_location_defaults_to_scope(self):
        out = make_registry().execute(
            "weather", {"days": 1}, ToolContext(role="manager", orchard_scope=["ORCHARD_3"]))
        self.assertIn("ORCHARD_3", out)

    def test_invalid_days_returns_error_observation(self):
        out = make_registry().execute(
            "weather", {"days": "abc"}, ToolContext(role="manager"))
        self.assertIn("参数错误", out)

    def test_days_clamped(self):
        provider = InMemoryWeather()
        calls = []

        class Tracking(InMemoryWeather):
            def forecast(self, location, days):
                calls.append(days)
                return super().forecast(location, days)

        make_registry(Tracking()).execute(
            "weather", {"location": "X", "days": 99}, ToolContext(role="manager"))
        self.assertEqual(calls, [7])   # 上限 7 天

    def test_unknown_location_returns_no_rain_default(self):
        out = make_registry().execute(
            "weather", {"location": "NOWHERE", "days": 2}, ToolContext(role="manager"))
        self.assertIn("无雨", out)


if __name__ == "__main__":
    unittest.main()
