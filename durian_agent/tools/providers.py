"""业务数据 Provider（架构文档 §17 工具层的数据源，依赖注入）。

- Protocol 定义各业务数据源接口（生产接线在阶段五服务拆分）；
- InMemory* 为开发/测试用确定性 mock——**不连任何真实系统**；
  与嵌入/LLM/VLM 通道同一注入模式。

时间语义：InMemory 提供器以 2026-09-10 为固定「今天」，保证确定性。
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Protocol

_TODAY = date(2026, 9, 10)


class WeatherProvider(Protocol):
    def forecast(self, location: str, days: int) -> List[Dict]: ...


class InMemoryWeather:
    """确定性天气预报 mock：单测/E2E 用。"""

    def __init__(self, today: date = _TODAY):
        self.today = today
        # 地点 → 逐日降雨序列（mm，-1 表示无雨）
        self._rain: Dict[str, List[float]] = {
            "ORCHARD_3": [0.0, 28.0, 15.0, 0.0, 0.0, 5.0, 0.0],
            "DEFAULT": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }

    def forecast(self, location: str, days: int) -> List[Dict]:
        days = max(1, min(int(days or 3), 7))
        rain = self._rain.get(location, self._rain["DEFAULT"])
        out = []
        for i in range(days):
            rainfall = rain[i % len(rain)]
            out.append({
                "date": (self.today + timedelta(days=i)).isoformat(),
                "location": location,
                "rainfall_mm": rainfall,
                "rain": rainfall > 0,
                "temp_c_max": 32,
                "temp_c_min": 24,
            })
        return out


class SensorProvider(Protocol):
    def soil_moisture(self, orchard: str,
                      plot: Optional[str] = None) -> Optional[Dict]: ...


class InMemorySensors:
    """确定性土壤传感器 mock。"""

    _PLOTS = {
        ("ORCHARD_3", "PLOT_5"): {"moisture_pct": 42.0, "temp_c": 27.5,
                                  "ph": 5.8, "updated": "2026-09-10T08:00"},
        ("ORCHARD_3", None): {"moisture_pct": 45.0, "temp_c": 27.0,
                              "ph": 5.9, "updated": "2026-09-10T08:00"},
    }

    def soil_moisture(self, orchard: str,
                      plot: Optional[str] = None) -> Optional[Dict]:
        data = self._PLOTS.get((orchard, plot))
        if data:
            return {**data, "orchard": orchard, "plot": plot}
        return None


class AssetProvider(Protocol):
    def query(self, asset_type: str,
              orchard: Optional[str] = None) -> List[Dict]: ...


class InMemoryAssets:
    """确定性资产 mock：园区/地块/设备/人员。"""

    _ASSETS = [
        {"type": "orchard", "id": "ORCHARD_3", "name": "三号园", "area_ha": 12.5},
        {"type": "orchard", "id": "ORCHARD_5", "name": "五号园", "area_ha": 8.0},
        {"type": "plot", "id": "PLOT_5", "orchard": "ORCHARD_3",
         "cultivar": "猫山王", "trees": 86},
        {"type": "plot", "id": "PLOT_12", "orchard": "ORCHARD_3",
         "cultivar": "金枕", "trees": 120},
        {"type": "device", "id": "SENSOR-31", "orchard": "ORCHARD_3",
         "kind": "土壤湿度传感器", "status": "在线"},
        {"type": "device", "id": "SENSOR-32", "orchard": "ORCHARD_3",
         "kind": "气象站", "status": "在线"},
        {"type": "worker", "id": "W-7", "orchard": "ORCHARD_3",
         "name": "阿明", "role": "果园技术员"},
    ]

    def query(self, asset_type: str, orchard: Optional[str] = None) -> List[Dict]:
        return [a for a in self._ASSETS
                if a["type"] == asset_type
                and (orchard is None or a.get("orchard") == orchard
                     or a.get("id") == orchard)]


class AlarmProvider(Protocol):
    def query(self, orchard: Optional[str] = None,
              status: Optional[str] = None) -> List[Dict]: ...


class InMemoryAlarms:
    """确定性告警 mock。"""

    _ALARMS = [
        {"id": "ALM-101", "orchard": "ORCHARD_3", "level": "高",
         "message": "PLOT_5 土壤湿度低于 30%，需检查滴灌", "status": "active",
         "raised": "2026-09-10T06:30"},
        {"id": "ALM-102", "orchard": "ORCHARD_3", "level": "中",
         "message": "SENSOR-31 离线超过 2 小时", "status": "active",
         "raised": "2026-09-10T07:10"},
        {"id": "ALM-103", "orchard": "ORCHARD_5", "level": "低",
         "message": "五号园东侧围栏传感器电量低", "status": "resolved",
         "raised": "2026-09-09T15:00"},
    ]

    def query(self, orchard: Optional[str] = None,
              status: Optional[str] = None) -> List[Dict]:
        return [a for a in self._ALARMS
                if (orchard is None or a["orchard"] == orchard)
                and (status is None or a["status"] == status)]
