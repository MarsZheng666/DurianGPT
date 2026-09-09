"""任务 #13 验证：RuleRouter 三路路由（架构文档 §12/§13/§15/§61）。

验证点：
1. 决策表正反例：COMPLEX 每条规则、MUST_RAG 每条规则、SIMPLE 兜底；
2. 优先级对齐 §61：COMPLEX 先于 MUST_RAG（双命中时走 COMPLEX，
   专业证据由 ToolPolicyGate 补齐）；
3. secondary intents 参与判定；
4. 与 #3 SemanticParser（规则层）的集成：真实四语问句端到端路由正确。
"""

import unittest

from durian_agent.router import route
from durian_agent.semantic.parse import SemanticParser


def schema(intent="unknown", secondary=None, risks=None):
    return {
        "intent": intent,
        "secondary_intents": secondary or [],
        "risk_features": risks or {},
    }


class TestRouteDecisionTable(unittest.TestCase):

    # ── COMPLEX_TASK ─────────────────────────────────────────

    def test_business_action_flag_routes_complex(self):
        result = route(schema(risks={"business_action_required": True}))
        self.assertEqual(result["route"], "COMPLEX_TASK")

    def test_realtime_flag_routes_complex(self):
        result = route(schema(risks={"realtime_data_required": True}))
        self.assertEqual(result["route"], "COMPLEX_TASK")

    def test_tool_required_intents_route_complex(self):
        for intent in ["asset_query", "alarm_query", "task_create",
                       "task_query", "task_update"]:
            result = route(schema(intent=intent))
            self.assertEqual(result["route"], "COMPLEX_TASK", msg=intent)

    # ── MUST_RAG ─────────────────────────────────────────────

    def test_must_rag_risk_flags(self):
        for flag in ["pesticide_related", "dosage_requested",
                     "regulation_related", "diagnosis_requested"]:
            result = route(schema(risks={flag: True}))
            self.assertEqual(result["route"], "MUST_RAG", msg=flag)

    def test_must_rag_intents(self):
        for intent in ["disease_diagnosis", "pest_diagnosis", "nutrient_diagnosis",
                       "disease_control", "pest_control"]:
            result = route(schema(intent=intent))
            self.assertEqual(result["route"], "MUST_RAG", msg=intent)

    def test_secondary_intent_triggers_must_rag(self):
        result = route(schema(intent="general_knowledge", secondary=["pest_control"]))
        self.assertEqual(result["route"], "MUST_RAG")

    def test_secondary_tool_intent_triggers_complex(self):
        result = route(schema(intent="weather_risk", secondary=["task_create"]))
        self.assertEqual(result["route"], "COMPLEX_TASK")

    # ── SIMPLE ───────────────────────────────────────────────

    def test_simple_default(self):
        for intent in ["general_knowledge", "pruning", "irrigation",
                       "variety_query", "flowering_management", "unknown"]:
            result = route(schema(intent=intent))
            self.assertEqual(result["route"], "SIMPLE", msg=intent)

    def test_simple_negative_control(self):
        """§14 示例类问题（无风险标志）不过度路由。"""
        result = route(schema(intent="general_knowledge"))
        self.assertEqual(result["route"], "SIMPLE")

    # ── 优先级（§61 顺序）────────────────────────────────────

    def test_complex_wins_over_must_rag(self):
        """双命中（业务操作+剂量）→ COMPLEX_TASK（§61 先判 COMPLEX）。"""
        result = route(schema(
            intent="irrigation",
            risks={"business_action_required": True, "dosage_requested": True},
        ))
        self.assertEqual(result["route"], "COMPLEX_TASK")
        self.assertIn("risk_flags", result["reason"])

    def test_reason_always_present(self):
        """reason 必须非空（路由可审计）。"""
        for s in [schema(), schema(risks={"pesticide_related": True}),
                  schema(intent="task_create")]:
            self.assertTrue(route(s)["reason"])


class TestRouterWithSemanticParser(unittest.TestCase):
    """规则层端到端：真实问句 → SemanticParse → Route。"""

    def setUp(self):
        self.parser = SemanticParser()   # 无 LLM：纯规则层

    def _route(self, query):
        return route(self.parser.parse(query))["route"]

    def test_end_to_end_routing_multilingual(self):
        cases = [
            ("帮我创建一个巡检工单", "COMPLEX_TASK"),          # 业务操作
            ("create an inspection task", "COMPLEX_TASK"),    # 业务操作(en)
            ("当前土壤湿度多少", "COMPLEX_TASK"),               # 实时数据
            ("炭疽病用什么农药防治，浓度多少倍", "MUST_RAG"),    # 农药+剂量
            ("what pesticide for anthracnose", "MUST_RAG"),   # 农药(en)
            ("炭疽病怎么防治", "MUST_RAG"),                    # 防治意图
            ("什么是榴莲坐果", "SIMPLE"),                       # 通识
            ("帮我查一下工单12的状态", "COMPLEX_TASK"),          # 工单查询
        ]
        for query, expected in cases:
            self.assertEqual(self._route(query), expected, msg=f"路由错误: {query!r}")


if __name__ == "__main__":
    unittest.main()
