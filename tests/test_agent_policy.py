"""任务 #38 验证：ToolPolicyGate（§16/§64）。"""

import unittest

from durian_agent.tools.policy import (
    gate_final_answer,
    has_valid_evidence,
    involves_professional_conclusion,
)


class TestProfessionalConclusion(unittest.TestCase):

    def test_section16_categories_detected(self):
        cases = [
            {"intent": "irrigation"},                        # 灌溉阈值
            {"intent": "fertilization"},                     # 施肥剂量
            {"intent": "disease_diagnosis"},                 # 病害诊断
            {"intent": "general_knowledge",
             "risk_features": {"pesticide_related": True}},  # 农药
            {"intent": "flowering_management"},              # 生育期管理
            {"intent": "weather_risk",
             "secondary_intents": ["irrigation"]},
        ]
        for semantic in cases:
            self.assertTrue(involves_professional_conclusion(semantic),
                            msg=f"§16 类别漏检: {semantic}")

    def test_non_professional_passes(self):
        for semantic in [{"intent": "task_create"},
                         {"intent": "asset_query"},
                         {"intent": "general_knowledge"},
                         None]:
            self.assertFalse(involves_professional_conclusion(semantic))


class TestGateFinalAnswer(unittest.TestCase):

    def test_professional_without_evidence_forces_rag(self):
        """§64：想下专业结论且无证据 → 强制 AgricultureRagTool。"""
        forced = gate_final_answer(
            {"intent": "irrigation",
             "risk_features": {"realtime_data_required": True}},
            state={}, query="3号园今天要灌水吗")
        self.assertIsNotNone(forced)
        self.assertEqual(forced["tool"], "agriculture_rag")
        self.assertEqual(forced["args"]["query"], "3号园今天要灌水吗")

    def test_professional_with_evidence_passes(self):
        forced = gate_final_answer(
            {"intent": "irrigation"},
            state={"evidence_sufficient": True,
                   "reranked_docs": [{"chunk_id": "c1", "text": "x"}]},
            query="要灌水吗")
        self.assertIsNone(forced)

    def test_evidence_flag_without_docs_not_valid(self):
        """evidence_sufficient=True 但无文档（状态损坏）→ 仍强制。"""
        forced = gate_final_answer(
            {"intent": "irrigation"}, state={"evidence_sufficient": True})
        self.assertIsNotNone(forced)

    def test_non_professional_passes_without_evidence(self):
        forced = gate_final_answer(
            {"intent": "asset_query"}, state={}, query="3号园有多少设备")
        self.assertIsNone(forced)

    def test_has_valid_evidence(self):
        self.assertTrue(has_valid_evidence(
            {"evidence_sufficient": True, "reranked_docs": [{"chunk_id": "c"}]}))
        self.assertFalse(has_valid_evidence({"evidence_sufficient": True}))
        self.assertFalse(has_valid_evidence({"reranked_docs": [{"chunk_id": "c"}]}))


if __name__ == "__main__":
    unittest.main()
