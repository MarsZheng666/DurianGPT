"""任务 #3 验证：SemanticParse → Canonical Schema（架构文档 §5/§6/§46）。

验证点：
1. §46 prompt 约束完整（不回答/不诊断/不推断农药剂量/JSON only）+ 封闭标签集；
2. 规则层独立可用（llm=None）：实体/症状/风险标志正确抽取；
3. LLM 层合成：规则实体覆盖 LLM 错误实体；risk OR 合并；非法输出清洗；
4. LLM 失败/乱码 → 规则层兜底，Schema 仍全键齐备；
5. 诱导诊断的 prompt 行为约束（prompt 文本层面）。
"""

import unittest

from durian_agent.llm import FakeLLM
from durian_agent.semantic.parse import (
    ENTITY_SLOTS,
    PLANT_PARTS,
    SEMANTIC_PARSE_SYSTEM_PROMPT,
    SemanticParser,
)

FULL_LLM_RESPONSE = """{
  "language": "th",
  "intent": "disease_diagnosis",
  "secondary_intents": ["disease_control"],
  "entities": {"cultivar": "Monthong", "orchard": null, "plot": null,
               "fertilizer": null, "pesticide": null, "disease": null, "pest": null},
  "growth_stage": "flowering",
  "plant_parts": ["leaf", "banana_peel", "root"],
  "symptoms": ["leaf_yellowing", ""],
  "environment": {"rainfall": "high", "soil_moisture": null, "temperature": null,
                  "humidity": null, "drainage": "unknown"},
  "risk_features": {"diagnosis_requested": true, "dosage_requested": false,
                    "pesticide_related": false, "regulation_related": false,
                    "weather_dependent": false, "realtime_data_required": false,
                    "domain_knowledge_required": true, "business_action_required": false}
}"""


class TestPrompt(unittest.TestCase):

    def test_prompt_contains_section46_constraints(self):
        for fragment in [
            "You do NOT answer the user's question.",
            "You only extract structured semantics.",
            "Do not diagnose diseases.",
            "Do not infer pesticide, dosage or treatment.",
            "Return JSON only.",
        ]:
            self.assertIn(fragment, SEMANTIC_PARSE_SYSTEM_PROMPT)
        # 封闭标签集进 prompt
        self.assertIn("disease_diagnosis", SEMANTIC_PARSE_SYSTEM_PROMPT)
        self.assertIn("flowering", SEMANTIC_PARSE_SYSTEM_PROMPT)
        self.assertIn("leaf", SEMANTIC_PARSE_SYSTEM_PROMPT)


class TestRulesOnly(unittest.TestCase):

    def setUp(self):
        self.parser = SemanticParser()   # 无 LLM：纯规则层

    def test_schema_complete_keys(self):
        schema = self.parser.parse("帮我看看3号园的猫山王")
        for key in ("language", "intent", "secondary_intents", "entities",
                    "growth_stage", "plant_parts", "symptoms", "environment",
                    "risk_features"):
            self.assertIn(key, schema)
        self.assertEqual(set(schema["entities"].keys()), set(ENTITY_SLOTS))

    def test_entities_and_stage_rules(self):
        schema = self.parser.parse("3号园猫山王开花期叶子发黄")
        self.assertEqual(schema["entities"]["cultivar"], "猫山王")
        self.assertEqual(schema["entities"]["orchard"], "ORCHARD_3")
        self.assertEqual(schema["growth_stage"], "flowering")

    def test_thai_symptoms_rule(self):
        schema = self.parser.parse("หมอนทองใบเหลือง")
        self.assertEqual(schema["entities"]["cultivar"], "金枕")
        self.assertIn("叶片发黄", schema["symptoms"])

    def test_risk_flags_rule(self):
        schema = self.parser.parse("炭疽病用什么农药，浓度多少倍")
        self.assertTrue(schema["risk_features"]["pesticide_related"])
        self.assertTrue(schema["risk_features"]["dosage_requested"])

    def test_intent_rule_fallback(self):
        schema = self.parser.parse("帮我创建一个巡检工单")
        self.assertEqual(schema["intent"], "task_create")


class TestLLMSynthesis(unittest.TestCase):

    def test_llm_full_schema_merged(self):
        # 查询文本明确含"猫山王"，LLM 却答 Monthong → 规则层必须覆盖
        fake = FakeLLM(FULL_LLM_RESPONSE)
        schema = SemanticParser(llm=fake).parse("猫山王开花期叶子发黄")
        self.assertEqual(schema["language"], "th")            # 语言取 LLM
        self.assertEqual(schema["intent"], "disease_diagnosis")
        self.assertEqual(schema["secondary_intents"], ["disease_control"])
        self.assertEqual(schema["entities"]["cultivar"], "猫山王")  # 规则覆盖 LLM
        self.assertEqual(schema["growth_stage"], "flowering")       # LLM 有效
        # 非法 plant_part（banana_peel）被过滤，合法的保留
        self.assertEqual(schema["plant_parts"], ["leaf", "root"])
        # 空串症状被剔除
        self.assertIn("leaf_yellowing", schema["symptoms"])
        self.assertEqual(schema["environment"]["rainfall"], "high")
        # risk：LLM diagnosis=true 保留
        self.assertTrue(schema["risk_features"]["diagnosis_requested"])

    def test_risk_or_merge(self):
        # LLM 漏了 dosage/pesticide，规则层命中 → OR 后为 True（安全侧）
        schema = SemanticParser(llm=FakeLLM(FULL_LLM_RESPONSE)).parse(
            "猫山王用25ml农药可以吗")
        self.assertTrue(schema["risk_features"]["dosage_requested"])
        self.assertTrue(schema["risk_features"]["pesticide_related"])

    def test_llm_garbage_falls_back(self):
        for garbage in ["完全不是JSON", "[]", '{"intent": "banana_care"}']:
            schema = SemanticParser(llm=FakeLLM(garbage)).parse("帮我创建一个巡检工单")
            self.assertEqual(schema["intent"], "task_create")   # 规则兜底
            self.assertEqual(set(schema["entities"].keys()), set(ENTITY_SLOTS))

    def test_llm_invalid_stage_rule_fallback(self):
        response = '{"language": "zh", "intent": "irrigation", "growth_stage": "blooming"}'
        schema = SemanticParser(llm=FakeLLM(response)).parse("开花期怎么浇水")
        self.assertEqual(schema["intent"], "irrigation")
        self.assertEqual(schema["growth_stage"], "flowering")  # LLM 非法→规则层


if __name__ == "__main__":
    unittest.main()
