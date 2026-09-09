"""任务 #5 验证：Intent 分类（架构文档 §7，20 个领域意图）。

验证点：
1. 标签集与 §7 完全一致（20 个）+ unknown 兜底；§6 示例别名收编；
2. 样例集卫生检查：20 意图 × ≥3 条，语言标签合法；
3. 规则兜底层：可确定性识别的样例（任务操作/告警/资产）分类正确；
4. LLM 路径契约：prompt 含封闭标签集与 JSON-only 约束；输出解析、
   非法输出降级、secondary intents 清洗；
5. 无 LLM 且规则无命中 → unknown（不猜 general_knowledge）。
"""

import json
import unittest
from pathlib import Path

from durian_agent.llm import FakeLLM
from durian_agent.semantic.intent import (
    INTENT_SYSTEM_PROMPT,
    IntentClassifier,
    _extract_json,
    _rule_classify,
)
from durian_agent.semantic.labels import (
    INTENTS,
    UNKNOWN_INTENT,
    is_valid_intent,
    normalize_intent,
)

SAMPLES_PATH = Path(__file__).resolve().parent.parent / "durian_agent" / "semantic" / "testdata" / "intent_samples.json"


def load_samples():
    data = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    return data["samples"]


class TestIntentLabels(unittest.TestCase):

    def test_intent_set_matches_doc_section7(self):
        expected = [
            "disease_diagnosis", "pest_diagnosis", "disease_control", "pest_control",
            "fertilization", "irrigation", "nutrient_diagnosis", "flowering_management",
            "fruit_management", "pruning", "soil_management", "weather_risk",
            "harvest", "variety_query", "general_knowledge", "asset_query",
            "alarm_query", "task_create", "task_query", "task_update",
        ]
        self.assertEqual(list(INTENTS), expected)
        self.assertEqual(len(INTENTS), 20)

    def test_doc_section6_alias_normalized(self):
        """§6 示例的 irrigation_decision 收编为 §7 的 irrigation。"""
        self.assertEqual(normalize_intent("irrigation_decision"), "irrigation")

    def test_invalid_intent_to_unknown(self):
        self.assertEqual(normalize_intent("banana_care"), UNKNOWN_INTENT)
        self.assertEqual(normalize_intent(""), UNKNOWN_INTENT)
        self.assertEqual(normalize_intent(None), UNKNOWN_INTENT)
        self.assertTrue(is_valid_intent(UNKNOWN_INTENT))


class TestSampleDataset(unittest.TestCase):

    def test_dataset_hygiene(self):
        samples = load_samples()
        self.assertGreaterEqual(len(samples), 60)
        by_intent = {}
        for s in samples:
            self.assertTrue(is_valid_intent(s["intent"]), msg=f"非法标签: {s}")
            self.assertIn(s["language"], {"zh", "en", "th", "ms"}, msg=f"非法语言: {s}")
            self.assertTrue(s["query"].strip(), msg=f"空样例: {s}")
            by_intent.setdefault(s["intent"], []).append(s)
        for intent in INTENTS:
            self.assertGreaterEqual(
                len(by_intent.get(intent, [])), 3,
                msg=f"意图 {intent} 样例不足 3 条",
            )


class TestRuleBackstop(unittest.TestCase):

    def test_rule_backstop_samples_classified(self):
        """标记 rule_backstop 的样例必须被规则层确定性识别。"""
        for s in load_samples():
            if s.get("rule_backstop"):
                self.assertEqual(
                    _rule_classify(s["query"]), s["intent"],
                    msg=f"规则兜底失败: {s['query']!r}",
                )

    def test_no_match_returns_none_not_guess(self):
        self.assertIsNone(_rule_classify("什么是榴莲坐果"))
        self.assertIsNone(_rule_classify("how often to irrigate"))


class TestLLMPath(unittest.TestCase):

    def test_prompt_contains_closed_set_and_constraints(self):
        for intent in INTENTS:
            self.assertIn(intent, INTENT_SYSTEM_PROMPT)
        self.assertIn("Return JSON only", INTENT_SYSTEM_PROMPT)
        self.assertIn("Do NOT answer", INTENT_SYSTEM_PROMPT)

    def test_llm_response_parsed(self):
        fake = FakeLLM('{"language": "th", "intent": "disease_diagnosis", "secondary_intents": ["pesticide_query_placeholder"]}')
        result = IntentClassifier(llm=fake).classify("ต้นทุเรียนใบเหลืองเป็นโรคอะไร")
        self.assertEqual(result["intent"], "disease_diagnosis")
        self.assertEqual(result["language"], "th")
        # 非法 secondary 被清洗掉
        self.assertEqual(result["secondary_intents"], [])
        # LLM 被实际调用且 prompt 正确
        self.assertEqual(len(fake.calls), 1)
        self.assertIn("disease_diagnosis", fake.calls[0]["system"])

    def test_llm_fenced_json_parsed(self):
        fake = FakeLLM('```json\n{"language": "en", "intent": "irrigation", "secondary_intents": ["weather_risk"]}\n```')
        result = IntentClassifier(llm=fake).classify("how often to irrigate")
        self.assertEqual(result["intent"], "irrigation")
        self.assertEqual(result["secondary_intents"], ["weather_risk"])

    def test_llm_invalid_output_falls_back_to_rules(self):
        fake = FakeLLM("我觉得这个问题讲的是浇水")
        result = IntentClassifier(llm=fake).classify("帮我创建一个巡检工单")
        self.assertEqual(result["intent"], "task_create")  # 规则兜底接住

    def test_llm_invalid_label_falls_back(self):
        fake = FakeLLM('{"language": "zh", "intent": "watering_help", "secondary_intents": []}')
        result = IntentClassifier(llm=fake).classify("帮我创建一个巡检工单")
        self.assertEqual(result["intent"], "task_create")

    def test_no_llm_no_rule_returns_unknown(self):
        result = IntentClassifier().classify("什么是榴莲坐果")
        self.assertEqual(result["intent"], UNKNOWN_INTENT)

    def test_extract_json_tolerates_noise(self):
        self.assertEqual(_extract_json('答案如下 {"a": 1} 请参考'), {"a": 1})
        self.assertIsNone(_extract_json("没有任何 JSON"))
        self.assertIsNone(_extract_json("[1, 2]"))


if __name__ == "__main__":
    unittest.main()
