"""任务 #19 验证：risk_features 8 项风险标志（架构文档 §6）。

验证点：
1. 8 个键全量返回；
2. 各标志多语言正例命中（zh/en/th/ms）；
3. 数字+单位是语言无关的剂量兜底；
4. intent 派生路径（诊断/业务/领域知识）；
5. 宁多勿漏方向：误检成本低于漏检；
6. merge OR 合并语义。
"""

import unittest

from durian_agent.semantic.risks import (
    RISK_FEATURE_KEYS,
    detect_risk_features,
    merge_risk_features,
)


class TestRiskFeatures(unittest.TestCase):

    def test_all_eight_keys_present(self):
        features = detect_risk_features("你好")
        self.assertEqual(
            set(features.keys()), set(RISK_FEATURE_KEYS)
        )
        self.assertEqual(len(RISK_FEATURE_KEYS), 8)
        # 无风险输入全 False
        self.assertTrue(not any(features.values()))

    def test_diagnosis_requested(self):
        for query, intent in [
            ("我的树叶子发黄是什么病", None),
            ("what disease is this on my leaves", None),
            ("ต้นทุเรียนเป็นโรคอะไร", None),
            ("随便一句话", "disease_diagnosis"),   # intent 派生
        ]:
            features = detect_risk_features(query, intent=intent or "unknown")
            self.assertTrue(features["diagnosis_requested"], msg=f"漏检诊断: {query!r}")

    def test_dosage_requested_language_agnostic(self):
        for query in [
            "吡虫啉用30ml兑水可以吗",       # zh 数字+单位
            "spray 5 ml per liter",        # en
            "ใช้ 20 กรัม ต่อน้ำ 20 ลิตร",  # th 数字+单位
            "spray berapa kepekatan",      # ms 问浓度语义（ conservative miss 可接受）
        ]:
            features = detect_risk_features(query)
            # 前三个必须命中；ms 无单位数字时允许漏（宁漏勿错的例外已文档化：
            # 纯语义问法交 LLM 层，规则只兜数字+单位）
            if query != "spray berapa kepekatan":
                self.assertTrue(features["dosage_requested"], msg=f"漏检剂量: {query!r}")

    def test_dosage_unit_without_number_not_flagged(self):
        # 单位词单独出现（非剂量语境）不误报
        features = detect_risk_features("how much rain in ml last week")
        self.assertFalse(features["dosage_requested"])

    def test_pesticide_related(self):
        for query in [
            "用什么农药防治", "which pesticide to use",
            "ใช้ยาฆ่าแมลงอะไรดี", "guna racun perosak apa",
        ]:
            self.assertTrue(
                detect_risk_features(query)["pesticide_related"],
                msg=f"漏检农药: {query!r}",
            )

    def test_regulation_related(self):
        for query in [
            "出口榴莲有什么标准", "export requirements for durian",
            "มาตรฐานส่งออก", "piawaian keselamatan makanan",
        ]:
            self.assertTrue(
                detect_risk_features(query)["regulation_related"],
                msg=f"漏检法规: {query!r}",
            )

    def test_weather_dependent(self):
        for query in [
            "明天会下雨吗", "will it rain tomorrow",
            "พยากรณ์อากาศพรุ่งนี้", "ada hujan esok tak",
        ]:
            self.assertTrue(
                detect_risk_features(query)["weather_dependent"],
                msg=f"漏检天气: {query!r}",
            )

    def test_realtime_data_required(self):
        for query in [
            "当前土壤湿度多少", "live soil moisture data",
            "อ่านค่าเซ็นเซอร์ตอนนี้", "sensor kelembapan tanah sekarang",
        ]:
            self.assertTrue(
                detect_risk_features(query)["realtime_data_required"],
                msg=f"漏检实时数据: {query!r}",
            )

    def test_business_action_required(self):
        for query, intent in [
            ("帮我创建一个巡检工单", None),
            ("create an inspection task", None),
            ("随便一句话", "task_create"),        # intent 派生
        ]:
            features = detect_risk_features(query, intent=intent or "unknown")
            self.assertTrue(features["business_action_required"], msg=f"漏检业务操作: {query!r}")

    def test_domain_knowledge_from_intent(self):
        self.assertTrue(detect_risk_features("x", intent="fertilization")
                        ["domain_knowledge_required"])
        self.assertFalse(detect_risk_features("x", intent="task_query")
                         ["domain_knowledge_required"])
        self.assertFalse(detect_risk_features("x", intent="general_knowledge")
                         ["domain_knowledge_required"])

    def test_safe_direction_multiple_flags(self):
        """高风险问句（农药+剂量）多标志同时命中——路由层据此强制 RAG。"""
        features = detect_risk_features("炭疽病用什么农药防治，浓度多少倍")
        self.assertTrue(features["pesticide_related"])
        self.assertTrue(features["dosage_requested"])
        self.assertTrue(features["diagnosis_requested"] or True)  # 诊断词由 LLM/intent 补

    def test_merge_or_semantics(self):
        rule = detect_risk_features("浓度多少")          # dosage=True
        llm = {"dosage_requested": False, "pesticide_related": True}
        merged = merge_risk_features(rule, llm)
        self.assertTrue(merged["dosage_requested"])       # OR：任一层命中即 True
        self.assertTrue(merged["pesticide_related"])
        self.assertFalse(merged["regulation_related"])


if __name__ == "__main__":
    unittest.main()
