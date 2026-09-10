"""任务 #30 验证：Evidence Check 五项判断（§29）。

验证点（五项各正反例）：
1. 关键实体覆盖；2. 核心症状词覆盖（双路径核心教训）；3. 数值条件覆盖；
4. 来源冲突；5. 单点相关；以及无证据/全满足场景。
"""

import unittest

from durian_agent.rag.evidence import check_evidence


def doc(text):
    return {"chunk_id": "c", "text": text, "score": 1.0}


class TestEvidenceCheck(unittest.TestCase):

    def test_no_evidence(self):
        result = check_evidence("炭疽病怎么治", {}, [])
        self.assertFalse(result["sufficient"])
        self.assertIn("无任何证据", result["reasons"])

    def test_entity_coverage(self):
        semantic = {"entities": {"cultivar": "猫山王", "disease": None}}
        miss = check_evidence("猫山王施肥", semantic,
                              [doc("金枕施肥以复合肥为主"), doc("幼树薄肥勤施")])
        self.assertFalse(miss["sufficient"])
        self.assertTrue(any("关键实体未覆盖" in r for r in miss["reasons"]))

        hit = check_evidence("猫山王施肥", semantic,
                             [doc("猫山王施肥以磷钾肥为主"), doc("注意排水")])
        self.assertTrue(hit["sufficient"], msg=hit["reasons"])

    def test_entity_alias_counts_as_coverage(self):
        """证据用英文别名（Musang King）也算覆盖了实体「猫山王」。"""
        semantic = {"entities": {"cultivar": "猫山王"}}
        result = check_evidence("猫山王施肥", semantic,
                                [doc("Musang King requires potash"),
                                 doc("fertilize along drip line")])
        self.assertTrue(result["sufficient"], msg=result["reasons"])

    def test_symptom_coverage_core_lesson(self):
        """核心症状词未被证据覆盖 → 不充分（KB 虫害关键词再多也不行）。"""
        semantic = {"entities": {}, "symptoms": ["叶片发黄"]}
        result = check_evidence("叶片发黄是什么病", semantic,
                                [doc("蚜虫用吡虫啉防治，稀释一千倍"),
                                 doc("炭疽病用波尔多液")])
        self.assertFalse(result["sufficient"])
        self.assertTrue(any("核心症状" in r for r in result["reasons"]))

    def test_numeric_condition(self):
        # 查询含剂量数字，证据无任何数字
        result = check_evidence("吡虫啉25ml兑水可以吗", {}, [doc("按标签浓度喷施")])
        self.assertFalse(result["sufficient"])
        self.assertTrue(any("数值条件" in r for r in result["reasons"]))

        # 证据有数字
        ok = check_evidence("吡虫啉25ml兑水可以吗", {},
                            [doc("每25ml兑水15升喷雾"), doc("安全间隔期7天")])
        self.assertTrue(ok["sufficient"], msg=ok["reasons"])

    def test_source_conflict(self):
        """证据数值与查询条件不一致 → 疑似来源冲突。"""
        result = check_evidence("株施复合肥2kg对吗", {},
                                [doc("每株施3公斤复合肥"), doc("建议5公斤每株")])
        self.assertFalse(result["sufficient"])
        self.assertTrue(any("来源冲突" in r for r in result["reasons"]))

    def test_single_point(self):
        result = check_evidence("炭疽病怎么治", {}, [doc("炭疽病用波尔多液防治")])
        self.assertFalse(result["sufficient"])
        self.assertTrue(any("单点" in r for r in result["reasons"]))

    def test_all_covered_sufficient(self):
        semantic = {"entities": {"disease": "炭疽病"}, "symptoms": []}
        result = check_evidence("炭疽病怎么治", semantic,
                                [doc("炭疽病可用波尔多液防治"),
                                 doc("炭疽病高发于雨季，注意排水")])
        self.assertTrue(result["sufficient"], msg=result["reasons"])
        self.assertEqual(result["evidence_count"], 2)


if __name__ == "__main__":
    unittest.main()
