"""任务 #12 验证：实体归一化（alias→canonical，架构文档 §9）。

验证点：
1. 同一实体的四语别名收敛到同一标准名（猫山王/Musang King/D197/Raja
   Kunyit/หมอนทอง → 猫山王）；
2. 大小写/全半角不漏（normalize 后匹配）；
3. 最长别名优先（Phytophthora palmivora 不被 Phytophthora 截断）；
4. 园区/地块编号多语言模式 → ORCHARD_n / PLOT_n 标准编码；
5. 无实体输入返回空 dict；多槽位并存。
"""

import unittest

from durian_agent.semantic.entities import detect_entities


class TestEntityNormalization(unittest.TestCase):

    def test_musang_king_aliases_converge(self):
        """同一实体的多语别名 → 同一标准名（§9 核心验收）。"""
        # 猫山王（马来西亚品种）：zh / en / 编号 / ms
        aliases = ["猫山王", "Musang King", "D197", "Raja Kunyit"]
        canonicals = {detect_entities(f"{a} 的施肥方案")["cultivar"] for a in aliases}
        self.assertEqual(canonicals, {"猫山王"})
        # 金枕（泰国品种）：th / en / en 别名
        aliases = ["หมอนทอง", "Monthong", "Golden Pillow", "金枕"]
        canonicals = {detect_entities(f"{a} 的价格")["cultivar"] for a in aliases}
        self.assertEqual(canonicals, {"金枕"})

    def test_case_and_fullwidth_insensitive(self):
        self.assertEqual(
            detect_entities("ｍｕｓａｎｇ ｋｉｎｇ 怎么种")["cultivar"],
            detect_entities("Musang King 怎么种")["cultivar"],
        )

    def test_longest_alias_wins(self):
        entities = detect_entities("Phytophthora palmivora 引起的病害")
        self.assertEqual(entities.get("disease"), "棕榈疫霉 (Phytophthora palmivora)")

    def test_disease_pest_fertilizer_slots(self):
        # 病害：英文别名 / 中文标准名直命中 / 泰文别名
        self.assertEqual(detect_entities("anthracnose on branches").get("disease"), "炭疽病")
        self.assertEqual(detect_entities("炭疽病怎么防治").get("disease"), "炭疽病")
        self.assertIn("disease", detect_entities("ต้นทุเรียนเป็นรากเน่า"))
        # 虫害（en 别名 / th 别名）
        self.assertIn("pest", detect_entities("stem borer damage on my durian"))
        self.assertIn("pest", detect_entities("หนอนเจาะผลทำลายผล"))
        # 肥料（nutrient 类目 → fertilizer 槽位；中文单字标准名直命中）
        self.assertEqual(detect_entities("缺钾怎么办").get("fertilizer"), "钾")
        self.assertIn("fertilizer", detect_entities("โพแทสเซียม ไม่พอ"))

    def test_orchard_patterns(self):
        cases = ["ORCHARD_03 的猫山王", "3号园需要巡检", "สวนที่ 3 มีปัญหา"]
        for case in cases:
            self.assertEqual(detect_entities(case).get("orchard"), "ORCHARD_3",
                             msg=f"园区编号未归一: {case}")

    def test_plot_patterns(self):
        cases = ["PLOT_12 土壤湿度多少", "5号地块该浇水吗", "แปลงที่ 2 ปลูกอะไรดี"]
        for case in cases:
            self.assertEqual(detect_entities(case).get("plot"), "PLOT_5"
                             if "5号" in case else ("PLOT_12" if "PLOT" in case.upper() else "PLOT_2"),
                             msg=f"地块编号未归一: {case}")

    def test_multi_slot_coexist(self):
        entities = detect_entities("3号园的猫山王得了炭疽病")
        self.assertEqual(entities.get("orchard"), "ORCHARD_3")
        self.assertEqual(entities.get("cultivar"), "猫山王")
        self.assertEqual(entities.get("disease"), "炭疽病")

    def test_no_entity_returns_empty(self):
        self.assertEqual(detect_entities("今天天气怎么样"), {})
        self.assertEqual(detect_entities(""), {})

    def test_pesticide_slot_not_rule_filled(self):
        """pesticide 槽位规则层不填（术语表无农药类目，LLM/#15 补足）。"""
        entities = detect_entities("用什么农药防治茎蛀虫")
        self.assertNotIn("pesticide", entities)
        # 但 pest（茎蛀虫 = stem borer 的标准名）应被识别
        self.assertEqual(entities.get("pest"), "茎蛀虫")


if __name__ == "__main__":
    unittest.main()
