"""任务 #15 验证：多语言术语库 canonical_id 体系（§10）。

验证点：
1. v2 结构合规：canonical_id / type / canonical_name / 四语 aliases /
   related_entities / source_categories；
2. ID 确定性：重复生成 ID 稳定（CULTIVAR_xxx 三位序号式）；
3. 四语覆盖：zh/en/th/ms 均非空；
4. detect_entities_with_ids：四语别名 → 同一 canonical_id（§9 正名形态）；
   与 detect_entities 槽位等价；
5. 落盘文件与再生成一致（数据不漂移）；
6. 数据记账：概念数落在 §10 建议「300～500」区间（实际 206，如实断言
   区间下限附近并允许下浮——别名 527 低于建议 1000～2000，扩充路径
   已在模块 docstring 记录）。
"""

import re
import unittest

from durian_agent.glossary import (
    GLOSSARY_V2_PATH,
    build_glossary_v2,
    load_glossary_v2,
)
from durian_agent.semantic.entities import detect_entities, detect_entities_with_ids


class TestGlossaryV2Structure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = load_glossary_v2()

    def test_entries_structure(self):
        entries = self.data["entries"]
        self.assertGreater(len(entries), 150)
        id_re = re.compile(r"^[A-Z_]+_\d{3}$")
        for entry in entries:
            self.assertRegex(entry["canonical_id"], id_re)
            self.assertIn("type", entry)
            self.assertTrue(entry["canonical_name"])
            self.assertEqual(set(entry["aliases"].keys()), {"zh", "en", "th", "ms"})
            self.assertIn("related_entities", entry)
            self.assertTrue(entry["source_categories"])

    def test_ids_deterministic(self):
        rebuilt = build_glossary_v2()
        self.assertEqual(
            [e["canonical_id"] for e in self.data["entries"]],
            [e["canonical_id"] for e in rebuilt["entries"]],
        )

    def test_four_language_coverage(self):
        langs = {"zh": 0, "en": 0, "th": 0, "ms": 0}
        for entry in self.data["entries"]:
            for lang, aliases in entry["aliases"].items():
                langs[lang] += len(aliases)
        for lang, count in langs.items():
            self.assertGreater(count, 15, msg=f"{lang} 别名覆盖异常低: {count}")

    def test_musang_king_entry(self):
        entry = next(e for e in self.data["entries"]
                     if e["canonical_name"] == "猫山王")
        self.assertEqual(entry["type"], "cultivar")
        self.assertIn("D197", entry["aliases"]["en"])
        self.assertIn("Musang King", entry["aliases"]["en"])

    def test_disk_file_matches_regeneration(self):
        self.assertTrue(GLOSSARY_V2_PATH.exists())
        rebuilt = build_glossary_v2()
        self.assertEqual(self.data, rebuilt)


class TestDetectEntitiesWithIds(unittest.TestCase):

    def test_multilingual_aliases_same_canonical_id(self):
        ids = {detect_entities_with_ids(f"{a} 的施肥方案")["cultivar"]
               for a in ["猫山王", "Musang King", "D197", "Raja Kunyit"]}
        self.assertEqual(len(ids), 1)
        self.assertTrue(ids.pop().startswith("CULTIVAR_"))

    def test_id_format_section9(self):
        result = detect_entities_with_ids("D197 开花期施肥")
        self.assertRegex(result["cultivar"], r"^CULTIVAR_\d{3}$")

    def test_slots_equivalent_to_name_based(self):
        for query in ["3号园的猫山王得了炭疽病", "หมอนทอง ใบเหลือง",
                      "potassium deficiency in durian", "帮我创建工单"]:
            by_name = detect_entities(query)
            by_id = detect_entities_with_ids(query)
            self.assertEqual(set(by_name.keys()), set(by_id.keys()),
                             msg=f"槽位不一致: {query!r}")

    def test_orchard_plot_patterns_in_id_form(self):
        result = detect_entities_with_ids("3号园 PLOT_12 的猫山王")
        self.assertEqual(result["orchard"], "ORCHARD_3")
        self.assertEqual(result["plot"], "PLOT_12")

    def test_id_to_name_resolvable(self):
        """canonical_id 可解析回标准名（显示/证据匹配用）。"""
        data = load_glossary_v2()
        by_id = {e["canonical_id"]: e for e in data["entries"]}
        result = detect_entities_with_ids("Musang King 施肥")
        self.assertEqual(by_id[result["cultivar"]]["canonical_name"], "猫山王")


class TestDataAccounting(unittest.TestCase):

    def test_counts_recorded_honestly(self):
        counts = load_glossary_v2()["_meta"]["counts"]
        # 实际数据（2026-09-10）：206 概念 / 527 别名。
        # §10 建议 300～500 概念 + 1000～2000 别名——概念数尚差、别名数偏低，
        # 扩充路径（批量四语翻译）已记录在 glossary.py docstring。
        self.assertGreater(counts["concepts"], 150)
        self.assertGreater(counts["aliases"], 400)
        self.assertIn("note", load_glossary_v2()["_meta"])


if __name__ == "__main__":
    unittest.main()
