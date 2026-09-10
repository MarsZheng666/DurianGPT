"""任务 #60 验证：多语言检索评估集（§55：1200+，语言×intent×difficulty）。"""

import unittest

from durian_agent.evaluation.dataset import (
    DIFFICULTIES,
    GOLD_PATH,
    coverage_report,
    load_gold_set,
)


class TestGoldSet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.entries = load_gold_set()

    def test_dataset_exists_and_size(self):
        self.assertTrue(GOLD_PATH.exists())
        self.assertGreaterEqual(len(self.entries), 1200)   # §55 目标

    def test_entry_format_section55(self):
        for entry in self.entries[:50]:
            for key in ("query", "language", "intent", "difficulty",
                        "gold_document_ids", "gold_chunk_ids"):
                self.assertIn(key, entry)
            self.assertIn(entry["language"], ("zh", "en", "th", "ms"))
            self.assertTrue(entry["query"])

    def test_four_language_coverage(self):
        report = coverage_report(self.entries)
        for lang in ("zh", "en", "th", "ms"):
            self.assertGreater(report["languages"].get(lang, 0), 50,
                               msg=f"{lang} 覆盖不足")

    def test_difficulty_dimensions(self):
        report = coverage_report(self.entries)
        used = set(report["difficulties"])
        self.assertTrue(used & {"Easy", "Alias", "Cross-language"},
                        msg=f"难度维度过窄: {used}")
        # 难度值限定 §55 六档（允许上游自带标签不在六档时如实保留）
        core = used & set(DIFFICULTIES)
        self.assertGreaterEqual(len(core), 3)

    def test_intent_coverage(self):
        report = coverage_report(self.entries)
        self.assertGreaterEqual(len(report["intents"]), 3)

    def test_limit_parameter(self):
        limited = load_gold_set(limit=100)
        self.assertEqual(len(limited), 100)


if __name__ == "__main__":
    unittest.main()
