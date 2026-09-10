"""任务 #57 验证：Token Budget（§34）。"""

import unittest

from durian_agent.memory.budget import (
    DEFAULT_BUDGET_RATIOS,
    TokenBudget,
    estimate_tokens,
)


class TestRatios(unittest.TestCase):

    def test_ratios_match_doc_section34(self):
        self.assertEqual(DEFAULT_BUDGET_RATIOS, {
            "system": 0.10, "summary": 0.15, "history": 0.25,
            "rag": 0.35, "query": 0.05, "output": 0.10,
        })
        self.assertAlmostEqual(sum(DEFAULT_BUDGET_RATIOS.values()), 1.0)

    def test_caps_from_total(self):
        budget = TokenBudget(total_tokens=8000)
        self.assertEqual(budget.cap("rag"), 2800)       # 35% 最大份额
        self.assertEqual(budget.cap("history"), 2000)
        self.assertEqual(budget.cap("system"), 800)
        self.assertEqual(budget.cap("summary"), 1200)
        self.assertEqual(budget.cap("query"), 400)      # 5% 最小
        self.assertEqual(budget.cap("output"), 800)

    def test_unknown_section_rejected(self):
        with self.assertRaises(KeyError):
            TokenBudget().cap("nope")


class TestEstimateTokens(unittest.TestCase):

    def test_cjk_one_per_char(self):
        self.assertEqual(estimate_tokens("榴莲施肥管理"), 6)

    def test_latin_four_per_token(self):
        self.assertEqual(estimate_tokens("abcdefgh"), 2)      # 8/4
        self.assertEqual(estimate_tokens("abc"), 1)           # 向上取整

    def test_mixed(self):
        # 4 CJK + (1空格 + 8 latin) = 4 + 9/4 向上取整 = 4 + 3
        self.assertEqual(estimate_tokens("榴莲施肥 abcdefgh"), 7)

    def test_empty(self):
        self.assertEqual(estimate_tokens(""), 0)


class TestFit(unittest.TestCase):

    def test_within_budget_untouched(self):
        budget = TokenBudget(total_tokens=8000)
        text = "短文本"
        self.assertEqual(budget.fit("query", text), text)

    def test_over_budget_truncated(self):
        budget = TokenBudget(total_tokens=1000)   # query cap = 50
        text = "问题" * 100                        # 200 chars
        fitted = budget.fit("query", text)
        self.assertLessEqual(len(fitted), 50)

    def test_sections_independent(self):
        budget = TokenBudget(total_tokens=1000)
        long_text = "x" * 4000
        self.assertLessEqual(len(budget.fit("rag", long_text)), 350)
        self.assertLessEqual(len(budget.fit("query", long_text)), 50)


class TestReport(unittest.TestCase):

    def test_report_flags_overuse(self):
        budget = TokenBudget(total_tokens=1000)
        report = budget.report({"query": "x" * 400, "system": "短"})
        self.assertTrue(report["query"]["over"])
        self.assertFalse(report["system"]["over"])
        self.assertEqual(report["query"]["cap"], 50)


if __name__ == "__main__":
    unittest.main()
