"""任务 #58 验证：生成质量指标（§54 五项，重点 Faithfulness/Abstention）。"""

import unittest

from durian_agent.evaluation.generation_metrics import (
    abstention_accuracy,
    citation_accuracy,
    evaluate_generation,
    evidence_coverage,
    is_abstention,
    llm_judge,
)
from durian_agent.llm import FakeLLM


class TestCitationAccuracy(unittest.TestCase):

    def test_valid_citations(self):
        self.assertEqual(
            citation_accuracy("x", ["c1", "c2"], ["c1", "c2", "c3"]), 1.0)

    def test_fabricated_citation_penalized(self):
        self.assertEqual(
            citation_accuracy("x", ["c1", "c9"], ["c1", "c2"]), 0.5)

    def test_no_citation(self):
        self.assertEqual(citation_accuracy("x", [], ["c1"]), 0.0)


class TestEvidenceCoverage(unittest.TestCase):

    def test_keyword_coverage(self):
        self.assertEqual(
            evidence_coverage(["波尔多液 防治", "排水"],
                              ["炭疽病用波尔多液防治", "注意排水"]), 1.0)

    def test_partial(self):
        self.assertEqual(
            evidence_coverage(["波尔多液", "修剪"],
                              ["用波尔多液防治"]), 0.5)


class TestAbstention(unittest.TestCase):

    def test_markers(self):
        self.assertTrue(is_abstention("知识库中暂无足够证据回答该问题。"))
        self.assertTrue(is_abstention("The knowledge base does not provide "
                                      "enough evidence."))
        self.assertFalse(is_abstention("波尔多液可防治炭疽病。"))

    def test_accuracy_both_directions(self):
        cases = [
            {"has_evidence": False, "answer": "暂无足够证据"},   # 应拒且拒 ✓
            {"has_evidence": True, "answer": "用波尔多液防治"},  # 有据且答 ✓
            {"has_evidence": False, "answer": "我觉得是多菌灵"},  # 应拒未拒 ✗
            {"has_evidence": True, "answer": "暂无足够证据"},     # 有据却拒 ✗
        ]
        self.assertEqual(abstention_accuracy(cases), 0.5)


class TestLLMJudge(unittest.TestCase):

    def test_score_parsed(self):
        fake = FakeLLM("0.8")
        score = llm_judge(fake, aspect="faithfulness",
                          question="q", reference="r",
                          evidence="e", answer="a")
        self.assertEqual(score, 0.8)
        self.assertIn("faithfulness", fake.calls[0]["user"])

    def test_invalid_score_none(self):
        self.assertIsNone(llm_judge(FakeLLM("无法评分"), aspect="x",
                                     question="q", reference="r",
                                     evidence="e", answer="a"))

    def test_judge_failure_none(self):
        class Broken(FakeLLM):
            def complete(self, system, user, *, temperature=0.0):
                raise RuntimeError("judge 不可用")
        self.assertIsNone(llm_judge(Broken(""), aspect="x", question="q",
                                    reference="r", evidence="e", answer="a"))


class TestEvaluateGeneration(unittest.TestCase):

    CASES = [{
        "question": "炭疽病怎么防治",
        "reference": "用波尔多液防治，注意排水",
        "evidence_texts": ["炭疽病发病初期用波尔多液防治，注意排水"],
        "provided_ids": ["c1"], "cited_ids": ["c1"],
        "gold_points": ["波尔多液", "排水"],
        "answer": "用波尔多液防治并注意排水 [c1]",
    }]

    def test_deterministic_metrics_without_llm(self):
        result = evaluate_generation(None, self.CASES)
        self.assertEqual(result["citation_accuracy"], 1.0)
        self.assertEqual(result["evidence_coverage"], 1.0)
        self.assertEqual(result["faithfulness"], 0.0)   # 无 judge 时无分
        self.assertEqual(result["answer_correctness"], 0.0)

    def test_with_judge(self):
        result = evaluate_generation(FakeLLM("0.9"), self.CASES)
        self.assertEqual(result["answer_correctness"], 0.9)
        self.assertEqual(result["faithfulness"], 0.9)


if __name__ == "__main__":
    unittest.main()
