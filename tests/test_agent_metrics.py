"""任务 #61 验证：RAG 离线指标（§53：四语种分别统计）。"""

import unittest

from durian_agent.evaluation.metrics import (
    evaluate_dataset,
    evaluate_query,
    hit_at_k,
    mrr,
    ndcg,
    recall_at_k,
)


class TestMetrics(unittest.TestCase):

    def test_hit_and_recall(self):
        ranked = ["a", "b", "c", "d"]
        gold = ["b", "d"]
        self.assertFalse(hit_at_k(ranked, gold, 1))
        self.assertTrue(hit_at_k(ranked, gold, 3))
        self.assertEqual(recall_at_k(ranked, gold, 4), 1.0)
        self.assertEqual(recall_at_k(ranked, gold, 2), 0.5)   # b 命中

    def test_mrr(self):
        self.assertEqual(mrr(["x", "a", "b"], ["a"]), 0.5)
        self.assertEqual(mrr(["a"], ["a"]), 1.0)
        self.assertEqual(mrr(["x", "y"], ["a"]), 0.0)

    def test_ndcg_hand_computed(self):
        # gold={a,b}，ranked 前3命中 a(rank2), b(rank3)
        # dcg = 1/log2(3) + 1/log2(4) ≈ 0.6309 + 0.5 = 1.1309
        # idcg（两个理想位）= 1/log2(2) + 1/log2(3) = 1 + 0.6309
        value = ndcg(["x", "a", "b"], ["a", "b"], k=3)
        expected = (1 / 1.58496 + 1 / 2.0) / (1 / 1.0 + 1 / 1.58496)
        self.assertAlmostEqual(value, expected, places=4)

    def test_perfect_ranking(self):
        result = evaluate_query(["a", "b"], ["a", "b"], k=2)
        self.assertEqual(result["hit@1"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["ndcg"], 1.0)

    def test_no_gold(self):
        result = evaluate_query(["a"], [], k=2)
        self.assertEqual(result["recall@2"], 0.0)


class TestDatasetEvaluation(unittest.TestCase):

    DATASET = [
        {"query": "zh 查询一", "language": "zh", "gold_ids": ["a"]},
        {"query": "zh 查询二", "language": "zh", "gold_ids": ["z"]},   # 不命中
        {"query": "en query", "language": "en", "gold_ids": ["b"]},
        {"query": "th query", "language": "th", "gold_ids": ["c"]},
        {"query": "ms query", "language": "ms", "gold_ids": ["d"]},
    ]

    @staticmethod
    def search(query):
        # 确定性检索：zh 首条命中，其余语言按查询尾部字符路由
        if "zh" in query:
            return ["a", "x"]
        if "en" in query:
            return ["x", "b"]
        if "th" in query:
            return ["c"]
        return ["d"]

    def test_per_language_report(self):
        report = evaluate_dataset(self.DATASET, self.search, k=2)
        # 四语种分别统计（§53）
        for lang in ("zh", "en", "th", "ms"):
            self.assertIn(lang, report)
        self.assertEqual(report["__count__"], {"zh": 2, "en": 1,
                                               "th": 1, "ms": 1})
        # zh：一命中一未命中 → hit@1 = 0.5
        self.assertEqual(report["zh"]["hit@1"], 0.5)
        # en：rank2 命中 → mrr = 0.5
        self.assertEqual(report["en"]["mrr"], 0.5)
        # th/ms：首位命中
        self.assertEqual(report["th"]["hit@1"], 1.0)
        self.assertEqual(report["ms"]["mrr"], 1.0)
        # overall 存在
        self.assertIn("overall", report)


if __name__ == "__main__":
    unittest.main()
