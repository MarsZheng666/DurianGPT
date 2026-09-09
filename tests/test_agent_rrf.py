"""任务 #27 验证：Weighted RRF（§26：score=Σ wi/(k+rank_i)）。

验证点：
1. 公式手算对照（单一文档、多路贡献）；
2. 多路共现在前（RRF 核心性质：被多路召回的文档融合分更高）；
3. 单路缺席贡献 0；未召回任何路的文档不出现；
4. rrf_rank 从 1 连续编号；sources 记录各路名次（§52 Trace）；
5. 权重可配置；k 可配置。
"""

import unittest

from durian_agent.rag.rrf import DEFAULT_K, DEFAULT_RRF_WEIGHTS, weighted_rrf


class TestWeightedRRF(unittest.TestCase):

    def test_formula_single_route(self):
        result = weighted_rrf({"dense_original": ["a", "b"]})
        # 只有一路时：score = w / (k + rank)
        self.assertAlmostEqual(result[0]["score"],
                               0.20 / (DEFAULT_K + 1))
        self.assertAlmostEqual(result[1]["score"],
                               0.20 / (DEFAULT_K + 2))

    def test_formula_multi_route_hand_computed(self):
        rankings = {
            "dense_canonical": ["x", "y"],     # x: rank1, y: rank2
            "bm25_expanded": ["y", "x"],       # y: rank1, x: rank2
        }
        result = {r["chunk_id"]: r for r in weighted_rrf(rankings)}
        expect_x = 0.30 / (DEFAULT_K + 1) + 0.35 / (DEFAULT_K + 2)
        expect_y = 0.30 / (DEFAULT_K + 2) + 0.35 / (DEFAULT_K + 1)
        self.assertAlmostEqual(result["x"]["score"], expect_x)
        self.assertAlmostEqual(result["y"]["score"], expect_y)
        # y 的 Expanded BM25 权重高 → y 排前
        self.assertEqual(result["y"]["rrf_rank"], 1)
        self.assertEqual(result["x"]["rrf_rank"], 2)

    def test_multi_route_presence_wins(self):
        """两路都召回的文档 > 只被一路召回的文档（即使单路名次靠后）。"""
        rankings = {
            "dense_original": ["solo", "shared"],
            "bm25_original": ["shared"],
        }
        result = weighted_rrf(rankings)
        self.assertEqual(result[0]["chunk_id"], "shared")

    def test_absent_route_contributes_zero(self):
        result = {r["chunk_id"]: r for r in weighted_rrf(
            {"dense_original": ["a"], "bm25_expanded": ["b"]})}
        self.assertAlmostEqual(result["a"]["score"], 0.20 / (DEFAULT_K + 1))
        self.assertAlmostEqual(result["b"]["score"], 0.35 / (DEFAULT_K + 1))

    def test_unranked_doc_absent_and_rank_sequential(self):
        result = weighted_rrf({"dense_original": ["a", "b"], "bm25_original": ["b"]})
        ids = [r["chunk_id"] for r in result]
        self.assertEqual(set(ids), {"a", "b"})
        self.assertEqual([r["rrf_rank"] for r in result], [1, 2])

    def test_sources_trace(self):
        result = {r["chunk_id"]: r for r in weighted_rrf(
            {"dense_canonical": ["x", "y"], "bm25_expanded": ["y", "x"]})}
        self.assertEqual(result["x"]["sources"], {"dense_canonical": 1, "bm25_expanded": 2})
        self.assertEqual(result["y"]["sources"], {"dense_canonical": 2, "bm25_expanded": 1})

    def test_custom_weights_and_k(self):
        result = weighted_rrf({"dense_original": ["a"]},
                              weights={"dense_original": 1.0}, k=1)
        self.assertAlmostEqual(result[0]["score"], 1.0 / 2)

    def test_unknown_route_zero_weight(self):
        """不在权重表里的路不参与融合（防拼写错误静默污染）。"""
        result = weighted_rrf({"typo_route": ["a"]})
        self.assertEqual(result, [])

    def test_default_weights_match_doc(self):
        self.assertEqual(DEFAULT_RRF_WEIGHTS, {
            "dense_original": 0.20, "dense_canonical": 0.30,
            "bm25_original": 0.15, "bm25_expanded": 0.35,
        })

    def test_top_n_truncation(self):
        result = weighted_rrf({"dense_original": ["a", "b", "c"]}, top_n=2)
        self.assertEqual([r["chunk_id"] for r in result], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
