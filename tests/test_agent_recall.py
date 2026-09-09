"""任务 #23 验证：四路召回（§25：Original/Canonical Dense + Original/Expanded BM25）。

验证点：
1. build_queries：实体标准名进 canonical、四语别名进 expanded；
   无实体时 canonical/expanded 退化为原查询；
2. 四路结构：rankings + hits 键齐备，rankings 直接喂 #27 RRF；
3. 跨语言桥接：中文语料文档可被英文别名查询经 canonical/expanded 命中；
4. §40 过滤表达式透传到两路 dense；
5. recall → weighted_rrf 端到端：正确文档排第一。
"""

import unittest
from pathlib import Path

from durian_agent.rag.index import BM25Index, VectorIndex
from durian_agent.rag.recall import ROUTES, FourWayRetriever, build_queries
from durian_agent.rag.rrf import weighted_rrf

RECORDS = [
    {"chunk_id": "c-mth", "document_id": "品种手册", "language": "zh",
     "domain": "variety", "source_type": "document",
     "role_scope": ["worker", "manager"], "orchard_scope": [],
     "text": "金枕果大肉厚，产量稳定，是泰国商业种植面积最大的品种，适合新果园起步。"},
    {"chunk_id": "c-fert", "document_id": "施肥手册", "language": "zh",
     "domain": "fertilization", "source_type": "sop",
     "role_scope": ["worker", "manager"], "orchard_scope": [],
     "text": "猫山王开花期施肥以磷钾肥为主，每株施入复合肥两公斤，沿滴水线开环沟施用。"},
    {"chunk_id": "c-irr", "document_id": "灌溉指南", "language": "zh",
     "domain": "irrigation", "source_type": "document",
     "role_scope": ["manager"], "orchard_scope": [],
     "text": "旱季每七天灌溉一次，每次灌透至根系分布层，雨季停灌并清理排水沟。"},
]


def fake_embed(texts):
    """确定性嵌入：只认中文内容词（模拟跨语言向量鸿沟——
    英文查询在中文语料上 dense 原路命不中，扩展路兜底）。
    5 维：猫山王/施肥/灌溉/金枕/其它，互不碰撞。"""
    vectors = []
    for t in texts:
        if "猫山王" in t:
            vectors.append([1.0, 0.0, 0.0, 0.0, 0.0])
        elif "施肥" in t:
            vectors.append([0.0, 1.0, 0.0, 0.0, 0.0])
        elif "灌溉" in t:
            vectors.append([0.0, 0.0, 1.0, 0.0, 0.0])
        elif "金枕" in t:
            vectors.append([0.0, 0.0, 0.0, 1.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 0.0, 0.0, 1.0])
    return vectors


class TestBuildQueries(unittest.TestCase):

    def test_canonical_and_expanded(self):
        queries = build_queries("Musang King 开花期施肥")
        self.assertIn("猫山王", queries["canonical"])
        # 扩展含多语别名
        for alias in ("Musang King", "D197", "Raja Kunyit"):
            self.assertIn(alias, queries["expanded"])
        self.assertIn("猫山王", queries["expanded"])

    def test_no_entity_degenerates_to_original(self):
        queries = build_queries("什么时候采收合适")
        self.assertEqual(queries["canonical"], queries["original"])
        self.assertEqual(queries["expanded"], queries["original"])

    def test_thai_alias_canonicalized(self):
        queries = build_queries("หมอนทอง ราคา")
        self.assertIn("金枕", queries["canonical"])


class TestFourWayRecall(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile

        cls.tmpdir = tempfile.mkdtemp()
        cls.retriever = FourWayRetriever(
            vector_index=VectorIndex(Path(cls.tmpdir) / "t.db",
                                     embed_fn=fake_embed, dim=5),
            bm25_index=BM25Index(RECORDS),
        )
        cls.retriever.vector_index.build(RECORDS)

    def test_four_routes_structure(self):
        result = self.retriever.recall("猫山王怎么施肥", top_k_each=5)
        self.assertEqual(tuple(result["rankings"].keys()), ROUTES)
        self.assertEqual(tuple(result["hits"].keys()), ROUTES)
        for route in ROUTES:
            self.assertLessEqual(len(result["rankings"][route]), 5)
            self.assertEqual(len(result["rankings"][route]), len(result["hits"][route]))

    def test_cross_lingual_bridge_via_canonical(self):
        """中文语料被英文别名查询命中（canonical dense / expanded bm25 兜底）。"""
        result = self.retriever.recall("Musang King fertilization", top_k_each=5)
        self.assertIn("c-fert", result["rankings"]["dense_canonical"])
        self.assertIn("c-fert", result["rankings"]["bm25_expanded"])
        # 原查询（英文）在中文语料上 dense 原路未必命中——这正是扩展路的意义
        self.assertNotIn("c-fert", result["rankings"]["dense_original"])

    def test_original_bm25_known_item(self):
        result = self.retriever.recall("旱季多久灌溉一次", top_k_each=3)
        self.assertEqual(result["rankings"]["bm25_original"][0], "c-irr")

    def test_expr_filter_applies_to_dense_only(self):
        result = self.retriever.recall(
            "旱季灌溉", top_k_each=5, expr='array_contains(role_scope, "worker")')
        self.assertNotIn("c-irr", result["rankings"]["dense_original"])
        self.assertNotIn("c-irr", result["rankings"]["dense_canonical"])
        # BM25 路不做权限过滤（过滤在 RRF 之后统一执行是 #47 的取舍，
        # 此处断言其行为：BM25 仍可召回，供上层按 scope 决策）
        self.assertIn("c-irr", result["rankings"]["bm25_original"])

    def test_recall_to_rrf_end_to_end(self):
        result = self.retriever.recall("Musang King fertilization", top_k_each=5)
        fused = weighted_rrf(result["rankings"])
        self.assertTrue(fused)
        self.assertEqual(fused[0]["chunk_id"], "c-fert")
        self.assertEqual(fused[0]["rrf_rank"], 1)
        # 融合结果含来源追溯
        self.assertIn("dense_canonical", fused[0]["sources"])


if __name__ == "__main__":
    unittest.main()
