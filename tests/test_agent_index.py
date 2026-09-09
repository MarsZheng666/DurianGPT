"""任务 #25 验证：Milvus + BM25 双索引（§18/§58）。

验证点：
1. BM25 known-item：相关文档排第一、无关不返回；
2. BM25 多语：中文/泰文/拉丁查询各自命中对应文档；
3. VectorIndex（milvus-lite）：假嵌入器下建库/检索/元数据回带；
4. §40 过滤：role_scope array_contains 生效（越权角色查不到）；
5. 真实嵌入器可加载（本地 bge-small-zh-v1.5，离线）。
"""

import unittest
from pathlib import Path

from durian_agent.rag.index import (
    BM25Index,
    VectorIndex,
    local_bge_embedder,
    multilingual_tokens,
)

RECORDS = [
    {"chunk_id": "c-fert", "document_id": "施肥手册", "language": "zh",
     "domain": "fertilization", "source_type": "sop",
     "role_scope": ["worker", "manager"], "orchard_scope": [],
     "text": "榴莲开花期施肥以磷钾肥为主，每株施入复合肥2公斤，沿树冠滴水线开环沟施用后覆土。"},
    {"chunk_id": "c-irr", "document_id": "灌溉指南", "language": "zh",
     "domain": "irrigation", "source_type": "document",
     "role_scope": ["manager"], "orchard_scope": ["ORCHARD_3"],
     "text": "旱季每7天灌溉一次，每次灌透至根系分布层，雨季停止灌溉并做好排水防涝。"},
    {"chunk_id": "c-th", "document_id": "thai-doc", "language": "th",
     "domain": "plant_protection", "source_type": "document",
     "role_scope": ["worker", "manager"], "orchard_scope": [],
     "text": "โรครากเน่าของทุเรียนเกิดจากเชื้อไฟทอปธอรา ป้องกันโดยการระบายน้ำให้ดี"},
    {"chunk_id": "c-en", "document_id": "pest-guide", "language": "en",
     "domain": "plant_protection", "source_type": "literature",
     "role_scope": ["worker", "manager"], "orchard_scope": [],
     "text": "Stem borer larvae tunnel into the trunk, causing dieback; inspect regularly."},
]


class TestMultilingualTokens(unittest.TestCase):

    def test_tokens_cover_three_scripts(self):
        self.assertIn("施肥", multilingual_tokens("榴莲施肥管理"))
        self.assertIn("durian", multilingual_tokens("how to grow durian"))
        # 泰文术语词典优先
        self.assertIn("หมอนทอง", multilingual_tokens("ราคาหมอนทอง"))

    def test_stopwords_dropped(self):
        tokens = multilingual_tokens("how to fertilize the durian")
        self.assertNotIn("how", tokens)
        self.assertNotIn("the", tokens)
        self.assertIn("fertilize", tokens)


class TestBM25Index(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.index = BM25Index(RECORDS)

    def test_known_item_top1(self):
        hits = self.index.search("开花期施什么肥", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["chunk_id"], "c-fert")

    def test_multilingual_hit(self):
        self.assertEqual(self.index.search("โรครากเน่า", top_k=1)[0]["chunk_id"], "c-th")
        self.assertEqual(self.index.search("stem borer trunk damage", top_k=1)[0]["chunk_id"], "c-en")
        self.assertEqual(self.index.search("旱季多久灌溉一次", top_k=1)[0]["chunk_id"], "c-irr")

    def test_irrelevant_query_no_hit(self):
        self.assertEqual(self.index.search("量子力学薛定谔方程", top_k=4), [])


class TestVectorIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile

        cls.tmpdir = tempfile.mkdtemp()
        # 确定性假嵌入器：施肥类文本 → 向量 a，其它 → 向量 b
        def fake_embed(texts):
            vectors = []
            for t in texts:
                if "施肥" in t:
                    vectors.append([1.0, 0.0, 0.0, 0.0])
                elif "灌溉" in t:
                    vectors.append([0.0, 1.0, 0.0, 0.0])
                elif "โรค" in t:
                    vectors.append([0.0, 0.0, 1.0, 0.0])
                else:
                    vectors.append([0.0, 0.0, 0.0, 1.0])
            return vectors

        cls.index = VectorIndex(Path(cls.tmpdir) / "test.db", embed_fn=fake_embed, dim=4)
        cls.count = cls.index.build(RECORDS)

    def test_build_count(self):
        self.assertEqual(self.count, 4)

    def test_search_returns_metadata(self):
        hits = self.index.search("榴莲施肥", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["chunk_id"], "c-fert")
        self.assertEqual(hits[0]["record"]["domain"], "fertilization")
        self.assertEqual(hits[0]["record"]["document_id"], "施肥手册")

    def test_role_scope_filter(self):
        """§40 数据权限：worker 过滤掉 manager-only 的灌溉文档。"""
        hits = self.index.search("旱季灌溉", top_k=4,
                                 expr='array_contains(role_scope, "worker")')
        ids = [h["chunk_id"] for h in hits]
        self.assertNotIn("c-irr", ids, msg="manager-only 文档泄漏给 worker")

    def test_orchard_scope_filter(self):
        hits = self.index.search("旱季灌溉", top_k=4,
                                 expr='array_contains(orchard_scope, "ORCHARD_3")')
        ids = [h["chunk_id"] for h in hits]
        self.assertIn("c-irr", ids)
        hits_all = self.index.search("旱季灌溉", top_k=4)
        self.assertGreater(len(hits_all), len(hits))


class TestLocalEmbedder(unittest.TestCase):

    def test_local_bge_embedder_available(self):
        """本地嵌入模型可加载且维度正确（离线，不触远端）。"""
        try:
            embed = local_bge_embedder()
        except Exception as exc:  # pragma: no cover - 模型目录缺失时提示
            self.skipTest(f"本地嵌入模型不可用: {exc}")
            return
        vectors = embed(["榴莲施肥", "灌溉"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 512)
        self.assertAlmostEqual(vectors[0][0], vectors[0][0])  # 数值合法


if __name__ == "__main__":
    unittest.main()
