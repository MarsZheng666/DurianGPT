"""任务 #32 验证：Cross Encoder Reranker（§27：RRF Top30-50 → Top3-5）。

验证点：
1. rerank_documents：按分数重排、Top-K 截取、附 rerank_score、不改入参；
2. 图 rerank 节点：注入 reranker 后生效（证据顺序按 rerank 分）；
3. 未注入时透传 RRF 序（向后兼容）；
4. 本地 bge-reranker-v2-m3 真实判别力（相关>0.5，无关<0.1）。
"""

import unittest

from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM
from durian_agent.rag.rerank import rerank_documents

DOCS = [
    {"chunk_id": "c-fert", "text": "开花期施肥以磷钾肥为主，沿滴水线开沟施用"},
    {"chunk_id": "c-dis", "text": "炭疽病发病初期用波尔多液防治，雨季注意排水"},
    {"chunk_id": "c-var", "text": "金枕果大肉厚，泰国主栽品种，商业价值高"},
]


def fake_reranker(query, texts):
    """确定性替身：与查询主题词重叠越多分越高。"""
    scores = []
    for text in texts:
        score = 0.0
        for word in ("炭疽", "病", "防治"):
            if word in query:
                score += 0.3 if word in text else 0.0
        scores.append(round(score, 2))
    return scores


class TestRerankDocuments(unittest.TestCase):

    def test_rerank_orders_and_truncates(self):
        result = rerank_documents("炭疽病怎么防治", DOCS, fake_reranker, top_k=2)
        self.assertEqual([d["chunk_id"] for d in result], ["c-dis", "c-fert"])
        # 附分数且降序
        self.assertEqual(result[0]["rerank_score"], 0.9)
        self.assertGreaterEqual(result[0]["rerank_score"], result[1]["rerank_score"])
        # 不改入参
        self.assertNotIn("rerank_score", DOCS[0])

    def test_empty_docs(self):
        self.assertEqual(rerank_documents("q", [], fake_reranker), [])

    def test_top_k_default_five(self):
        many = [{"chunk_id": f"c{i}", "text": "文本"} for i in range(10)]
        result = rerank_documents("q", many, lambda q, ts: [1.0] * len(ts))
        self.assertEqual(len(result), 5)


class TestGraphRerankNode(unittest.TestCase):

    @staticmethod
    def _hits():
        return {
            "hits": {
                "bm25_original": [
                    {"chunk_id": "c-fert", "score": 3.0, "text": "开花期施肥以磷钾肥为主",
                     "record": {"document_id": "施肥手册"}},
                    {"chunk_id": "c-dis", "score": 2.0, "text": "炭疽病发病初期用波尔多液防治",
                     "record": {"document_id": "植保手册"}},
                ],
                "dense_canonical": [], "dense_original": [], "bm25_expanded": [],
            },
            "rankings": {"bm25_original": ["c-fert", "c-dis"], "dense_canonical": [],
                         "dense_original": [], "bm25_expanded": []},
            "queries": {},
        }

    def test_reranker_reorders_evidence(self):
        class Retriever:
            def __init__(self, result):
                self.result = result

            def recall(self, q, *, top_k_each=10, expr=None):
                return self.result

        retriever = Retriever(self._hits())
        graph = DurianAgentGraph(
            llm=FakeLLM("波尔多液。"), retriever=retriever, reranker=fake_reranker)
        result = graph.invoke("炭疽病怎么防治", thread_id="t-rr")
        # rerank 后炭疽病文档排第一（RRF 序里它本是第二）
        self.assertEqual(result["reranked_docs"][0]["chunk_id"], "c-dis")
        self.assertIn("rerank_score", result["reranked_docs"][0])

    def test_no_reranker_passthrough(self):
        class Retriever:
            def __init__(self, result):
                self.result = result

            def recall(self, q, *, top_k_each=10, expr=None):
                return self.result

        retriever = Retriever(self._hits())
        graph = DurianAgentGraph(llm=FakeLLM("x"), retriever=retriever)
        result = graph.invoke("炭疽病怎么防治", thread_id="t-norr")
        # 未注入：RRF 序保持（施肥文档在 bm25 里 rank 更靠前）
        self.assertEqual(result["reranked_docs"][0]["chunk_id"], "c-fert")


class TestLocalRerankerReal(unittest.TestCase):
    """本地 bge-reranker-v2-m3 真实判别力（约 8s 加载，服务进程内一次）。"""

    def test_real_model_discrimination(self):
        try:
            from durian_agent.rag.rerank import local_bge_reranker
            rerank = local_bge_reranker()
        except FileNotFoundError:
            self.skipTest("本地重排模型不存在")
        scores = rerank("榴莲炭疽病怎么防治", [
            "炭疽病发病初期用波尔多液防治，雨季注意排水。",
            "金枕果大肉厚产量稳定，适合商业种植。",
        ])
        self.assertGreater(scores[0], 0.5)   # 相关
        self.assertLess(scores[1], 0.1)      # 无关


if __name__ == "__main__":
    unittest.main()
