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
        # min_score=0：本测试只验证排序与截取（阈值行为在 TestRerankThreshold）
        result = rerank_documents("炭疽病怎么防治", DOCS, fake_reranker,
                                  top_k=2, min_score=0.0)
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


class TestRerankThreshold(unittest.TestCase):
    """任务 #26：阈值机制（§28：低于阈值不进最终上下文）。"""

    def test_below_threshold_filtered(self):
        docs = [
            {"chunk_id": "c-good", "text": "炭疽病用波尔多液防治"},
            {"chunk_id": "c-bad", "text": "金枕品种介绍"},
        ]
        # c-bad 得 0.0（无主题词），低于默认阈值 0.3 → 被过滤
        result = rerank_documents("炭疽病怎么防治", docs, fake_reranker)
        self.assertEqual([d["chunk_id"] for d in result], ["c-good"])

    def test_all_below_threshold_returns_empty(self):
        docs = [{"chunk_id": "c", "text": "无关文本"}]
        result = rerank_documents("炭疽病怎么防治", docs, fake_reranker)
        self.assertEqual(result, [])

    def test_threshold_configurable(self):
        from durian_agent.rag.rerank import RERANK_THRESHOLD
        docs = [
            {"chunk_id": "c-mid", "text": "雨季排水防治"},        # fake 分 0.3（防治）
            {"chunk_id": "c-high", "text": "炭疽病防治要点"},     # fake 分 0.9
        ]
        strict = rerank_documents("炭疽病怎么防治", docs, fake_reranker,
                                  min_score=0.7)
        self.assertEqual([d["chunk_id"] for d in strict], ["c-high"])
        loose = rerank_documents("炭疽病怎么防治", docs, fake_reranker,
                                 min_score=0.3)
        self.assertEqual([d["chunk_id"] for d in loose], ["c-high", "c-mid"])
        self.assertEqual(RERANK_THRESHOLD, 0.3)

    def test_threshold_filters_drive_retry_chain(self):
        """阈值过滤后证据不足 → 图走改写重试链（与 #29 循环衔接）。"""
        class Retriever:
            def __init__(self):
                self.calls = 0

            def recall(self, q, *, top_k_each=10, expr=None):
                self.calls += 1
                return {
                    "hits": {"bm25_original": [
                        {"chunk_id": "c-weak", "score": 3.0,
                         "text": "榴莲品种介绍", "record": {}}]},
                    "rankings": {"bm25_original": ["c-weak"]},
                    "queries": {},
                }

        retriever = Retriever()
        graph = DurianAgentGraph(
            llm=FakeLLM("x"), retriever=retriever, reranker=fake_reranker)
        result = graph.invoke("炭疽病怎么防治", thread_id="t-thr")
        # 弱证据被阈值过滤 → 证据不足 → 重试 → 最终拒答
        self.assertFalse(result["evidence_sufficient"])
        self.assertGreaterEqual(retriever.calls, 2)
        self.assertIn("暂无足够证据", result["final_answer"])
