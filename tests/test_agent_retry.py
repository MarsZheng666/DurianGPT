"""任务 #29 验证：Evidence Retry 循环（§29/§63 伪代码）。

§63：
    while (retrievalCount < MAX_RETRIEVAL):
        ragSearch(...)
        if sufficient: break
        rewrite(...); retrievalCount++
    if !sufficient: abstain()

验证点：
1. 改写救回：首轮检索空 → 改写重查命中 → 正常带引用回答（循环的核心价值）；
2. 重试耗尽：始终空 → 拒答，禁退化为自由回答；
3. 计数簿记：retrieval_count/retry_count 与轮次一致；
4. max_retrievals 可配置（2~3，§29 建议区间）。
"""

import unittest

from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM

HITS_SECOND_ROUND = {
    "hits": {
        "bm25_original": [
            {"chunk_id": "c1", "score": 4.0, "text": "炭疽病发病初期用波尔多液防治",
             "record": {"document_id": "植保手册"}},
            {"chunk_id": "c2", "score": 2.0, "text": "炭疽病雨季高发，注意排水通风",
             "record": {"document_id": "栽培指南"}},
        ],
        "dense_canonical": [], "dense_original": [], "bm25_expanded": [],
    },
    "rankings": {"bm25_original": ["c1", "c2"], "dense_canonical": [],
                 "dense_original": [], "bm25_expanded": []},
    "queries": {},
}


class FlakyRetriever:
    """首轮空、之后命中的检索替身（模拟改写后换词命中）。"""

    def __init__(self, results):
        self.results = results   # 每次调用依次取用；越界取最后一个
        self.calls = []

    def recall(self, query, *, top_k_each=10, expr=None):
        self.calls.append(query)
        index = min(len(self.calls) - 1, len(self.results) - 1)
        return self.results[index]


class TestEvidenceRetryLoop(unittest.TestCase):

    def test_retry_rescues_after_rewrite(self):
        """首轮空 → 改写重查命中 → 正常回答（§63 循环的核心价值）。"""
        retriever = FlakyRetriever([None, HITS_SECOND_ROUND])
        graph = DurianAgentGraph(
            llm=FakeLLM("波尔多液可防治炭疽病。"), retriever=retriever)
        result = graph.invoke("炭疽病用什么药防治", thread_id="t-rescue")
        self.assertTrue(result["evidence_sufficient"])
        self.assertEqual(result["retrieval_count"], 2)
        self.assertEqual(result["retry_count"], 1)
        # 两轮用了不同的查询（第二轮是改写结果）
        self.assertEqual(len(retriever.calls), 2)
        self.assertNotEqual(retriever.calls[0], retriever.calls[1])
        # 最终带引用
        self.assertTrue(result["rag_sources"])

    def test_retry_exhausts_then_abstains(self):
        retriever = FlakyRetriever([None, None, None])
        graph = DurianAgentGraph(
            llm=FakeLLM("x"), retriever=retriever, max_retrievals=2)
        result = graph.invoke("炭疽病用什么药防治", thread_id="t-exhaust")
        self.assertFalse(result["evidence_sufficient"])
        self.assertEqual(result["retrieval_count"], 2)   # 上限即停
        self.assertEqual(len(retriever.calls), 2)
        self.assertIn("暂无足够证据", result["final_answer"])
        self.assertEqual(result["last_error"], "RAG_NO_RESULT")

    def test_no_retry_when_first_round_sufficient(self):
        retriever = FlakyRetriever([HITS_SECOND_ROUND])
        graph = DurianAgentGraph(
            llm=FakeLLM("波尔多液。"), retriever=retriever)
        result = graph.invoke("炭疽病用什么药防治", thread_id="t-direct")
        self.assertEqual(result["retrieval_count"], 1)
        self.assertEqual(result.get("retry_count", 0), 0)
        self.assertEqual(len(retriever.calls), 1)

    def test_max_retrievals_configurable_to_three(self):
        retriever = FlakyRetriever([None, None, None])
        graph = DurianAgentGraph(
            llm=FakeLLM("x"), retriever=retriever, max_retrievals=3)
        result = graph.invoke("炭疽病用什么药防治", thread_id="t-max3")
        self.assertEqual(result["retrieval_count"], 3)
        self.assertIn("暂无足够证据", result["final_answer"])


if __name__ == "__main__":
    unittest.main()
