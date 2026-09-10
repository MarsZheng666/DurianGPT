"""任务 #59 验证：全链路 Trace（§52 15 字段）。"""

import unittest

from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM
from durian_agent.observability import (
    TraceCollector,
    record_from_state,
    timed_invoke,
)


class SmartLLM(FakeLLM):
    """语义解析走规则层；ReAct 按脚本。"""

    def __init__(self, responses):
        super().__init__("")
        self.responses = list(responses)
        self.react_calls = 0

    def complete(self, system, user, *, temperature=0.0):
        if "Respond with EXACTLY ONE JSON" in system:
            index = min(self.react_calls, len(self.responses) - 1)
            self.react_calls += 1
            return self.responses[index]
        return "不是JSON"


HITS = {
    "hits": {
        "bm25_original": [
            {"chunk_id": "CHUNK_1", "score": 4.0,
             "text": "炭疽病发病初期用波尔多液防治，注意排水",
             "record": {"document_id": "灌溉手册"}},
            {"chunk_id": "CHUNK_2", "score": 2.0,
             "text": "炭疽病雨季高发，及时清园减少病源", "record": {}},
        ],
        "dense_canonical": [], "dense_original": [], "bm25_expanded": [],
    },
    "rankings": {"bm25_original": ["CHUNK_1", "CHUNK_2"]},
    "queries": {},
}


class FakeRetriever:
    def __init__(self, result):
        self.result = result

    def recall(self, query, *, top_k_each=10, expr=None):
        return self.result


class TestTraceRecord(unittest.TestCase):

    def _run(self, responses, query, **invoke_kwargs):
        graph = DurianAgentGraph(
            llm=SmartLLM(responses), retriever=FakeRetriever(HITS))
        return graph, timed_invoke(graph, query, **invoke_kwargs)

    def test_fifteen_fields_present(self):
        _, result = self._run(
            ['{"final_answer": "湿度42%高于阈值，不需灌水 [CHUNK_1]"}'],
            "根据未来三天天气和当前土壤湿度，判断3号园猫山王今天要不要灌水",
            thread_id="t-trace", user_id="u-9", tenant_id="IOI",
            role="manager")
        graph = _  # noqa
        # 取图实例的 last_trace —— 重新构造一次拿 trace
        graph2, result2 = self._run(
            ['{"final_answer": "不需灌水 [CHUNK_1]"}'],
            "根据天气和当前土壤湿度判断要不要灌水",
            thread_id="t-trace", user_id="u-9", tenant_id="IOI",
            role="manager")
        trace = graph2.last_trace
        data = trace.to_dict()
        # §52 15 项字段全部在场且有值
        for key in ("trace_id", "thread_id", "user_id", "tenant_id", "route",
                    "intent", "tool_sequence", "retrieval_latency_ms",
                    "retrieval_sources", "rrf_ranks", "reranker_scores",
                    "evidence_decision", "token_input", "token_output",
                    "memory_compressions", "fallback_count", "final_success"):
            self.assertIn(key, data, msg=f"缺字段: {key}")
        self.assertEqual(trace.route, "COMPLEX_TASK")
        self.assertTrue(trace.final_success)
        self.assertGreaterEqual(trace.retrieval_latency_ms, 0.0)
        self.assertEqual(trace.evidence_decision, "sufficient")
        self.assertIn("agriculture_rag", trace.tool_sequence)  # 强制取证留痕

    def test_must_rag_path_trace(self):
        graph = DurianAgentGraph(
            llm=FakeLLM("波尔多液防治 [CHUNK_1]。"),
            retriever=FakeRetriever(HITS))
        timed_invoke(graph, "炭疽病用什么农药防治", thread_id="t-rag")
        trace = graph.last_trace
        self.assertEqual(trace.route, "MUST_RAG")
        self.assertEqual(trace.evidence_decision, "sufficient")
        self.assertIn("bm25_original", trace.retrieval_sources)
        self.assertEqual(trace.rrf_ranks.get("CHUNK_1"), 1)

    def test_failure_trace(self):
        graph = DurianAgentGraph(llm=FakeLLM("x"), retriever=FakeRetriever(None))
        timed_invoke(graph, "炭疽病用什么农药防治", thread_id="t-fail")
        trace = graph.last_trace
        self.assertFalse(trace.final_success)
        self.assertEqual(trace.last_error, "RAG_NO_RESULT")
        self.assertGreaterEqual(trace.fallback_count, 1)

    def test_collector_jsonl(self):
        graph = DurianAgentGraph(llm=FakeLLM("好的。"), retriever=None)
        timed_invoke(graph, "什么是榴莲坐果", thread_id="t-c")
        collector = TraceCollector()
        collector.add(graph.last_trace)
        collector.add(record_from_state({"thread_id": "x"}, latency_ms=1))
        lines = collector.to_jsonl().strip().split("\n")
        self.assertEqual(len(lines), 2)
        self.assertIn("t-c", lines[0])


if __name__ == "__main__":
    unittest.main()
