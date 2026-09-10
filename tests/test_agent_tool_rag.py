"""任务 #36 验证：AgricultureRagTool（§16/§17）。"""

import unittest

from durian_agent.tools import ToolContext, ToolRegistry
from durian_agent.tools.rag_tool import SPEC, register, search_for_api

HITS = {
    "hits": {
        "bm25_original": [
            {"chunk_id": "CHUNK_1", "score": 4.0,
             "text": "猫山王开花期施肥以磷钾肥为主，每株复合肥两公斤",
             "record": {"document_id": "施肥手册"}},
            {"chunk_id": "CHUNK_2", "score": 2.0,
             "text": "猫山王花期注意排水，忌积水烂根",
             "record": {"document_id": "栽培指南"}},
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


class TestAgricultureRagTool(unittest.TestCase):

    def _run(self, args, retriever_result=HITS, semantic=None, state=None):
        registry = ToolRegistry()
        register(registry, FakeRetriever(retriever_result), semantic=semantic)
        state = state if state is not None else {}
        ctx = ToolContext(state=state)
        out = registry.execute("agriculture_rag", args, ctx)
        return out, state

    def test_observation_with_chunk_annotations(self):
        out, _ = self._run({"query": "猫山王开花期施肥"})
        self.assertIn("[CHUNK_1]", out)
        self.assertIn("磷钾肥", out)

    def test_evidence_written_to_state(self):
        """证据回写 ctx.state（图 answer 节点引用生成用）。"""
        _, state = self._run({"query": "猫山王开花期施肥"})
        self.assertTrue(state["evidence_sufficient"])
        self.assertEqual({d["chunk_id"] for d in state["reranked_docs"]},
                         {"CHUNK_1", "CHUNK_2"})

    def test_no_result_observation(self):
        out, state = self._run({"query": "量子力学"}, retriever_result=None)
        self.assertIn("未检索到", out)
        self.assertFalse(state["evidence_sufficient"])

    def test_insufficient_evidence_noted(self):
        """单点证据 → 证据不足附注（不冒充充分）。"""
        single = {
            "hits": {"bm25_original": [HITS["hits"]["bm25_original"][0]]},
            "rankings": {"bm25_original": ["CHUNK_1"]},
        }
        out, state = self._run({"query": "猫山王开花期施肥"},
                               retriever_result=single)
        self.assertIn("证据可能不足", out)
        self.assertFalse(state["evidence_sufficient"])

    def test_query_required(self):
        out, _ = self._run({})
        self.assertIn("query 必填", out)

    def test_no_retriever_configured(self):
        registry = ToolRegistry()
        register(registry, None)
        out = registry.execute("agriculture_rag", {"query": "x"},
                               ToolContext())
        self.assertIn("不可用", out)

    def test_spec_description_mentions_evidence_duty(self):
        self.assertIn("必须先用它取证", SPEC.description)

    def test_search_for_api_contract(self):
        """§60 /api/rag/search 契约（#11 复用）：documents+evidence_sufficient。"""
        result = search_for_api(FakeRetriever(HITS), None, "猫山王开花期施肥")
        self.assertEqual(result["documents"][0]["chunk_id"], "CHUNK_1")
        self.assertEqual(result["documents"][0]["document_id"], "施肥手册")
        self.assertTrue(result["evidence_sufficient"])


if __name__ == "__main__":
    unittest.main()
