"""任务 #1 验证：LangGraph 图拓扑（§44 19 节点 + §45 全边分支）。

验证点：
1. 拓扑：§44 的 19 个节点全部注册；
2. 边分支覆盖：
   - MUST_RAG happy path（检索→RRF→证据充分→带引用回答）
   - MUST_RAG 不足→改写→重试→仍不足→诚实拒答（fallback）
   - SIMPLE → simpleAgent 直答
   - COMPLEX_TASK → reactAgent（阶段三前的降级提示）
3. 状态字段按 §4 填充（route/semantic/retrieval_count/…）；
4. thread 隔离（checkpointer）。
"""

import unittest

from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM

EXPECTED_NODES = {
    "contextInit", "normalize", "semanticParse", "route", "ragQueryBuild",
    "retrieve", "rrf", "rerank", "evidenceCheck", "rewriteQuery",
    "simpleAgent", "reactAgent", "toolPolicy", "toolNode", "permissionCheck",
    "confirmation", "memoryCompress", "answer", "fallback",
}

HITS = {
    "hits": {
        "dense_canonical": [
            {"chunk_id": "c1", "score": 0.9, "text": "炭疽病用波尔多液防治",
             "record": {"document_id": "植保手册", "language": "zh"}},
        ],
        "bm25_original": [
            {"chunk_id": "c1", "score": 3.2, "text": "炭疽病用波尔多液防治",
             "record": {"document_id": "植保手册"}},
            {"chunk_id": "c2", "score": 1.1, "text": "雨季注意排水防涝",
             "record": {"document_id": "灌溉指南"}},
        ],
        "dense_original": [],
        "bm25_expanded": [],
    },
    "rankings": {"dense_canonical": ["c1"], "bm25_original": ["c1", "c2"],
                 "dense_original": [], "bm25_expanded": []},
    "queries": {"original": "炭疽病", "canonical": "炭疽病", "expanded": "炭疽病"},
}


class FakeRetriever:
    """四路召回替身：可控命中。"""

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def recall(self, query, *, top_k_each=10, expr=None):
        self.calls.append(query)
        return self.result


def make_graph(retriever_result=None, llm_response="依据证据回答。"):
    retriever = FakeRetriever(retriever_result)
    graph = DurianAgentGraph(
        llm=FakeLLM(llm_response), retriever=retriever, max_retrievals=2)
    return graph, retriever


class TestTopology(unittest.TestCase):

    def test_all_19_nodes_registered(self):
        graph, _ = make_graph()
        nodes = set(graph.graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
        self.assertEqual(nodes, EXPECTED_NODES)
        self.assertEqual(len(EXPECTED_NODES), 19)


class TestContextInit(unittest.TestCase):
    """任务 #9：ContextInit 节点（§3/§44：初始化权限、记忆、业务状态）。"""

    def setUp(self):
        graph, _ = make_graph()
        self.node = graph._context_init

    def test_defaults_filled_for_missing_keys(self):
        patch = self.node({"original_query": "你好", "thread_id": "t",
                           "user_id": "u", "role": "worker", "language": "zh"})
        # §4 安全默认值全部补齐
        self.assertEqual(patch["retrieval_count"], 0)
        self.assertEqual(patch["retry_count"], 0)
        self.assertEqual(patch["degrade_level"], 0)
        self.assertFalse(patch["evidence_sufficient"])
        self.assertEqual(patch["allowed_tools"], set())
        self.assertEqual(patch["allowed_knowledge_partitions"], set())
        self.assertEqual(patch["history_summary"], "")

    def test_existing_values_not_overwritten(self):
        patch = self.node({"original_query": "你好", "thread_id": "t",
                           "role": "manager", "retrieval_count": 3,
                           "evidence_sufficient": True})
        # 已有值不进 patch（LangGraph 语义下即保留原值）
        self.assertNotIn("role", patch)
        self.assertNotIn("retrieval_count", patch)
        self.assertNotIn("evidence_sufficient", patch)

    def test_minimal_privilege_default_role(self):
        patch = self.node({"original_query": "你好"})
        self.assertEqual(patch["role"], "worker")

    def test_human_message_appended(self):
        """每个 invoke 落一条 HumanMessage（messages 通道被 LangGraph
        预初始化为空列表，"not in state" 判不出首次——无条件追加）。"""
        patch = self.node({"original_query": "什么是榴莲坐果"})
        self.assertEqual(len(patch["messages"]), 1)
        self.assertEqual(patch["messages"][0].content, "什么是榴莲坐果")


class TestEdgeBranches(unittest.TestCase):

    def test_must_rag_happy_path(self):
        graph, retriever = make_graph(HITS)
        result = graph.invoke("炭疽病用什么农药防治，浓度多少倍", thread_id="t1")
        self.assertEqual(result["route"], "MUST_RAG")
        self.assertTrue(result["evidence_sufficient"])
        self.assertEqual(result["final_answer"], "依据证据回答。")
        # 引用返回（§31）
        self.assertTrue(result["rag_sources"])
        self.assertEqual(result["rag_sources"][0]["chunk_id"], "c1")
        # 证据进了生成 prompt
        self.assertEqual(result["retrieval_count"], 1)
        # 状态按 §4 填充
        self.assertIn("semantic", result)
        self.assertIn("search_queries", result)
        self.assertTrue(result["reranked_docs"])

    def test_must_rag_insufficient_retry_then_fallback(self):
        """空检索：改写重试到上限 → 诚实拒答（§29/§30）。"""
        graph, retriever = make_graph(None)   # 检索无命中
        result = graph.invoke("炭疽病用什么农药防治", thread_id="t2")
        self.assertEqual(result["route"], "MUST_RAG")
        self.assertFalse(result["evidence_sufficient"])
        self.assertEqual(result["retrieval_count"], 2)      # max_retrievals=2
        self.assertEqual(result["last_error"], "RAG_NO_RESULT")
        self.assertIn("暂无足够证据", result["final_answer"])
        # 重试确实换了查询（改写占位：第二轮用扩展查询）
        self.assertEqual(len(retriever.calls), 2)
        self.assertNotEqual(retriever.calls[0], retriever.calls[1])

    def test_simple_path(self):
        graph, retriever = make_graph(None, llm_response="坐果是受精后幼果形成的过程。")
        result = graph.invoke("什么是榴莲坐果", thread_id="t3")
        self.assertEqual(result["route"], "SIMPLE")
        self.assertEqual(result["final_answer"], "坐果是受精后幼果形成的过程。")

    def test_complex_task_degrades_gracefully(self):
        graph, _ = make_graph(None)
        result = graph.invoke("帮我创建一个巡检工单", thread_id="t4")
        self.assertEqual(result["route"], "COMPLEX_TASK")
        self.assertIn("第三阶段", result["final_answer"])   # reactAgent 占位降级


class TestThreadIsolation(unittest.TestCase):

    def test_two_threads_independent(self):
        from langgraph.checkpoint.memory import MemorySaver

        graph, _ = make_graph(None, llm_response="好的。")
        graph_with_cp = DurianAgentGraph(
            llm=FakeLLM("好的。"), retriever=FakeRetriever(None),
            checkpointer=MemorySaver())
        r1 = graph_with_cp.invoke("什么是榴莲坐果", thread_id="thread-a")
        r2 = graph_with_cp.invoke("为什么需要修剪", thread_id="thread-b")
        self.assertEqual(r1["original_query"], "什么是榴莲坐果")
        self.assertEqual(r2["original_query"], "为什么需要修剪")
        # 各自线程的消息互不掺杂
        self.assertEqual(len(r1["messages"]), 2)   # human + ai
        self.assertEqual(len(r2["messages"]), 2)
        graph.graph  # noqa: B018  原图仍可用


if __name__ == "__main__":
    unittest.main()
