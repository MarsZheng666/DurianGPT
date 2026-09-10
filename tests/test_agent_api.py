"""任务 #4 验证：POST /api/chat 主接口（§60 契约）。

验证点：
1. 契约结构：answer/route/sources/pending_confirmation 四字段 + thread_id 扩展；
2. 三路由形态（MUST_RAG 拒答与带引用、SIMPLE、COMPLEX 降级）；
3. 网关联动：角色头透传、非法角色 400、缺省 thread 自动分配并回传；
4. 空 message 422（pydantic 校验）；
5. 离线模式（未注入任何依赖）契约仍完整。
"""

import unittest

from fastapi.testclient import TestClient

from durian_agent.api.app import create_app
from durian_agent.api.gateway import ConversationGateway
from durian_agent.graph import DurianAgentGraph
from durian_agent.llm import FakeLLM


class FakeRetriever:
    def __init__(self, result):
        self.result = result

    def recall(self, query, *, top_k_each=10, expr=None):
        return self.result


HITS = {
    "hits": {
        "dense_canonical": [
            {"chunk_id": "c1", "score": 0.9, "text": "炭疽病用波尔多液防治",
             "record": {"document_id": "植保手册"}},
        ],
        "bm25_original": [
            {"chunk_id": "c2", "score": 1.5, "text": "炭疽病雨季高发，注意排水通风",
             "record": {"document_id": "栽培指南"}},
        ],
        "dense_original": [], "bm25_expanded": [],
    },
    "rankings": {"dense_canonical": ["c1"], "bm25_original": ["c2"],
                 "dense_original": [], "bm25_expanded": []},
}


def make_client(**graph_kwargs):
    graph = DurianAgentGraph(
        llm=FakeLLM(graph_kwargs.pop("llm_response", "好的。")),
        retriever=FakeRetriever(graph_kwargs.pop("retriever_result", None)),
        **graph_kwargs,
    )
    return TestClient(create_app(graph=graph))


class TestChatContract(unittest.TestCase):

    def test_response_shape_simple(self):
        client = make_client(llm_response="坐果是受精后幼果形成的过程。")
        resp = client.post("/api/chat", json={
            "thread_id": "t-1", "message": "什么是榴莲坐果", "language": "zh"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        # §60 四字段 + thread_id 扩展
        self.assertEqual(
            set(body.keys()),
            {"answer", "route", "sources", "pending_confirmation", "thread_id"})
        self.assertEqual(body["route"], "SIMPLE")
        self.assertEqual(body["answer"], "坐果是受精后幼果形成的过程。")
        self.assertEqual(body["sources"], [])
        self.assertIsNone(body["pending_confirmation"])

    def test_must_rag_with_sources(self):
        client = make_client(retriever_result=HITS, llm_response="波尔多液可防治 [c1]。")
        resp = client.post("/api/chat", json={
            "thread_id": "t-2", "message": "炭疽病用什么农药防治"})
        body = resp.json()
        self.assertEqual(body["route"], "MUST_RAG")
        self.assertEqual(body["sources"][0]["chunk_id"], "c1")
        self.assertEqual(body["sources"][0]["document_id"], "植保手册")

    def test_must_rag_fallback_abstains(self):
        client = make_client(retriever_result=None)
        resp = client.post("/api/chat", json={
            "thread_id": "t-3", "message": "炭疽病用什么农药防治"})
        body = resp.json()
        self.assertEqual(body["route"], "MUST_RAG")
        self.assertIn("暂无足够证据", body["answer"])
        self.assertEqual(body["sources"], [])

    def test_complex_task_shape(self):
        client = make_client()
        resp = client.post("/api/chat", json={
            "thread_id": "t-4", "message": "帮我创建一个巡检工单"})
        body = resp.json()
        self.assertEqual(body["route"], "COMPLEX_TASK")
        self.assertIn("未能完成", body["answer"])


class TestGatewayIntegration(unittest.TestCase):

    def test_role_header_passthrough(self):
        client = make_client()
        resp = client.post("/api/chat",
                           json={"message": "你好"},
                           headers={"X-User-Id": "u-9", "X-Role": "manager"})
        self.assertEqual(resp.status_code, 200)

    def test_invalid_role_rejected_400(self):
        client = make_client()
        resp = client.post("/api/chat",
                           json={"message": "你好"},
                           headers={"X-Role": "hacker"})
        self.assertEqual(resp.status_code, 400)

    def test_thread_id_auto_assigned_and_returned(self):
        client = make_client()
        resp = client.post("/api/chat", json={"message": "你好"})
        body = resp.json()
        self.assertTrue(body["thread_id"].startswith("thread-"))

    def test_empty_message_422(self):
        client = make_client()
        resp = client.post("/api/chat", json={"message": ""})
        self.assertEqual(resp.status_code, 422)


class TestOfflineMode(unittest.TestCase):

    def test_no_deps_contract_complete(self):
        """未注入任何依赖（无 LLM/检索）契约仍完整可响应。"""
        client = TestClient(create_app())
        resp = client.post("/api/chat", json={
            "thread_id": "t-off", "message": "什么是榴莲坐果"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["route"], "SIMPLE")
        self.assertIn("需要 LLM", body["answer"])

    def test_health(self):
        client = TestClient(create_app())
        self.assertEqual(client.get("/health").json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()


class TestRagSearchEndpoint(unittest.TestCase):
    """任务 #11：POST /api/rag/search（§60 独立 RAG 契约）。"""

    def test_search_returns_documents_and_evidence(self):
        client = make_client(retriever_result=HITS)
        resp = client.post("/api/rag/search", json={
            "query": "炭疽病怎么防治", "top_k": 2})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("documents", body)
        self.assertIn("evidence_sufficient", body)
        self.assertLessEqual(len(body["documents"]), 2)
        self.assertEqual(body["documents"][0]["chunk_id"], "c1")

    def test_empty_query_422(self):
        client = make_client()
        resp = client.post("/api/rag/search", json={"query": ""})
        self.assertEqual(resp.status_code, 422)

    def test_no_retriever_409(self):
        client = TestClient(create_app())   # 离线模式无检索器
        resp = client.post("/api/rag/search", json={"query": "x"})
        self.assertEqual(resp.status_code, 409)
